"""2D replay data: player positions per round, mapped onto the radar image.

Positions aren't part of the normal ingest (they'd make every parse slower and every cache bigger
for a feature only the web app uses). The first time a round of a demo is replayed, the original
.dem is read once more for position snapshots 8 times a second (every 8 ticks at 64 tick), plus
smokes and molotov fires, and cached next to the demo's other tables (~2 MB per match).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import polars as pl

from ..analysis import live_rounds, weapon_label
from ..demo_store import DemoStore
from ..errors import AnalystError
from ..sides import norm_side

SNAPSHOT_EVERY = 8  # ticks: 8 frames a second at 64 tick
REPLAY_FILES = ("replay_positions", "replay_smokes", "replay_fires")
POSITION_COLS = ["tick", "steamid", "name", "side", "x", "y", "z", "yaw", "health", "active_weapon",
                 "has_bomb", "is_defusing", "flash_duration"]
# awpy-data release to fall back to when GitHub's "latest release" API can't be reached.
FALLBACK_DATA_VERSION = int(os.getenv("CS2_AWPY_DATA_VERSION", "2000935"))


def _replay_dir(store: DemoStore, demo_id: str) -> Path:
    return store.cache_dir / demo_id


def ensure_replay_cache(store: DemoStore, demo_id: str) -> None:
    d = _replay_dir(store, demo_id)
    if all((d / f"{n}.parquet").exists() for n in REPLAY_FILES):
        return
    source = Path(store.meta(demo_id).get("source") or "")
    if not source.is_file():
        raise AnalystError(f"The replay needs the original demo file, which is no longer at '{source}'. "
                           "Put it back (or ingest it again from its new location).")
    from awpy import Demo

    demo = Demo(source)
    snaps = demo.snapshots(every=SNAPSHOT_EVERY)
    keep = [c for c in POSITION_COLS if c in snaps.columns]
    pos = snaps.select(keep).filter(pl.col("steamid") != 0).with_columns(
        pl.col("side").map_elements(norm_side, return_dtype=pl.Utf8)
    )
    demo.load("smokes", "fires")
    pos.write_parquet(d / "replay_positions.parquet")
    demo.smokes.select("start_tick", "end_tick", "x", "y").write_parquet(d / "replay_smokes.parquet")
    demo.fires.select("start_tick", "end_tick", "x", "y").write_parquet(d / "replay_fires.parquet")
    del demo


@functools.cache
def map_transform(map_name: str) -> dict[str, float]:
    """World -> radar pixel transform for a map (radar images are 1024x1024)."""
    import awpy.data as D

    try:
        table = D.map_data()
    except Exception:
        table = D.map_data(version=FALLBACK_DATA_VERSION)
    if map_name not in table:
        raise AnalystError(f"No radar data for map '{map_name}'.")
    return table[map_name]


def radar_file(map_name: str) -> Path:
    import awpy.data as D

    try:
        return D.radar_path(map_name)
    except FileNotFoundError:
        raise AnalystError(f"No radar image for map '{map_name}'.") from None
    except Exception:
        return D.radar_path(map_name, version=FALLBACK_DATA_VERSION)


def to_radar(x: float, y: float, t: dict[str, float]) -> tuple[float, float]:
    return round((x - t["pos_x"]) / t["scale"], 1), round((t["pos_y"] - y) / t["scale"], 1)


def round_replay(store: DemoStore, demo_id: str, round_num: int, transform: dict[str, float]) -> dict[str, Any]:
    """Everything the 2D replay draws for one round, in radar pixels, frame-aligned."""
    ensure_replay_cache(store, demo_id)
    d = _replay_dir(store, demo_id)
    meta = store.meta(demo_id)
    tick_rate = meta["tick_rate"]
    rounds = live_rounds(store.table(demo_id, "rounds"))
    r = rounds.filter(pl.col("round_num") == round_num)
    if r.is_empty():
        raise AnalystError(f"Round {round_num} does not exist.")
    r = r.row(0, named=True)
    start = r["freeze_end_tick"]
    end = r["end_tick"] + int(3 * tick_rate)  # a few seconds after the round is decided

    pos = pl.read_parquet(d / "replay_positions.parquet").filter(pl.col("tick").is_between(start, end))
    ticks = sorted(set(pos["tick"].to_list()))
    index = {t: i for i, t in enumerate(ticks)}
    roster = store.table(demo_id, "round_players").filter(pl.col("round_num") == round_num)
    players = []
    for p in roster.sort(["side", "name"]).iter_rows(named=True):
        mine = pos.filter(pl.col("steamid") == p["steamid"]).sort("tick")
        frames: list[list[Any] | None] = [None] * len(ticks)
        for row in mine.iter_rows(named=True):
            if row["health"] is None or row["health"] <= 0:
                continue
            px, py = to_radar(row["x"], row["y"], transform)
            frames[index[row["tick"]]] = [
                px, py, round(row.get("yaw") or 0.0), int(row["health"]), row.get("active_weapon"),
                bool(row.get("has_bomb")), bool(row.get("is_defusing")), round(float(row.get("flash_duration") or 0), 1),
            ]
        if any(f is not None for f in frames):
            players.append({"steamid": str(p["steamid"]), "name": p["name"], "side": p["side"], "frames": frames})

    kills = store.table(demo_id, "kills").filter(pl.col("round_num") == round_num).sort("tick")
    kill_rows = []
    for k in kills.with_columns(weapon_label()).iter_rows(named=True):
        vx, vy = to_radar(k["victim_x"], k["victim_y"], transform)
        ax, ay = (to_radar(k["attacker_x"], k["attacker_y"], transform)
                  if k.get("attacker_x") is not None else (None, None))
        kill_rows.append({"tick": k["tick"], "attacker": k["attacker_name"], "attacker_side": k["attacker_side"],
                          "victim": k["victim_name"], "victim_side": k["victim_side"], "weapon": k["weapon"],
                          "headshot": k["headshot"], "victim_xy": [vx, vy], "attacker_xy": [ax, ay],
                          "after_round_end": k["after_round_end"]})

    def areas(name: str) -> list[dict[str, Any]]:
        df = pl.read_parquet(d / f"{name}.parquet").filter(
            (pl.col("end_tick") >= start) & (pl.col("start_tick") <= end))
        return [{"start": a["start_tick"], "end": a["end_tick"], "xy": list(to_radar(a["x"], a["y"], transform))}
                for a in df.iter_rows(named=True)]

    bomb = store.table(demo_id, "bomb").filter(
        (pl.col("round_num") == round_num) & pl.col("event").is_in(["finish_plant", "defuse"]))
    bomb_rows = [{"tick": b["tick"], "event": b["event"], "name": b["name"],
                  "xy": list(to_radar(b["x"], b["y"], transform)) if b.get("x") is not None else None}
                 for b in bomb.iter_rows(named=True)]
    return {
        "round": round_num, "map": meta.get("map"), "tick_rate": tick_rate, "frame_every": SNAPSHOT_EVERY,
        "units_per_px": transform["scale"],
        "start_tick": start, "end_tick": r["end_tick"], "winner_side": r["winner_side"], "reason": r["reason_name"],
        "ticks": ticks, "players": players, "kills": kill_rows,
        "smokes": areas("replay_smokes"), "fires": areas("replay_fires"), "bomb": bomb_rows,
        "frame_fields": ["x", "y", "yaw", "hp", "weapon", "has_bomb", "defusing", "flashed_s"],
    }
