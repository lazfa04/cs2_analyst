"""Demo layer: parse a .dem once, cache analysis-ready tables as parquet.

Why a cache: parsing is the expensive step (seconds to tens of seconds per
demo). Every tool call after ingest only reads small parquet files, so the
agent can ask dozens of questions about one match cheaply. Same idea as an
ETL step feeding a query layer.

Cache layout:  <cache_dir>/<demo_id>/{meta.json, rounds.parquet, kills.parquet, ...}
demo_id is a content hash, so re-ingesting the same file is a no-op and two
copies of one demo map to the same id.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from .errors import AnalystError, DemoNotFound
from .sides import norm_side

# Bump when the cached table layout changes; older caches get re-ingested.
# v2: added grenade_throws. v3: coaches and spectators (dead at freeze end) left out of round_players.
SCHEMA_VERSION = 3

TABLES = ("rounds", "kills", "damages", "blinds", "bomb", "stats", "players", "round_economy", "round_players",
          "grenade_throws")

# Grenade throws come from the weapon_fire events (awpy's `shots`), which always name the thrower.
# awpy's own grenade/smoke tables leave the thrower empty on many demos.
THROW_TYPES = {
    "weapon_smokegrenade": "smoke", "weapon_flashbang": "flash", "weapon_hegrenade": "HE",
    "weapon_molotov": "molotov/incendiary", "weapon_incgrenade": "molotov/incendiary", "weapon_decoy": "decoy",
}

# Per-player state captured at each round's freeze-end tick (the finished buy).
SNAPSHOT_COLS = [
    "steamid", "name", "side", "money", "equipment_value", "primary_weapon",
    "secondary_weapon", "armor", "has_helmet", "has_defuser", "has_bomb",
    "flashbangs", "smoke_grenades", "he_grenades", "fire_grenades", "health",
]


class DemoLike(Protocol):
    """The slice of awpy.Demo this module needs. Lets tests pass a fake."""

    header: dict[str, Any]
    tick_rate: float
    convars: dict[str, str]
    rounds: pl.DataFrame
    kills: pl.DataFrame
    damages: pl.DataFrame
    blinds: pl.DataFrame
    bomb: pl.DataFrame
    stats: pl.DataFrame
    players: pl.DataFrame
    round_economy: pl.DataFrame
    shots: pl.DataFrame

    def load(self, *datasets: str) -> None: ...
    def snapshots(self, *, ticks: Any = ...) -> pl.DataFrame: ...


def demo_id_for(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def add_round_num(df: pl.DataFrame, rounds: pl.DataFrame, tick_col: str = "tick") -> pl.DataFrame:
    """Tag each event row with the round it happened in.

    awpy's event tables (kills, damages, ...) carry ticks but no round number,
    so we interval-join on the round windows. A row belongs to round r when
    start(r) <= tick <= official_end(r). Rows outside every window (warmup,
    between-map junk) are dropped. `after_round_end` flags exit frags: the
    round was already decided when the event happened.
    """
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("round_num"),
            pl.lit(None, dtype=pl.Boolean).alias("after_round_end"),
        )
    windows = (
        rounds.select(
            "round_num",
            pl.coalesce("start_tick", "freeze_end_tick").cast(pl.Int64).alias("_w_start"),
            "end_tick",
            # last round has no official end; leave it open-ended
            pl.coalesce(
                pl.col("official_end_tick") if "official_end_tick" in rounds.columns else pl.lit(None),
                pl.lit(2**31 - 1),
            ).cast(pl.Int64).alias("_w_end"),
        )
        .sort("_w_end")
    )
    out = (
        df.with_columns(pl.col(tick_col).cast(pl.Int64).alias("_tick"))
        .sort("_tick")
        .join_asof(windows, left_on="_tick", right_on="_w_end", strategy="forward")
        .filter(pl.col("round_num").is_not_null() & (pl.col("_tick") >= pl.col("_w_start")))
        .with_columns((pl.col("_tick") > pl.col("end_tick")).alias("after_round_end"))
        .drop("_tick", "_w_start", "_w_end", "end_tick")
    )
    return out


def _round_players(demo: DemoLike, rounds: pl.DataFrame) -> pl.DataFrame:
    """Each player's side, money and loadout at freeze end of every round.

    This is the ground truth for "who was on which side in round N", which
    the event tables can't tell us for players who did nothing that round.
    """
    freeze_ticks = rounds.drop_nulls("freeze_end_tick").select("round_num", "freeze_end_tick")
    if freeze_ticks.is_empty():
        return pl.DataFrame()
    snaps = demo.snapshots(ticks=freeze_ticks["freeze_end_tick"].to_list())
    keep = [c for c in SNAPSHOT_COLS if c in snaps.columns]
    snaps = snaps.select(["tick", *keep])
    out = snaps.join(freeze_ticks, left_on="tick", right_on="freeze_end_tick", how="inner").drop("tick")
    out = out.with_columns(pl.col("side").map_elements(norm_side, return_dtype=pl.Utf8)).filter(pl.col("steamid") != 0)
    # Coaches sit on a team's side in the demo but are never alive. Everyone playing has health at freeze end.
    if "health" in out.columns:
        out = out.filter(pl.col("health") > 0).drop("health")
    return out


def _grenade_throws(shots: pl.DataFrame, rounds: pl.DataFrame) -> pl.DataFrame:
    throws = shots.filter(pl.col("weapon").is_in(list(THROW_TYPES))).select(
        "tick", "steamid", "name",
        pl.col("side").map_elements(norm_side, return_dtype=pl.Utf8),
        pl.col("weapon").replace_strict(THROW_TYPES, return_dtype=pl.Utf8).alias("grenade"),
    )
    return add_round_num(throws, rounds).filter(~pl.col("after_round_end"))


def ingest_demo_object(demo: DemoLike, demo_id: str, out_dir: Path, source: str) -> dict[str, Any]:
    """Pull every table we need out of a parsed demo and write the cache."""
    t0 = time.perf_counter()
    demo.load("rounds", "kills", "damages", "blinds", "bomb", "stats", "players", "round_economy", "shots")
    rounds = demo.rounds.with_columns(pl.col("winner_side").map_elements(norm_side, return_dtype=pl.Utf8))

    tables: dict[str, pl.DataFrame] = {
        "rounds": rounds,
        "kills": add_round_num(demo.kills, rounds),
        "damages": add_round_num(demo.damages, rounds),
        "blinds": add_round_num(demo.blinds, rounds),
        "bomb": add_round_num(demo.bomb, rounds),
        "stats": demo.stats,
        "players": demo.players.filter(pl.col("steamid") != 0),
        "round_economy": demo.round_economy.with_columns(
            pl.col("side").map_elements(norm_side, return_dtype=pl.Utf8)
        ),
        "round_players": _round_players(demo, rounds),
        "grenade_throws": _grenade_throws(demo.shots, rounds),
    }
    for col_side in ("attacker_side", "victim_side", "assister_side"):
        for name in ("kills", "damages", "blinds"):
            df = tables[name]
            if col_side in df.columns:
                tables[name] = df.with_columns(pl.col(col_side).map_elements(norm_side, return_dtype=pl.Utf8))

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.write_parquet(tmp / f"{name}.parquet")

    header = demo.header
    meta = {
        "demo_id": demo_id,
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "map": header.get("map_name"),
        "server": header.get("server_name"),
        "tick_rate": demo.tick_rate,
        "duration_s": header.get("playback_time"),
        "max_rounds": _int_or_none(demo.convars.get("mp_maxrounds")),
        "rounds_played": rounds.height,
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parse_seconds": round(time.perf_counter() - t0, 2),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    # Atomic-ish swap so a crash mid-ingest never leaves a half-written cache.
    if out_dir.exists():
        for f in out_dir.iterdir():
            f.unlink()
        out_dir.rmdir()
    tmp.rename(out_dir)
    return meta


def _int_or_none(v: str | None) -> int | None:
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


class DemoStore:
    """Owns the on-disk cache. Tools talk to this, never to awpy directly."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, path: str | Path, force: bool = False) -> dict[str, Any]:
        p = Path(path).expanduser()
        if not p.is_file():
            raise DemoNotFound(
                f"No file at '{p}'. Pass the full path to a .dem file "
                f"(CS2 saves your matches under game/csgo/replays/)."
            )
        demo_id = demo_id_for(p)
        out = self.cache_dir / demo_id
        if not force and (out / "meta.json").exists():
            meta = json.loads((out / "meta.json").read_text())
            if meta.get("schema_version") == SCHEMA_VERSION:
                return {**meta, "cached": True}

        from awpy import Demo, InvalidDemoError  # imported lazily: heavy native module

        # awpy only checks the header when opening; a truncated file (an interrupted download)
        # passes that and fails later, while the tables are decoded. Catch both stages.
        #
        # awpy memory-maps the file, and Windows won't let anyone delete a mapped file. So on
        # failure we must not keep the Demo alive: raising `from e` would keep the original
        # exception, its traceback, its frames and through them `demo`, locking the file for as
        # long as the process runs (in Claude Desktop, until it restarts).
        demo, error = None, None
        try:
            demo = Demo(p)
            meta = ingest_demo_object(demo, demo_id, out, source=str(p))
        except InvalidDemoError as e:
            error = str(e)
        finally:
            demo = None
        if error is not None:
            gc.collect()  # unmap now rather than whenever the collector gets to it
            size_mb = p.stat().st_size / 1e6
            raise AnalystError(
                f"'{p.name}' ({size_mb:.1f} MB) couldn't be parsed: {error}. If it was just downloaded, the download "
                "is probably incomplete or corrupted: delete it and download it again. CS:GO demos are not supported."
            )
        self._load.cache_clear()
        return {**meta, "cached": False}

    def source_path(self, demo_id: str) -> Path:
        """Where the demo's .dem is now. If it was moved (e.g. into a subfolder), look for the same file
        name in CS2_DEMO_DIR, and remember the new place."""
        meta_file = self.cache_dir / demo_id / "meta.json"
        m = json.loads(meta_file.read_text())
        p = Path(m.get("source") or "")
        if p.is_file():
            return p
        from .config import get_settings

        demo_dir = get_settings().demo_dir
        moved = demo_dir / p.name if demo_dir and p.name else None
        if moved and moved.is_file():
            meta_file.write_text(json.dumps({**m, "source": str(moved)}, indent=2))
            return moved
        return p

    def refresh_stale(self, on_progress=lambda name: None) -> list[str]:
        """Re-parse cached demos written by an older SCHEMA_VERSION, when their .dem is still around.
        Returns the demo_ids refreshed."""
        done = []
        for meta_file in sorted(self.cache_dir.glob("*/meta.json")):
            m = json.loads(meta_file.read_text())
            source = self.source_path(m["demo_id"]) if m.get("schema_version") != SCHEMA_VERSION else None
            if source is not None and source.is_file():
                on_progress(source.name)
                self.ingest(source, force=True)
                done.append(m["demo_id"])
        return done

    def list_demos(self) -> list[dict[str, Any]]:
        out = []
        for meta_file in sorted(self.cache_dir.glob("*/meta.json")):
            m = json.loads(meta_file.read_text())
            out.append({k: m.get(k) for k in ("demo_id", "map", "rounds_played", "source", "ingested_at")})
        return out

    def meta(self, demo_id: str) -> dict[str, Any]:
        f = self.cache_dir / demo_id / "meta.json"
        if not f.exists():
            known = [d["demo_id"] for d in self.list_demos()]
            hint = f"Ingested demos: {known}." if known else "No demos ingested yet."
            raise DemoNotFound(f"Unknown demo_id '{demo_id}'. {hint} Call ingest_demo with a .dem path first.")
        return json.loads(f.read_text())

    def table(self, demo_id: str, name: str) -> pl.DataFrame:
        self.meta(demo_id)  # raises a guiding error for unknown ids
        return self._load(demo_id, name)

    @lru_cache(maxsize=64)
    def _load(self, demo_id: str, name: str) -> pl.DataFrame:
        if name not in TABLES:
            raise ValueError(f"unknown table {name}")
        return pl.read_parquet(self.cache_dir / demo_id / f"{name}.parquet")
