"""Player positions over time, read from the original .dem on first use and cached next to the demo.

Positions aren't part of the normal ingest (they'd make every parse slower and every cache bigger).
The 2D replay and the position-based facts (spacing at death, retakes) read them through here: the
first request for a demo reads the .dem once more for snapshots 8 times a second (every 8 ticks at
64 tick), plus smokes and molotov fires (~2 MB per match).

Also: walking distance between two points, over the map's navigation mesh. Straight-line distance
ignores walls; two players 400 units apart through a wall can be 1,500 units apart on foot, and a
trade needs the on-foot distance.
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Callable
from pathlib import Path

import polars as pl

from .demo_store import DemoStore
from .errors import AnalystError
from .sides import norm_side

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
    source = store.source_path(demo_id)
    if not source.is_file():
        raise AnalystError(f"Player positions need the original demo file, which is no longer at '{source}'. "
                           "Put it back, or put it in CS2_DEMO_DIR.")
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


def positions(store: DemoStore, demo_id: str) -> pl.DataFrame:
    ensure_replay_cache(store, demo_id)
    return pl.read_parquet(_replay_dir(store, demo_id) / "replay_positions.parquet")


Point = tuple[float, float, float]


def straight(a: Point, b: Point) -> float:
    return math.dist(a, b)


@functools.cache
def walking_distance(map_name: str) -> tuple[Callable[[Point, Point], float], str]:
    """A distance function for the map, and what kind it is ("walking" or "straight-line").

    Walking distance follows the nav mesh: the shortest chain of walkable areas, measured between
    area centres, plus the bits from each player to their area's centre. Falls back to straight-line
    when the map's nav mesh can't be loaded or a point is off the mesh (mid-jump, on a ladder).
    """
    try:
        import awpy.data as D
        from awpy import NavMesh

        try:
            nav = NavMesh(D.nav_path(map_name))
        except Exception:
            nav = NavMesh(D.nav_path(map_name, version=FALLBACK_DATA_VERSION))
    except Exception:
        return straight, "straight-line"

    centre = {}

    def c(area_id: int) -> Point:
        if area_id not in centre:
            centre[area_id] = tuple(nav.area(area_id)["centroid"])
        return centre[area_id]

    def dist(a: Point, b: Point) -> float:
        try:
            fa, fb = nav.find_area(a), nav.find_area(b)
            if fa is None or fb is None:
                return straight(a, b)
            if fa == fb:
                return straight(a, b)
            path = nav.find_path(fa, fb)
            if not path:
                return straight(a, b)
            pts = [c(x) for x in path]
            return straight(a, pts[0]) + sum(straight(p, q) for p, q in zip(pts, pts[1:])) + straight(pts[-1], b)
        except Exception:
            return straight(a, b)

    return dist, "walking"
