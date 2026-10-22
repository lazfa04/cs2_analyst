from pathlib import Path

import polars as pl
import pytest

from cs2_analyst.demo_store import DemoStore, add_round_num
from cs2_analyst.errors import DemoNotFound

from .conftest import DEMO_ID


def test_add_round_num_drops_warmup_and_flags_exit_frags(store):
    kills = store.table(DEMO_ID, "kills")
    assert 50 not in kills["tick"].to_list(), "warmup kill must be dropped"
    by_tick = {r["tick"]: r for r in kills.iter_rows(named=True)}
    assert by_tick[400]["round_num"] == 1 and by_tick[400]["after_round_end"] is False
    assert by_tick[1100]["round_num"] == 1 and by_tick[1100]["after_round_end"] is True
    assert by_tick[3550]["round_num"] == 4, "last round has no official end and must stay open"


def test_add_round_num_handles_empty_frame():
    rounds = pl.DataFrame({"round_num": [1], "start_tick": [0], "freeze_end_tick": [10],
                           "end_tick": [100], "official_end_tick": [120]})
    out = add_round_num(pl.DataFrame({"tick": []}, schema={"tick": pl.Int32}), rounds)
    assert out.is_empty() and "round_num" in out.columns


def test_sides_are_normalised_everywhere(store):
    assert set(store.table(DEMO_ID, "kills")["attacker_side"].unique()) == {"CT", "T"}
    assert set(store.table(DEMO_ID, "rounds")["winner_side"].unique()) == {"CT", "T"}
    rp = store.table(DEMO_ID, "round_players")
    assert set(rp["side"].unique()) == {"CT", "T"}
    assert 0 not in rp["steamid"].to_list(), "GOTV bot must be filtered out"


def test_round_players_track_halftime_swap(store):
    rp = store.table(DEMO_ID, "round_players")
    side = {(r["round_num"], r["name"]): r["side"] for r in rp.iter_rows(named=True)}
    assert side[(1, "alpha")] == "CT" and side[(3, "alpha")] == "T"


def test_meta_and_listing(store):
    meta = store.meta(DEMO_ID)
    assert meta["map"] == "de_test" and meta["rounds_played"] == 4 and meta["max_rounds"] == 4
    assert [d["demo_id"] for d in store.list_demos()] == [DEMO_ID]


def test_unknown_demo_id_lists_known_ids(store):
    with pytest.raises(DemoNotFound, match=DEMO_ID):
        store.table("nope", "kills")


def test_ingest_missing_file_gives_a_hint(tmp_path: Path):
    with pytest.raises(DemoNotFound, match="replays"):
        DemoStore(tmp_path).ingest(tmp_path / "missing.dem")


def test_truncated_demo_gives_a_redownload_hint(tmp_path: Path):
    """An interrupted download passes awpy's header check and fails later, while decoding."""
    import os

    import pytest

    from cs2_analyst.errors import AnalystError

    src = os.getenv("CS2_TEST_DEMO")
    if not src:
        pytest.skip("set CS2_TEST_DEMO to a .dem path")
    truncated = tmp_path / "truncated.dem"
    with open(src, "rb") as f:
        truncated.write_bytes(f.read(2_600_000))
    with pytest.raises(AnalystError, match="download is probably incomplete"):
        DemoStore(tmp_path / "cache").ingest(truncated)


def test_failed_parse_releases_the_demo_file(tmp_path: Path):
    """On Windows a still-mapped demo can't be deleted; a failed ingest must not keep it open."""
    import gc
    import os
    import weakref

    import pytest

    import awpy

    from cs2_analyst.errors import AnalystError

    src = os.getenv("CS2_TEST_DEMO")
    if not src:
        pytest.skip("set CS2_TEST_DEMO to a .dem path")
    truncated = tmp_path / "truncated.dem"
    with open(src, "rb") as f:
        truncated.write_bytes(f.read(2_600_000))

    opened = []
    real_demo = awpy.Demo

    def tracking_demo(path):
        d = real_demo(path)
        opened.append(weakref.ref(d))
        return d

    awpy.Demo = tracking_demo
    try:
        with pytest.raises(AnalystError) as err:
            DemoStore(tmp_path / "cache").ingest(truncated)
    finally:
        awpy.Demo = real_demo
    assert err.value.__cause__ is None and err.value.__context__ is None, "no chained traceback keeping frames alive"
    gc.collect()
    assert opened and opened[0]() is None, "the Demo (and its memory map) must be released"
