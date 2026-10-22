"""Optional check against a real demo with a known result.

Skipped unless CS2_TEST_DEMO points at a .dem file. Defaults below are for the
public awpy fixture `faceit-de_mirage-6de46e85.dem` (FACEIT, Mirage, 16-14):
https://github.com/pnxenopoulos/awpy-fixtures
"""

import os
from pathlib import Path

import pytest

from cs2_analyst import analysis as A
from cs2_analyst.demo_store import DemoStore

DEMO = os.getenv("CS2_TEST_DEMO")
pytestmark = [pytest.mark.demo, pytest.mark.skipif(not DEMO, reason="set CS2_TEST_DEMO to a .dem path")]


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    store = DemoStore(tmp_path_factory.mktemp("cache"))
    meta = store.ingest(Path(DEMO))
    return store, meta["demo_id"]


def test_final_score_matches_published_result(real):
    store, did = real
    t = lambda n: store.table(did, n)
    ov = A.match_overview(store.meta(did), t("rounds"), t("stats"), t("players"), t("round_players"))
    assert sorted(s["rounds_won"] for s in ov["score"].values()) == [14, 16]
    assert ov["rounds_played"] == 30


def test_scoreboard_matches_faceit_ground_truth(real):
    store, did = real
    stats = {r["name"]: r for r in store.table(did, "stats").iter_rows(named=True)}
    assert (stats["babaya0310"]["kills"], stats["babaya0310"]["deaths"]) == (21, 25)
    assert (stats["cntoz"]["kills"], stats["cntoz"]["deaths"]) == (19, 19)


def test_every_round_has_ten_players_at_freeze_end(real):
    store, did = real
    counts = store.table(did, "round_players").group_by("round_num").len()["len"]
    assert counts.min() == 10


def test_opening_duels_agree_with_awpy_scoreboard(real):
    """Our first-kill logic must agree with awpy's independent opening_kills/deaths."""
    store, did = real
    t = lambda n: store.table(did, n)
    for row in t("stats").iter_rows(named=True):
        od = A.opening_duels(t("kills"), t("rounds"), row["steamid"], store.meta(did)["tick_rate"])
        assert (od["overall"]["won"], od["overall"]["lost"]) == (row["opening_kills"], row["opening_deaths"]), row["name"]


def test_agent_checks_resolve_against_real_tool_output(real):
    """A scripted agent run on the real demo: paths and values match real result shapes."""
    import anyio

    from cs2_analyst.agent import AgentConfig, run_agent
    from cs2_analyst.agent.toolbox import Toolbox
    from cs2_analyst.config import Settings
    from cs2_analyst.server import build_server

    from .test_agent import FakeLLM, fact, report, tool

    store, did = real
    llm = FakeLLM(
        [tool("get_match_overview", demo_id=did)],
        [tool("get_opening_duels", demo_id=did, player="babaya0310")],
        [report(
            fact("c1", "laiq's team won 16 rounds", ("t1", "score.team_laiq__.rounds_won", "16")),
            fact("c2", "babaya0310 got 21 kills", ("t1", "scoreboard[name=babaya0310].kills", "21")),
            fact("c3", "he won 6 opening duels", ("t2", "overall.won", "6")),
            fact("c4", "his round 5 duel was against kzri", ("t2", "duels[round=5].opponent", "kzri")),
        )],
    )
    settings = Settings(cache_dir=store.cache_dir, demo_dir=None, faceit_api_key=None,
                        leetify_api_key=None, http_cache_ttl_s=0)

    async def go():
        async with Toolbox.connect(build_server(settings=settings, store=store)) as tb:
            demo = {"demo_id": did, "map": "de_mirage", "rounds_played": 30}
            return await run_agent("How did babaya do?", demo, llm, tb, AgentConfig(judge_model=None))

    res = anyio.run(go)
    assert [r.status for r in res.verification.results] == ["verified"] * 4
