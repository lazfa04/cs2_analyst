"""End-to-end through the MCP protocol with an in-memory client session."""

import json

import anyio
import pytest
from mcp import Client

from cs2_analyst.config import Settings
from cs2_analyst.server import build_server

from .conftest import DEMO_ID

EXPECTED_TOOLS = {
    "list_demos", "ingest_demo", "get_match_overview", "get_round_summary", "get_opening_duels",
    "get_utility_stats", "find_kills", "get_faceit_profile", "get_recent_faceit_matches", "get_leetify_profile",
}


@pytest.fixture
def server(store, tmp_path):
    settings = Settings(cache_dir=store.cache_dir, demo_dir=None, faceit_api_key=None,
                        leetify_api_key=None, http_cache_ttl_s=0)
    return build_server(settings=settings, store=store)


def call(server, name, args):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(name, args)
    return anyio.run(go)


def test_all_tools_registered_with_descriptions(server):
    async def go():
        async with Client(server) as c:
            return (await c.list_tools()).tools
    tools = anyio.run(go)
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(t.description and len(t.description) > 40 for t in tools), "descriptions are prompts; keep them real"


def test_overview_through_protocol(server):
    res = call(server, "get_match_overview", {"demo_id": DEMO_ID})
    assert not res.is_error
    body = json.loads(res.content[0].text)
    assert body["score"]["Team B (started T)"]["rounds_won"] == 3


@pytest.mark.parametrize("name,args,hint", [
    ("get_utility_stats", {"demo_id": "nope", "player": "alpha"}, "Call ingest_demo"),
    ("get_opening_duels", {"demo_id": DEMO_ID, "player": "zulu"}, "Players: "),
    ("get_round_summary", {"demo_id": DEMO_ID, "round": 9}, "Valid rounds"),
    ("get_faceit_profile", {"nickname": "x"}, "FACEIT_API_KEY"),
])
def test_errors_reach_the_model_with_a_recovery_hint(server, name, args, hint):
    res = call(server, name, args)
    assert res.is_error
    assert hint in res.content[0].text, "a generic 'Error executing tool' would leave the model stuck"


def test_schema_validation_rejects_bad_args(server):
    res = call(server, "get_round_summary", {"demo_id": DEMO_ID, "round": 0})
    assert res.is_error
