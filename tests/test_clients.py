"""API clients tested against httpx.MockTransport: no network, no keys needed."""

import httpx
import pytest

from cs2_analyst.clients import FaceitClient, LeetifyClient
from cs2_analyst.clients import http as http_mod
from cs2_analyst.errors import ApiError, MissingApiKey

STEAM = "76561198000000001"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(http_mod.time, "sleep", lambda s: None)


def faceit_handler(request: httpx.Request) -> httpx.Response:
    assert request.headers["Authorization"] == "Bearer k"
    path = request.url.path
    if path.endswith("/players") and request.url.params.get("nickname") == "ghost":
        return httpx.Response(404, json={"errors": []})
    if path.endswith("/players"):
        return httpx.Response(200, json={
            "player_id": "pid", "nickname": "Afzal", "country": "bh", "steam_id_64": STEAM,
            "games": {"cs2": {"skill_level": 10, "faceit_elo": 2400}},
        })
    if path.endswith("/stats/cs2"):
        return httpx.Response(200, json={"lifetime": {"Matches": "812", "Win Rate %": "55", "Average K/D Ratio": "1.21"}})
    if path.endswith("/games/cs2/stats"):
        return httpx.Response(200, json={"items": [
            {"stats": {"Match Id": "m1", "Map": "de_mirage", "Result": "1", "Score": "13 / 9", "Kills": "24", "ADR": "91.4"}},
            {"stats": {"Match Id": "m2", "Map": "de_nuke", "Result": "0", "Score": "8 / 13", "Kills": "12"}},
        ]})
    return httpx.Response(500)


def faceit(handler=faceit_handler) -> FaceitClient:
    return FaceitClient("k", transport=httpx.MockTransport(handler))


def test_faceit_profile_maps_fields():
    p = faceit().profile("Afzal")
    assert (p["level"], p["elo"], p["steam_id_64"]) == (10, 2400, STEAM)
    assert p["lifetime"]["matches"] == 812 and p["lifetime"]["avg_kd"] == 1.21


def test_faceit_recent_matches_normalises_results():
    r = faceit().recent_matches("Afzal", limit=2)
    assert [m["result"] for m in r["matches"]] == ["win", "loss"]
    assert r["wins"] == 1 and r["matches"][0]["kills"] == 24 and r["matches"][0]["adr"] == 91.4


def test_faceit_unknown_player_is_guided():
    with pytest.raises(ApiError, match="case-sensitive") as e:
        faceit().profile("ghost")
    assert e.value.status == 404


def test_faceit_requires_key():
    with pytest.raises(MissingApiKey, match="Demo analysis tools still work"):
        FaceitClient(None)


def test_retries_on_429_then_succeeds_and_caches():
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return faceit_handler(request)

    client = faceit(flaky)
    client._c.get("/players", {"nickname": "Afzal"})
    client._c.get("/players", {"nickname": "Afzal"})  # served from TTL cache
    assert calls["n"] == 2


def test_persistent_429_surfaces_rate_limit_message():
    client = faceit(lambda r: httpx.Response(429))
    with pytest.raises(ApiError, match="rate limit"):
        client.profile("Afzal")


def test_bad_key_message():
    client = faceit(lambda r: httpx.Response(401))
    with pytest.raises(ApiError, match="API key"):
        client.profile("Afzal")


def test_leetify_profile_and_key_header():
    def handler(request):
        assert request.headers["_leetify_key"] == "lk"
        assert request.url.params["steam64_id"] == STEAM
        return httpx.Response(200, json={
            "name": "afz", "winrate": 0.55, "total_matches": 900,
            "ranks": {"faceit": 10}, "rating": {"aim": 80.1},
            "stats": {"reaction_time_ms": 520.0, "noise": 1},
            "recent_matches": [{"id": "x", "map_name": "de_ancient", "outcome": "win", "score": [13, 7]}] * 8,
        })

    p = LeetifyClient("lk", transport=httpx.MockTransport(handler)).profile(STEAM, recent=3)
    assert p["stats"] == {"reaction_time_ms": 520.0}, "only whitelisted stats are returned"
    assert len(p["recent_matches"]) == 3


def test_leetify_rejects_non_steam64_before_calling():
    client = LeetifyClient(transport=httpx.MockTransport(lambda r: pytest.fail("should not call API")))
    with pytest.raises(ApiError, match="17 digits"):
        client.profile("afzal")


@pytest.mark.parametrize("raw,result,own", [
    ("13 / 11", "loss", "11-13"),   # FACEIT lists team 1 first; this player was on the losing team
    ("7 / 13", "win", "13-7"),
    ("13 / 9", "win", "13-9"),
    ("8 / 13", "loss", "8-13"),
    ("weird", "win", "weird"),
])
def test_faceit_score_is_from_the_players_side(raw, result, own):
    assert FaceitClient._own_score(raw, result) == own


def test_leetify_skips_practice_matches_and_flags_stale_faceit_rank():
    def handler(request):
        return httpx.Response(200, json={"name": "x", "ranks": {"faceit_elo": 1596}, "recent_matches": [
            {"id": "a", "map_name": "de_inferno", "data_source": "matchmaking"},
            {"id": "b", "map_name": "rush_001", "data_source": "manual_rush"},
            {"id": "c", "map_name": "de_dust2", "data_source": "faceit"}]})
    p = LeetifyClient("lk", transport=httpx.MockTransport(handler)).profile("76561198802039309")
    assert [m["id"] for m in p["recent_matches"]] == ["a", "c"]
    assert "out of date" in p["note"]


def test_faceit_by_steam_id_and_recent_averages():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/players"):
            seen.append(dict(request.url.params))
            if request.url.params.get("game_player_id") == "76561198000000999":
                return httpx.Response(404, json={"errors": []})
        return faceit_handler(request)

    c = faceit(handler)
    p = c.profile(steam64=STEAM)
    assert p["nickname"] == "Afzal" and seen[-1] == {"game": "cs2", "game_player_id": STEAM}
    r = c.recent_matches(steam64=STEAM, limit=2)
    assert r["averages"]["kills"] == 18 and r["averages"]["adr"] == 91.4, "averages skip missing values"
    with pytest.raises(ApiError, match="No FACEIT account is linked") as e:
        c.profile(steam64="76561198000000999")
    assert e.value.status == 404
