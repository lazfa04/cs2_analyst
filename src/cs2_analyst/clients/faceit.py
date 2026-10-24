"""FACEIT Data API v4 (https://docs.faceit.com). Needs a server-side API key."""

from __future__ import annotations

from typing import Any

import httpx

from ..errors import ApiError, MissingApiKey
from .http import CachedClient

BASE_URL = "https://open.faceit.com/data/v4"
GAME = "cs2"


def _num(v: Any) -> float | int | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else round(f, 2)


class FaceitClient:
    def __init__(self, api_key: str | None, ttl_s: int = 300, transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise MissingApiKey(
                "FACEIT_API_KEY is not set, so FACEIT tools are unavailable. "
                "Demo analysis tools still work."
            )
        self._c = CachedClient(
            BASE_URL, headers={"Authorization": f"Bearer {api_key}"}, ttl_s=ttl_s,
            transport=transport, service="FACEIT",
        )

    def _player(self, nickname: str | None = None, steam64: str | None = None) -> dict[str, Any]:
        """By nickname, or by Steam64 ID (how a demo knows its players)."""
        params = {"nickname": nickname, "game": GAME} if nickname else {"game": GAME, "game_player_id": steam64}
        try:
            return self._c.get("/players", params)
        except ApiError as e:
            if e.status == 404:
                msg = (f"FACEIT has no player '{nickname}'. Nicknames are case-sensitive; check the exact spelling."
                       if nickname else f"No FACEIT account is linked to Steam ID {steam64}.")
                raise ApiError(msg, 404) from e
            raise

    def profile(self, nickname: str | None = None, *, steam64: str | None = None) -> dict[str, Any]:
        p = self._player(nickname, steam64)
        cs2 = (p.get("games") or {}).get(GAME) or {}
        life = (self._c.get(f"/players/{p['player_id']}/stats/{GAME}") or {}).get("lifetime", {})
        return {
            "nickname": p.get("nickname"),
            "player_id": p.get("player_id"),
            "country": p.get("country"),
            "steam_id_64": p.get("steam_id_64"),
            "level": cs2.get("skill_level"),
            "elo": cs2.get("faceit_elo"),
            "lifetime": {
                "matches": _num(life.get("Matches")),
                "win_rate_pct": _num(life.get("Win Rate %")),
                "avg_kd": _num(life.get("Average K/D Ratio")),
                "avg_hs_pct": _num(life.get("Average Headshots %")),
                "adr": _num(life.get("ADR")),
                "recent_results": life.get("Recent Results"),
            },
        }

    @staticmethod
    def _own_score(score: Any, result: str | None) -> Any:
        """FACEIT's Score is in the match's team order, not the player's: a loss can read "13 / 11".
        Put the player's team first, using the result to tell which side is theirs."""
        try:
            a, b = (int(x) for x in str(score).replace(" ", "").split("/"))
        except ValueError:
            return score
        if result in ("win", "loss") and a != b and (a > b) != (result == "win"):
            a, b = b, a
        return f"{a}-{b}"

    def recent_matches(self, nickname: str | None = None, limit: int = 10, *, steam64: str | None = None) -> dict[str, Any]:
        p = self._player(nickname, steam64)
        data = self._c.get(f"/players/{p['player_id']}/games/{GAME}/stats", {"limit": max(1, min(limit, 30))})
        matches = []
        for item in data.get("items", []):
            s = item.get("stats", {})
            result = {"1": "win", "0": "loss"}.get(str(s.get("Result")), s.get("Result"))
            matches.append({
                "match_id": s.get("Match Id"),
                "map": s.get("Map"),
                "result": result,
                "score": self._own_score(s.get("Score"), result),  # player's team first
                "kills": _num(s.get("Kills")),
                "deaths": _num(s.get("Deaths")),
                "kd": _num(s.get("K/D Ratio")),
                "adr": _num(s.get("ADR")),
                "hs_pct": _num(s.get("Headshots %")),
            })
        wins = sum(m["result"] == "win" for m in matches)

        def avg(k: str) -> float | None:
            vals = [m[k] for m in matches if isinstance(m[k], (int, float))]
            return round(sum(vals) / len(vals), 2) if vals else None

        return {"nickname": p.get("nickname"), "count": len(matches), "wins": wins,
                "averages": {"kd": avg("kd"), "adr": avg("adr"), "hs_pct": avg("hs_pct"), "kills": avg("kills")},
                "matches": matches}
