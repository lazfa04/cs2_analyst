"""Leetify public CS API. Works without a key (lower rate limits); a key is sent as `_leetify_key`."""

from __future__ import annotations

from typing import Any

import httpx

from ..errors import ApiError
from .http import CachedClient

BASE_URL = "https://api-public.cs-prod.leetify.com"

# The full stats block is ~25 numbers; these are the ones that map to coaching questions.
KEY_STATS = (
    "accuracy_head", "spray_accuracy", "preaim", "reaction_time_ms", "counter_strafing_good_shots_ratio",
    "ct_opening_duel_success_percentage", "t_opening_duel_success_percentage",
    "trade_kills_success_percentage", "traded_deaths_success_percentage",
    "flashbang_hit_foe_per_flashbang", "flashbang_leading_to_kill", "utility_on_death_avg",
)


class LeetifyClient:
    def __init__(self, api_key: str | None = None, ttl_s: int = 300, transport: httpx.BaseTransport | None = None):
        headers = {"_leetify_key": api_key} if api_key else {}
        self._c = CachedClient(BASE_URL, headers=headers, ttl_s=ttl_s, transport=transport, service="Leetify")

    def profile(self, steam64_id: str, recent: int = 5) -> dict[str, Any]:
        if not steam64_id.isdigit() or len(steam64_id) != 17:
            raise ApiError(
                f"'{steam64_id}' is not a Steam64 ID (17 digits, starts with 7656). "
                "get_faceit_profile returns steam_id_64 if you only know a FACEIT nickname."
            )
        try:
            p = self._c.get("/v3/profile", {"steam64_id": steam64_id})
        except ApiError as e:
            if e.status == 404:
                raise ApiError(
                    f"Leetify has no public profile for {steam64_id} (private or never signed up).", 404
                ) from e
            raise
        stats = p.get("stats") or {}
        return {
            "name": p.get("name"),
            "privacy_mode": p.get("privacy_mode"),
            "total_matches": p.get("total_matches"),
            "winrate": p.get("winrate"),
            "ranks": p.get("ranks"),
            "rating": p.get("rating"),
            "stats": {k: stats.get(k) for k in KEY_STATS if k in stats},
            "recent_matches": [
                {k: m.get(k) for k in ("id", "finished_at", "map_name", "outcome", "score", "leetify_rating", "data_source")}
                for m in (p.get("recent_matches") or [])[:recent]
            ],
        }
