"""A demo player's FACEIT and Leetify context, next to how they played in this match. Used by the web app.

Demos know players by Steam64 ID, which both services accept, so this works for anyone in the match who has
an account. Each service fails on its own (no key, private profile, not on FACEIT) and says why; the match
numbers always come from the demo.

The comparison is "this match vs. usual": FACEIT's last N matches and lifetime averages for K/D, ADR and
headshots, and Leetify's averages for trading, opening duels and utility at death. Leetify computes those
with its own definitions, so they're labelled as roughly comparable, not identical.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import analysis, deaths
from .demo_store import DemoStore
from .errors import AnalystError, ApiError, MissingApiKey  # noqa: F401  (AnalystError raised below)

RECENT = 20


def _pct(v: Any) -> float | None:
    """Leetify mixes 0-1 ratios and 0-100 percentages; show both as percentages."""
    if not isinstance(v, (int, float)):
        return None
    return round(v * 100 if abs(v) <= 1 else v, 1)


def _service(fn: Callable[[], dict[str, Any]], key_hint: str) -> dict[str, Any]:
    try:
        return fn()
    except MissingApiKey:
        return {"error": f"Add {key_hint} to .env and restart the app to see this."}
    except ApiError as e:
        return {"error": str(e), "status": getattr(e, "status", None)}


def this_match(store: DemoStore, demo_id: str, steamid: int) -> dict[str, Any]:
    t = lambda n: store.table(demo_id, n)  # noqa: E731
    ov = analysis.match_overview(store.meta(demo_id), t("rounds"), t("stats"), t("players"), t("round_players"),
                                 kills=t("kills"), damages=t("damages"), bomb=t("bomb"))
    name = dict(t("players").select("steamid", "name").iter_rows()).get(steamid)
    row = next((r for r in ov["scoreboard"] if r["name"] == name), None)
    if row is None:
        raise AnalystError(f"{name or steamid} isn't on this match's scoreboard.")
    rev = deaths.death_review(t("kills"), t("rounds"), t("round_players"), t("grenade_throws"), t("blinds"),
                              store.meta(demo_id)["tick_rate"], steamid=steamid, bomb=t("bomb"), limit=10_000)
    dd = rev["deaths_detail"]
    duels = row["opening_kills"] + row["opening_deaths"]
    return {
        "name": row["name"], "team": row["team"], "rating": row["rating"], "kills": row["kills"], "deaths": row["deaths"],
        "kd": round(row["kills"] / row["deaths"], 2) if row["deaths"] else float(row["kills"]),
        "adr": row["adr"], "hs_pct": row["hs_pct"], "kast": row["kast"],
        "opening_duels": f"{row['opening_kills']}-{row['opening_deaths']}",
        "opening_win_pct": round(100 * row["opening_kills"] / duels, 1) if duels else None,
        "traded_deaths_pct": round(100 * sum(d["traded"] for d in dd) / len(dd), 1) if dd else None,
        "utility_on_death_avg": round(sum(d["utility_value"] for d in dd) / len(dd)) if dd else None,
    }


def player_context(store: DemoStore, demo_id: str, player: str, faceit: Callable[[], Any],
                   leetify: Callable[[], Any], recent: int = RECENT) -> dict[str, Any]:
    info = analysis.resolve_player(store.table(demo_id, "players"), player)
    sid = int(info["steamid"])
    steam64 = str(sid)
    match = this_match(store, demo_id, sid)

    def fc() -> dict[str, Any]:
        client = faceit()
        prof = client.profile(steam64=steam64)
        prof["recent"] = client.recent_matches(steam64=steam64, limit=recent)
        return prof

    fp = _service(fc, "FACEIT_API_KEY")
    lp = _service(lambda: leetify().profile(steam64), "LEETIFY_API_KEY")

    life = fp.get("lifetime") or {}
    rec = (fp.get("recent") or {}).get("averages") or {}
    ls = lp.get("stats") or {}
    rows = [
        {"stat": "K/D", "this_match": match["kd"], "recent": rec.get("kd"), "usual": life.get("avg_kd"),
         "usual_source": "FACEIT lifetime", "higher_is_better": True},
        {"stat": "ADR", "this_match": match["adr"], "recent": rec.get("adr"), "usual": life.get("adr"),
         "usual_source": "FACEIT lifetime", "higher_is_better": True},
        {"stat": "Headshot %", "this_match": match["hs_pct"], "recent": rec.get("hs_pct"), "usual": life.get("avg_hs_pct"),
         "usual_source": "FACEIT lifetime", "higher_is_better": True},
        {"stat": "Opening duels won %", "this_match": match["opening_win_pct"], "recent": None,
         "usual": _avg(_pct(ls.get("t_opening_duel_success_percentage")), _pct(ls.get("ct_opening_duel_success_percentage"))),
         "usual_source": "Leetify (T and CT averaged)", "higher_is_better": True},
        {"stat": "Deaths traded %", "this_match": match["traded_deaths_pct"], "recent": None,
         "usual": _pct(ls.get("traded_deaths_success_percentage")), "usual_source": "Leetify (own definition)",
         "higher_is_better": True},
        {"stat": "Utility held at death ($)", "this_match": match["utility_on_death_avg"], "recent": None,
         "usual": ls.get("utility_on_death_avg"), "usual_source": "Leetify (own definition)", "higher_is_better": False},
    ]
    return {
        "player": match["name"], "steam64": steam64, "this_match": match,
        "faceit": fp, "leetify": lp, "comparison": rows,
        "recent_count": (fp.get("recent") or {}).get("count"),
    }


def _avg(a: float | None, b: float | None) -> float | None:
    vals = [v for v in (a, b) if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None

