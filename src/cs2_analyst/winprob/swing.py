"""Round swing: how much each action changed the team's chance to win the round, credited to players.

This is the idea behind HLTV's Rating 3.0 (its exact formula isn't public). A kill that turns a 2v2
post-plant into a 2v1 is worth far more than an exit-frag-like kill in a 5v2, and plain kill counts
can't tell them apart. For every event:

  kill    the killer's team gains delta, the victim's team loses it. The killer gets the gain (75% of
          it when someone assisted; the assister gets 25%), the victim gets the loss.
  plant   the planter gets T's gain.
  defuse  the defuser gets CT's gain (everything that was left to win).

What isn't credited: the starting probability (that's the economy, not a play) and the last jump when
a round ends on time or on the bomb exploding. Teamkills only charge the victim's team, to the victim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from . import model as M
from .features import FEATURES

ASSIST_SHARE = 0.25


def with_probabilities(model, states: pl.DataFrame) -> pl.DataFrame:
    p = M.predict(model, states)
    df = states.with_columns(pl.Series("p_t", p))
    return df.with_columns(
        (pl.col("p_t") - pl.col("p_t").shift(1).over(["demo_id", "round"])).fill_null(0.0).alias("delta_t")
    )


def credits(states: pl.DataFrame) -> list[dict[str, Any]]:
    """One row per (player, event) credit, in win-probability points (0-1)."""
    out: list[dict[str, Any]] = []
    for e in states.filter(pl.col("step") > 0).iter_rows(named=True):
        d = e["delta_t"]

        def team(side: str | None) -> float:
            return d if side == "T" else -d

        base = {"round": e["round"], "tick": e["tick"], "event": e["event"], "p_t_after": e["p_t"]}
        if e["event"] == "kill":
            fair_kill = e["actor_id"] is not None and e["actor_side"] != e["victim_side"]
            if fair_kill:
                gain = team(e["actor_side"])
                share = 1 - ASSIST_SHARE if e["assister_id"] is not None else 1.0
                out.append({**base, "steamid": e["actor_id"], "name": e["actor"], "role": "kill",
                            "other": e["victim"], "swing": gain * share})
                if e["assister_id"] is not None:
                    out.append({**base, "steamid": e["assister_id"], "name": e["assister"], "role": "assist",
                                "other": e["victim"], "swing": gain * ASSIST_SHARE})
            out.append({**base, "steamid": e["victim_id"], "name": e["victim"], "role": "death",
                        "other": e["actor"], "swing": team(e["victim_side"])})
        elif e["actor_id"] is not None:  # plant or defuse
            out.append({**base, "steamid": e["actor_id"], "name": e["actor"], "role": e["event"], "other": None,
                        "swing": team(e["actor_side"])})
    return out


def swing_ratings(states: pl.DataFrame, round_players: pl.DataFrame, labels: dict[int, str],
                  ratings: dict[int, float] | None = None, top: int = 3) -> dict[str, Any]:
    """Per player: average swing per round in percentage points, plus their biggest moments."""
    cr = credits(states)
    rounds_played = round_players.group_by("steamid").agg(pl.col("round_num").n_unique()).iter_rows()
    played = {sid: n for sid, n in rounds_played}
    names = dict(round_players.select("steamid", "name").unique("steamid").iter_rows())
    rows = []
    for sid, n in played.items():
        mine = [c for c in cr if c["steamid"] == sid]
        total = sum(c["swing"] for c in mine)
        by_role = {}
        for c in mine:
            by_role[c["role"]] = by_role.get(c["role"], 0.0) + c["swing"]
        best = sorted(mine, key=lambda c: -c["swing"])[:top]
        worst = sorted(mine, key=lambda c: c["swing"])[:top]
        moment = lambda c: {"round": c["round"], "tick": c["tick"], "what": c["role"], "other": c["other"],  # noqa: E731
                            "swing_pp": round(100 * c["swing"], 1)}
        rows.append({
            "name": names.get(sid), "team": labels.get(sid),
            "swing_per_round_pp": round(100 * total / n, 2) if n else 0.0,
            "total_swing_pp": round(100 * total, 1), "rounds": n,
            "from_kills_pp": round(100 * (by_role.get("kill", 0) + by_role.get("assist", 0)), 1),
            "from_deaths_pp": round(100 * by_role.get("death", 0), 1),
            "from_bomb_pp": round(100 * (by_role.get("plant", 0) + by_role.get("defuse", 0)), 1),
            "rating": (ratings or {}).get(sid),
            "biggest_plays": [moment(c) for c in best if c["swing"] > 0],
            "costliest_moments": [moment(c) for c in worst if c["swing"] < 0],
        })
    rows.sort(key=lambda r: -r["swing_per_round_pp"])
    for i, r in enumerate(rows, 1):
        r["swing_rank"] = i
    if ratings:
        by_rating = sorted(rows, key=lambda r: -(r["rating"] or 0))
        for i, r in enumerate(by_rating, 1):
            r["rating_rank"] = i
    return {
        "players": rows,
        "note": "swing_per_round_pp: average change in the team's chance to win the round caused by this player's "
                "kills, assists, deaths, plants and defuses, in percentage points per round (0 = neutral). It rewards "
                "kills that mattered (opening kills, clutch kills, post-plant kills) over kills in decided rounds. "
                "Built on a model trained on a few hundred rounds, so treat small gaps between players as noise.",
    }


def round_timeline(states: pl.DataFrame, model, round_num: int, tick_rate: float) -> dict[str, Any]:
    r = states.filter(pl.col("round") == round_num).sort("step")
    if r.is_empty():
        return {}
    X = r.select(FEATURES).to_numpy().astype(float)
    contribs = M.explain(model, X)
    start = r["tick"][0]
    steps = []
    for i, e in enumerate(r.iter_rows(named=True)):
        steps.append({
            "step": e["step"], "tick": e["tick"], "time_s": round((e["tick"] - start) / tick_rate, 1),
            "event": e["event"], "actor": e["actor"], "actor_side": e["actor_side"], "victim": e["victim"],
            "alive": f"{e['t_alive']}v{e['ct_alive']} (T v CT)", "bomb_planted": bool(e["bomb_planted"]),
            "p_t_win": round(float(e["p_t"]), 3), "p_ct_win": round(1 - float(e["p_t"]), 3),
            "swing_t_pp": round(100 * float(e["delta_t"]), 1),
            "why": M.top_reasons(contribs[i], X[i]),
        })
    biggest = max(steps[1:], key=lambda s: abs(s["swing_t_pp"])) if len(steps) > 1 else None
    return {
        "round": round_num, "winner_side": "T" if r["t_win"][0] else "CT",
        "start_p_t_win": steps[0]["p_t_win"], "steps": steps,
        "biggest_swing": biggest,
        "note": "p_t_win is the model's chance that T wins after each event. swing_t_pp is the change in T's "
                "chance (negative = good for CT). `why` lists the features pushing the estimate most (TreeSHAP).",
    }
