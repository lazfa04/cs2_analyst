"""Coach cards: a player's costliest flagged deaths, written from the death review's facts. No LLM, no cost.

A card leads with the death's headline flag (deaths.HEADLINE_FLAGS: why it happened and whose it was to fix),
lists the measured facts behind it and the context that changes its meaning (after the plant, the round's
result, utility still held), and ends with a question instead of a verdict. The question is the honest part:
the demo shows that the nearest teammate was 1,200 units away, not whether the fight was the plan.

Deaths that aren't shown are counted, so nothing disappears silently: isolated deaths while lurking (the role),
deaths whose only flags are utility or blind time (state, not cause), and unflagged deaths.

score_labels() measures the cards against expert labels (evals/death_labels.toml): for each labelled death,
did the headline point at what actually went wrong?
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from . import deaths as D
from .demo_store import DemoStore
from .errors import AnalystError

MAX_CARDS = 8

TITLES = {
    "isolated": "Died where nobody could trade",
    "untraded_with_support": "The trade was there and didn't come",
    "dry_opening": "Opening death without support",
}


def player_review(store: DemoStore, demo_id: str, steamid: int, round_num: int | None = None,
                  model: Any = None) -> dict[str, Any]:
    """death_review with everything it can use: positions and walking distance when the .dem can be found,
    swing when a win probability model is given. Missing pieces are noted, never guessed."""
    from . import replay_cache

    meta = store.meta(demo_id)
    try:
        pos = replay_cache.positions(store, demo_id)
        dist, kind = replay_cache.walking_distance(meta["map"])
    except AnalystError:
        pos, dist, kind = None, None, "unavailable"
    swing = None
    if model is not None:
        from .winprob import swing as sw
        from .winprob.features import states_for

        states = sw.with_probabilities(model, states_for(store, demo_id))
        swing = {(c["round"], c["tick"]): c["swing"] for c in sw.credits(states)
                 if c["role"] == "death" and c["steamid"] == steamid}
    t = lambda n: store.table(demo_id, n)  # noqa: E731
    out = D.death_review(t("kills"), t("rounds"), t("round_players"), t("grenade_throws"), t("blinds"), meta["tick_rate"],
                         positions=pos, distance=dist, distance_kind=kind, steamid=steamid, round_num=round_num,
                         swing=swing, bomb=t("bomb"), limit=10_000)
    if pos is None:
        out["note_positions"] = "The original .dem wasn't found, so spacing (teammate distances) is missing."
    if swing is None:
        out["note_swing"] = "No win probability model yet (cs2-agent model train), so swing costs are missing."
    return out


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def card(d: dict[str, Any], distance_kind: str | None) -> dict[str, Any]:
    head = d["headline"]
    kind = "on foot" if distance_kind == "walking" else "straight-line"
    facts = []
    if head == "isolated":
        facts.append(f"Nearest living teammate {d['nearest_teammate_units']} units away ({kind}); "
                     f"none within {int(D.TRADE_RANGE)} to trade. Not traded.")
    elif head == "untraded_with_support":
        facts.append(f"{_plural(d['teammates_within_trade_range'], 'teammate')} within {int(D.TRADE_RANGE)} units "
                     f"(nearest {d['nearest_teammate_units']}), and nobody traded.")
    elif head == "dry_opening":
        facts.append(f"First death of the round. No teammate flash in the {D.SUPPORT_WINDOW_S:g}s before, "
                     "and the killer wasn't blind.")
    if d.get("carrying_bomb"):
        facts.append("Carrying the bomb: it dropped where you died.")
    if "dry_opening" in d["flags"] and head != "dry_opening":
        facts.append("It was the round's opening death, with no flash support.")
    if "died_with_utility" in d["flags"]:
        held = ", ".join(f"{n} {g}" for g, n in d["utility_held"].items())
        facts.append(f"Still holding {held} (${d['utility_value']}).")
    if "died_blind" in d["flags"]:
        facts.append(f"Blind for another {d['blind_left_s']}s.")

    context = [f"{d['time_s']}s into the round" if d.get("time_s") is not None else None,
               ("after the plant" if d["after_plant"] else None),
               ("opening death" if d["opening_death"] and "dry_opening" not in d["flags"] else None),
               f"{_plural(d['teammates_alive'], 'teammate')} alive",
               f"round {'won' if d['round_won'] else 'lost'}"]
    if head == "isolated":
        question = ("The bomb was down: were you retaking alone? Could you have waited to retake with the team?"
                    if d["after_plant"] and d["side"] == "CT" else
                    "The bomb was down: did this fight need taking, or could you have played around the bomb with the team?"
                    if d["after_plant"] else
                    "You had the bomb: did this fight need taking alone, or could you have waited for the team?"
                    if d.get("carrying_bomb") else
                    "Was this fight part of the plan, or could you have taken it closer to a teammate?")
    elif head == "untraded_with_support":
        question = "Who was in position to trade, and what stopped them: timing, angle, or utility?"
    else:
        question = "Could a flash, a teammate's swing, or a different angle have set this duel up?"
    return {
        "round": d["round"], "tick": d["tick"], "time_s": d.get("time_s"), "killer": d["killer"], "weapon": d["weapon"],
        "side": d["side"], "headline": head, "points_at": d["flag_points_at"], "title": TITLES[head],
        "facts": facts, "context": [c for c in context if c], "question": question,
        "swing_pp": d.get("swing_pp"), "flags": d["flags"],
    }


def coach_cards(review: dict[str, Any], limit: int = MAX_CARDS) -> dict[str, Any]:
    rows = review["deaths_detail"]
    flagged = [d for d in rows if d["headline"]]
    has_swing = any("swing_pp" in d for d in rows)
    flagged.sort(key=(lambda d: (d.get("swing_pp", 0.0), d["round"])) if has_swing else (lambda d: d["round"]))
    lurk = [d["round"] for d in rows if d.get("isolated_while_lurking")]
    state_only = [d["round"] for d in rows if not d["headline"] and d["flags"]]
    clean = [d["round"] for d in rows if not d["flags"] and not d.get("isolated_while_lurking")]
    not_shown = []
    if lurk:
        not_shown.append(f"{_plural(len(lurk), 'isolated death')} while lurking (a round spent apart from the team, before the plant) "
                         f"(rounds {', '.join(map(str, lurk))})")
    if state_only:
        not_shown.append(f"{_plural(len(state_only), 'death')} flagged only for utility held or blind time, which say what the "
                         f"death cost, not why it happened (rounds {', '.join(map(str, state_only))})")
    if clean:
        not_shown.append(f"{_plural(len(clean), 'death')} with no flag (rounds {', '.join(map(str, clean))})")
    return {
        "deaths": review["deaths"],
        "cards": [card(d, review.get("distance_kind")) for d in flagged[:limit]],
        "more_flagged": max(0, len(flagged) - limit),
        "by_whom": {w: sum(d["flag_points_at"] == w for d in flagged) for w in ("player", "teammates")},
        "not_shown": not_shown,
        "lurker": next((p["lurker"] for p in review.get("spacing_profile", {}).values()), None),
        "sorted_by": "swing (costliest first)" if has_swing else "round",
        "notes": [review[k] for k in ("note_positions", "note_swing") if k in review],
    }


# --------------------------------------------------------------- scoring against expert labels

def load_labels(path: Path) -> list[dict[str, Any]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out = data.get("death", [])
    for i, lab in enumerate(out):
        missing = {"demo", "player", "round", "flag_was_the_issue", "mistake_by", "flags"} - set(lab)
        if missing:
            raise ValueError(f"{path}: death #{i + 1} is missing {sorted(missing)}.")
        if lab["mistake_by"] not in ("player", "teammates", "none"):
            raise ValueError(f"{path}: death #{i + 1} mistake_by must be player, teammates or none.")
    return out


def score_labels(labels: list[dict[str, Any]], rows: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Each label against the review's row for that death (None if it wasn't found).

    Two ways of reading the review, so the change is visible:
      any_flag  every flag is an alarm (how the review read before headlines and the lurker profile)
      headline  only the headline flag is an alarm
    A label with flag_was_the_issue = true wants an alarm on one of the flags it names (a hit, else a miss);
    false wants silence (else a false alarm). Blame: the label's mistake_by vs the headline's points_at.
    """
    per, tallies = [], {m: {"hit": 0, "miss": 0, "false_alarm": 0, "quiet": 0} for m in ("any_flag", "headline")}
    blame_agree = blame_n = 0
    for lab, row in zip(labels, rows):
        entry = {"demo": lab["demo"], "player": lab["player"], "round": lab["round"],
                 "issue": lab["flag_was_the_issue"], "label_flags": lab["flags"], "mistake_by": lab["mistake_by"]}
        if row is None:
            per.append({**entry, "found": False})
            continue
        named = set(lab["flags"])
        alarms = {"any_flag": set(row["flags"]), "headline": {row["headline"]} - {None}}
        outcomes = {}
        for mode, alarm in alarms.items():
            if lab["flag_was_the_issue"]:
                o = "hit" if alarm & named else "miss"
            else:
                o = "false_alarm" if alarm else "quiet"
            tallies[mode][o] += 1
            outcomes[mode] = o
        whose = row["flag_points_at"] or "none"
        blame_n += 1
        blame_agree += whose == lab["mistake_by"]
        per.append({**entry, "found": True, "flags_now": row["flags"], "headline": row["headline"], "points_at": whose,
                    "lurking": row.get("lurking"), "nearest_teammate_units": row.get("nearest_teammate_units"),
                    "outcomes": outcomes})

    def rates(t: dict[str, int]) -> dict[str, Any]:
        alarms = t["hit"] + t["false_alarm"]
        issues = t["hit"] + t["miss"]
        return {**t, "precision": round(t["hit"] / alarms, 3) if alarms else None,
                "recall": round(t["hit"] / issues, 3) if issues else None}

    return {"labels": len(labels), "found": sum(p["found"] for p in per), "per_death": per,
            "modes": {m: rates(t) for m, t in tallies.items()},
            "blame_agreement": round(blame_agree / blame_n, 3) if blame_n else None}


def _file_name(path: str | None) -> str:
    """The file name from a path written on any OS (meta.json keeps the path as the ingesting machine wrote it)."""
    return re.split(r"[\\/]", path or "")[-1]


def find_demo(store: DemoStore, name: str, demo_dir: Path | None = None) -> str | None:
    """A cached demo by demo_id or file name (case-insensitive). If it isn't cached but the file is in
    demo_dir, it's parsed now."""
    for d in store.list_demos():
        if name == d["demo_id"] or name.lower() == _file_name(d.get("source")).lower():
            return d["demo_id"]
    if demo_dir is not None and (demo_dir / name).is_file():
        return store.ingest(demo_dir / name)["demo_id"]
    return None


def label_rows(store: DemoStore, labels: list[dict[str, Any]], model: Any = None,
               demo_dir: Path | None = None) -> tuple[list[dict | None], dict]:
    """The review row for every labelled death, reviewing each (demo, player) once. Also each player's
    spacing profile, since it decides the lurker context."""
    from .analysis import resolve_player

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict | None] = []
    profiles: dict[str, Any] = {}
    for lab in labels:
        did = find_demo(store, lab["demo"], demo_dir)
        if did is None:
            rows.append(None)
            continue
        key = (did, lab["player"])
        if key not in cache:
            sid = resolve_player(store.table(did, "players"), lab["player"])["steamid"]
            cache[key] = player_review(store, did, sid, model=model)
            profiles[f"{lab['demo']} / {lab['player']}"] = {
                **next(iter(cache[key].get("spacing_profile", {}).values()), {}),
                **{k: cache[key][k] for k in ("note_positions",) if k in cache[key]}}
        rows.append(next((d for d in cache[key]["deaths_detail"] if d["round"] == lab["round"]), None))
    return rows, profiles

