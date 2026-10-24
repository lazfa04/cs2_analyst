"""Death review: the measurable context of every death, the raw material for coaching.

For each death it records what the demo can actually show, never a judgement:
  spacing      walking distance to the nearest living teammate, and how many were close enough
               to trade (TRADE_RANGE), from the position snapshots and the map's nav mesh
  traded       whether a teammate traded the death
  opening      whether it was the round's first kill, and if so whether anyone flashed for the
               player or the killer was blind (a "dry" opening death has neither)
  blind        whether the player was flashed when they died, and how much blind time was left
  utility      grenades still held (freeze-end buy minus what they threw before dying), and value
  swing        how much the death cost the team's chance to win the round (when the model exists)

Flags are thresholds on those facts, named for what they measure:
  isolated           untraded, and the nearest teammate was more than TRADE_RANGE away on foot
  dry_opening        died first, with no flash from a teammate and the killer not blind
  died_with_utility  held UTILITY_FLAG_VALUE or more in grenades
  died_blind         still blind for BLIND_FLAG_S or more
  untraded_with_support  untraded although a teammate was within TRADE_RANGE: the trade was there to take
after_plant says whether the bomb was already down: an isolated fight after your own team's plant is a
different thing from an isolated lurk before it (the expert labels separated exactly those two).
A flag is a fact to look at, not a verdict: an isolated lurker death can be the plan, and a dry
opening can be the right call on an eco. The agent and the coach have to say it that way.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

from .analysis import live_rounds, weapon_label

TRADE_RANGE = 800.0          # walking units; at ~250 u/s that's ~3s, about the trade window
SUPPORT_WINDOW_S = 3.0       # a teammate's flash this soon before an opening death counts as support
UTILITY_FLAG_VALUE = 500     # $ of grenades still held at death
BLIND_FLAG_S = 0.5
NADE_VALUE = {"flash": 200, "smoke": 300, "HE": 300}
FIRE_VALUE = {"T": 400, "CT": 500}  # molotov vs incendiary
BUY_COLS = {"flash": "flashbangs", "smoke": "smoke_grenades", "HE": "he_grenades", "fire": "fire_grenades"}


def _frame(positions: pl.DataFrame, tick: int, max_age: int) -> pl.DataFrame:
    """Everyone's latest snapshot at or before `tick`, if it's recent (snapshots are 8 ticks apart;
    anything older belongs to another moment, or another round)."""
    before = positions.filter(pl.col("tick").is_between(tick - max_age, tick))
    if before.is_empty():
        return before
    last = before["tick"].max()
    return before.filter(pl.col("tick") == last)


def death_review(
    kills: pl.DataFrame,
    rounds: pl.DataFrame,
    round_players: pl.DataFrame,
    grenade_throws: pl.DataFrame,
    blinds: pl.DataFrame,
    tick_rate: float,
    positions: pl.DataFrame | None = None,
    distance: Callable[[tuple, tuple], float] | None = None,
    distance_kind: str = "straight-line",
    steamid: int | None = None,
    round_num: int | None = None,
    swing: dict[tuple[int, int], float] | None = None,
    bomb: pl.DataFrame | None = None,
    limit: int = 40,
) -> dict[str, Any]:
    live = live_rounds(rounds).select("round_num", "winner_side")
    winners = dict(live.iter_rows())
    feed = kills.filter(~pl.col("after_round_end")).sort("tick")
    firsts = {r["round_num"]: r["tick"] for r in feed.group_by("round_num", maintain_order=True).first().iter_rows(named=True)}
    deaths = feed.filter(pl.col("round_num").is_in(list(winners)))
    late = kills.filter(pl.col("after_round_end") & pl.col("round_num").is_in(list(winners)))
    if steamid is not None:
        deaths = deaths.filter(pl.col("victim_steamid") == steamid)
        late = late.filter(pl.col("victim_steamid") == steamid)
    if round_num is not None:
        deaths = deaths.filter(pl.col("round_num") == round_num)
        late = late.filter(pl.col("round_num") == round_num)
    roster = {(r["round_num"], r["steamid"]): r for r in round_players.iter_rows(named=True)}
    plants: dict[int, int] = {}
    if bomb is not None and bomb.height:
        for b in bomb.filter(pl.col("event") == "finish_plant").sort("tick").iter_rows(named=True):
            plants.setdefault(b["round_num"], b["tick"])
    window = int(SUPPORT_WINDOW_S * tick_rate)

    rows = []
    for d in deaths.with_columns(weapon_label()).iter_rows(named=True):
        rnd, tick, vic, side = d["round_num"], d["tick"], d["victim_steamid"], d["victim_side"]
        killer = d.get("attacker_steamid")
        here = (d.get("victim_x"), d.get("victim_y"), d.get("victim_z") or 0.0)

        # teammates alive at the moment of death, and how far they were
        dead_before = set(feed.filter((pl.col("round_num") == rnd) & (pl.col("tick") < tick))["victim_steamid"].to_list())
        mates = [sid for (r, sid), p in roster.items() if r == rnd and p["side"] == side and sid != vic and sid not in dead_before]
        nearest = within = None
        if positions is not None and mates and here[0] is not None:
            frame = _frame(positions.filter(pl.col("steamid").is_in(mates) & (pl.col("health") > 0)), tick,
                           max_age=int(tick_rate))
            measure = distance or (lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5)
            dists = [measure(here, (p["x"], p["y"], p["z"] or 0.0)) for p in frame.iter_rows(named=True)]
            if dists:
                nearest = round(min(dists))
                within = sum(x <= TRADE_RANGE for x in dists)

        # opening duel support: a teammate's flash just before, or the killer blind at the time
        opening = firsts.get(rnd) == tick
        team_flash = killer_blind = False
        if opening:
            team_flash = grenade_throws.filter(
                (pl.col("round_num") == rnd) & (pl.col("side") == side) & (pl.col("steamid") != vic)
                & (pl.col("grenade") == "flash") & pl.col("tick").is_between(tick - window, tick)
            ).height > 0
        if killer is not None:
            kb = blinds.filter(
                (pl.col("round_num") == rnd) & (pl.col("victim_steamid") == killer) & (pl.col("tick") <= tick)
                & ((pl.col("tick") + pl.col("duration") * tick_rate) >= tick))
            killer_blind = kb.height > 0
        vb = blinds.filter(
            (pl.col("round_num") == rnd) & (pl.col("victim_steamid") == vic) & (pl.col("tick") <= tick)
            & ((pl.col("tick") + pl.col("duration") * tick_rate) >= tick))
        blind_left = round(max(((b["tick"] + b["duration"] * tick_rate - tick) / tick_rate for b in vb.iter_rows(named=True)),
                               default=0.0), 1)

        # utility still held: freeze-end buy minus throws before death
        buy = roster.get((rnd, vic), {})
        thrown = dict(grenade_throws.filter((pl.col("round_num") == rnd) & (pl.col("steamid") == vic) & (pl.col("tick") < tick))
                      .group_by("grenade").len().iter_rows())
        held = {}
        for g, col in BUY_COLS.items():
            name = "molotov/incendiary" if g == "fire" else g
            held[name] = max(0, int(buy.get(col) or 0) - int(thrown.get(name, 0)))
        value = sum(n * (FIRE_VALUE.get(side, 450) if g == "molotov/incendiary" else NADE_VALUE[g]) for g, n in held.items())

        traded = bool(d.get("victim_traded"))
        flags = []
        if not traded and nearest is not None and nearest > TRADE_RANGE:
            flags.append("isolated")
        if opening and not team_flash and not killer_blind:
            flags.append("dry_opening")
        if value >= UTILITY_FLAG_VALUE:
            flags.append("died_with_utility")
        if blind_left >= BLIND_FLAG_S:
            flags.append("died_blind")
        if not traded and within:
            flags.append("untraded_with_support")
        row = {
            "round": rnd, "tick": tick, "victim": d["victim_name"], "side": side, "killer": d.get("attacker_name"),
            "weapon": d["weapon"], "round_won": winners.get(rnd) == side, "traded": traded,
            "opening_death": opening, "after_plant": rnd in plants and plants[rnd] < tick,
            "teammates_alive": len(mates),
            "nearest_teammate_units": nearest, "teammates_within_trade_range": within,
            "team_flash_before": team_flash if opening else None, "killer_was_blind": killer_blind,
            "blind_left_s": blind_left, "utility_held": {g: n for g, n in held.items() if n}, "utility_value": value,
            "flags": flags,
        }
        if swing is not None:
            row["swing_pp"] = round(100 * swing.get((rnd, tick), 0.0), 1)
        rows.append(row)

    def summary(flag: str) -> dict[str, Any]:
        hit = [r for r in rows if flag in r["flags"]]
        out = {"count": len(hit), "rounds": [r["round"] for r in hit]}
        if swing is not None:
            out["swing_pp"] = round(sum(r.get("swing_pp", 0.0) for r in hit), 1)
        return out

    costly = sorted(rows, key=lambda r: r.get("swing_pp", 0.0))[:5] if swing is not None else []
    return {
        "deaths": len(rows),
        "untraded": sum(not r["traded"] for r in rows),
        "flags": {f: summary(f) for f in ("isolated", "untraded_with_support", "dry_opening", "died_with_utility",
                                          "died_blind")},
        # The scoreboard counts these too, so its death total can be higher than `deaths`.
        "deaths_after_round_end": {"count": late.height, "rounds": sorted(late["round_num"].to_list())},
        "costliest_deaths": [{k: r[k] for k in ("round", "tick", "killer", "swing_pp", "flags")} for r in costly],
        "distance_kind": distance_kind if positions is not None else None,
        "deaths_detail": rows[:limit],
        "truncated": len(rows) > limit,
        "rules": (f"isolated: untraded and the nearest living teammate was more than {int(TRADE_RANGE)} units away "
                  f"({distance_kind} distance). dry_opening: died first in the round with no teammate flash in the "
                  f"{SUPPORT_WINDOW_S:g}s before and the killer not blind. died_with_utility: held ${UTILITY_FLAG_VALUE}+ of "
                  f"grenades. died_blind: {BLIND_FLAG_S}s+ of blind left. untraded_with_support: untraded although a "
                  f"teammate was within {int(TRADE_RANGE)} units (the trade was there; a teammate's miss, not the victim's). "
                  "Deaths after the round was decided (saving, hunted down) aren't reviewed but are counted in "
                  "deaths_after_round_end, which is why the scoreboard's death total can be higher. These are facts to look at, not verdicts: "
                  "an isolated lurk can be the plan, and a dry peek can be right on an eco or a retake."),
    }
