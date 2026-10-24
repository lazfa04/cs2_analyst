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
Headline flags say why the death happened and whose it was to fix (isolated, dry_opening: the player's;
untraded_with_support: the teammates'). died_with_utility and died_blind describe what the death cost or the
state the player was in, so they're shown with a headline but never lead a coaching card on their own: in the
expert labels both utility flags were factually right and neither was the mistake.

lurking: on T, before the plant, in a round the player spent apart from the team (spacing_profile). An isolated
death in that round is the lurk, not a mistake; it stays in the facts (nearest teammate, lurking) but not in the
flags. Leaving the team and dying alone is still flagged. spacing_profile also says who lurks habitually.

traded uses the usual definition (the killer died to the victim's team within TRADE_WINDOW_S), because awpy's
flag credits a revenge kill to only one of the killer's victims. carrying_bomb: the bomb dropped where they died.

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
TRADE_WINDOW_S = 5.25        # the killer dying to the victim's team this soon after counts as a trade (awpy's window)
SUPPORT_WINDOW_S = 3.0       # a teammate's flash this soon before an opening death counts as support
UTILITY_FLAG_VALUE = 500     # $ of grenades still held at death
BLIND_FLAG_S = 0.5
NADE_VALUE = {"flash": 200, "smoke": 300, "HE": 300}
FIRE_VALUE = {"T": 400, "CT": 500}  # molotov vs incendiary
BUY_COLS = {"flash": "flashbangs", "smoke": "smoke_grenades", "HE": "he_grenades", "fire": "fire_grenades"}
HEADLINE_FLAGS = {"isolated": "player", "dry_opening": "player", "untraded_with_support": "teammates"}

LURK_RANGE = 1000.0      # straight-line units to the nearest living teammate
LURK_SKIP_S = 10.0       # ignore the walk out of spawn
LURK_SHARE = 0.4         # apart from the team before contact in at least this share of T rounds
LURK_MIN_ROUNDS = 4


def spacing_profile(positions: pl.DataFrame, rounds: pl.DataFrame, kills: pl.DataFrame,
                    round_players: pl.DataFrame, tick_rate: float, side: str = "T",
                    bomb: pl.DataFrame | None = None) -> dict[int, dict[str, Any]]:
    """How far each player is from their team on one side, round by round, before the plant.

    Per round: the median straight-line distance to the nearest living teammate, over the snapshots where the
    player is alive, from LURK_SKIP_S after freeze end until the plant (or the round's end). A round counts as
    "apart" when that median is over LURK_RANGE: the player spent most of that round away from everyone, which
    is a lurk or a split, a choice rather than a moment. A lurker (the role) is apart in LURK_SHARE+ of their
    rounds on the side and is the most apart player on their team.

    The window used to end at the round's first kill. In pugs the first kill often comes 5-9 seconds in, so
    many rounds had no window at all and the rest measured a few seconds near spawn; found on the author's
    own demo, where it reported him never apart in a round he lurked. Straight-line distance understates how
    far apart players are through walls, so this errs towards not calling a round a lurk.
    """
    live = live_rounds(rounds).select("round_num", "freeze_end_tick", "end_tick")
    plants = (bomb.filter(pl.col("event") == "finish_plant").group_by("round_num").agg(pl.col("tick").min().alias("plant"))
              if bomb is not None and bomb.height else pl.DataFrame(schema={"round_num": pl.Int32, "plant": pl.Int32}))
    win = (live.join(plants.with_columns(pl.col("round_num").cast(live["round_num"].dtype)), on="round_num", how="left")
           .with_columns((pl.col("freeze_end_tick") + int(LURK_SKIP_S * tick_rate)).cast(pl.Int64).alias("lo"),
                         pl.coalesce("plant", "end_tick").cast(pl.Int64).alias("hi"))
           .select("round_num", "lo", "hi").sort("lo"))
    pos = (positions.filter((pl.col("side") == side) & (pl.col("health") > 0))
           .select(pl.col("tick").cast(pl.Int64), pl.col("steamid").cast(pl.UInt64), "x", "y", pl.col("z").fill_null(0.0))
           .sort("tick"))
    if pos.is_empty() or win.is_empty():
        return {}
    pos = (pos.join_asof(win, left_on="tick", right_on="lo", strategy="backward")
           .filter(pl.col("round_num").is_not_null() & (pl.col("tick") < pl.col("hi")))
           .select("tick", "round_num", "steamid", "x", "y", "z"))
    pairs = (pos.join(pos.select("tick", pl.col("steamid").alias("mate"), pl.col("x").alias("mx"),
                                 pl.col("y").alias("my"), pl.col("z").alias("mz")), on="tick")
             .filter(pl.col("steamid") != pl.col("mate"))
             .with_columns(((pl.col("x") - pl.col("mx")) ** 2 + (pl.col("y") - pl.col("my")) ** 2
                            + (pl.col("z") - pl.col("mz")) ** 2).sqrt().alias("d")))
    if pairs.is_empty():
        return {}
    per_round = (pairs.group_by("round_num", "steamid", "tick").agg(pl.col("d").min())
                 .group_by("round_num", "steamid").agg(pl.col("d").median().alias("median")))
    names = dict(round_players.select(pl.col("steamid").cast(pl.UInt64), "name").unique("steamid").iter_rows())
    # Team key: the first round a player was on this side (both halves of a team share it).
    team = dict(round_players.filter(pl.col("side") == side).group_by(pl.col("steamid").cast(pl.UInt64))
                .agg(pl.col("round_num").min()).iter_rows())
    out: dict[int, dict[str, Any]] = {}
    for sid, g in per_round.group_by("steamid"):
        sid = sid[0]
        apart = sorted(g.filter(pl.col("median") > LURK_RANGE)["round_num"].to_list())
        out[sid] = {"name": names.get(sid), "rounds": g.height, "rounds_apart": len(apart), "apart_rounds": apart,
                    "share_apart": round(len(apart) / g.height, 2), "median_units": round(g["median"].median()),
                    "lurker": False}
    for key in set(team.values()):
        members = [s for s in out if team.get(s) == key]
        best = max(members, key=lambda s: (out[s]["share_apart"], out[s]["median_units"]), default=None)
        if best is not None and out[best]["rounds"] >= LURK_MIN_ROUNDS and out[best]["share_apart"] >= LURK_SHARE:
            out[best]["lurker"] = True
    return out


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
    freeze = dict(live_rounds(rounds).select("round_num", "freeze_end_tick").iter_rows())
    profile = spacing_profile(positions, rounds, kills, round_players, tick_rate, bomb=bomb) if positions is not None else {}
    trade_ticks = int(TRADE_WINDOW_S * tick_rate)
    drops = ({(b["round_num"], b["steamid"], b["tick"]) for b in bomb.filter(pl.col("event") == "drop").iter_rows(named=True)}
             if bomb is not None and bomb.height and "steamid" in bomb.columns else set())

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

        # Traded: the killer died to the victim's team within TRADE_WINDOW_S. awpy's victim_traded credits one
        # revenge kill to only the first of the killer's victims, so a double kill avenged counts one trade.
        revenge = feed.filter((pl.col("round_num") == rnd) & (pl.col("victim_steamid") == killer)
                              & (pl.col("attacker_side") == side) & pl.col("tick").is_between(tick + 1, tick + trade_ticks))
        traded = bool(d.get("victim_traded")) or (killer is not None and revenge.height > 0)
        traded_after = round((revenge["tick"].min() - tick) / tick_rate, 1) if revenge.height else None
        carrying_bomb = any((rnd, vic, t) in drops for t in range(tick - 2, tick + 3))
        after_plant = rnd in plants and plants[rnd] < tick
        prof = profile.get(vic)
        # Lurking: on T, before the plant, in a round the player spent apart from the team. Isolation that lasted
        # the round is the plan; the mistake pattern is leaving the team and dying alone, which this doesn't excuse.
        lurking = (None if positions is None else
                   bool(side == "T" and prof and rnd in prof["apart_rounds"] and not after_plant))
        flags = []
        isolated = not traded and nearest is not None and nearest > TRADE_RANGE
        if isolated and not lurking:
            flags.append("isolated")
        if opening and not team_flash and not killer_blind:
            flags.append("dry_opening")
        if value >= UTILITY_FLAG_VALUE:
            flags.append("died_with_utility")
        if blind_left >= BLIND_FLAG_S:
            flags.append("died_blind")
        if not traded and within:
            flags.append("untraded_with_support")
        headline = next((f for f in flags if f in HEADLINE_FLAGS), None)
        row = {
            "round": rnd, "tick": tick, "time_s": round((tick - freeze[rnd]) / tick_rate, 1) if rnd in freeze else None,
            "victim": d["victim_name"], "side": side, "killer": d.get("attacker_name"),
            "weapon": d["weapon"], "round_won": winners.get(rnd) == side, "traded": traded, "traded_after_s": traded_after,
            "carrying_bomb": carrying_bomb,
            "opening_death": opening, "after_plant": after_plant, "lurking": lurking,
            "isolated_while_lurking": bool(isolated and lurking),
            "teammates_alive": len(mates),
            "nearest_teammate_units": nearest, "teammates_within_trade_range": within,
            "team_flash_before": team_flash if opening else None, "killer_was_blind": killer_blind,
            "blind_left_s": blind_left, "utility_held": {g: n for g, n in held.items() if n}, "utility_value": value,
            "flags": flags, "headline": headline, "flag_points_at": HEADLINE_FLAGS.get(headline),
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
        "isolated_while_lurking": {"count": sum(r["isolated_while_lurking"] for r in rows),
                                   "rounds": [r["round"] for r in rows if r["isolated_while_lurking"]]},
        "spacing_profile": {v["name"]: {k: v[k] for k in ("rounds", "rounds_apart", "share_apart", "median_units", "lurker")}
                            for sid, v in profile.items() if steamid is None or sid == steamid},
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
                  f"lurking: on T, before the plant, in a round the player spent apart from the team (median distance to the nearest "
                  f"teammate over {int(LURK_RANGE)} units from {LURK_SKIP_S:g}s in until the plant); isolated deaths then are "
                  f"counted in isolated_while_lurking instead of flagged. traded: the killer died to the victim's team within "
                  f"{TRADE_WINDOW_S:g}s. headline: the flag that says why the death happened; "
                  "died_with_utility and died_blind never headline. "
                  "Deaths after the round was decided (saving, hunted down) aren't reviewed but are counted in "
                  "deaths_after_round_end, which is why the scoreboard's death total can be higher. These are facts to look at, not verdicts: "
                  "an isolated lurk can be the plan, and a dry peek can be right on an eco or a retake."),
    }
