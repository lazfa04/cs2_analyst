"""Analysis layer: pure functions from Polars frames to small, JSON-ready dicts.

Rules every function here follows (they are the tool contract):
  * No I/O. Frames in, dicts out. That makes everything unit-testable and
    reusable by the agent, the evals and the later ML pipeline.
  * Output is a SUMMARY, never a raw table. Tool results land in the model's
    context window, so 20 well-chosen rows beat 2,000 raw ones.
  * Anything that points at a moment in the demo carries `round` and `tick`.
    That's what lets an agent cite evidence and a verifier re-check it.
  * Lists are capped with `limit`, and the result says when it was truncated.
"""

from __future__ import annotations

import difflib
from typing import Any

import polars as pl

from .errors import PlayerNotFound, RoundNotFound

GRENADE_WEAPONS = {"hegrenade", "inferno", "molotov", "incgrenade", "flashbang", "smokegrenade", "decoy"}

# Team/self blinds shorter than this are "minor": usually the edge of a flash while turned away.
MINOR_BLIND_S = 1.5

# The engine calls burning fire "inferno", which reads like the map. All fire variants get one
# player-facing name, so fire damage isn't split three ways either.
WEAPON_LABELS = {"inferno": "molotov/incendiary", "molotov": "molotov/incendiary", "incgrenade": "molotov/incendiary"}


def weapon_label(col: str = "weapon") -> pl.Expr:
    return pl.col(col).replace(WEAPON_LABELS)


# ---------------------------------------------------------------- helpers

def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    return _clean(df.to_dicts())


def _seconds(ticks: int | None, tick_rate: float) -> float | None:
    return None if ticks is None else round(ticks / tick_rate, 1)


def live_rounds(rounds: pl.DataFrame) -> pl.DataFrame:
    """Rounds that count: knife side-decider rounds are excluded."""
    if "is_knife_round" in rounds.columns:
        return rounds.filter(~pl.col("is_knife_round"))
    return rounds


# ---------------------------------------------------------------- players & teams

def resolve_player(players: pl.DataFrame, query: str) -> dict[str, Any]:
    """Map whatever the model typed (name, partial name, steamid) to one player.

    Fails with the full roster in the message so the model can retry with an
    exact name instead of guessing.
    """
    q = query.strip()
    roster = players.select("steamid", "name").unique("steamid")
    if q.isdigit():
        hit = roster.filter(pl.col("steamid") == int(q))
        if hit.height:
            return hit.row(0, named=True)
    lowered = roster.with_columns(pl.col("name").str.to_lowercase().alias("_l"))
    for cond in (pl.col("_l") == q.lower(), pl.col("_l").str.contains(q.lower(), literal=True)):
        hit = lowered.filter(cond)
        if hit.height == 1:
            return hit.drop("_l").row(0, named=True)
        if hit.height > 1:
            names = hit["name"].to_list()
            raise PlayerNotFound(f"'{query}' matches several players: {names}. Use one exact name.")
    names = roster["name"].to_list()
    close = difflib.get_close_matches(q, names, n=3, cutoff=0.5)
    hint = f" Did you mean {close}?" if close else ""
    raise PlayerNotFound(f"No player '{query}' in this demo.{hint} Players: {names}.")


def team_labels(round_players: pl.DataFrame, players: pl.DataFrame) -> dict[int, str]:
    """steamid -> stable team label for the whole match.

    Sides swap at halftime, so "CT" is not a team. We anchor each player to
    the side they had in the first round they appear in, and use the clan
    tag when the demo has one (pro/league demos do, matchmaking doesn't).
    """
    first = (
        round_players.sort("round_num")
        .group_by("steamid", maintain_order=True)
        .first()
        .select("steamid", pl.col("side").alias("start_side"))
    )
    clans = players.select("steamid", "team_clan_name") if "team_clan_name" in players.columns else None
    labels: dict[int, str] = {}
    for row in first.iter_rows(named=True):
        label = f"Team {'A' if row['start_side'] == 'CT' else 'B'} (started {row['start_side']})"
        labels[row["steamid"]] = label
    if clans is not None:
        # Clan tags are recorded per player; use the most common tag per starting side.
        tagged = first.join(clans, on="steamid").drop_nulls("team_clan_name")
        for side, grp in tagged.group_by("start_side"):
            tag = grp["team_clan_name"].mode().first()
            for sid in first.filter(pl.col("start_side") == side[0])["steamid"]:
                labels[sid] = tag
    return labels


def _round_winner_teams(rounds: pl.DataFrame, round_players: pl.DataFrame, labels: dict[int, str]) -> dict[int, str]:
    """round_num -> label of the team that won it (side is per-round; team is not)."""
    rp = round_players.with_columns(
        pl.col("steamid").replace_strict(labels, default=None, return_dtype=pl.Utf8).alias("team")
    )
    joined = live_rounds(rounds).select("round_num", "winner_side").join(
        rp, left_on=["round_num", "winner_side"], right_on=["round_num", "side"], how="left"
    )
    out = joined.group_by("round_num").agg(pl.col("team").drop_nulls().mode().first())
    return {r["round_num"]: r["team"] for r in out.iter_rows(named=True)}


# ---------------------------------------------------------------- tools' logic

def match_overview(
    meta: dict[str, Any],
    rounds: pl.DataFrame,
    stats: pl.DataFrame,
    players: pl.DataFrame,
    round_players: pl.DataFrame,
    kills: pl.DataFrame | None = None,
    damages: pl.DataFrame | None = None,
    bomb: pl.DataFrame | None = None,
) -> dict[str, Any]:
    labels = team_labels(round_players, players)
    winners = _round_winner_teams(rounds, round_players, labels)
    live = live_rounds(rounds)

    teams: dict[str, dict[str, int]] = {}
    for r in live.iter_rows(named=True):
        team = winners.get(r["round_num"])
        if team is None:
            continue
        t = teams.setdefault(team, {"rounds_won": 0, "ct_rounds_won": 0, "t_rounds_won": 0})
        t["rounds_won"] += 1
        t[f"{r['winner_side'].lower()}_rounds_won"] += 1

    if kills is not None:
        # awpy's clutch columns count every last-alive situation (even 1v5s on an eco and post-round
        # deaths). Replace them with the clutch rules get_clutches uses, so the two never disagree.
        counts = clutch_counts(last_alive_situations(kills, rounds, round_players, damages, bomb))
        stats = stats.with_columns(
            pl.col("steamid").map_elements(lambda s_: counts.get(s_, (0, 0))[0], return_dtype=pl.Int64).alias("clutches_won"),
            pl.col("steamid").map_elements(lambda s_: counts.get(s_, (0, 0))[1], return_dtype=pl.Int64).alias("clutches_played"),
        )
    multi = {"k3": {}, "k4": {}, "aces": {}}
    if kills is not None:
        for r in _multikill_rounds(kills, rounds):
            key = {3: "k3", 4: "k4"}.get(min(r["kills"], 5), "aces" if r["kills"] >= 5 else None)
            if key:
                multi[key][r["_sid"]] = multi[key].get(r["_sid"], 0) + 1
    stats = stats.with_columns(
        pl.col("steamid").map_elements(lambda s_, m=m: m.get(s_, 0), return_dtype=pl.Int64).alias(name)
        for name, m in multi.items()
    )
    board = (
        with_rating(stats)
        .with_columns(
            pl.col("steamid").replace_strict(labels, default="unknown", return_dtype=pl.Utf8).alias("team")
        )
        .with_columns(
            pl.col("rating_raw").rank("ordinal", descending=True).cast(pl.Int32).alias("match_rank"),
            pl.col("rating_raw").rank("ordinal", descending=True).over("team").cast(pl.Int32).alias("team_rank"),
        )
        .sort(["team", "rating_raw"], descending=[False, True])
        .select(
            "name", "team", "rating", "team_rank", "match_rank", "impact", "kills", "deaths", "assists",
            pl.col("adr").round(1), pl.col("kast").round(1), pl.col("headshot_pct").round(1).alias("hs_pct"),
            "opening_kills", "opening_deaths", "traded_deaths",
            "clutches_won", "clutches_played", "k3", "k4", "aces", "utility_damage",
        )
    )
    return {
        "demo_id": meta["demo_id"],
        "map": meta.get("map"),
        "rounds_played": live.height,
        "score": dict(sorted(teams.items(), key=lambda kv: -kv[1]["rounds_won"])),
        "scoreboard": _rows(board),
        "note": (
            "Knife rounds excluded. Teams are labelled by starting side because sides swap at halftime. "
            "Players are ranked by `rating`, an approximation of HLTV Rating 2.0 (1.00 is average); "
            "rank players by rating, not by kills."
        ),
    }


def with_rating(stats: pl.DataFrame) -> pl.DataFrame:
    """Add `impact` and `rating`: the public approximation of HLTV Rating 2.0.

    Same formula awpy shipped in its stats module: per-round kills (KPR), deaths (DPR),
    assists (APR), plus KAST % and ADR. 1.00 is roughly average. HLTV's current Rating 3.0
    is built around "round swing" (how each action moved the team's chance to win the
    round) and its exact formula isn't published, so this is the closest open stand-in.
    """
    rounds = pl.col("rounds_played").cast(pl.Float64)
    kpr, dpr, apr = pl.col("kills") / rounds, pl.col("deaths") / rounds, pl.col("assists") / rounds
    impact = 2.13 * kpr + 0.42 * apr - 0.41
    rating = 0.0073 * pl.col("kast") + 0.3591 * kpr - 0.5329 * dpr + 0.2372 * impact + 0.0032 * pl.col("adr") + 0.1587
    # The formula is linear, so a 0-13 game with no damage goes below zero, which real ratings
    # never do. Display is floored at 0.00; `rating_raw` keeps the order for ranking.
    return stats.with_columns(
        impact.round(2).alias("impact"),
        rating.clip(lower_bound=0).round(2).alias("rating"),
        rating.alias("rating_raw"),
    )


def round_summary(
    round_num: int,
    rounds: pl.DataFrame,
    kills: pl.DataFrame,
    round_players: pl.DataFrame,
    round_economy: pl.DataFrame,
    bomb: pl.DataFrame,
    tick_rate: float,
) -> dict[str, Any]:
    r = rounds.filter(pl.col("round_num") == round_num)
    if r.is_empty():
        raise RoundNotFound(
            f"Round {round_num} does not exist. Valid rounds: {rounds['round_num'].min()}-{rounds['round_num'].max()}."
        )
    r = r.row(0, named=True)
    start = r["freeze_end_tick"]

    feed = kills.filter(pl.col("round_num") == round_num).sort("tick")
    kill_feed = feed.select(
        pl.lit(round_num).alias("round"),
        "tick",
        ((pl.col("tick") - start) / tick_rate).round(1).alias("time_s"),
        pl.col("attacker_name").alias("attacker"),
        pl.col("attacker_side").alias("attacker_side"),
        pl.col("victim_name").alias("victim"),
        weapon_label(), "headshot", "is_trade", "victim_traded", "after_round_end",
    )

    econ = {
        e["side"]: {"buy_type": e["buy_type"], "equipment_value": e["equipment_value"]}
        for e in round_economy.filter(pl.col("round_num") == round_num).iter_rows(named=True)
    }
    loadouts = (
        round_players.filter(pl.col("round_num") == round_num)
        .sort(["side", "equipment_value"], descending=[False, True])
        .select("name", "side", "money", "equipment_value", "primary_weapon", "secondary_weapon",
                "armor", "has_helmet", "has_defuser")
    )
    bomb_events = bomb.filter(
        (pl.col("round_num") == round_num) & pl.col("event").is_in(["finish_plant", "defuse"])
    ).select(
        "event", "tick", ((pl.col("tick") - start) / tick_rate).round(1).alias("time_s"), "name", "bombsite"
    )
    return {
        "round": round_num,
        "winner_side": r["winner_side"],
        "reason": r["reason_name"],
        "is_knife_round": r.get("is_knife_round", False),
        "duration_s": _seconds(r["end_tick"] - start if start is not None else None, tick_rate),
        "ticks": {"freeze_end": start, "end": r["end_tick"]},
        "economy": econ,
        "loadouts": _rows(loadouts),
        "kills": _rows(kill_feed),
        "bomb": _rows(bomb_events),
    }


def opening_duels(
    kills: pl.DataFrame,
    rounds: pl.DataFrame,
    steamid: int,
    tick_rate: float,
    limit: int = 10,
) -> dict[str, Any]:
    """First kill of each round where the player was the killer or the victim.

    Opening duels decide a big share of rounds (the team that gets the first
    kill plays 5v4), so this is usually the first place to look.
    """
    live = live_rounds(rounds).select("round_num", "freeze_end_tick", "winner_side")
    firsts = (
        kills.filter(~pl.col("after_round_end"))
        .sort("tick")
        .group_by("round_num", maintain_order=True)
        .first()
        .join(live, on="round_num", how="inner")
    )
    mine = firsts.filter((pl.col("attacker_steamid") == steamid) | (pl.col("victim_steamid") == steamid))
    duels = mine.with_columns(
        (pl.col("attacker_steamid") == steamid).alias("won"),
    ).with_columns(
        pl.when(pl.col("won")).then(pl.col("attacker_side")).otherwise(pl.col("victim_side")).alias("side"),
        pl.when(pl.col("won")).then(pl.col("victim_name")).otherwise(pl.col("attacker_name")).alias("opponent"),
        ((pl.col("tick") - pl.col("freeze_end_tick")) / tick_rate).round(1).alias("time_s"),
        pl.when(pl.col("won")).then(pl.col("attacker_x")).otherwise(pl.col("victim_x")).round(0).alias("x"),
        pl.when(pl.col("won")).then(pl.col("attacker_y")).otherwise(pl.col("victim_y")).round(0).alias("y"),
    ).with_columns(
        (pl.col("winner_side") == pl.col("side")).alias("round_won"),
    ).sort("round_num")

    def _split(df: pl.DataFrame) -> dict[str, Any]:
        won = int(df["won"].sum()) if df.height else 0
        return {"taken": df.height, "won": won, "lost": df.height - won}

    detail = duels.select(
        pl.col("round_num").alias("round"), "tick", "side", "won", "opponent", weapon_label(),
        "headshot", "time_s", "x", "y", "round_won",
        # if the player died first, did a teammate trade them?
        pl.when(~pl.col("won")).then(pl.col("victim_traded")).otherwise(None).alias("traded"),
    )
    return {
        "overall": _split(duels),
        "by_side": {side: _split(duels.filter(pl.col("side") == side)) for side in ("CT", "T")},
        "rounds_won_after_opening": {
            "after_winning_duel": int(duels.filter(pl.col("won"))["round_won"].sum()),
            "after_losing_duel": int(duels.filter(~pl.col("won"))["round_won"].sum()),
        },
        "rounds_with_any_opening_duel": firsts.height,
        "duels": _rows(detail.head(limit)),
        "truncated": detail.height > limit,
    }


def utility_summary(
    stats: pl.DataFrame,
    blinds: pl.DataFrame,
    damages: pl.DataFrame,
    steamid: int,
    kills: pl.DataFrame | None = None,
    tick_rate: float = 64.0,
) -> dict[str, Any]:
    s = stats.filter(pl.col("steamid") == steamid)
    base = s.select(
        "flashes_thrown", "enemies_flashed", "flash_assists", "utility_damage",
        pl.col("flash_duration_dealt").round(1), "rounds_played",
    ).row(0, named=True) if s.height else {}

    thrown = blinds.filter(pl.col("attacker_steamid") == steamid)
    enemy = thrown.filter(pl.col("attacker_side") != pl.col("victim_side"))
    team_all = thrown.filter((pl.col("attacker_side") == pl.col("victim_side")) & (pl.col("victim_steamid") != steamid))
    self_all = thrown.filter(pl.col("victim_steamid") == steamid)
    # Short blinds (catching the edge of a flash while turned away) aren't real team or self
    # flashes. Most self-blinds in real demos are under a second, so they're counted apart.
    significant = pl.col("duration") >= MINOR_BLIND_S
    team, selff = team_all.filter(significant), self_all.filter(significant)

    nade_dmg = (
        damages.filter((pl.col("attacker_steamid") == steamid) & pl.col("weapon").is_in(list(GRENADE_WEAPONS)))
        .filter(pl.col("attacker_side") != pl.col("victim_side"))
        .with_columns(weapon_label())
        .group_by("weapon")
        .agg(pl.col("dmg_health").sum().alias("damage"), pl.len().alias("hits"))
        .sort("damage", descending=True)
    )
    team = _with_death_while_blind(team, kills, tick_rate)
    costly = team.filter(pl.col("teammate_died_while_blind"))
    worst_team_flashes = team.sort("duration", descending=True).head(5).select(
        pl.col("round_num").alias("round"), "tick", pl.col("victim_name").alias("teammate"),
        pl.col("duration").round(2).alias("blind_s"), "teammate_died_while_blind",
    )
    rounds_played = base.get("rounds_played") or 0
    return {
        **base,
        "enemy_blinds": enemy.height,
        "avg_enemy_blind_s": round(float(enemy["duration"].mean()), 2) if enemy.height else 0.0,
        "teammates_flashed": team.height,
        "self_flashes": selff.height,
        "minor_team_blinds": team_all.height - team.height,
        "minor_self_blinds": self_all.height - selff.height,
        "team_self_flash_rule": f"teammates_flashed and self_flashes count blinds of {MINOR_BLIND_S}s or longer; "
                                "shorter ones are listed as minor and are not mistakes worth calling out.",
        "utility_damage_per_round": round(base.get("utility_damage", 0) / rounds_played, 1) if rounds_played else None,
        "grenade_damage_by_type": _rows(nade_dmg),
        "worst_team_flashes": _rows(worst_team_flashes),
        "costly_team_flashes": costly.height,
        "costly_team_flash_list": _rows(costly.sort("tick").select(
            pl.col("round_num").alias("round"), "tick", pl.col("victim_name").alias("teammate"),
            pl.col("duration").round(2).alias("blind_s"), pl.col("_killer").alias("killed_by"),
            pl.col("_death_tick").alias("death_tick"),
        )),
        "costly_rule": f"A team flash is costly when the blinded teammate died before the blind wore off "
                       f"(+{COSTLY_GRACE_S}s). That's the only kind worth calling a mistake; most team flashes cost nothing.",
    }


COSTLY_GRACE_S = 0.5


def _with_death_while_blind(team_blinds: pl.DataFrame, kills: pl.DataFrame | None, tick_rate: float) -> pl.DataFrame:
    """Flag team blinds where the blinded teammate died before the blind (plus a short grace) wore off."""
    if kills is None or team_blinds.is_empty():
        return team_blinds.with_columns(
            pl.lit(False).alias("teammate_died_while_blind"),
            pl.lit(None, dtype=pl.Utf8).alias("_killer"),
            pl.lit(None, dtype=pl.Int64).alias("_death_tick"),
        )
    deaths = kills.select(
        pl.col("victim_steamid"), pl.col("tick").cast(pl.Int64).alias("_death_tick"),
        pl.col("attacker_name").alias("_killer"),
    )
    b = team_blinds.with_row_index("_i").with_columns(
        (pl.col("tick").cast(pl.Int64) + ((pl.col("duration") + COSTLY_GRACE_S) * tick_rate).cast(pl.Int64)).alias("_until")
    )
    hits = (
        b.select("_i", "victim_steamid", "tick", "_until")
        .join(deaths, on="victim_steamid", how="inner")
        .filter((pl.col("_death_tick") >= pl.col("tick").cast(pl.Int64)) & (pl.col("_death_tick") <= pl.col("_until")))
        .group_by("_i").agg(pl.col("_death_tick").min(), pl.col("_killer").first())
    )
    return (
        b.join(hits, on="_i", how="left")
        .with_columns(pl.col("_death_tick").is_not_null().alias("teammate_died_while_blind"))
        .drop("_i", "_until")
    )


def _kill_tags(kills: pl.DataFrame, rounds: pl.DataFrame | None) -> pl.DataFrame:
    """Add `opening` (first kill of the round) and `attacker_won_round` to every live kill."""
    df = kills.filter(~pl.col("after_round_end")).sort("tick")
    df = df.with_columns((pl.col("tick") == pl.col("tick").min().over("round_num")).alias("opening"))
    if rounds is not None:
        df = df.join(live_rounds(rounds).select("round_num", "winner_side"), on="round_num", how="left").with_columns(
            (pl.col("attacker_side") == pl.col("winner_side")).alias("attacker_won_round")
        ).drop("winner_side")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Boolean).alias("attacker_won_round"))
    return df


def _group_counts(df: pl.DataFrame, name_col: str, key: str) -> list[dict[str, Any]]:
    out = (
        df.drop_nulls(name_col)
        .group_by(name_col)
        .agg(pl.len().alias("count"), pl.col("round_num").unique().sort().alias("rounds"))
        .sort(["count", name_col], descending=[True, False])
        .rename({name_col: key})
    )
    return _rows(out)


def find_kills(
    kills: pl.DataFrame,
    *,
    rounds: pl.DataFrame | None = None,
    steamid: int | None = None,
    role: str = "any",
    round_num: int | None = None,
    side: str | None = None,
    weapon: str | None = None,
    headshot: bool | None = None,
    traded: bool | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Filterable kill feed with pagination, plus counts over ALL matching kills (not just the page).

    The counts exist so nobody has to tally rows by hand: "who killed me most", "how many on T side"
    and "which rounds" are read straight from `by_attacker`, `by_side` and `rounds`.
    """
    exit_frags = kills.filter(pl.col("after_round_end"))
    df = _kill_tags(kills, rounds)
    if steamid is not None:
        cond = {
            "attacker": pl.col("attacker_steamid") == steamid,
            "victim": pl.col("victim_steamid") == steamid,
        }.get(role, (pl.col("attacker_steamid") == steamid) | (pl.col("victim_steamid") == steamid))
        df = df.filter(cond)
        exit_frags = exit_frags.filter(cond)
    if round_num is not None:
        df = df.filter(pl.col("round_num") == round_num)
    if side:
        df = df.filter(pl.col("attacker_side") == side.upper())
    if weapon:
        w = weapon.lower()
        df = df.filter(pl.col("weapon").str.contains(w, literal=True) | weapon_label().str.contains(w, literal=True))
    if headshot is not None:
        df = df.filter(pl.col("headshot") == headshot)
    if traded is not None:
        df = df.filter(pl.col("victim_traded") == traded)

    df = df.sort("tick")
    # Side split: with a player and a role, it's the side THAT PLAYER was on; otherwise the killer's side.
    if steamid is not None and role == "victim":
        side_col, side_of = pl.col("victim_side"), "the player (the victim)"
    elif steamid is not None and role == "attacker":
        side_col, side_of = pl.col("attacker_side"), "the player (the killer)"
    elif steamid is not None:
        side_col = pl.when(pl.col("attacker_steamid") == steamid).then(pl.col("attacker_side")).otherwise(pl.col("victim_side"))
        side_of = "the player"
    else:
        side_col, side_of = pl.col("attacker_side"), "the killer"
    by_side = {s_: 0 for s_ in ("CT", "T")}
    for row in df.group_by(side_col.alias("_s")).len().iter_rows(named=True):
        if row["_s"] in by_side:
            by_side[row["_s"]] = row["len"]

    page = df.slice(offset, limit).select(
        pl.col("round_num").alias("round"), "tick",
        pl.col("attacker_name").alias("attacker"), "attacker_side",
        pl.col("victim_name").alias("victim"), weapon_label(), "headshot", "thrusmoke",
        "opening", "is_trade", "victim_traded", "attacker_won_round",
    )
    return {
        "total": df.height,
        "rounds": sorted(set(df["round_num"].to_list())),
        "by_side": by_side,
        "by_side_is_side_of": side_of,
        "by_attacker": _group_counts(df, "attacker_name", "attacker"),
        "by_victim": _group_counts(df, "victim_name", "victim"),
        "tags": {
            "openings": int(df["opening"].sum()),
            "trades": int(df["is_trade"].sum()),
            "in_rounds_the_killer_won": int(df["attacker_won_round"].sum()) if rounds is not None else None,
        },
        "exit_frags_excluded": exit_frags.height,
        "offset": offset,
        "returned": page.height,
        "next_offset": offset + limit if offset + limit < df.height else None,
        "kills": _rows(page),
        "note": "total, rounds, by_side, by_attacker, by_victim and tags count ALL matching kills, not just this page. "
                "Cite them for counts instead of counting rows.",
    }


# ---------------------------------------------------------------- clutches

# How a CS player counts clutches (Afzal's rules, from labelling the Falcons eval):
#   * any last-alive situation the player's side WINS is a clutch won, even a 1v5. If he died and the
#     bomb won it, it still counts, flagged won_after_death ("last to die, then the bomb went off")
#   * a lost 1v4 or 1v5 is just "last alive", unless he clearly went for it: got a kill or planted the
#     bomb (m0NESY's 1v4 on Anubis r18: 2 kills and a plant). Damage alone isn't enough at those odds
#   * surviving a lost round (saving the gun) isn't a lost clutch at all
#   * a lost 1v1 is always a lost clutch (unless saved)
#   * a lost 1v2/1v3 is a lost clutch only if he attempted it: got a kill, did damage, planted the bomb
#     (T), or died near the planted bomb going for the defuse (CT). Otherwise it's "last alive".
BIG_DISADVANTAGE = 4  # 1v4 and 1v5
# A CT who dies within this distance of the planted bomb was going for the defuse (CS units; a bombsite
# is roughly 1000-1500 units across). A heuristic: there's no start-defuse event in the demo.
DEFUSE_ATTEMPT_RADIUS = 800.0


def last_alive_situations(
    kills: pl.DataFrame,
    rounds: pl.DataFrame,
    round_players: pl.DataFrame,
    damages: pl.DataFrame | None = None,
    bomb: pl.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Every moment a player became the last one alive on their side with enemies still up.

    Replays each round's kill feed against the roster at freeze end. Both sides can end up in one
    in the same round (a 1v3 that turns into a 1v1). Raw situations, before deciding which count as
    clutches; awpy's scoreboard counts all of these.
    """
    live = live_rounds(rounds).select("round_num", "winner_side")
    winners = dict(live.iter_rows())
    roster: dict[int, dict[str, set[int]]] = {}
    names: dict[int, str] = {}
    for r in round_players.iter_rows(named=True):
        if r["round_num"] in winners and r["side"] in ("CT", "T"):
            roster.setdefault(r["round_num"], {"CT": set(), "T": set()})[r["side"]].add(r["steamid"])
            names[r["steamid"]] = r["name"]

    feed = kills.filter(~pl.col("after_round_end")).sort("tick")
    by_round: dict[int, list[dict[str, Any]]] = {}
    for k in feed.iter_rows(named=True):
        by_round.setdefault(k["round_num"], []).append(k)

    found: list[dict[str, Any]] = []
    for rnd, sides in sorted(roster.items()):
        alive = {s_: set(v) for s_, v in sides.items()}
        side_of = {sid: s_ for s_, ids in sides.items() for sid in ids}
        started: dict[str, dict[str, Any]] = {}
        for k in by_round.get(rnd, []):
            victim = k["victim_steamid"]
            vside = side_of.get(victim)
            if vside is None:
                continue
            for c in started.values():
                if k["attacker_steamid"] == c["_sid"] and vside != c["side"]:
                    c["kills"] += 1
                if victim == c["_sid"]:
                    c["_death"] = (k["tick"], k.get("victim_x"), k.get("victim_y"))
            alive[vside].discard(victim)
            enemy = "T" if vside == "CT" else "CT"
            if vside not in started and len(alive[vside]) == 1 and alive[enemy]:
                sid = next(iter(alive[vside]))
                started[vside] = {"round": rnd, "tick": k["tick"], "player": names[sid], "side": vside,
                                  "vs": len(alive[enemy]), "kills": 0, "_sid": sid}
        for c in sorted(started.values(), key=lambda c: c["tick"]):
            c["won"] = winners[rnd] == c["side"]
            c["survived"] = c["_sid"] in alive[c["side"]]
            c["situation"] = f"1v{c['vs']}"
            c["attempt"] = _attempt_signals(c, damages, bomb)
            found.append(c)
    return found


def _attempt_signals(c: dict[str, Any], damages: pl.DataFrame | None, bomb: pl.DataFrame | None) -> list[str]:
    """What shows the player actually went for it after becoming last alive."""
    signals = ["kill"] if c["kills"] else []
    rnd, sid, start = c["round"], c["_sid"], c["tick"]
    if damages is not None and damages.height:
        hit = damages.filter(
            (pl.col("round_num") == rnd) & (pl.col("attacker_steamid") == sid) & (pl.col("tick") >= start)
            & (pl.col("victim_side") != c["side"]) & (pl.col("dmg_health") > 0) & ~pl.col("after_round_end")
        )
        if hit.height:
            signals.append("damage")
    if bomb is not None and bomb.height:
        in_round = bomb.filter(pl.col("round_num") == rnd)
        if c["side"] == "T" and in_round.filter(
            pl.col("event").is_in(["start_plant", "finish_plant"]) & (pl.col("steamid") == sid) & (pl.col("tick") >= start)
        ).height:
            signals.append("planted")
        death = c.get("_death")
        plant = in_round.filter(pl.col("event") == "finish_plant").sort("tick")
        if c["side"] == "CT" and death and plant.height and {"x", "y"} <= set(plant.columns) and death[1] is not None:
            p = plant.row(0, named=True)
            if p["tick"] <= death[0] and p["x"] is not None:
                dist = ((p["x"] - death[1]) ** 2 + (p["y"] - death[2]) ** 2) ** 0.5
                if dist <= DEFUSE_ATTEMPT_RADIUS:
                    signals.append("went_for_defuse")
    return signals


def clutch_outcome(c: dict[str, Any]) -> str:
    if c["won"]:
        return "won"
    if c["survived"]:
        return "saved"
    attempt = set(c.get("attempt") or [])
    if c["vs"] >= BIG_DISADVANTAGE:
        return "lost" if c["kills"] >= 1 or "planted" in attempt else "last_alive"
    attempted = bool(attempt) or c["kills"] >= 1
    if attempted or c["vs"] == 1:
        return "lost"
    return "last_alive"


def clutch_counts(found: list[dict[str, Any]]) -> dict[int, tuple[int, int]]:
    """steamid -> (clutches won, clutches played) under the rules above."""
    out: dict[int, list[int]] = {}
    for c in found:
        o = clutch_outcome(c)
        if o in ("won", "lost"):
            w, p = out.setdefault(c["_sid"], [0, 0])
            out[c["_sid"]] = [w + (o == "won"), p + 1]
    return {k: (v[0], v[1]) for k, v in out.items()}


def clutches(
    kills: pl.DataFrame,
    rounds: pl.DataFrame,
    round_players: pl.DataFrame,
    steamid: int | None = None,
    limit: int = 30,
    damages: pl.DataFrame | None = None,
    bomb: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Clutches the way players count them, plus the last-alive situations that don't count.

    Starts from every last-alive situation, then: won = clutch won (any odds); lost = an attempt
    (died after a kill, or any 1v1 died); saved = survived a lost round; last_alive = a lost 1v4/1v5
    or dying in a 1v2/1v3 without a kill.
    """
    found = last_alive_situations(kills, rounds, round_players, damages, bomb)
    if steamid is not None:
        found = [c for c in found if c["_sid"] == steamid]
    for c in found:
        c["outcome"] = clutch_outcome(c)
        c["won_after_death"] = c["won"] and not c["survived"]
    real = [c for c in found if c["outcome"] in ("won", "lost")]
    other = [c for c in found if c["outcome"] not in ("won", "lost")]

    def _summary(all_: list[dict[str, Any]]) -> dict[str, Any]:
        cs = [c for c in all_ if c["outcome"] in ("won", "lost")]
        situations: dict[str, dict[str, int]] = {}
        for c in cs:
            s_ = situations.setdefault(c["situation"], {"played": 0, "won": 0})
            s_["played"] += 1
            s_["won"] += int(c["won"])
        return {"played": len(cs), "won": sum(c["won"] for c in cs), "lost": sum(not c["won"] for c in cs),
                "by_situation": dict(sorted(situations.items())),
                "won_after_death": sum(c["outcome"] == "won" and not c["survived"] for c in all_),
                "saved": sum(c["outcome"] == "saved" for c in all_),
                "last_alive_not_clutch": sum(c["outcome"] == "last_alive" for c in all_)}

    by_player: dict[str, list[dict[str, Any]]] = {}
    for c in found:
        by_player.setdefault(c["player"], []).append(c)
    strip = lambda cs: [{k: v for k, v in c.items() if not k.startswith("_")} for c in cs]  # noqa: E731
    return {
        "summary": _summary(found),
        "by_player": sorted(
            ({"name": n, **_summary(cs)} for n, cs in by_player.items()),
            key=lambda x: (-x["played"], -x["won"], x["name"]),
        ) if steamid is None else None,
        "clutches": strip(real)[:limit],
        "not_clutches": strip(other)[:limit],
        "truncated": len(real) > limit or len(other) > limit,
        "rule": "A last-alive situation starts when a player is the last one alive on their side with enemies up. "
                "outcome: won = side won the round (any odds, also by bomb or time) and counts as a clutch won; "
                "won_after_death means he died and the bomb or time won it (he was last to die, describe it that "
                "way); lost = a failed attempt and counts as a clutch lost: any lost 1v1, or a 1v2/1v3 where `attempt` "
                "shows he went for it (kill, damage, planted, went_for_defuse), or a 1v4/1v5 where he got a kill or "
                "planted; saved = survived a lost round (not a clutch); last_alive = any other lost 1v4/1v5, or a "
                "1v2/1v3 he didn't attempt (not a clutch). The "
                "scoreboard's clutch columns use these same rules.",
    }


# ---------------------------------------------------------------- multi-kills

def _multikill_rounds(kills: pl.DataFrame, rounds: pl.DataFrame) -> list[dict[str, Any]]:
    """Every (player, round) with 2+ kills on enemies. Kills after the round was decided count, like on
    HLTV: an ace finished on the last enemy after the bomb went off is still an ace."""
    live = live_rounds(rounds).select("round_num", "winner_side")
    df = (
        kills.filter(pl.col("attacker_steamid").is_not_null() & (pl.col("attacker_side") != pl.col("victim_side")))
        .join(live, on="round_num", how="inner")
        .sort("tick")
    )
    out = []
    for (sid, rnd), g in df.group_by(["attacker_steamid", "round_num"], maintain_order=True):
        if g.height < 2:
            continue
        out.append({
            "_sid": sid, "round": rnd, "player": g["attacker_name"][0], "side": g["attacker_side"][0],
            "kills": g.height, "round_won": g["winner_side"][0] == g["attacker_side"][0],
            "victims": g["victim_name"].to_list(), "weapons": g.select(weapon_label())["weapon"].to_list(),
            "ticks": g["tick"].to_list(), "after_round_end": int(g["after_round_end"].sum()),
        })
    return sorted(out, key=lambda r: (r["round"], r["ticks"][0]))


MULTI_NAMES = {2: "2k", 3: "3k", 4: "4k", 5: "ace"}


def multikills(kills: pl.DataFrame, rounds: pl.DataFrame, steamid: int | None = None,
               min_kills: int = 2, limit: int = 40) -> dict[str, Any]:
    """Multi-kill rounds (2k, 3k, 4k, aces) per player, each with the round, victims and weapons."""
    found = _multikill_rounds(kills, rounds)
    if steamid is not None:
        found = [r for r in found if r["_sid"] == steamid]
    by_player: dict[str, dict[str, Any]] = {}
    for r in found:
        p = by_player.setdefault(r["player"], {"name": r["player"], "2k": 0, "3k": 0, "4k": 0, "ace": 0,
                                               "rounds": {"2k": [], "3k": [], "4k": [], "ace": []}})
        kind = MULTI_NAMES[min(r["kills"], 5)]
        p[kind] += 1
        p["rounds"][kind].append(r["round"])
    shown = [{**{k: v for k, v in r.items() if not k.startswith("_")}, "type": MULTI_NAMES[min(r["kills"], 5)]}
             for r in found if r["kills"] >= min_kills]
    shown.sort(key=lambda r: (-r["kills"], r["round"]))
    return {
        "by_player": sorted(by_player.values(), key=lambda p: (-p["ace"], -p["4k"], -p["3k"], -p["2k"], p["name"])),
        "rounds": shown[:limit],
        "truncated": len(shown) > limit,
        "note": "Kills on enemies in one round (teamkills don't count). Kills after the round was decided count, "
                "as on HLTV; after_round_end says how many of them came after. ticks cite each kill.",
    }


# ---------------------------------------------------------------- round by round

def round_list(
    rounds: pl.DataFrame,
    kills: pl.DataFrame,
    round_players: pl.DataFrame,
    players: pl.DataFrame,
    round_economy: pl.DataFrame,
    bomb: pl.DataFrame,
    max_rounds: int | None = None,
    overtime_rounds: int = 6,
) -> dict[str, Any]:
    """Every round in one compact table, plus halves and how each team did per buy type.

    Answers "how did the match flow" and "did economy decide it" in one call instead of
    pulling every round summary.
    """
    labels = team_labels(round_players, players)
    live = live_rounds(rounds).sort("round_num")
    # side -> team for each round, from the freeze-end roster
    side_team: dict[tuple[int, str], str] = {}
    rp = round_players.with_columns(
        pl.col("steamid").replace_strict(labels, default=None, return_dtype=pl.Utf8).alias("team")
    ).drop_nulls("team")
    for row in rp.group_by(["round_num", "side"]).agg(pl.col("team").mode().first()).iter_rows(named=True):
        side_team[(row["round_num"], row["side"])] = row["team"]

    econ = {(e["round_num"], e["side"]): e for e in round_economy.iter_rows(named=True)}
    firsts = {
        r["round_num"]: r for r in
        kills.filter(~pl.col("after_round_end")).sort("tick").group_by("round_num", maintain_order=True)
        .first().iter_rows(named=True)
    }
    plants = {
        r["round_num"]: r["bombsite"] for r in
        bomb.filter(pl.col("event") == "finish_plant").group_by("round_num").first().iter_rows(named=True)
    }
    team_names = sorted(set(labels.values()))
    score = {t: 0 for t in team_names}
    rows: list[dict[str, Any]] = []
    for r in live.iter_rows(named=True):
        n = r["round_num"]
        winner = side_team.get((n, r["winner_side"]))
        if winner in score:
            score[winner] += 1
        fk = firsts.get(n)
        teams = []
        for side in ("CT", "T"):
            e = econ.get((n, side), {})
            teams.append({"team": side_team.get((n, side)), "side": side, "buy": e.get("buy_type"),
                          "equipment": e.get("equipment_value")})
        rows.append({
            "round": n, "winner": winner, "winner_side": r["winner_side"], "reason": r["reason_name"],
            "first_kill_team": side_team.get((n, fk["attacker_side"])) if fk else None,
            "bomb_planted": plants.get(n),
            "teams": teams,
            "score_after": dict(score),
        })

    # Halves: a new one starts when the sides swap, and at the start of every overtime period (teams keep
    # their sides going into overtime, so the swap alone would merge it with the second half).
    def _new_period(n: int) -> bool:
        return bool(max_rounds) and n > max_rounds and (n - max_rounds - 1) % overtime_rounds == 0

    halves: list[dict[str, Any]] = []
    anchor = team_names[0] if team_names else None
    for row in rows:
        side = next((t["side"] for t in row["teams"] if t["team"] == anchor), None)
        if not halves or halves[-1]["_side"] != side or _new_period(row["round"]):
            halves.append({"_side": side, "rounds": [row["round"], row["round"]],
                           "won": {t: 0 for t in team_names}, "sides": {}})
        h = halves[-1]
        h["rounds"][1] = row["round"]
        if row["winner"] in h["won"]:
            h["won"][row["winner"]] += 1
        for t in row["teams"]:
            if t["team"]:
                h["sides"][t["team"]] = t["side"]
    halves_out = [{"half": i + 1, "rounds": f"{h['rounds'][0]}-{h['rounds'][1]}",
                   "overtime": bool(max_rounds) and h["rounds"][0] > max_rounds,
                   "sides": h["sides"], "rounds_won": h["won"]} for i, h in enumerate(halves)]

    by_buy: dict[str, dict[str, dict[str, int]]] = {t: {} for t in team_names}
    for row in rows:
        for t in row["teams"]:
            if t["team"] not in by_buy or t["buy"] is None:
                continue
            b = by_buy[t["team"]].setdefault(t["buy"], {"played": 0, "won": 0})
            b["played"] += 1
            b["won"] += int(row["winner"] == t["team"])
    return {
        "rounds_played": len(rows),
        "final_score": score,
        "halves": halves_out,
        "rounds_won_by_buy_type": by_buy,
        "first_kill_conversion": {
            t: {"first_kills": sum(1 for r in rows if r["first_kill_team"] == t),
                "rounds_won_after_first_kill": sum(1 for r in rows if r["first_kill_team"] == t and r["winner"] == t)}
            for t in team_names
        },
        "rounds": rows,
        "note": "Buy types: pistol, eco, force, full (each team's freeze-end equipment). first_kill_team got the "
                "round's first kill. Use get_round_summary only for the detail of a single round.",
    }


# ---------------------------------------------------------------- grenades thrown

GRENADE_TYPES = ("smoke", "flash", "HE", "molotov/incendiary", "decoy")


def grenade_usage(
    throws: pl.DataFrame,
    bomb: pl.DataFrame,
    round_players: pl.DataFrame,
    players: pl.DataFrame,
    steamid: int,
    rounds_played: int,
) -> dict[str, Any]:
    """What a player threw, split by side and by before/after the bomb plant, compared with teammates.

    Smokes and molotovs before a plant on T side are execute utility; after a plant they hold the
    post-plant. On CT side, before a plant is holding a site, after a plant is the retake.
    """
    plants = bomb.filter(pl.col("event") == "finish_plant").group_by("round_num").agg(pl.col("tick").min().alias("_plant"))
    t = throws.join(plants, on="round_num", how="left").with_columns(
        pl.when(pl.col("_plant").is_null()).then(pl.lit("no_plant"))
        .when(pl.col("tick") < pl.col("_plant")).then(pl.lit("before_plant"))
        .otherwise(pl.lit("after_plant")).alias("phase")
    )
    mine = t.filter(pl.col("steamid") == steamid)

    def _counts(df: pl.DataFrame) -> dict[str, int]:
        got = dict(df.group_by("grenade").len().iter_rows())
        return {g: int(got.get(g, 0)) for g in GRENADE_TYPES}

    by_side = {side: _counts(mine.filter(pl.col("side") == side)) for side in ("CT", "T")}
    phases = {
        side: {ph: _counts(mine.filter((pl.col("side") == side) & (pl.col("phase") == ph)))
               for ph in ("before_plant", "after_plant", "no_plant")}
        for side in ("CT", "T")
    }
    labels = team_labels(round_players, players)
    my_team = labels.get(steamid)
    mates = [sid for sid, team in labels.items() if team == my_team]
    names = dict(players.select("steamid", "name").unique("steamid").iter_rows())
    team_rows = []
    for sid in mates:
        c = _counts(t.filter(pl.col("steamid") == sid))
        team_rows.append({"name": names.get(sid), "total": sum(c.values()), **c})
    team_rows.sort(key=lambda r: (-r["total"], r["name"] or ""))
    for i, r in enumerate(team_rows, 1):
        r["team_rank_total"] = i
    total = mine.height
    return {
        "thrown": _counts(mine),
        "total_thrown": total,
        "per_round": round(total / rounds_played, 2) if rounds_played else None,
        "by_side": by_side,
        "by_side_and_plant": phases,
        "team_comparison": team_rows,
        "plant_rule": "before_plant / after_plant are relative to the bomb plant in that round; no_plant means the "
                      "bomb was never planted. T side before_plant = execute utility, CT side after_plant = retake utility.",
    }
