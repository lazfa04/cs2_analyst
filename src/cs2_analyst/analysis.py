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
            "clutches_won", "clutches_played", "utility_damage",
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
        "weapon", "headshot", "is_trade", "victim_traded", "after_round_end",
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
        pl.col("round_num").alias("round"), "tick", "side", "won", "opponent", "weapon",
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
        .group_by("weapon")
        .agg(pl.col("dmg_health").sum().alias("damage"), pl.len().alias("hits"))
        .sort("damage", descending=True)
    )
    worst_team_flashes = team.sort("duration", descending=True).head(5).select(
        pl.col("round_num").alias("round"), "tick", pl.col("victim_name").alias("teammate"),
        pl.col("duration").round(2).alias("blind_s"),
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
    }


def find_kills(
    kills: pl.DataFrame,
    *,
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
    """Filterable kill feed with pagination. `role` is attacker / victim / any."""
    df = kills.filter(~pl.col("after_round_end"))
    if steamid is not None:
        cond = {
            "attacker": pl.col("attacker_steamid") == steamid,
            "victim": pl.col("victim_steamid") == steamid,
        }.get(role, (pl.col("attacker_steamid") == steamid) | (pl.col("victim_steamid") == steamid))
        df = df.filter(cond)
    if round_num is not None:
        df = df.filter(pl.col("round_num") == round_num)
    if side:
        df = df.filter(pl.col("attacker_side") == side.upper())
    if weapon:
        df = df.filter(pl.col("weapon").str.contains(weapon.lower(), literal=True))
    if headshot is not None:
        df = df.filter(pl.col("headshot") == headshot)
    if traded is not None:
        df = df.filter(pl.col("victim_traded") == traded)

    df = df.sort("tick")
    page = df.slice(offset, limit).select(
        pl.col("round_num").alias("round"), "tick",
        pl.col("attacker_name").alias("attacker"), "attacker_side",
        pl.col("victim_name").alias("victim"), "weapon", "headshot", "thrusmoke",
        "is_trade", "victim_traded",
    )
    return {
        "total": df.height,
        "offset": offset,
        "returned": page.height,
        "next_offset": offset + limit if offset + limit < df.height else None,
        "kills": _rows(page),
    }
