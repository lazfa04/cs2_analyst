"""Turn a parsed demo into game states: one row per moment the round changed.

A round starts at freeze end and changes at every kill, the bomb plant and the defuse. Each row is
the state right AFTER that event, so the win probability before an event is the previous row's and
the event's swing is the difference. The label is whether T won the round.

Features are things a caster can see on the HUD at that moment: players alive, their health, what
they bought, defuse kits, whether the bomb is down, and the clock. Equipment is the freeze-end buy of
the players still alive (weapons picked up mid-round aren't tracked).
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ..analysis import live_rounds

# CS2 competitive defaults (mp_roundtime_defuse 1:55, mp_c4timer 40).
ROUND_TIME_S = 115.0
BOMB_TIME_S = 40.0

FEATURES = [
    "t_alive", "ct_alive", "t_hp", "ct_hp", "t_equip", "ct_equip", "ct_defusers",
    "bomb_planted", "round_time_left_s", "bomb_time_left_s",
]

# Direction each feature must push P(T wins), all else equal. With a few hundred rounds a tree model
# can learn nonsense like "a 6th CT alive helps T" from noise; these constraints rule that out.
MONOTONE = {
    "t_alive": 1, "ct_alive": -1, "t_hp": 1, "ct_hp": -1, "t_equip": 1, "ct_equip": -1,
    "ct_defusers": -1, "bomb_planted": 1,
    "round_time_left_s": 1,   # before the plant, time is on CT's side: less of it hurts T
    "bomb_time_left_s": -1,   # after the plant, the less bomb time left, the better for T
}

LABELS = {
    "t_alive": "Ts alive", "ct_alive": "CTs alive", "t_hp": "T health", "ct_hp": "CT health",
    "t_equip": "T equipment", "ct_equip": "CT equipment", "ct_defusers": "CT defuse kits",
    "bomb_planted": "bomb planted", "round_time_left_s": "round time left", "bomb_time_left_s": "bomb time left",
}


def round_states(
    demo_id: str,
    rounds: pl.DataFrame,
    kills: pl.DataFrame,
    damages: pl.DataFrame,
    bomb: pl.DataFrame,
    round_players: pl.DataFrame,
    tick_rate: float,
) -> pl.DataFrame:
    """All game states of every live round in one demo."""
    live = live_rounds(rounds).drop_nulls("freeze_end_tick").select("round_num", "freeze_end_tick", "winner_side")
    roster: dict[int, list[dict[str, Any]]] = {}
    for p in round_players.iter_rows(named=True):
        if p["side"] in ("CT", "T"):
            roster.setdefault(p["round_num"], []).append(p)

    def by_round(df: pl.DataFrame) -> dict[int, list[dict[str, Any]]]:
        out: dict[int, list[dict[str, Any]]] = {}
        if df.is_empty():
            return out
        for row in df.filter(~pl.col("after_round_end")).sort("tick").iter_rows(named=True):
            out.setdefault(row["round_num"], []).append(row)
        return out

    kills_r, dmg_r = by_round(kills), by_round(damages)
    bomb_r = by_round(bomb.filter(pl.col("event").is_in(["finish_plant", "defuse"])) if bomb.height else bomb)

    rows: list[dict[str, Any]] = []
    for rnd, freeze, winner in live.iter_rows():
        players = roster.get(rnd)
        if not players or winner not in ("CT", "T"):
            continue
        side = {p["steamid"]: p["side"] for p in players}
        name = {p["steamid"]: p["name"] for p in players}
        equip = {p["steamid"]: p.get("equipment_value") or 0 for p in players}
        kit = {p["steamid"]: bool(p.get("has_defuser")) for p in players}
        alive = set(side)
        hp = {sid: 100 for sid in side}
        plant_tick: int | None = None

        events: list[tuple[int, int, str, dict[str, Any]]] = []
        for k in kills_r.get(rnd, []):
            events.append((k["tick"], 1, "kill", k))
        for b in bomb_r.get(rnd, []):
            events.append((b["tick"], 2, "plant" if b["event"] == "finish_plant" else "defuse", b))
        events.sort(key=lambda e: (e[0], e[1]))
        damage = dmg_r.get(rnd, [])
        di = 0

        def state(tick: int, event: str, info: dict[str, Any]) -> dict[str, Any]:
            t_ids = [s for s in alive if side[s] == "T"]
            ct_ids = [s for s in alive if side[s] == "CT"]
            elapsed = max(0.0, (tick - freeze) / tick_rate)
            planted = plant_tick is not None
            return {
                "demo_id": demo_id, "round": rnd, "tick": tick, "event": event, **info,
                "t_alive": len(t_ids), "ct_alive": len(ct_ids),
                "t_hp": sum(hp[s] for s in t_ids), "ct_hp": sum(hp[s] for s in ct_ids),
                "t_equip": sum(equip[s] for s in t_ids), "ct_equip": sum(equip[s] for s in ct_ids),
                "ct_defusers": sum(kit[s] for s in ct_ids),
                "bomb_planted": int(planted),
                "round_time_left_s": 0.0 if planted else max(0.0, ROUND_TIME_S - elapsed),
                "bomb_time_left_s": max(0.0, BOMB_TIME_S - (tick - plant_tick) / tick_rate) if planted else 0.0,
                "t_win": int(winner == "T"),
            }

        no_actor = {"actor": None, "actor_id": None, "actor_side": None, "victim": None, "victim_id": None,
                    "victim_side": None, "assister": None, "assister_id": None}
        rows.append(state(freeze, "start", no_actor))
        for tick, _, kind, ev in events:
            while di < len(damage) and damage[di]["tick"] <= tick:
                d = damage[di]
                if d["victim_steamid"] in hp and d.get("health_post") is not None:
                    hp[d["victim_steamid"]] = max(0, int(d["health_post"]))
                di += 1
            if kind == "kill":
                v = ev["victim_steamid"]
                if v not in alive:
                    continue
                alive.discard(v)
                hp[v] = 0
                a, asst = ev.get("attacker_steamid"), ev.get("assister_steamid")
                info = {"actor": name.get(a), "actor_id": a if a in side else None, "actor_side": side.get(a),
                        "victim": name.get(v), "victim_id": v, "victim_side": side[v],
                        "assister": name.get(asst), "assister_id": asst if asst in side else None}
            elif kind == "plant":
                if plant_tick is not None:
                    continue
                plant_tick = tick
                sid = ev.get("steamid")
                info = {**no_actor, "actor": name.get(sid), "actor_id": sid if sid in side else None, "actor_side": "T"}
            else:  # defuse ends the round: a terminal state, used for credit but not for training
                sid = ev.get("steamid")
                info = {**no_actor, "actor": name.get(sid), "actor_id": sid if sid in side else None, "actor_side": "CT"}
            rows.append(state(tick, kind, info))

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.int_range(pl.len()).over(["demo_id", "round"]).alias("step")
    )


def states_for(store, demo_id: str) -> pl.DataFrame:
    t = lambda n: store.table(demo_id, n)  # noqa: E731
    return round_states(demo_id, t("rounds"), t("kills"), t("damages"), t("bomb"), t("round_players"),
                        store.meta(demo_id)["tick_rate"])


def training_rows(states: pl.DataFrame) -> pl.DataFrame:
    """States a model should learn from: not the defuse (the round is already over) and not states
    where one side is wiped out (also decided)."""
    return states.filter((pl.col("event") != "defuse") & (pl.col("t_alive") > 0) & (pl.col("ct_alive") > 0))
