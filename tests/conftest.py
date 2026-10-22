"""A tiny, hand-checkable 2v2 match that runs through the REAL ingest pipeline.

The fake demo returns tables in awpy's raw format (sides spelled
"counter-terrorist", team numbers 2/3, no round numbers on events), then
`ingest_demo_object` normalises and caches them exactly like a real demo.
So every analysis test also exercises ingest.

Match script (tick rate 64, 4 rounds, halftime after round 2):
  Team A = alpha(1), bravo(2), starts CT.  Team B = charlie(3), delta(4), starts T.
  r1 CT win  (A)   alpha opens vs charlie (win), delta trades, bravo kills delta, exit frag at 1100
  r2 T  win  (B)   charlie opens vs alpha (alpha dies untraded), bomb planted A
  r3 CT win  (B)   sides swapped; alpha (now T) opens vs charlie (win)
  r4 CT win  (B)   delta opens vs alpha (alpha dies, bravo trades him)
  Final: Team B 3 - 1 Team A.  Plus one warmup kill at tick 50 that must be dropped.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cs2_analyst.demo_store import DemoStore, ingest_demo_object

CT, T = "counter-terrorist", "terrorist"
NAMES = {1: "alpha", 2: "bravo", 3: "charlie", 4: "delta"}
DEMO_ID = "fakedemo0001"


def _side(sid: int, round_num: int) -> str:
    team_a = sid in (1, 2)
    first_half = round_num <= 2
    return CT if team_a == first_half else T


def _kill(tick, att, vic, weapon="ak47", hs=False, is_trade=False, victim_traded=False, rnd=1):
    return {
        "tick": tick, "attacker_steamid": att, "attacker_name": NAMES[att], "attacker_side": _side(att, rnd),
        "attacker_x": 100.0, "attacker_y": 200.0, "victim_steamid": vic, "victim_name": NAMES[vic],
        "victim_side": _side(vic, rnd), "victim_x": 110.0, "victim_y": 210.0, "assister_side": None,
        "weapon": weapon, "headshot": hs, "thrusmoke": False, "is_trade": is_trade, "victim_traded": victim_traded,
    }


class FakeDemo:
    header = {"map_name": "de_test", "server_name": "pytest", "playback_time": 70.0}
    tick_rate = 64.0
    convars = {"mp_maxrounds": "4"}

    rounds = pl.DataFrame({
        "round_num": [1, 2, 3, 4],
        "start_tick": [100, 1200, 2200, 3200],
        "freeze_end_tick": [200, 1300, 2300, 3300],
        "end_tick": [1000, 2000, 3000, 4000],
        "official_end_tick": [1200, 2200, 3200, None],
        "winner": [3, 2, 3, 3],
        "winner_side": [CT, T, CT, CT],
        "reason_name": ["ct_win", "terrorists_win", "ct_win", "ct_win"],
        "is_knife_round": [False] * 4,
    })
    kills = pl.DataFrame([
        _kill(50, 3, 1),                                                   # warmup: dropped
        _kill(400, 1, 3, hs=True, victim_traded=True, rnd=1),              # r1 opening, alpha wins
        _kill(450, 4, 1, is_trade=True, rnd=1),                            # delta trades charlie
        _kill(600, 2, 4, weapon="awp", rnd=1),
        _kill(1100, 2, 3, rnd=1),                                          # exit frag after r1 ended
        _kill(1500, 3, 1, rnd=2),                                          # r2 opening, alpha dies untraded
        _kill(1600, 4, 2, rnd=2),
        _kill(2500, 1, 3, hs=True, rnd=3),                                 # r3 opening, alpha (T) wins
        _kill(3500, 4, 1, weapon="m4a1", victim_traded=True, rnd=4),       # r4 opening, alpha dies...
        _kill(3550, 2, 4, is_trade=True, rnd=4),                           # ...and bravo trades
    ])
    blinds = pl.DataFrame({
        "tick": [300, 310, 2400],
        "attacker_steamid": [1, 1, 1], "attacker_name": ["alpha"] * 3, "attacker_side": [CT, CT, T],
        "victim_steamid": [3, 2, 1], "victim_name": ["charlie", "bravo", "alpha"], "victim_side": [T, CT, T],
        "duration": [2.0, 1.5, 1.0],
    })
    damages = pl.DataFrame({
        "tick": [320, 330, 2450],
        "attacker_steamid": [1, 1, 1], "attacker_name": ["alpha"] * 3, "attacker_side": [CT, CT, T],
        "victim_steamid": [3, 2, 4], "victim_name": ["charlie", "bravo", "delta"], "victim_side": [T, CT, CT],
        "weapon": ["hegrenade", "hegrenade", "ak47"], "dmg_health": [40, 10, 27],
    })
    bomb = pl.DataFrame({
        "tick": [1700], "event": ["finish_plant"], "steamid": [3], "name": ["charlie"], "bombsite": ["A"],
    })
    stats = pl.DataFrame({
        "steamid": [1, 2, 3, 4], "name": list(NAMES.values()), "rounds_played": [4] * 4,
        "kills": [2, 3, 1, 3], "deaths": [3, 1, 3, 2], "assists": [0, 0, 0, 0],
        "flash_assists": [0, 0, 0, 0], "headshot_pct": [100.0, 0.0, 0.0, 0.0], "adr": [60.0, 70.0, 40.0, 80.0],
        "kast": [75.0, 100.0, 50.0, 75.0], "opening_kills": [2, 0, 1, 1], "opening_deaths": [2, 0, 2, 0],
        "traded_deaths": [1, 0, 1, 0], "clutches_won": [0, 0, 0, 0], "clutches_played": [0, 0, 0, 0],
        "utility_damage": [40, 0, 0, 0], "flashes_thrown": [3, 0, 0, 0], "enemies_flashed": [1, 0, 0, 0],
        "flash_duration_dealt": [2.0, 0.0, 0.0, 0.0],
    }, schema_overrides={"steamid": pl.UInt64})
    players = pl.DataFrame({
        "steamid": [1, 2, 3, 4, 0], "name": [*NAMES.values(), "DemoRecorder"],
        "side": [T, T, CT, CT, None], "team_clan_name": [None] * 5,
    }, schema_overrides={"steamid": pl.UInt64, "team_clan_name": pl.Utf8})
    round_economy = pl.DataFrame({
        "round_num": [1, 1, 2, 2], "side": [CT, T, CT, T],
        "equipment_value": [4000, 3000, 9000, 20000], "buy_type": ["pistol", "pistol", "eco", "full"],
        "n_players": [2, 2, 2, 2],
    })

    def load(self, *datasets: str) -> None:
        pass

    def snapshots(self, *, ticks=None) -> pl.DataFrame:
        freeze = dict(zip(self.rounds["freeze_end_tick"], self.rounds["round_num"]))
        rows = []
        for tick in ticks:
            rnd = freeze[tick]
            for sid, name in NAMES.items():
                rows.append({
                    "tick": tick, "steamid": sid, "name": name,
                    "side": 3 if _side(sid, rnd) == CT else 2,  # team numbers, like the engine
                    "money": 800, "equipment_value": 1000, "primary_weapon": None, "secondary_weapon": "glock",
                    "armor": 100, "has_helmet": False, "has_defuser": False, "has_bomb": False,
                })
        rows.append({"tick": ticks[0], "steamid": 0, "name": "DemoRecorder", "side": None})  # GOTV bot
        return pl.DataFrame(rows, schema_overrides={"steamid": pl.UInt64})


# Normalise the kill tables' steamid dtype to match awpy (UInt64).
for _name in ("kills", "blinds", "damages", "bomb"):
    _df = getattr(FakeDemo, _name)
    setattr(FakeDemo, _name, _df.with_columns(pl.col(pl.Int64).exclude("tick", "dmg_health").cast(pl.UInt64)))


@pytest.fixture
def store(tmp_path: Path) -> DemoStore:
    s = DemoStore(tmp_path / "cache")
    ingest_demo_object(FakeDemo(), DEMO_ID, s.cache_dir / DEMO_ID, source="fake.dem")
    return s


@pytest.fixture
def tables(store: DemoStore):
    return lambda name: store.table(DEMO_ID, name)
