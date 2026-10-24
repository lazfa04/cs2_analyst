"""Death review on the fake 2v2 match, with hand-placed positions."""

import polars as pl

from cs2_analyst import deaths as D

TICK_RATE = 64.0


def positions(bravo_x: float) -> pl.DataFrame:
    """Round 1 snapshots: everyone at (110, 210) except bravo, who is at (bravo_x, 210)."""
    rows = []
    for t in (200, 300, 400, 440):
        for sid, name, side in [(1, "alpha", "CT"), (2, "bravo", "CT"), (3, "charlie", "T"), (4, "delta", "T")]:
            rows.append({"tick": t, "steamid": sid, "name": name, "side": side, "x": bravo_x if sid == 2 else 110.0,
                         "y": 210.0, "z": 0.0, "health": 100})
    return pl.DataFrame(rows, schema_overrides={"steamid": pl.UInt64})


def untraded(tables) -> pl.DataFrame:
    """The fixture's kills with bravo's kill of delta moved from 600 to 900: past the trade window, so alpha's
    round 1 death (450) is untraded, as the spacing tests need."""
    k = tables("kills")
    return k.with_columns(pl.when(pl.col("tick") == 600).then(900).otherwise(pl.col("tick")).cast(k["tick"].dtype).alias("tick"))


def review(tables, **kw):
    return D.death_review(untraded(tables), tables("rounds"), tables("round_players"), tables("grenade_throws"),
                          tables("blinds"), TICK_RATE, **kw)


def test_isolated_needs_untraded_and_a_far_teammate(tables):
    # alpha dies at 450 (round 1), untraded; bravo is his only living teammate
    far = review(tables, positions=positions(3000.0), steamid=1, round_num=1)["deaths_detail"][0]
    assert far["nearest_teammate_units"] == 2890 and far["teammates_within_trade_range"] == 0
    assert "isolated" in far["flags"]
    near = review(tables, positions=positions(400.0), steamid=1, round_num=1)["deaths_detail"][0]
    assert near["nearest_teammate_units"] == 290 and "isolated" not in near["flags"]


def test_dry_opening_and_blind_time(tables):
    charlie = review(tables, steamid=3, round_num=1)["deaths_detail"][0]
    assert charlie["opening_death"] and charlie["team_flash_before"] is False and charlie["killer_was_blind"] is False
    assert "dry_opening" in charlie["flags"]
    # alpha flashed charlie at tick 300 for 2.0s: blind until 428, killed at 400 -> 0.4s left, under the 0.5s flag
    assert charlie["blind_left_s"] == 0.4 and "died_blind" not in charlie["flags"]


def test_a_teammates_flash_makes_an_opening_supported(tables):
    flash = pl.DataFrame({"tick": [360], "steamid": [4], "name": ["delta"], "side": ["T"], "grenade": ["flash"],
                          "round_num": [1], "after_round_end": [False]},
                         schema_overrides={"steamid": pl.UInt64, "round_num": pl.Int32})
    throws = pl.concat([tables("grenade_throws"), flash], how="diagonal_relaxed")
    r = D.death_review(tables("kills"), tables("rounds"), tables("round_players"), throws, tables("blinds"),
                       TICK_RATE, steamid=3, round_num=1)["deaths_detail"][0]
    assert r["team_flash_before"] is True and "dry_opening" not in r["flags"]


def test_utility_held_is_the_buy_minus_what_was_thrown(tables):
    rp = tables("round_players").with_columns(pl.lit(2).alias("flashbangs"), pl.lit(1).alias("smoke_grenades"),
                                              pl.lit(0).alias("he_grenades"), pl.lit(1).alias("fire_grenades"))
    r = D.death_review(untraded(tables), tables("rounds"), rp, tables("grenade_throws"), tables("blinds"), TICK_RATE,
                       steamid=1, round_num=1)["deaths_detail"][0]
    # alpha threw a smoke and a flash in round 1 before dying at 450
    assert r["utility_held"] == {"flash": 1, "molotov/incendiary": 1} and r["utility_value"] == 200 + 500
    assert "died_with_utility" in r["flags"]


def test_summary_counts_flags_and_swing(tables):
    swing = {(1, 450): -0.2, (2, 1500): -0.1, (4, 3500): -0.05}
    out = review(tables, positions=positions(3000.0), steamid=1, swing=swing)
    assert out["deaths"] == 3 and out["untraded"] == 2
    assert out["flags"]["isolated"] == {"count": 1, "rounds": [1], "swing_pp": -20.0}
    assert out["costliest_deaths"][0]["round"] == 1
    assert out["distance_kind"] == "straight-line"


def test_without_positions_spacing_is_missing_not_guessed(tables):
    r = review(tables, steamid=1, round_num=1)
    assert r["deaths_detail"][0]["nearest_teammate_units"] is None and r["distance_kind"] is None
    assert "isolated" not in r["deaths_detail"][0]["flags"]


def test_untraded_with_support_and_post_round_deaths(tables):
    # alpha dies untraded at 450 with bravo 290 units away: the trade was there to take
    near = review(tables, positions=positions(400.0), steamid=1, round_num=1)
    assert "untraded_with_support" in near["deaths_detail"][0]["flags"]
    # charlie's death at 1100 came after round 1 was decided: not reviewed, but counted
    out = review(tables, steamid=3)
    assert out["deaths_after_round_end"] == {"count": 1, "rounds": [1]}
    assert all(d["tick"] != 1100 for d in out["deaths_detail"])


def test_after_plant(tables):
    # round 2: plant at 1700; alpha died at 1500 (before), bravo at 1600 (before)
    r2 = review(tables, round_num=2, bomb=tables("bomb"))["deaths_detail"]
    assert [d["after_plant"] for d in r2] == [False, False]
    late_plant = tables("bomb").with_columns(pl.lit(1550).cast(tables("bomb")["tick"].dtype).alias("tick"))
    r2 = review(tables, round_num=2, bomb=late_plant)["deaths_detail"]
    assert [d["after_plant"] for d in r2] == [False, True]


def three_t_players(tables, delta_x: float):
    """Rounds 1-2 with a third T player, echo(5). charlie and echo stand together at x=3000/3100 all round;
    delta stands at delta_x. delta dies in round 1 at tick 600, at (110, 210) per the kill feed."""
    rp = tables("round_players")
    echo = rp.filter((pl.col("steamid") == 3) & (pl.col("round_num") <= 2)).with_columns(
        pl.lit(5, dtype=pl.UInt64).alias("steamid"), pl.lit("echo").alias("name"))
    rows = []
    for lo, hi in ((200, 1000), (1300, 2000)):
        for t in range(lo, hi, 8):
            for sid, name, x in [(3, "charlie", 3000.0), (5, "echo", 3100.0), (4, "delta", delta_x)]:
                rows.append({"tick": t, "steamid": sid, "name": name, "side": "T", "x": x, "y": 0.0, "z": 0.0, "health": 100})
    return pl.concat([rp, echo]), pl.DataFrame(rows, schema_overrides={"steamid": pl.UInt64})


def test_spacing_profile_finds_the_rounds_a_player_spent_apart(tables, monkeypatch):
    monkeypatch.setattr(D, "LURK_MIN_ROUNDS", 2)
    monkeypatch.setattr(D, "LURK_SKIP_S", 2.0)  # the fake rounds are short: round 2 is planted 6s in
    rp, pos = three_t_players(tables, delta_x=0.0)
    prof = D.spacing_profile(pos, tables("rounds"), tables("kills"), rp, TICK_RATE, bomb=tables("bomb"))
    assert prof[4]["lurker"] and prof[4]["apart_rounds"] == [1, 2] and prof[4]["share_apart"] == 1.0
    assert not prof[3]["lurker"] and not prof[5]["lurker"] and prof[3]["rounds_apart"] == 0
    rp, pos = three_t_players(tables, delta_x=3200.0)
    assert not any(p["lurker"] for p in D.spacing_profile(pos, tables("rounds"), tables("kills"), rp, TICK_RATE).values())


def test_spacing_window_ends_at_the_plant(tables, monkeypatch):
    monkeypatch.setattr(D, "LURK_SKIP_S", 2.0)
    # round 2 is planted at 1700: after it, everyone stacks on the bomb, which mustn't hide a lurk before it
    rp, pos = three_t_players(tables, delta_x=0.0)
    stacked = pos.with_columns(pl.when(pl.col("tick") >= 1700).then(3000.0).otherwise(pl.col("x")).alias("x"))
    prof = D.spacing_profile(stacked, tables("rounds"), tables("kills"), rp, TICK_RATE, bomb=tables("bomb"))
    assert 2 in prof[4]["apart_rounds"]


def test_isolated_in_a_round_spent_apart_is_the_lurk_not_a_flag(tables):
    def delta(delta_x):
        rp, pos = three_t_players(tables, delta_x)
        return D.death_review(untraded(tables), tables("rounds"), rp, tables("grenade_throws"), tables("blinds"),
                              TICK_RATE, positions=pos, steamid=4, round_num=1)

    out = delta(0.0)
    d = out["deaths_detail"][0]
    assert d["nearest_teammate_units"] == 2997 and d["lurking"] is True  # charlie is dead: (110, 210) to echo (3100, 0)
    assert "isolated" not in d["flags"] and d["isolated_while_lurking"]
    assert out["isolated_while_lurking"] == {"count": 1, "rounds": [1]}
    # with the team all round, then dying alone: that is flagged
    d = delta(3200.0)["deaths_detail"][0]
    assert d["lurking"] is False and d["headline"] == "isolated" and d["flag_points_at"] == "player"


def test_headline_ignores_utility_and_blind_flags(tables):
    rp = tables("round_players").with_columns(pl.lit(2).alias("flashbangs"), pl.lit(1).alias("smoke_grenades"),
                                              pl.lit(0).alias("he_grenades"), pl.lit(1).alias("fire_grenades"))
    r = D.death_review(untraded(tables), tables("rounds"), rp, tables("grenade_throws"), tables("blinds"), TICK_RATE,
                       steamid=1, round_num=1)["deaths_detail"][0]
    assert r["flags"] == ["died_with_utility"] and r["headline"] is None and r["flag_points_at"] is None
    assert r["time_s"] == round((450 - 200) / TICK_RATE, 1)
    near = review(tables, positions=positions(400.0), steamid=1, round_num=1)["deaths_detail"][0]
    assert near["headline"] == "untraded_with_support" and near["flag_points_at"] == "teammates"


def test_a_double_kill_avenged_trades_both_deaths(tables):
    # Fixture as scripted: delta kills alpha at 450, bravo kills delta at 600 (2.3s later). awpy's flag says
    # alpha wasn't traded; by the usual definition (killer dies to the team within 5.25s) he was.
    r = D.death_review(tables("kills"), tables("rounds"), tables("round_players"), tables("grenade_throws"),
                       tables("blinds"), TICK_RATE, steamid=1, round_num=1)["deaths_detail"][0]
    assert r["traded"] is True and r["traded_after_s"] == round(150 / TICK_RATE, 1)
    assert review(tables, steamid=1, round_num=1)["deaths_detail"][0]["traded"] is False, "too late with the fix-up"


def test_carrying_bomb(tables):
    drop = pl.DataFrame({"tick": [450], "event": ["drop"], "steamid": [1], "name": ["alpha"], "bombsite": [None],
                         "round_num": [1], "after_round_end": [False]})
    bomb = pl.concat([tables("bomb"), drop], how="diagonal_relaxed")
    r = review(tables, steamid=1, round_num=1, bomb=bomb)["deaths_detail"][0]
    assert r["carrying_bomb"] is True
    assert review(tables, steamid=1, round_num=1, bomb=tables("bomb"))["deaths_detail"][0]["carrying_bomb"] is False
