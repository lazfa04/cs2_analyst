import polars as pl
import pytest

from cs2_analyst import analysis as A
from cs2_analyst.errors import PlayerNotFound, RoundNotFound

TICK_RATE = 64.0


# ---------------------------------------------------------------- resolve_player

@pytest.mark.parametrize("query", ["alpha", "ALPHA", "alp", "1"])
def test_resolve_player_accepts_names_partials_and_ids(tables, query):
    assert A.resolve_player(tables("players"), query)["steamid"] == 1


def test_resolve_player_ambiguous_lists_candidates(tables):
    with pytest.raises(PlayerNotFound, match="several players"):
        A.resolve_player(tables("players"), "a")  # alpha, bravo, charlie, delta


def test_resolve_player_unknown_suggests_close_match(tables):
    with pytest.raises(PlayerNotFound, match="Did you mean.*alpha"):
        A.resolve_player(tables("players"), "alpah")


# ---------------------------------------------------------------- overview

def test_team_labels_survive_halftime(tables):
    labels = A.team_labels(tables("round_players"), tables("players"))
    assert labels[1] == labels[2] == "Team A (started CT)"
    assert labels[3] == labels[4] == "Team B (started T)"


def test_match_overview_score_attributes_rounds_to_teams_not_sides(tables, store):
    ov = A.match_overview(store.meta("fakedemo0001"), tables("rounds"), tables("stats"),
                          tables("players"), tables("round_players"))
    assert ov["score"]["Team B (started T)"] == {"rounds_won": 3, "ct_rounds_won": 2, "t_rounds_won": 1}
    assert ov["score"]["Team A (started CT)"] == {"rounds_won": 1, "ct_rounds_won": 1, "t_rounds_won": 0}
    assert list(ov["score"])[0] == "Team B (started T)", "winner listed first"
    assert len(ov["scoreboard"]) == 4


def test_rating_matches_hltv_2_approximation(tables):
    # alpha: 2 kills, 3 deaths, 0 assists in 4 rounds, KAST 75, ADR 60
    alpha = A.with_rating(tables("stats")).filter(pl.col("name") == "alpha").row(0, named=True)
    kpr, dpr = 2 / 4, 3 / 4
    impact = 2.13 * kpr - 0.41
    expected = 0.0073 * 75 + 0.3591 * kpr - 0.5329 * dpr + 0.2372 * impact + 0.0032 * 60 + 0.1587
    assert alpha["rating"] == pytest.approx(expected, abs=0.006)
    assert alpha["impact"] == pytest.approx(impact, abs=0.006)


def test_scoreboard_ranks_by_rating_not_kills(tables, store):
    ov = A.match_overview(store.meta("fakedemo0001"), tables("rounds"), tables("stats"),
                          tables("players"), tables("round_players"))
    board = ov["scoreboard"]
    assert sorted(r["match_rank"] for r in board) == [1, 2, 3, 4]
    for team in {r["team"] for r in board}:
        rows = [r for r in board if r["team"] == team]
        assert [r["team_rank"] for r in rows] == [1, 2], "listed best-rated first within each team"
        assert rows[0]["rating"] >= rows[1]["rating"]
    # bravo and delta both have 3 kills; bravo died once, delta twice, so bravo must rate higher
    by_name = {r["name"]: r for r in board}
    assert by_name["bravo"]["match_rank"] < by_name["delta"]["match_rank"]
    assert "rating" in ov["note"]


# ---------------------------------------------------------------- round summary

def test_round_summary_has_feed_economy_and_bomb(tables):
    rs = A.round_summary(2, tables("rounds"), tables("kills"), tables("round_players"),
                         tables("round_economy"), tables("bomb"), TICK_RATE)
    assert rs["winner_side"] == "T"
    assert rs["economy"]["T"]["buy_type"] == "full"
    assert [k["tick"] for k in rs["kills"]] == [1500, 1600]
    assert rs["kills"][0]["time_s"] == round((1500 - 1300) / 64, 1)
    assert rs["bomb"][0]["bombsite"] == "A"
    assert len(rs["loadouts"]) == 4


def test_round_summary_unknown_round(tables):
    with pytest.raises(RoundNotFound, match="1-4"):
        A.round_summary(9, tables("rounds"), tables("kills"), tables("round_players"),
                        tables("round_economy"), tables("bomb"), TICK_RATE)


# ---------------------------------------------------------------- opening duels

def test_opening_duels_counts_by_side_and_outcome(tables):
    od = A.opening_duels(tables("kills"), tables("rounds"), 1, TICK_RATE)
    assert od["overall"] == {"taken": 4, "won": 2, "lost": 2}
    assert od["by_side"]["CT"] == {"taken": 2, "won": 1, "lost": 1}
    assert od["by_side"]["T"] == {"taken": 2, "won": 1, "lost": 1}
    # r1 won duel -> won round; r3 won duel on T but CT won the round.
    assert od["rounds_won_after_opening"] == {"after_winning_duel": 1, "after_losing_duel": 0}


def test_opening_duels_reports_trades_and_citations(tables):
    duels = {d["round"]: d for d in A.opening_duels(tables("kills"), tables("rounds"), 1, TICK_RATE)["duels"]}
    assert duels[2]["traded"] is False and duels[4]["traded"] is True
    assert duels[1]["traded"] is None, "trade flag only applies to lost duels"
    assert all("tick" in d and "round" in d for d in duels.values())


def test_opening_duels_limit_truncates(tables):
    od = A.opening_duels(tables("kills"), tables("rounds"), 1, TICK_RATE, limit=1)
    assert len(od["duels"]) == 1 and od["truncated"] is True


# ---------------------------------------------------------------- utility

def test_utility_splits_enemy_team_and_self_flashes(tables):
    u = A.utility_summary(tables("stats"), tables("blinds"), tables("damages"), 1)
    # team blind on bravo is 1.5s (counts); self blind is 1.0s (minor, not a self-flash)
    assert (u["enemy_blinds"], u["teammates_flashed"], u["self_flashes"]) == (1, 1, 0)
    assert (u["minor_team_blinds"], u["minor_self_blinds"]) == (0, 1)
    assert u["worst_team_flashes"][0]["teammate"] == "bravo"
    # team damage from the HE on bravo is excluded; only 40 on charlie counts
    assert u["grenade_damage_by_type"] == [{"weapon": "hegrenade", "damage": 40, "hits": 1}]
    assert u["utility_damage_per_round"] == 10.0


# ---------------------------------------------------------------- find_kills

def test_find_kills_untraded_deaths(tables):
    res = A.find_kills(tables("kills"), steamid=1, role="victim", traded=False)
    assert [k["tick"] for k in res["kills"]] == [450, 1500]


def test_find_kills_excludes_exit_frags_and_filters(tables):
    all_kills = A.find_kills(tables("kills"), limit=50)
    assert 1100 not in [k["tick"] for k in all_kills["kills"]]
    assert A.find_kills(tables("kills"), weapon="awp")["total"] == 1
    assert A.find_kills(tables("kills"), side="ct")["total"] == 3  # ticks 400, 600, 3500


def test_find_kills_paginates(tables):
    p1 = A.find_kills(tables("kills"), limit=3)
    p2 = A.find_kills(tables("kills"), limit=3, offset=p1["next_offset"])
    assert p1["total"] == 8 and p1["next_offset"] == 3
    assert {k["tick"] for k in p1["kills"]}.isdisjoint({k["tick"] for k in p2["kills"]})


def test_rating_is_floored_at_zero_but_ranking_keeps_order():
    stats = pl.DataFrame({
        "name": ["a", "b"], "rounds_played": [13, 13], "kills": [1, 0], "deaths": [13, 13],
        "assists": [0, 0], "kast": [7.7, 0.0], "adr": [14.2, 0.0],
    })
    rated = A.with_rating(stats)
    assert rated["rating"].to_list() == [0.0, 0.0], "a stomp can't produce a negative rating"
    assert rated["rating_raw"][0] > rated["rating_raw"][1], "1-13 still ranks above 0-13"


def test_fire_is_labelled_molotov_incendiary_not_inferno(tables):
    """'inferno' is the engine's name for burning fire; to a player it reads like the map."""
    damages = pl.DataFrame({
        "tick": [320, 330, 340, 350],
        "attacker_steamid": [1, 1, 1, 1], "attacker_side": ["CT"] * 4,
        "victim_steamid": [3, 3, 4, 4], "victim_side": ["T"] * 4,
        "weapon": ["inferno", "molotov", "incgrenade", "hegrenade"], "dmg_health": [30, 5, 20, 40],
    }, schema_overrides={"attacker_steamid": pl.UInt64, "victim_steamid": pl.UInt64})
    u = A.utility_summary(tables("stats"), tables("blinds"), damages, 1)
    by_type = {g["weapon"]: g["damage"] for g in u["grenade_damage_by_type"]}
    assert by_type == {"molotov/incendiary": 55, "hegrenade": 40}, "fire variants merged under one label"

    kills = tables("kills").with_columns(pl.lit("inferno").alias("weapon"))
    res = A.find_kills(kills, weapon="molotov", limit=1)
    assert res["total"] == 8 and res["kills"][0]["weapon"] == "molotov/incendiary"
    assert A.find_kills(kills, weapon="inferno")["total"] == 8, "the engine name still works as a filter"


# ---------------------------------------------------------------- counts so the model never tallies by hand

def test_find_kills_breakdown_counts_all_matches_not_just_the_page(tables):
    res = A.find_kills(tables("kills"), rounds=tables("rounds"), steamid=1, role="victim", limit=1)
    assert res["returned"] == 1 and res["total"] == 3
    assert res["rounds"] == [1, 2, 4]
    assert res["by_attacker"] == [{"attacker": "delta", "count": 2, "rounds": [1, 4]},
                                  {"attacker": "charlie", "count": 1, "rounds": [2]}]
    # alpha was CT in rounds 1-2 and T in round 4: the split is by the VICTIM's side here
    assert res["by_side"] == {"CT": 2, "T": 1} and "victim" in res["by_side_is_side_of"]
    # r2 and r4 deaths were openings; the r1 death was a trade; killers won r2 and r4
    assert res["tags"] == {"openings": 2, "trades": 1, "in_rounds_the_killer_won": 2}


def test_find_kills_tags_each_kill(tables):
    kills = {k["tick"]: k for k in A.find_kills(tables("kills"), rounds=tables("rounds"), limit=50)["kills"]}
    assert kills[400]["opening"] and kills[400]["attacker_won_round"]
    assert not kills[450]["opening"] and kills[450]["is_trade"] and not kills[450]["attacker_won_round"]
    assert A.find_kills(tables("kills"), rounds=tables("rounds"))["exit_frags_excluded"] == 1


# ---------------------------------------------------------------- clutches

def test_last_alive_situations_replay_each_round(tables):
    raw = A.last_alive_situations(tables("kills"), tables("rounds"), tables("round_players"))
    assert len(raw) == 6, "every round of a 2v2 leaves someone last alive; r1 and r4 have one per side"


def test_clutches_count_like_a_player_would(tables):
    c = A.clutches(tables("kills"), tables("rounds"), tables("round_players"))
    # r1 delta 1v2, 1 kill, died: lost.  r1 bravo 1v1 won.  r2 bravo 1v2, no kill, died: last alive.
    # r3 delta 1v2 won.  r4 bravo 1v2, 1 kill, survived the lost round: saved.  r4 charlie 1v1 won.
    assert (c["summary"]["played"], c["summary"]["won"], c["summary"]["lost"]) == (4, 3, 1)
    assert (c["summary"]["saved"], c["summary"]["last_alive_not_clutch"]) == (1, 1)
    by = {p["name"]: p for p in c["by_player"]}
    assert (by["bravo"]["played"], by["bravo"]["won"], by["bravo"]["saved"], by["bravo"]["last_alive_not_clutch"]) == (1, 1, 1, 1)
    assert (by["delta"]["played"], by["delta"]["won"]) == (2, 1)
    assert by["delta"]["by_situation"] == {"1v2": {"played": 2, "won": 1}}


def test_clutches_both_sides_can_clutch_in_one_round(tables):
    r1 = [c for c in A.clutches(tables("kills"), tables("rounds"), tables("round_players"))["clutches"] if c["round"] == 1]
    # delta is left 1v2 and gets a kill; that leaves bravo 1v1, who wins it
    assert [(c["player"], c["situation"], c["kills"], c["outcome"]) for c in r1] == [
        ("delta", "1v2", 1, "lost"), ("bravo", "1v1", 1, "won")]


def test_clutches_for_one_player_lists_saves_and_last_alive_apart(tables):
    c = A.clutches(tables("kills"), tables("rounds"), tables("round_players"), steamid=2)
    assert [x["round"] for x in c["clutches"]] == [1] and c["by_player"] is None
    assert [(x["round"], x["outcome"]) for x in c["not_clutches"]] == [(2, "last_alive"), (4, "saved")]


@pytest.mark.parametrize("vs,kills,won,survived,outcome", [
    (5, 0, True, True, "won"),          # a won 1v5 is a clutch won
    (5, 0, False, False, "last_alive"), # a lost 1v5 without a fight isn't a lost clutch
    (4, 2, False, False, "lost"),       # ...but a 1v4 with kills is an attempt (m0NESY, Anubis r18)
    (4, 0, False, False, "last_alive"),
    (2, 1, False, True, "saved"),       # survived a lost round: saved the gun
    (2, 1, False, False, "lost"),       # an attempt: got a kill, then died
    (3, 0, False, False, "last_alive"), # died without a kill
    (1, 0, False, False, "lost"),       # a 1v1 is always an attempt
    (3, 0, True, False, "won"),         # won by the bomb after dying still counts
])
def test_clutch_outcome_rules(vs, kills, won, survived, outcome):
    assert A.clutch_outcome({"vs": vs, "kills": kills, "won": won, "survived": survived}) == outcome


@pytest.mark.parametrize("attempt", [["planted"], ["went_for_defuse"], ["damage"]])
def test_an_attempt_without_a_kill_is_still_a_lost_clutch(attempt):
    assert A.clutch_outcome({"vs": 2, "kills": 0, "won": False, "survived": False, "attempt": attempt}) == "lost"


@pytest.mark.parametrize("attempt,outcome", [(["planted"], "lost"), (["damage"], "last_alive"),
                                             (["went_for_defuse"], "last_alive")])
def test_at_1v4_odds_only_a_kill_or_a_plant_counts_as_an_attempt(attempt, outcome):
    assert A.clutch_outcome({"vs": 4, "kills": 0, "won": False, "survived": False, "attempt": attempt}) == outcome


def _bravo_r2(tables, damages=None, bomb=None):
    """Round 2: bravo (CT) is left 1v2 at tick 1500 and dies at 1600 without a kill."""
    raw = A.last_alive_situations(tables("kills"), tables("rounds"), tables("round_players"), damages, bomb)
    return next(c for c in raw if c["round"] == 2)


def test_damage_after_becoming_last_alive_is_an_attempt(tables):
    assert _bravo_r2(tables)["attempt"] == [] and A.clutch_outcome(_bravo_r2(tables)) == "last_alive"
    dmg = pl.DataFrame({"tick": [1550], "round_num": [2], "attacker_steamid": [2], "victim_side": ["T"],
                        "dmg_health": [27], "after_round_end": [False]},
                       schema_overrides={"attacker_steamid": pl.UInt64, "round_num": pl.Int32})
    c = _bravo_r2(tables, damages=dmg)
    assert c["attempt"] == ["damage"] and A.clutch_outcome(c) == "lost"


@pytest.mark.parametrize("plant_xy,attempted", [((300.0, 300.0), True), ((3000.0, 3000.0), False)])
def test_ct_dying_near_the_planted_bomb_went_for_the_defuse(tables, plant_xy, attempted):
    # bravo dies at (110, 210); the bomb was planted at tick 1550, before he died
    bomb = pl.DataFrame({"tick": [1550], "round_num": [2], "event": ["finish_plant"], "steamid": [3],
                         "x": [plant_xy[0]], "y": [plant_xy[1]]},
                        schema_overrides={"steamid": pl.UInt64, "round_num": pl.Int32})
    c = _bravo_r2(tables, bomb=bomb)
    assert ("went_for_defuse" in c["attempt"]) is attempted


def test_t_planting_after_becoming_last_alive_is_an_attempt():
    bomb = pl.DataFrame({"tick": [1550], "round_num": [2], "event": ["start_plant"], "steamid": [2]},
                        schema_overrides={"steamid": pl.UInt64, "round_num": pl.Int32})
    clutch = {"round": 2, "tick": 1500, "_sid": 2, "side": "T", "kills": 0}
    assert A._attempt_signals(clutch, None, bomb) == ["planted"]


def test_scoreboard_clutch_columns_use_the_same_rules(tables, store):
    ov = A.match_overview(store.meta("fakedemo0001"), tables("rounds"), tables("stats"), tables("players"),
                          tables("round_players"), kills=tables("kills"))
    board = {r["name"]: (r["clutches_won"], r["clutches_played"]) for r in ov["scoreboard"]}
    assert board == {"alpha": (0, 0), "bravo": (1, 1), "charlie": (1, 1), "delta": (1, 2)}


# ---------------------------------------------------------------- costly team flashes

def test_team_flash_is_costly_only_if_teammate_died_while_blind(tables):
    def deaths(tick):
        return pl.DataFrame({"victim_steamid": [2], "tick": [tick], "attacker_name": ["charlie"]},
                            schema_overrides={"victim_steamid": pl.UInt64})
    # bravo is blinded at tick 310 for 1.5s: blind + 0.5s grace runs to tick 438
    costly = A.utility_summary(tables("stats"), tables("blinds"), tables("damages"), 1, kills=deaths(350))
    assert costly["costly_team_flashes"] == 1
    assert costly["costly_team_flash_list"][0]["killed_by"] == "charlie"
    assert costly["worst_team_flashes"][0]["teammate_died_while_blind"] is True
    harmless = A.utility_summary(tables("stats"), tables("blinds"), tables("damages"), 1, kills=deaths(500))
    assert harmless["costly_team_flashes"] == 0 and harmless["teammates_flashed"] == 1


# ---------------------------------------------------------------- grenades thrown

def test_grenade_usage_splits_by_side_and_plant(tables):
    g = A.grenade_usage(tables("grenade_throws"), tables("bomb"), tables("round_players"), tables("players"), 1, 4)
    assert g["thrown"] == {"smoke": 2, "flash": 1, "HE": 1, "molotov/incendiary": 0, "decoy": 0}
    assert g["total_thrown"] == 4 and g["per_round"] == 1.0, "rifle shot and warmup throw ignored"
    assert g["by_side_and_plant"]["CT"]["after_plant"]["smoke"] == 1, "r2 smoke after the plant is a retake smoke"
    assert g["by_side_and_plant"]["CT"]["no_plant"]["flash"] == 1
    assert g["by_side"]["T"]["HE"] == 1
    assert [r["name"] for r in g["team_comparison"]] == ["alpha", "bravo"]
    assert g["team_comparison"][0]["team_rank_total"] == 1


def test_grenade_usage_execute_vs_post_plant(tables):
    g = A.grenade_usage(tables("grenade_throws"), tables("bomb"), tables("round_players"), tables("players"), 3, 4)
    assert g["by_side_and_plant"]["T"]["before_plant"]["smoke"] == 1
    assert g["by_side_and_plant"]["T"]["after_plant"]["molotov/incendiary"] == 1


# ---------------------------------------------------------------- round list

def test_round_list_score_halves_and_economy(tables):
    rl = A.round_list(tables("rounds"), tables("kills"), tables("round_players"), tables("players"),
                      tables("round_economy"), tables("bomb"))
    a, b = "Team A (started CT)", "Team B (started T)"
    assert rl["final_score"] == {a: 1, b: 3}
    assert rl["halves"] == [
        {"half": 1, "rounds": "1-2", "overtime": False, "sides": {a: "CT", b: "T"}, "rounds_won": {a: 1, b: 1}},
        {"half": 2, "rounds": "3-4", "overtime": False, "sides": {a: "T", b: "CT"}, "rounds_won": {a: 0, b: 2}},
    ]
    assert rl["rounds_won_by_buy_type"][a] == {"pistol": {"played": 1, "won": 1}, "eco": {"played": 1, "won": 0}}
    assert rl["first_kill_conversion"][b] == {"first_kills": 2, "rounds_won_after_first_kill": 2}
    r2 = rl["rounds"][1]
    assert r2["bomb_planted"] == "A" and r2["winner"] == b and r2["score_after"] == {a: 1, b: 1}


def test_round_list_overtime_starts_a_new_half_even_without_a_side_swap(tables):
    # Pretend regulation was 3 rounds with 1-round overtime halves: round 4 keeps round 3's sides
    # but is overtime, so it must not be merged into the second half.
    rl = A.round_list(tables("rounds"), tables("kills"), tables("round_players"), tables("players"),
                      tables("round_economy"), tables("bomb"), max_rounds=3, overtime_rounds=1)
    assert [(h["rounds"], h["overtime"]) for h in rl["halves"]] == [("1-2", False), ("3-3", False), ("4-4", True)]
