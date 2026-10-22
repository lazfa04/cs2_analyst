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
