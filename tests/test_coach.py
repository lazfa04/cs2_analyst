"""Coach cards and scoring them against expert labels. Pure functions on death review rows."""

from pathlib import Path

import pytest

from cs2_analyst import coach as C


def row(rnd, flags, *, headline=None, swing=None, lurk=False, **kw):
    pts = {"isolated": "player", "dry_opening": "player", "untraded_with_support": "teammates"}
    r = {"round": rnd, "tick": 1000 * rnd, "time_s": 30.0, "victim": "me", "side": "T", "killer": "them",
         "weapon": "AK-47", "round_won": False, "traded": False, "opening_death": False, "after_plant": False,
         "lurking": lurk, "isolated_while_lurking": lurk, "teammates_alive": 3, "nearest_teammate_units": 1200,
         "teammates_within_trade_range": 0, "team_flash_before": None, "killer_was_blind": False, "blind_left_s": 0.0,
         "utility_held": {"smoke": 1, "flash": 2}, "utility_value": 700, "flags": flags, "headline": headline,
         "flag_points_at": pts.get(headline), **kw}
    if swing is not None:
        r["swing_pp"] = swing
    return r


def review(rows):
    return {"deaths": len(rows), "deaths_detail": rows, "distance_kind": "walking",
            "spacing_profile": {"me": {"lurker": True}}}


def test_cards_lead_with_the_headline_costliest_first_and_count_what_they_skip():
    rows = [row(3, ["isolated"], headline="isolated", swing=-10.0, after_plant=True),
            row(5, [], lurk=True, swing=-30.0),
            row(7, ["untraded_with_support"], headline="untraded_with_support", swing=-25.0,
                teammates_within_trade_range=2, nearest_teammate_units=400),
            row(12, ["died_with_utility"], swing=-15.0),
            row(14, [], swing=-5.0)]
    out = C.coach_cards(review(rows))
    assert [c["round"] for c in out["cards"]] == [7, 3], "costliest first, only headline flags"
    r3 = out["cards"][1]
    assert r3["title"] == "Died where nobody could trade" and "on foot" in r3["facts"][0]
    assert "bomb was down" in r3["question"] and "after the plant" in r3["context"]
    assert out["by_whom"] == {"player": 1, "teammates": 1} and out["lurker"] is True
    assert len(out["not_shown"]) == 3 and "lurking" in out["not_shown"][0] and "rounds 12" in out["not_shown"][1]
    assert out["sorted_by"].startswith("swing")


def test_after_plant_questions_depend_on_the_side():
    t = C.card(row(3, ["isolated"], headline="isolated", after_plant=True), "walking")
    ct = C.card(row(3, ["isolated"], headline="isolated", after_plant=True, side="CT"), "walking")
    bomb = C.card(row(3, ["isolated"], headline="isolated", carrying_bomb=True), "walking")
    assert "around the bomb" in t["question"] and "retak" in ct["question"] and "You had the bomb" in bomb["question"]
    assert "Carrying the bomb: it dropped where you died." in bomb["facts"]


def test_utility_and_blind_are_facts_on_a_card_never_the_card():
    c = C.card(row(4, ["dry_opening", "died_with_utility", "died_blind"], headline="dry_opening", blind_left_s=1.2,
                   opening_death=True), "straight-line")
    assert c["title"] == "Opening death without support"
    assert "Still holding 1 smoke, 2 flash ($700)." in c["facts"] and "Blind for another 1.2s." in c["facts"]
    assert "opening death" not in c["context"], "already the headline"


LABELS = [
    {"demo": "d.dem", "player": "me", "round": 3, "flag_was_the_issue": True, "flags": ["isolated"], "mistake_by": "player"},
    {"demo": "d.dem", "player": "me", "round": 5, "flag_was_the_issue": False, "flags": ["isolated", "died_with_utility"],
     "mistake_by": "none"},
    {"demo": "d.dem", "player": "me", "round": 7, "flag_was_the_issue": True, "flags": ["untraded_with_support"],
     "mistake_by": "teammates"},
    {"demo": "d.dem", "player": "me", "round": 12, "flag_was_the_issue": False, "flags": ["died_with_utility"],
     "mistake_by": "none"},
    {"demo": "d.dem", "player": "me", "round": 16, "flag_was_the_issue": False, "flags": ["isolated"], "mistake_by": "player"},
]


def test_score_labels_any_flag_vs_headline():
    rows = [row(3, ["isolated"], headline="isolated"),
            row(5, ["died_with_utility"], lurk=True),                      # lurk cleared the isolated flag
            row(7, ["untraded_with_support"], headline="untraded_with_support"),
            row(12, ["died_with_utility"]),
            row(16, ["isolated"], headline="isolated")]
    s = C.score_labels(LABELS, rows)
    assert s["modes"]["any_flag"] == {"hit": 2, "miss": 0, "false_alarm": 3, "quiet": 0, "precision": 0.4, "recall": 1.0}
    assert s["modes"]["headline"] == {"hit": 2, "miss": 0, "false_alarm": 1, "quiet": 2, "precision": 0.667, "recall": 1.0}
    assert s["blame_agreement"] == 1.0, "round 16: the label blames the player, and so does the isolated headline"
    assert s["per_death"][1]["outcomes"] == {"any_flag": "false_alarm", "headline": "quiet"}


def test_unfound_deaths_are_reported_not_scored():
    s = C.score_labels(LABELS[:1], [None])
    assert s["found"] == 0 and s["modes"]["headline"]["precision"] is None and s["per_death"][0]["found"] is False


def test_load_labels_validates(tmp_path: Path):
    p = tmp_path / "l.toml"
    p.write_text('[[death]]\ndemo = "d.dem"\nplayer = "me"\nround = 1\n')
    with pytest.raises(ValueError, match="missing"):
        C.load_labels(p)
    assert len(C.load_labels(Path("evals/death_labels.toml"))) == 5


def test_find_demo_by_file_name_from_any_os(store):
    import json

    meta = store.cache_dir / "fakedemo0001" / "meta.json"
    m = json.loads(meta.read_text())
    m["source"] = "C:\\\\Users\\\\me\\\\Downloads\\\\test-demos\\\\D2.dem"
    meta.write_text(json.dumps(m))
    assert C.find_demo(store, "d2.dem") == "fakedemo0001"
    assert C.find_demo(store, "fakedemo0001") == "fakedemo0001"
    assert C.find_demo(store, "other.dem") is None


def test_label_rows_finds_the_death(store):
    import json

    meta = store.cache_dir / "fakedemo0001" / "meta.json"
    m = json.loads(meta.read_text())
    m["source"] = "/demos/fake.dem"
    meta.write_text(json.dumps(m))
    labs = [{"demo": "fake.dem", "player": "alpha", "round": 2, "flag_was_the_issue": False, "flags": [],
             "mistake_by": "none"}]
    rows, _ = C.label_rows(store, labs)
    assert rows[0] is not None and rows[0]["round"] == 2 and rows[0]["victim"] == "alpha"
