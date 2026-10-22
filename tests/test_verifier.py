import pytest

from cs2_analyst.agent.findings import Check, Claim, Report
from cs2_analyst.agent.pricing import cost_usd
from cs2_analyst.agent.verifier import PathError, compare, feedback, parse_path, resolve, verify

RESULT = {
    "overall": {"taken": 7, "won": 6},
    "score": {"Team B (started T)": {"rounds_won": 16}},
    "duels": [
        {"round": 5, "won": True, "opponent": "kzri", "time_s": 21.2, "traded": None},
        {"round": 9, "won": False, "opponent": "organner.pl free trial", "time_s": 8.0, "traded": True},
    ],
}


# ---------------------------------------------------------------- paths

@pytest.mark.parametrize("path,value", [
    ("overall.won", 6),
    ("score.Team B (started T).rounds_won", 16),
    ("duels.1.opponent", "organner.pl free trial"),
    ("duels[1].won", False),
    ("duels[round=9].traded", True),
    ("duels[opponent=organner.pl free trial].round", 9),
    ("duels[round=9,won=false].time_s", 8.0),
])
def test_resolve_paths(path, value):
    assert resolve(RESULT, path) == value


@pytest.mark.parametrize("path,msg", [
    ("overall.lost", "available"),
    ("duels[round=4].won", "no item matches"),
    ("duels.7", "out of range"),
    ("duels.won", "use an index"),
    ("overall.won.x", "plain value"),
    ("duels[round].won", "key=value"),
    ("duels[round=4", "unclosed"),
])
def test_resolve_errors_explain_themselves(path, msg):
    with pytest.raises(PathError, match=msg):
        resolve(RESULT, path)


def test_parse_path_tokens():
    assert parse_path("a.b[2].c[x=1,y=two]") == ["a", "b", 2, "c", {"x": "1", "y": "two"}]


# ---------------------------------------------------------------- comparison

@pytest.mark.parametrize("actual,expected,ok", [
    (6, "6", True), (6, "6.0", True), (6, "5", False),          # counts are exact
    (89.83, "89.8", True), (89.83, "90", True),                 # floats: precision-aware rounding
    (89.83, "89", False), (89.83, "89.9", False), (89.83, "89.83", True),
    (2.5, "3", True), (2.5, "2", False), (73.25, "73.3", True),   # half-up, not banker's rounding
    (45.0, "45%", True),
    (True, "true", True), (True, "yes", True), (False, "true", False),
    (None, "null", True), (None, "0", False),
    ("kzri", "KZRI", True), ("kzri", "kzr", False),
])
def test_compare(actual, expected, ok):
    assert compare(actual, expected)[0] is ok


def test_compare_rejects_containers():
    ok, reason = compare({"a": 1}, "1")
    assert not ok and "single value" in reason


# ---------------------------------------------------------------- verify

LEDGER = {"t1": {"name": "get_opening_duels", "result": RESULT, "is_error": False},
          "t2": {"name": "get_opening_duels", "result": "No player 'zulu'", "is_error": True}}


def fact(cid, *checks):
    return Claim(id=cid, text=cid, kind="fact", checks=[Check(call_id=c, path=p, expected=e) for c, p, e in checks])


def interp(cid, *supports):
    return Claim(id=cid, text=cid, kind="interpretation", supports=list(supports))


def test_verify_statuses():
    report = Report(summary="s", claims=[
        fact("ok", ("t1", "overall.won", "6"), ("t1", "overall.taken", "7")),
        fact("wrong", ("t1", "overall.won", "7")),
        fact("bad_call", ("t9", "overall.won", "6")),
        fact("err_call", ("t2", "x", "1")),
        fact("no_checks"),
        interp("i_good", "ok"),
        interp("i_bad", "ok", "wrong"),
        interp("i_none"),
    ])
    st = {r.claim.id: r.status for r in verify(report, LEDGER).results}
    assert st == {"ok": "verified", "wrong": "failed", "bad_call": "failed", "err_call": "failed",
                  "no_checks": "unchecked", "i_good": "not_judged", "i_bad": "unsupported", "i_none": "unsupported"}


def test_feedback_includes_actual_values():
    v = verify(Report(summary="s", claims=[fact("c1", ("t1", "overall.won", "7"))]), LEDGER)
    text = feedback(v)
    assert "c1" in text and "actual value: 6" in text and "submit_report again" in text


# ---------------------------------------------------------------- pricing

def test_cost_sonnet():
    # 10k fresh in at $2, 2k out at $10, 50k cache reads at $0.10 per MTok
    assert cost_usd("claude-sonnet-5-5", 10_000, 2_000, cache_read=50_000) == pytest.approx(0.02 + 0.02 + 0.005)


def test_cost_haiku_long_context_tier_and_unknown_model():
    assert cost_usd("claude-haiku-5-5", 150_000, 0) == pytest.approx(0.075)  # $0.50 tier above 100k
    assert cost_usd("some-other-model", 1, 1) is None
