"""Red-team scoring for the judge, with scripted judges (no API)."""

import json
from pathlib import Path

import anyio
import pytest

from cs2_analyst.agent.llm import Block, LLMResponse
from cs2_analyst.evals.redteam import RedTeamError, load_redteam, misses, run_redteam, score

USAGE = {"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
REAL_FILE = Path(__file__).parent.parent / "evals" / "redteam.toml"

SMALL = """
[[case]]
id = "a"
facts = ["alpha won 4 of 8 opening duels.", "The team won all 4 rounds after his opening wins."]
summary = "alpha went 4 for 8 in openings. He also had 30 kills."

[[case.interpretation]]
text = "His opening wins were valuable."
supports = ["f2"]
expected = true
category = "fair"

[[case.interpretation]]
text = "He dominated the opening duels."
supports = ["f1"]
expected = false
category = "overreach"

[[case.planted]]
kind = "unbacked"
match = ["30 kills"]

[[case]]
id = "b"
facts = ["bravo died 10 times."]
summary = "bravo died 10 times."

[[case.interpretation]]
text = "His teammates refused to trade him."
supports = ["f1"]
expected = false
category = "unmeasured"
"""


def _resp(input_):
    blocks = [{"type": "tool_use", "id": "tu_v", "name": "submit_verdicts", "input": input_}]
    return LLMResponse([Block(**b) for b in blocks], blocks, dict(USAGE), "tool_use")


class ScriptedJudge:
    """Approves everything and flags nothing: the rubber stamp the red team must expose."""

    def __init__(self, flag=None):
        self.flag = flag or {}
        self.payloads = []

    async def create(self, *, model, system, tools, messages, max_tokens):
        payload = json.loads(messages[0]["content"])
        self.payloads.append(payload)
        verdicts = [{"claim_id": i["claim_id"], "supported": True, "reason": "ok"} for i in payload["interpretations"]]
        flags = self.flag.get(payload["summary"], {})
        return _resp({"verdicts": verdicts,
                      "unbacked_summary_statements": flags.get("unbacked", []),
                      "contradicted_summary_statements": flags.get("contradicted", [])})


@pytest.fixture
def small(tmp_path):
    p = tmp_path / "rt.toml"
    p.write_text(SMALL)
    return load_redteam(p)


def test_the_shipped_red_team_file_is_valid_and_balanced():
    cases = load_redteam(REAL_FILE)
    interps = [i for c in cases for i in c.interpretations]
    bad = [i for i in interps if not i.expected]
    assert len(bad) >= 15 and len(interps) - len(bad) >= 10, "needs plenty of both, or one side can't be measured"
    assert {i.category for i in bad} == {"overreach", "contradicts", "unmeasured", "one_event", "causal",
                                         "rating_only"}
    assert {c.tier for c in cases} == {"easy", "hard"}
    assert any(not c.planted for c in cases) and any(p.kind == "contradicted" for c in cases for p in c.planted)


def test_a_rubber_stamp_judge_scores_50_balanced(small):
    results = anyio.run(lambda: run_redteam(small, ScriptedJudge(), "claude-haiku-5-5", "v3"))
    m = score(results)
    assert m["conclusions"]["fair_accepted_pct"] == 100.0
    assert m["conclusions"]["bad_rejected_pct"] == 0.0
    assert m["conclusions"]["balanced_accuracy_pct"] == 50.0, "always-yes must not look good"
    assert m["summaries"]["planted_caught_pct"] == 0.0
    assert m["conclusions"]["bad_rejected_by_category"] == {"overreach": {"rejected": 0, "total": 1},
                                                             "unmeasured": {"rejected": 0, "total": 1}}
    lines = misses(results)
    assert any("accepted [overreach, should be rejected] He dominated" in line for line in lines)
    assert m["conclusions"]["balanced_accuracy_by_tier_pct"] == {"easy": 50.0}
    assert any("missed planted unbacked" in line for line in lines)


def test_planted_mistakes_and_false_alarms(small):
    judge = ScriptedJudge(flag={
        "alpha went 4 for 8 in openings. He also had 30 kills.": {"unbacked": ["He also had 30 kills"]},
        "bravo died 10 times.": {"contradicted": ["bravo died 10 times"]},  # a false alarm on a clean summary
    })
    results = anyio.run(lambda: run_redteam(small, judge, "claude-haiku-5-5", "v3"))
    m = score(results)["summaries"]
    assert m["unbacked_caught_pct"] == 100.0 and m["clean_summaries_flagged_pct"] == 100.0
    assert any("false alarm" in line for line in misses(results))
    # the judge saw the facts and each conclusion with only its own supporting facts
    first = judge.payloads[0]
    assert first["verified_claims"] == ["alpha won 4 of 8 opening duels.", "The team won all 4 rounds after his opening wins."]
    assert [f["text"] for f in first["interpretations"][1]["facts"]] == ["alpha won 4 of 8 opening duels."]


@pytest.mark.parametrize("broken,error", [
    (SMALL.replace('supports = ["f2"]', 'supports = ["f9"]'), "don't exist"),
    (SMALL.replace('category = "overreach"', 'category = "vibes"'), "unknown category"),
    (SMALL.replace('expected = true\ncategory = "fair"', 'expected = false\ncategory = "fair"'), "expected="),
    (SMALL.replace('kind = "unbacked"', 'kind = "wrong"'), "planted kind"),
])
def test_bad_red_team_files_fail_loudly(tmp_path, broken, error):
    p = tmp_path / "rt.toml"
    p.write_text(broken)
    with pytest.raises(RedTeamError, match=error):
        load_redteam(p)


def test_redteam_cli_writes_results_and_lists_misses(tmp_path, monkeypatch, capsys):
    from cs2_analyst.agent import cli
    from cs2_analyst.agent import llm as llm_mod
    from cs2_analyst.evals import cli as eval_cli

    p = tmp_path / "rt.toml"
    p.write_text(SMALL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(llm_mod, "AnthropicLLM", lambda: ScriptedJudge())
    monkeypatch.setattr(eval_cli, "RUNS", tmp_path / "runs")
    cli.main(["eval", "redteam", "--file", str(p), "-y", "--tag", "rt"])
    out = capsys.readouterr().out
    assert "Balanced accuracy" in out and "What the judge got wrong" in out
    run = next((tmp_path / "runs").glob("*-rt"))
    assert len((run / "redteam.jsonl").read_text().splitlines()) == 2
    assert json.loads((run / "redteam_metrics.json").read_text())["conclusions"]["balanced_accuracy_pct"] == 50.0


class FlippingJudge(ScriptedJudge):
    """Approves everything on even calls, rejects everything on odd ones: a maximally noisy judge."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def create(self, *, model, system, tools, messages, max_tokens):
        payload = json.loads(messages[0]["content"])
        yes = self.calls % 2 == 0
        self.calls += 1
        verdicts = [{"claim_id": i["claim_id"], "supported": yes, "reason": "x"} for i in payload["interpretations"]]
        return _resp({"verdicts": verdicts, "unbacked_summary_statements": [], "contradicted_summary_statements": []})


def test_stability_measures_how_much_verdicts_move_between_repeats(small):
    from cs2_analyst.evals.redteam import stability

    steady = [anyio.run(lambda: run_redteam(small, ScriptedJudge(), "m", "v4")) for _ in range(2)]
    s = stability(steady)
    assert s["unstable_conclusions"] == 0 and s["balanced_accuracy_spread_pct"] == 0.0

    judge = FlippingJudge()  # one case per run, so run 1 approves everything and run 2 rejects everything
    runs = [anyio.run(lambda: run_redteam(small[:1], judge, "m", "v4", concurrency=1)) for _ in range(2)]
    s = stability(runs)
    assert s["unstable_conclusions"] == 2, "both conclusions in case a flipped between the two runs"
    assert s["wrong_counts"] == {"a: His opening wins were valuable.": 1, "a: He dominated the opening duels.": 1}


def test_redteam_cli_repeats(tmp_path, monkeypatch, capsys):
    from cs2_analyst.agent import cli
    from cs2_analyst.agent import llm as llm_mod
    from cs2_analyst.evals import cli as eval_cli

    p = tmp_path / "rt.toml"
    p.write_text(SMALL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(llm_mod, "AnthropicLLM", lambda: ScriptedJudge())
    monkeypatch.setattr(eval_cli, "RUNS", tmp_path / "runs")
    cli.main(["eval", "redteam", "--file", str(p), "-y", "--tag", "rt3", "--repeats", "3"])
    out = capsys.readouterr().out
    assert "per repeat (3 runs)" in out and "times wrong out of 3" in out
    run = next((tmp_path / "runs").glob("*-rt3"))
    assert len((run / "redteam.jsonl").read_text().splitlines()) == 6
