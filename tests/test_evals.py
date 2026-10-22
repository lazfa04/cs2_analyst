"""Phase 3 evals: case loading, the runner, metrics, labels and reports. No API key needed."""

import json
from pathlib import Path

import anyio
import pytest
from rich.console import Console

from cs2_analyst.agent.llm import Block, LLMResponse
from cs2_analyst.agent.prompts import JUDGE_SYSTEM
from cs2_analyst.config import Settings
from cs2_analyst.evals import labels as L
from cs2_analyst.evals.cases import CasesError, load_cases
from cs2_analyst.evals.metrics import compute, load_results
from cs2_analyst.evals.report import markdown, render, render_compare
from cs2_analyst.evals.runner import RunConfig, run_eval
from cs2_analyst.server import build_server

from .conftest import DEMO_ID

USAGE = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 500, "cache_creation_input_tokens": 0}

CASES_TOML = f"""
[[demo]]
name = "fake"
file = "{DEMO_ID}"
player = "alpha"

[[demo]]
name = "missing"
file = "does-not-exist.dem"
player = "alpha"

[[demo]]
name = "noplayer"
file = "{DEMO_ID}"

[[question]]
id = "overview"
text = "How did {{player}} play?"

[[question]]
id = "trades"
text = "Which of {{player}}'s deaths weren't traded?"

[[question]]
id = "best"
text = "Who was the best player?"
"""


def _resp(*blocks):
    stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    return LLMResponse([Block(**b) for b in blocks], list(blocks), dict(USAGE), stop)


def _report_call(expected_rounds: str):
    claims = [
        {"id": "c1", "kind": "fact", "text": f"The match had {expected_rounds} rounds", "supports": [],
         "checks": [{"call_id": "t1", "path": "rounds_played", "expected": expected_rounds}]},
        {"id": "c2", "kind": "interpretation", "text": "It was a short match", "checks": [], "supports": ["c1"]},
    ]
    return {"type": "tool_use", "id": "tu_report", "name": "submit_report",
            "input": {"summary": "A short match.", "claims": claims}}


class ScriptedAgent:
    """Behaves like the agent on any case. On 'traded' questions its first draft gets the
    round count wrong (5 instead of 4) and it fixes it after the verifier's feedback."""

    def __init__(self, fail_on: str | None = None):
        self.calls = 0
        self.fail_on = fail_on

    async def create(self, *, model, system, tools, messages, max_tokens):
        self.calls += 1
        if system == JUDGE_SYSTEM:
            return _resp({"type": "tool_use", "id": "tu_v", "name": "submit_verdicts",
                          "input": {"verdicts": [{"claim_id": "c2", "supported": True, "reason": "ok"}],
                                    "unbacked_summary_statements": []}})
        first = messages[0]["content"]
        if self.fail_on and self.fail_on in first:
            raise RuntimeError("simulated API outage")
        if len(messages) == 1:
            return _resp({"type": "tool_use", "id": "tu_ov", "name": "get_match_overview",
                          "input": {"demo_id": DEMO_ID}})
        last = messages[-1]["content"]
        feedback = isinstance(last, list) and any("failed verification" in str(c.get("content", "")) for c in last)
        wrong_first = "traded" in first and not feedback
        return _resp(_report_call("5" if wrong_first else "4"))


@pytest.fixture
def cases_file(tmp_path):
    p = tmp_path / "cases.toml"
    p.write_text(CASES_TOML)
    return p


@pytest.fixture
def execute(store, tmp_path):
    settings = Settings(cache_dir=store.cache_dir, demo_dir=None, faceit_api_key=None,
                        leetify_api_key=None, http_cache_ttl_s=0)

    def _run(cases, llm, run_dir=None, **cfg):
        run_dir = run_dir or tmp_path / "run1"
        anyio.run(lambda: run_eval(cases, RunConfig(concurrency=2, **cfg), run_dir, llm,
                                   lambda: build_server(settings=settings, store=store)))
        return run_dir

    return _run


# ---------------------------------------------------------------- cases

def test_cases_expand_demos_by_questions(cases_file):
    cases = load_cases(cases_file)
    ids = [c.id for c in cases]
    assert "fake/overview" in ids and "missing/trades" in ids
    assert ids.count("noplayer/best") == 1 and "noplayer/overview" not in ids, "{player} questions need a player"
    assert next(c for c in cases if c.id == "fake/overview").question == "How did alpha play?"


def test_case_filters_and_errors(cases_file, tmp_path):
    assert [c.id for c in load_cases(cases_file, demos=["fake"], questions=["trades"])] == ["fake/trades"]
    with pytest.raises(CasesError, match="Unknown demo"):
        load_cases(cases_file, demos=["nope"])
    dup = tmp_path / "dup.toml"
    dup.write_text('[[demo]]\nname="a"\nfile="x"\n[[demo]]\nname="a"\nfile="y"\n[[question]]\nid="q"\ntext="t"\n')
    with pytest.raises(CasesError, match="Duplicate demo"):
        load_cases(dup)


# ---------------------------------------------------------------- runner + metrics

def test_run_records_every_case_and_isolates_failures(cases_file, execute):
    run_dir = execute(load_cases(cases_file), ScriptedAgent())
    results = {r["case_id"]: r for r in load_results(run_dir)}
    assert len(results) == 7
    assert results["missing/overview"]["error"].startswith("demo failed to load")
    ok = results["fake/overview"]
    assert ok["stop_reason"] == "submitted" and ok["first_draft"] == {"verified": 1, "not_judged": 1}
    assert {c["status"] for c in ok["claims"]} == {"verified", "supported"}
    assert len(list((run_dir / "traces").glob("*.jsonl"))) == 4, "one trace per case that ran (3 fake + 1 noplayer)"


def test_metrics_show_what_the_verifier_caught(cases_file, execute):
    m = compute(load_results(execute(load_cases(cases_file, demos=["fake"]), ScriptedAgent())))
    assert m["cases"] == 3 and m["errors"] == 0
    assert m["facts"]["verified_pct"] == 100.0
    # 'trades' got a wrong fact in its first draft, fixed after feedback: 1 of 3 first drafts wrong, 0 final
    assert m["first_draft"]["answers_with_wrong_fact_pct"] == pytest.approx(33.3)
    assert m["first_draft"]["final_answers_with_wrong_fact_pct"] == 0.0
    assert m["first_draft"]["revision_rate_pct"] == pytest.approx(33.3)
    assert m["interpretations"]["supported_pct"] == 100.0
    assert m["ops"]["cache_hit_pct"] == pytest.approx(33.3)
    assert set(m["per_question"]) == {"overview", "trades", "best"}


def test_resume_only_runs_missing_or_failed_cases(cases_file, execute):
    cases = load_cases(cases_file, demos=["fake"])
    flaky = ScriptedAgent(fail_on="Who was the best player")
    run_dir = execute(cases, flaky)
    assert sum(1 for r in load_results(run_dir) if r.get("error")) == 1

    retry = ScriptedAgent()
    execute(cases, retry, run_dir=run_dir)
    results = load_results(run_dir)
    assert not any(r.get("error") for r in results) and len(results) == 3
    assert retry.calls == 3, "only the failed case ran again (2 agent turns + 1 judge)"


# ---------------------------------------------------------------- labels + agreement

def test_label_keys_ignore_case_whitespace_and_fact_order():
    assert L.item_key("It was  short", ["b", "A"]) == L.item_key("it was short", ["a", "B"])
    assert L.item_key("x", ["a"]) != L.item_key("y", ["a"])


def test_judge_agreement_from_labels(cases_file, execute, tmp_path):
    results = load_results(execute(load_cases(cases_file, demos=["fake"]), ScriptedAgent()))
    items = L.judged_items(results)
    assert len(items) == 3 and all(it["judge_supported"] for it in items)
    assert items[0]["facts"][0]["values"] == [4], "labeller sees the verified values"

    labels_file = tmp_path / "labels.jsonl"
    L.save_label(labels_file, items[0], agree=True)
    L.save_label(labels_file, items[1], agree=False)  # the same item text: later label overrides
    agreement = compute(results, L.load_labels(labels_file))["judge_agreement"]
    # all three items share one key (same interpretation + facts), so the latest label (disagree) applies
    assert agreement["labelled"] == 3 and agreement["too_lenient"] == 3 and agreement["agreement_pct"] == 0.0
    assert agreement["trivial_baseline_pct"] == 100.0, "always-'no' would match all three labels"


# ---------------------------------------------------------------- reports

def test_report_compare_and_markdown_render(cases_file, execute, tmp_path):
    a = compute(load_results(execute(load_cases(cases_file, demos=["fake"]), ScriptedAgent())))
    b = compute(load_results(execute(load_cases(cases_file, demos=["fake"], questions=["overview"]),
                                     ScriptedAgent(), run_dir=tmp_path / "run2")))
    console = Console(record=True, width=200, color_system=None)
    render("run1", a, console)
    render_compare("run1", a, "run2", b, console)
    out = console.export_text()
    assert "Facts verified (%)" in out and "First drafts with a wrong fact" in out and "run2" in out
    md = markdown("run1", a)
    assert md.startswith("### Eval run1") and "| Facts verified (%) | 100 |" in md


def test_eval_report_cli(cases_file, execute, monkeypatch, tmp_path, capsys):
    from cs2_analyst.agent import cli
    from cs2_analyst.evals import cli as eval_cli

    runs = tmp_path / "runs"
    execute(load_cases(cases_file, demos=["fake"]), ScriptedAgent(), run_dir=runs / "20261022-000000-test")
    monkeypatch.setattr(eval_cli, "RUNS", runs)
    monkeypatch.setattr(eval_cli, "LABELS", tmp_path / "labels.jsonl")
    cli.main(["eval", "report"])
    assert (runs / "20261022-000000-test" / "report.md").exists()
    assert "3 judged interpretations aren't labelled yet" in capsys.readouterr().out


def test_label_cli_hides_judge_verdict_and_saves_answers(cases_file, execute, monkeypatch, tmp_path):
    from rich.prompt import Prompt

    from cs2_analyst.agent import cli
    from cs2_analyst.evals import cli as eval_cli

    runs = tmp_path / "runs"
    execute(load_cases(cases_file, demos=["fake"]), ScriptedAgent(), run_dir=runs / "20261022-000000-test")
    labels_file = tmp_path / "labels.jsonl"
    monkeypatch.setattr(eval_cli, "RUNS", runs)
    monkeypatch.setattr(eval_cli, "LABELS", labels_file)
    answers = iter(["n", "too vague"])  # one unique item (identical across cases): reject it, add a note
    monkeypatch.setattr(Prompt, "ask", lambda *a, **k: next(answers))

    console = Console(record=True, width=200, color_system=None)
    monkeypatch.setattr(eval_cli, "Console", lambda: console)
    cli.main(["eval", "label"])
    out = console.export_text()
    assert "It was a short match" in out and "The match had 4 rounds" in out
    assert "ok" not in out.split("Supported?")[0].split("Rests on:")[-1], "judge's reason must stay hidden"
    saved = L.load_labels(labels_file)
    assert len(saved) == 1 and next(iter(saved.values()))["supported"] is False
    assert "Judge agrees with expert (%)" in out and "too_lenient" not in out


# ---------------------------------------------------------------- rejudge

class RejectingJudge:
    """A judge that rejects every interpretation and records which prompt it was given."""

    def __init__(self):
        self.systems = []

    async def create(self, *, model, system, tools, messages, max_tokens):
        self.systems.append(system)
        payload = json.loads(messages[0]["content"])
        verdicts = [{"claim_id": i["claim_id"], "supported": False, "reason": "no"} for i in payload["interpretations"]]
        return _resp({"type": "tool_use", "id": "tu_v", "name": "submit_verdicts",
                      "input": {"verdicts": verdicts, "unbacked_summary_statements": ["something"]}})


def test_rejudge_redoes_only_the_judges_calls(cases_file, execute, tmp_path):
    from cs2_analyst.agent.prompts import JUDGE_PROMPTS
    from cs2_analyst.evals.rejudge import rejudge_run

    src = execute(load_cases(cases_file, demos=["fake"]), ScriptedAgent())
    judge_llm = RejectingJudge()
    dst = anyio.run(lambda: rejudge_run(src, tmp_path / "rj", judge_llm, "claude-haiku-5-5", "v1"))

    before, after = load_results(src), {r["case_id"]: r for r in load_results(dst)}
    assert judge_llm.systems and all(s == JUDGE_PROMPTS["v1"] for s in judge_llm.systems)
    for rec in before:
        new = after[rec["case_id"]]
        facts_b = [(c["id"], c["status"]) for c in rec["claims"] if c["kind"] == "fact"]
        facts_a = [(c["id"], c["status"]) for c in new["claims"] if c["kind"] == "fact"]
        assert facts_a == facts_b, "facts are never re-judged"
        assert [c["status"] for c in new["claims"] if c["kind"] == "interpretation"] == ["unsupported"]
        assert new["summary_issues"] == ["something"] and new["summary"] == rec["summary"]
    # same interpretations -> same label keys, so agreement is comparable across the two runs
    assert [i["key"] for i in L.judged_items(before)] == [i["key"] for i in L.judged_items(list(after.values()))]
    assert json.loads((dst / "config.json").read_text())["rejudged_from"] == src.name


def test_rejudge_keeps_verifier_rejections():
    from cs2_analyst.evals.rejudge import rebuild

    rec = {"summary": "s", "claims": [
        {"id": "c1", "kind": "fact", "text": "f", "status": "failed", "judged": False, "supports": [],
         "checks": [{"call_id": "t1", "path": "x", "expected": "1", "actual": 2, "ok": False}]},
        {"id": "c2", "kind": "interpretation", "text": "i", "status": "unsupported", "judged": False,
         "note": "supporting facts not verified: ['c1']", "supports": ["c1"], "checks": []},
        {"id": "c3", "kind": "interpretation", "text": "j", "status": "supported", "judged": True,
         "note": "fine", "supports": [], "checks": []},
    ]}
    _, v = rebuild(rec)
    st = {r.claim.id: r.status for r in v.results}
    assert st == {"c1": "failed", "c2": "unsupported", "c3": "not_judged"}, "only the judge's own calls reset"
    assert v.results[0].checks[0].actual == 2


def test_root_cause_unwraps_nested_exception_groups():
    from cs2_analyst.evals.runner import root_cause

    err = ConnectionError("network down")
    wrapped = ExceptionGroup("outer", [ExceptionGroup("inner", [err])])
    assert root_cause(wrapped) is err
