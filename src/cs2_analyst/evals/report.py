"""Print a run's metrics, or compare two runs side by side; also writes report.md for the README."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

ROWS: list[tuple[str, str, str, bool]] = [
    # (label, section, key, higher_is_better)
    ("Facts verified (%)", "facts", "verified_pct", True),
    ("Facts failed (%)", "facts", "failed_pct", False),
    ("Facts unchecked (%)", "facts", "unchecked_pct", False),
    ("Facts per answer", "facts", "per_answer", True),
    ("First drafts with a wrong fact (%)", "first_draft", "answers_with_wrong_fact_pct", False),
    ("Final answers with a wrong fact (%)", "first_draft", "final_answers_with_wrong_fact_pct", False),
    ("Answers needing a revision (%)", "first_draft", "revision_rate_pct", False),
    ("Interpretations supported (%)", "interpretations", "supported_pct", True),
    ("Rejected by judge (%)", "interpretations", "rejected_by_judge_pct", False),
    ("Rejected by verifier (%)", "interpretations", "rejected_by_verifier_pct", False),
    ("Answers with unbacked summary text (%)", "summary_issues", "answers_flagged_pct", False),
    ("Judge agrees with expert (%)", "judge_agreement", "agreement_pct", True),
    ("  a judge that always gives the majority answer (%)", "judge_agreement", "trivial_baseline_pct", True),
    ("  labelled items", "judge_agreement", "labelled", True),
    ("  judge too lenient", "judge_agreement", "too_lenient", False),
    ("  judge too strict", "judge_agreement", "too_strict", False),
    ("Avg cost per question ($)", "ops", "avg_cost_usd", False),
    ("Total cost ($)", "ops", "total_cost_usd", False),
    ("Median time (s)", "ops", "p50_seconds", False),
    ("p95 time (s)", "ops", "p95_seconds", False),
    ("Avg tool calls", "ops", "avg_tool_calls", False),
    ("Prompt cache hit (%)", "ops", "cache_hit_pct", True),
]


def _fmt(v: Any) -> str:
    return "–" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def render(name: str, m: dict[str, Any], console: Console) -> None:
    t = Table(title=f"Eval {name} · {m['cases']} cases · {m['errors']} errors", title_justify="left")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for label, sec, key, _ in ROWS:
        t.add_row(label, _fmt(m[sec].get(key)))
    console.print(t)
    q = Table(title="Per question", title_justify="left")
    for col in ("question", "cases", "facts verified %", "interp. supported %", "summary flagged", "avg $"):
        q.add_column(col)
    for qid, v in m["per_question"].items():
        q.add_row(qid, str(v["cases"]), _fmt(v["facts_verified_pct"]), _fmt(v["interpretations_supported_pct"]),
                  str(v["summary_flagged"]), _fmt(v["avg_cost_usd"]))
    console.print(q)
    if m["stop_reasons"]:
        console.print("[dim]stop reasons: " + ", ".join(f"{k} {v}" for k, v in m["stop_reasons"].items()) + "[/]")


def render_compare(a_name: str, a: dict[str, Any], b_name: str, b: dict[str, Any], console: Console) -> None:
    t = Table(title=f"{a_name}  vs  {b_name}", title_justify="left")
    t.add_column("Metric")
    t.add_column(a_name, justify="right")
    t.add_column(b_name, justify="right")
    t.add_column("Change", justify="right")
    for label, sec, key, higher_better in ROWS:
        va, vb = a[sec].get(key), b[sec].get(key)
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != vb:
            d = vb - va
            good = (d > 0) == higher_better
            delta = f"[{'green' if good else 'red'}]{d:+.4g}[/]"
        t.add_row(label, _fmt(va), _fmt(vb), delta)
    console.print(t)


def markdown(name: str, m: dict[str, Any]) -> str:
    lines = [f"### Eval {name}", "", f"{m['cases']} cases, {m['errors']} errors.", "",
             "| Metric | Value |", "| --- | --- |"]
    lines += [f"| {label.strip()} | {_fmt(m[sec].get(key))} |" for label, sec, key, _ in ROWS]
    lines += ["", "| Question | Cases | Facts verified % | Interpretations supported % | Avg $ |",
              "| --- | --- | --- | --- | --- |"]
    lines += [f"| {q} | {v['cases']} | {_fmt(v['facts_verified_pct'])} | {_fmt(v['interpretations_supported_pct'])} "
              f"| {_fmt(v['avg_cost_usd'])} |" for q, v in m["per_question"].items()]
    return "\n".join(lines) + "\n"
