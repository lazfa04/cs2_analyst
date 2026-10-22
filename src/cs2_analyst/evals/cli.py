"""`cs2-agent eval ...` commands: run, report, compare, label."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt

from ..agent.prompts import DEFAULT_JUDGE_PROMPT, JUDGE_PROMPTS
from ..config import get_settings
from ..server import build_server
from . import labels as L
from .cases import CasesError, load_cases
from .metrics import compute, load_results
from .report import markdown, render, render_compare
from .runner import RunConfig, run_eval

CASES = Path("evals/cases.toml")
RUNS = Path("evals/runs")
LABELS = Path("evals/labels.jsonl")
DEFAULT_COST_PER_CASE = 0.05  # rough guess until a previous run gives a real average


def add_parser(sub: argparse._SubParsersAction) -> None:
    e = sub.add_parser("eval", help="Run and score the eval set.")
    esub = e.add_subparsers(dest="eval_command", required=True)

    r = esub.add_parser("run", help="Run the agent on every case in the eval set.")
    r.add_argument("--cases", type=Path, default=CASES)
    r.add_argument("--demo", action="append", help="Only this demo (repeatable).")
    r.add_argument("--question", action="append", help="Only this question id (repeatable).")
    r.add_argument("--model", default="claude-sonnet-5-5")
    r.add_argument("--judge-model", default="claude-haiku-5-5")
    r.add_argument("--no-judge", action="store_true")
    r.add_argument("--judge-prompt", choices=sorted(JUDGE_PROMPTS), default=DEFAULT_JUDGE_PROMPT)
    r.add_argument("--max-steps", type=int, default=12)
    r.add_argument("--max-revisions", type=int, default=1)
    r.add_argument("--concurrency", type=int, default=3)
    r.add_argument("--tag", help="Name for this run, e.g. 'baseline' or 'prompt-v2'.")
    r.add_argument("--resume", help="Run name to resume: only missing or failed cases run.")
    r.add_argument("--yes", "-y", action="store_true", help="Don't ask for confirmation before spending.")

    rep = esub.add_parser("report", help="Show the metrics of a run.")
    rep.add_argument("run", nargs="?", default="latest")

    c = esub.add_parser("compare", help="Compare two runs side by side.")
    c.add_argument("a")
    c.add_argument("b", nargs="?", default="latest")

    rj = esub.add_parser("rejudge", help="Re-run only the judge on a past run (same items, new judge).")
    rj.add_argument("run", help="The run whose answers to re-judge.")
    rj.add_argument("--judge-prompt", choices=sorted(JUDGE_PROMPTS), default=DEFAULT_JUDGE_PROMPT)
    rj.add_argument("--judge-model", default="claude-haiku-5-5")
    rj.add_argument("--tag", help="Name for the new run (default: rejudge-<prompt>).")
    rj.add_argument("--yes", "-y", action="store_true")

    lab = esub.add_parser("label", help="Label interpretations so judge agreement can be measured.")
    lab.add_argument("run", nargs="?", default="latest")
    lab.add_argument("--relabel", action="store_true", help="Also show items you already labelled.")


def _resolve_run(name: str) -> Path:
    runs = sorted((p for p in RUNS.glob("*") if (p / "results.jsonl").exists()), key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f"No eval runs in {RUNS}/ yet. Start one with: cs2-agent eval run")
    if name == "latest":
        return runs[-1]
    match = [p for p in runs if p.name == name or p.name.endswith(name)]
    if not match:
        sys.exit(f"No run '{name}'. Runs: {[p.name for p in runs]}")
    return match[-1]


def _avg_cost_from_history() -> float:
    costs = [rec["usage"]["cost_usd"] for d in RUNS.glob("*") for rec in load_results(d)
             if not rec.get("error") and rec.get("usage", {}).get("cost_usd") is not None]
    return sum(costs) / len(costs) if costs else DEFAULT_COST_PER_CASE


def cmd_run(args: argparse.Namespace, console: Console) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to your .env file (see .env.example).")
    try:
        cases = load_cases(args.cases, args.demo, args.question)
    except CasesError as e:
        sys.exit(str(e))

    if args.resume:
        run_dir = _resolve_run(args.resume)
    else:
        name = time.strftime("%Y%m%d-%H%M%S") + "-" + (args.tag or args.model.removeprefix("claude-"))
        run_dir = RUNS / name
    est = len(cases) * _avg_cost_from_history()
    console.print(f"{len(cases)} cases across {len({c.demo.name for c in cases})} demos, "
                  f"model {args.model}, estimated cost ~${est:.2f}. Results: {run_dir}")
    if not args.yes and Prompt.ask("Start?", choices=["y", "n"], default="n") != "y":
        sys.exit("Cancelled.")

    from ..agent.llm import AnthropicLLM

    cfg = RunConfig(model=args.model, judge_model=None if args.no_judge else args.judge_model,
                    max_steps=args.max_steps, max_revisions=args.max_revisions,
                    concurrency=args.concurrency, tag=args.tag, judge_prompt=args.judge_prompt)
    settings = get_settings()
    done = [0]

    def progress(case_id: str, rec: dict) -> None:
        done[0] += 1
        status = f"[red]error: {escape(str(rec['error'])[:80])}[/]" if rec.get("error") else rec["stop_reason"]
        cost = rec.get("usage", {}).get("cost_usd")
        console.print(f"[{done[0]}/{len(cases)}] {case_id}: {status}" + (f" · ${cost:.3f}" if cost else ""))

    asyncio.run(run_eval(cases, cfg, run_dir, AnthropicLLM(), lambda: build_server(settings, log_level="WARNING"), progress))
    _report(run_dir, console)


def _report(run_dir: Path, console: Console) -> None:
    m = compute(load_results(run_dir), L.load_labels(LABELS))
    render(run_dir.name, m, console)
    (run_dir / "report.md").write_text(markdown(run_dir.name, m), encoding="utf-8")
    console.print(f"[dim]report: {run_dir / 'report.md'}[/]")
    if m["judge_agreement"]["unlabelled"]:
        console.print(f"[dim]{m['judge_agreement']['unlabelled']} judged interpretations aren't labelled yet: "
                      f"cs2-agent eval label {run_dir.name}[/]")


def cmd_label(args: argparse.Namespace, console: Console) -> None:
    run_dir = _resolve_run(args.run)
    existing = L.load_labels(LABELS)
    seen, items = set(), []
    for it in L.judged_items(load_results(run_dir)):
        if it["key"] in seen or (it["key"] in existing and not args.relabel):
            continue
        seen.add(it["key"])
        items.append(it)
    if not items:
        console.print("Nothing new to label in this run.")
        return
    console.print(f"{len(items)} interpretations to label. For each: do the facts support the conclusion?\n"
                  "Judge it as a CS2 player would. The judge's own verdict is hidden so it can't sway you.\n")
    for i, it in enumerate(items, 1):
        facts = "\n".join(f"  • {escape(f['text'])}" for f in it["facts"]) or "  (no facts)"
        console.print(Panel(f"[dim]{escape(it['question'])}[/]\n\n[bold]{escape(it['interpretation'])}[/]\n\n"
                            f"Rests on:\n{facts}", title=f"{i}/{len(items)} · {it['case_id']}", title_align="left"))
        choice = Prompt.ask("Supported?", choices=["y", "n", "s", "q"], default="s")
        if choice == "q":
            break
        if choice == "s":
            continue
        note = Prompt.ask("Note (optional, Enter to skip)", default="")
        L.save_label(LABELS, it, agree=(choice == "y"), note=note)
    _report(run_dir, console)


def cmd_rejudge(args: argparse.Namespace, console: Console) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to your .env file (see .env.example).")
    src = _resolve_run(args.run)
    dst = RUNS / (time.strftime("%Y%m%d-%H%M%S") + "-" + (args.tag or f"rejudge-{args.judge_prompt}"))
    n = sum(1 for r in load_results(src) if not r.get("error"))
    console.print(f"Re-judging {n} answers from {src.name} with judge prompt {args.judge_prompt} "
                  f"({args.judge_model}). One small judge call per answer. Results: {dst}")
    if not args.yes and Prompt.ask("Start?", choices=["y", "n"], default="n") != "y":
        sys.exit("Cancelled.")

    from ..agent.llm import AnthropicLLM
    from .rejudge import rejudge_run

    asyncio.run(rejudge_run(src, dst, AnthropicLLM(), args.judge_model, args.judge_prompt))
    labels = L.load_labels(LABELS)
    render_compare(src.name, compute(load_results(src), labels), dst.name, compute(load_results(dst), labels), console)
    (dst / "report.md").write_text(markdown(dst.name, compute(load_results(dst), labels)), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    console = Console()
    if args.eval_command == "run":
        cmd_run(args, console)
    elif args.eval_command == "report":
        _report(_resolve_run(args.run), console)
    elif args.eval_command == "compare":
        a, b = _resolve_run(args.a), _resolve_run(args.b)
        labels = L.load_labels(LABELS)
        render_compare(a.name, compute(load_results(a), labels), b.name, compute(load_results(b), labels), console)
    elif args.eval_command == "rejudge":
        cmd_rejudge(args, console)
    elif args.eval_command == "label":
        cmd_label(args, console)
