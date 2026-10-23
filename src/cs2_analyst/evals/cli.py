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

REDTEAM = Path("evals/redteam.toml")

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
    r.add_argument("--critic-revisions", type=int, default=1,
                   help="Times the judge's findings go back to the agent (0 = report only, the pre-v3 behaviour).")
    r.add_argument("--concurrency", type=int, default=3)
    r.add_argument("--tag", help="Name for this run, e.g. 'baseline' or 'prompt-v2'.")
    r.add_argument("--resume", help="Run name to resume: only missing or failed cases run.")
    r.add_argument("--yes", "-y", action="store_true", help="Don't ask for confirmation before spending.")

    rep = esub.add_parser("report", help="Show the metrics of a run.")
    rep.add_argument("run", nargs="?", default="latest")

    c = esub.add_parser("compare", help="Compare two runs side by side.")
    c.add_argument("a")
    c.add_argument("b", nargs="?", default="latest")
    c.add_argument("--all", action="store_true", help="Compare every case, not just the ones both runs have.")

    rj = esub.add_parser("rejudge", help="Re-run only the judge on a past run (same items, new judge).")
    rj.add_argument("run", help="The run whose answers to re-judge.")
    rj.add_argument("--judge-prompt", choices=sorted(JUDGE_PROMPTS), default=DEFAULT_JUDGE_PROMPT)
    rj.add_argument("--judge-model", default="claude-haiku-5-5")
    rj.add_argument("--tag", help="Name for the new run (default: rejudge-<prompt>).")
    rj.add_argument("--yes", "-y", action="store_true")

    rt = esub.add_parser("redteam", help="Score the judge on conclusions with known right answers (cheap: no agent).")
    rt.add_argument("--file", type=Path, default=None, help="Red-team cases (default: evals/redteam.toml).")
    rt.add_argument("--judge-prompt", choices=sorted(JUDGE_PROMPTS), default=DEFAULT_JUDGE_PROMPT)
    rt.add_argument("--judge-model", default="claude-haiku-5-5")
    rt.add_argument("--tag", help="Name for this run (default: redteam-<prompt>).")
    rt.add_argument("--repeats", type=int, default=1,
                    help="Run the whole set N times to measure how much verdicts move between identical runs.")
    rt.add_argument("--yes", "-y", action="store_true")

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
                    critic_revisions=args.critic_revisions,
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


def cmd_redteam(args: argparse.Namespace, console: Console) -> None:
    import json

    from rich.table import Table

    from .redteam import RedTeamError, load_redteam, misses, run_redteam, score, stability

    try:
        cases = load_redteam(args.file or REDTEAM)
    except RedTeamError as e:
        sys.exit(str(e))
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to your .env file (see .env.example).")
    n_interp = sum(len(c.interpretations) for c in cases)
    per_call = 0.0007 if "haiku" in args.judge_model else 0.007  # per case, measured on real runs
    dst = RUNS / (time.strftime("%Y%m%d-%H%M%S") + "-" + (args.tag or f"redteam-{args.judge_prompt}"))
    console.print(f"{len(cases)} red-team cases ({n_interp} conclusions), judge {args.judge_model} prompt "
                  f"{args.judge_prompt}, one judge call per case, {args.repeats}x, "
                  f"estimated cost ~${len(cases) * per_call * args.repeats:.2f}.")
    if not args.yes and Prompt.ask("Start?", choices=["y", "n"], default="n") != "y":
        sys.exit("Cancelled.")

    from ..agent.llm import AnthropicLLM

    llm = AnthropicLLM()
    runs = [asyncio.run(run_redteam(cases, llm, args.judge_model, args.judge_prompt))
            for _ in range(max(1, args.repeats))]
    results = [r for rep in runs for r in rep]  # pooled: rates below are averages over the repeats
    m = score(results)
    stab = stability(runs) if len(runs) > 1 else None
    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "redteam.jsonl").open("w", encoding="utf-8") as f:
        for k, rep in enumerate(runs):
            for r in rep:
                f.write(json.dumps({**r, "repeat": k}, default=str) + "\n")
    (dst / "redteam_metrics.json").write_text(json.dumps({**m, "stability": stab}, indent=2), encoding="utf-8")

    c, sm = m["conclusions"], m["summaries"]
    t = Table(title=f"Red team · {dst.name}", title_justify="left")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    rows = [
        (f"Bad conclusions rejected (of {c['bad']}) %", c["bad_rejected_pct"]),
        (f"Fair conclusions accepted (of {c['fair']}) %", c["fair_accepted_pct"]),
        ("Balanced accuracy % (50 = always yes or always no)", c["balanced_accuracy_pct"]),
        *[(f"  {tier} cases", v) for tier, v in c["balanced_accuracy_by_tier_pct"].items()],
        *[(f"  rejected: {cat}", f"{v['rejected']}/{v['total']}") for cat, v in c["bad_rejected_by_category"].items()],
        (f"Planted summary mistakes caught (of {sm['planted']}) %", sm["planted_caught_pct"]),
        ("  contradictions caught %", sm["contradictions_caught_pct"]),
        ("  contradictions filed as contradictions %", sm["contradictions_filed_correctly_pct"]),
        ("  unbacked statements caught %", sm["unbacked_caught_pct"]),
        (f"Clean summaries falsely flagged (of {sm['clean']}) %", sm["clean_summaries_flagged_pct"]),
        ("Cost ($)", m["total_cost_usd"]),
    ]
    if stab:
        rows += [
            (f"Balanced accuracy per repeat ({stab['repeats']} runs)",
             " / ".join(str(b) for b in stab["balanced_accuracy_each_pct"])),
            ("  spread between repeats (points)", stab["balanced_accuracy_spread_pct"]),
            (f"  conclusions whose verdict changed (of {stab['conclusions']})", stab["unstable_conclusions"]),
        ]
    for label, v in rows:
        t.add_row(label, "–" if v is None else str(v))
    console.print(t)
    if stab:
        if stab["wrong_counts"]:
            console.print(f"[bold]Conclusions the judge got wrong (times wrong out of {stab['repeats']}):[/]")
            for item, n in sorted(stab["wrong_counts"].items(), key=lambda kv: -kv[1]):
                console.print(f"  [yellow]{n}/{stab['repeats']}[/] {escape(item)}")
        sm_wrong = [line for line in misses(results) if "planted" in line or "false alarm" in line]
        for line in sorted(set(sm_wrong)):
            console.print(f"  [yellow]•[/] {escape(line)}")
    else:
        wrong = misses(results)
        if wrong:
            console.print("[bold]What the judge got wrong:[/]")
            for line in wrong:
                console.print(f"  [yellow]•[/] {escape(line)}")
    console.print(f"[dim]results: {dst / 'redteam.jsonl'}[/]")


def run(args: argparse.Namespace) -> None:
    console = Console()
    if args.eval_command == "run":
        cmd_run(args, console)
    elif args.eval_command == "report":
        _report(_resolve_run(args.run), console)
    elif args.eval_command == "compare":
        a, b = _resolve_run(args.a), _resolve_run(args.b)
        labels = L.load_labels(LABELS)
        ra, rb = load_results(a), load_results(b)
        if not args.all:
            # Only the cases both runs answered, so a bigger question set doesn't skew the numbers.
            shared = {r["case_id"] for r in ra} & {r["case_id"] for r in rb}
            extra_a, extra_b = len(ra) - len(shared), len(rb) - len(shared)
            ra, rb = [r for r in ra if r["case_id"] in shared], [r for r in rb if r["case_id"] in shared]
            if extra_a or extra_b:
                console.print(f"[dim]Comparing the {len(shared)} cases both runs have "
                              f"({extra_a} only in {a.name}, {extra_b} only in {b.name}; --all to include them).[/]")
        render_compare(a.name, compute(ra, labels), b.name, compute(rb, labels), console)
    elif args.eval_command == "rejudge":
        cmd_rejudge(args, console)
    elif args.eval_command == "label":
        cmd_label(args, console)
    elif args.eval_command == "redteam":
        cmd_redteam(args, console)
