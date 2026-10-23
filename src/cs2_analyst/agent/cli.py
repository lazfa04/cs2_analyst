"""`cs2-agent ask <demo> "<question>"`: run the agent on one demo from the terminal.

The demo argument is a path, a file name inside CS2_DEMO_DIR, or a demo_id
that's already cached. The CLI ingests the demo itself (a fixed setup step,
no model needed), runs the agent, prints a verified report and writes the
trace to ./traces.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ..config import get_settings
from ..server import build_server
from .llm import AnthropicLLM
from .prompts import DEFAULT_JUDGE_PROMPT, JUDGE_PROMPTS
from .loop import AgentConfig, AgentResult, run_agent
from .toolbox import Toolbox

ICONS = {
    "verified": "[green]✓ verified[/]",
    "supported": "[green]✓ supported[/]",
    "failed": "[red]✗ failed[/]",
    "unsupported": "[yellow]⚠ unsupported[/]",
    "unchecked": "[yellow]? unchecked[/]",
    "not_judged": "[dim]– not judged[/]",
}


class CliError(Exception):
    """A user-facing problem: print the message, no traceback."""


async def _resolve_demo(toolbox: Toolbox, demo: str) -> dict | str:
    """The demo's info, or an error message. Returned, not raised: an exception raised inside
    the MCP client's task group comes out wrapped in a long ExceptionGroup traceback."""
    if re.fullmatch(r"[0-9a-f]{12}", demo):
        listing, _ = await toolbox.call("list_demos", {})
        for d in listing.get("demos", []):
            if d["demo_id"] == demo:
                return d
    result, is_error = await toolbox.call("ingest_demo", {"path": demo})
    if is_error:
        return str(result).removeprefix("Error executing tool ingest_demo: ")
    return result


async def ask(args: argparse.Namespace) -> AgentResult:
    from anthropic import APIError

    settings = get_settings()
    error = None
    async with Toolbox.connect(build_server(settings, log_level="WARNING")) as toolbox:
        demo = await _resolve_demo(toolbox, args.demo)
        if isinstance(demo, str):
            error = f"Could not load demo: {demo}"
        else:
            cfg = AgentConfig(
                model=args.model, judge_model=None if args.no_judge else args.judge_model,
                max_steps=args.max_steps, max_revisions=args.max_revisions, player=args.player,
                judge_prompt=args.judge_prompt, critic_revisions=args.critic_revisions,
            )
            try:
                return await run_agent(args.question, demo, AnthropicLLM(), toolbox, cfg)
            except APIError as e:  # bad key, no credit, unknown model, outage...
                status = getattr(e, "status_code", None)
                error = f"Anthropic API error{f' {status}' if status else ''}: {e.message}"
    raise CliError(error)  # raised after the MCP session has closed cleanly


def render(result: AgentResult, console: Console) -> None:
    console.print(Panel(escape(result.question), title=escape(f"{result.demo.get('map')} · {result.demo['demo_id']}"), expand=False))
    if result.report is None:
        console.print(f"[red]No report produced[/] (stopped: {result.stop_reason}).")
    else:
        title = "Answer" if result.stop_reason == "submitted" else "Answer [yellow](unverified)[/]"
        console.print(Panel(escape(result.report.summary), title=title))

    shown = [r for r in result.verification.results if r.status not in ("failed", "unsupported")]
    dropped = [r for r in result.verification.results if r.status in ("failed", "unsupported")]
    if shown:
        t = Table(show_header=True, header_style="bold", expand=True)
        t.add_column("", width=14)
        t.add_column("Claim")
        t.add_column("Evidence", style="dim", max_width=40)
        for r in shown:
            ev = ", ".join(f"{c.call_id}:{c.path}" for c in r.checks) or ("supports " + ", ".join(r.claim.supports))
            t.add_row(ICONS[r.status], escape(r.claim.text), escape(ev))
        console.print(t)
    if result.verification.summary_contradictions:
        console.print("[bold]In the answer but CONTRADICTS a verified claim:[/]")
        for issue in result.verification.summary_contradictions:
            console.print(f"  [red]✗[/] {escape(issue)}")
    if result.verification.summary_issues:
        console.print("[bold]In the answer but not backed by a verified claim:[/]")
        for issue in result.verification.summary_issues:
            console.print(f"  [yellow]⚠[/] {escape(issue)}")
    if dropped:
        console.print("[bold]Did not hold up:[/]")
        for r in dropped:
            why = r.note or "; ".join(f"{c.path}: expected {c.expected!r}, got {c.actual!r}" for c in r.checks if not c.ok)
            console.print(f"  {ICONS[r.status]} {escape(r.claim.text)}  [dim]({escape(why)})[/]")

    s = result.trace.summary()
    cost = f"${s['cost_usd']:.4f}" if s["cost_usd"] is not None else "unknown"
    console.print(
        f"[dim]{s['llm_calls']} model calls · {s['tool_calls']} tool calls · "
        f"{s['input_tokens'] + s['cache_read_tokens'] + s['cache_write_tokens']:,} in / {s['output_tokens']:,} out tokens "
        f"({s['cache_read_tokens']:,} cached) · {cost} · {s['elapsed_s']}s · revisions {result.revisions}"
        f" · critic revisions {result.critic_revisions}[/]"
    )


def to_json(result: AgentResult) -> dict:
    return {
        "question": result.question,
        "demo": result.demo,
        "stop_reason": result.stop_reason,
        "revisions": result.revisions,
        "critic_revisions": result.critic_revisions,
        "first_review": result.first_review,
        "summary": result.report.summary if result.report else None,
        "claims": [
            {"id": r.claim.id, "kind": r.claim.kind, "text": r.claim.text, "status": r.status, "note": r.note,
             "judged": r.judged, "checks": [vars(c) for c in r.checks], "supports": r.claim.supports}
            for r in result.verification.results
        ],
        "summary_issues": result.verification.summary_issues,
        "summary_contradictions": result.verification.summary_contradictions,
        "usage": result.trace.summary(),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cs2-agent", description="Ask questions about a CS2 demo, or run the evals.")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("ask", help="Ask one question about one demo.")
    a.add_argument("demo", help="Path to a .dem, a file name in CS2_DEMO_DIR, or a cached demo_id.")
    a.add_argument("question")
    a.add_argument("--player", help="Your in-game name, so 'I' and 'my' resolve to you.")
    a.add_argument("--model", default="claude-sonnet-5-5")
    a.add_argument("--judge-model", default="claude-haiku-5-5")
    a.add_argument("--no-judge", action="store_true", help="Skip the interpretation judge.")
    a.add_argument("--judge-prompt", choices=sorted(JUDGE_PROMPTS), default=DEFAULT_JUDGE_PROMPT)
    a.add_argument("--max-steps", type=int, default=12)
    a.add_argument("--max-revisions", type=int, default=1)
    a.add_argument("--critic-revisions", type=int, default=1,
                   help="How many times the judge's findings go back to the agent for a fix (0 = only report them).")
    a.add_argument("--json", action="store_true", help="Print the result as JSON.")
    a.add_argument("--trace-dir", default=os.getenv("CS2_TRACE_DIR", "traces"))
    from ..evals import cli as eval_cli

    eval_cli.add_parser(sub)
    args = p.parse_args(argv)
    if args.command == "eval":
        eval_cli.run(args)
        return

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to your .env file (see .env.example).")

    console = Console()
    try:
        with console.status("Analysing..."):
            result = asyncio.run(ask(args))
    except CliError as e:
        sys.exit(str(e))
    trace_path = result.trace.write(Path(args.trace_dir))
    if args.json:
        print(json.dumps(to_json(result), indent=2, default=str))
    else:
        render(result, console)
        console.print(f"[dim]trace: {escape(str(trace_path))}[/]")


if __name__ == "__main__":
    main()
