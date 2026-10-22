"""Run the eval set: every case through the real agent, one JSON line per result.

  * Demos are ingested once, up front and one at a time; cases then run concurrently
    (capped, to stay under API rate limits).
  * Each result is appended to results.jsonl as soon as its case finishes, so an
    interrupted run loses nothing and `--resume` only runs what's missing.
  * A failing case (API error, crash) is recorded as an error and the run continues.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..agent import AgentConfig, AgentResult, run_agent
from ..agent.llm import LLM
from ..agent.prompts import AGENT_PROMPT_VERSION, DEFAULT_JUDGE_PROMPT
from ..agent.toolbox import Toolbox
from ..agent.trace import Trace
from .cases import Case


@dataclass
class RunConfig:
    model: str = "claude-sonnet-5-5"
    judge_model: str | None = "claude-haiku-5-5"
    max_steps: int = 12
    max_revisions: int = 1
    concurrency: int = 3
    tag: str | None = None
    judge_prompt: str = DEFAULT_JUDGE_PROMPT


def result_record(case: Case, res: AgentResult) -> dict[str, Any]:
    v = res.verification
    return {
        "case_id": case.id,
        "demo": case.demo.name,
        "question_id": case.question_id,
        "question": case.question,
        "stop_reason": res.stop_reason,
        "revisions": res.revisions,
        "error": None,
        "summary": res.report.summary if res.report else None,
        "summary_issues": v.summary_issues,
        "first_draft": res.first_draft,
        "final_counts": v.counts(),
        "claims": [
            {
                "id": r.claim.id, "kind": r.claim.kind, "text": r.claim.text, "status": r.status,
                "judged": r.judged, "note": r.note, "supports": r.claim.supports,
                "checks": [{"call_id": c.call_id, "path": c.path, "expected": c.expected,
                            "actual": c.actual, "ok": c.ok} for c in r.checks],
            }
            for r in v.results
        ],
        "usage": res.trace.summary(),
    }


def root_cause(e: BaseException) -> BaseException:
    """The MCP client wraps failures in (nested) ExceptionGroups; dig out the first real error."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return e


def error_record(case: Case, error: str) -> dict[str, Any]:
    return {"case_id": case.id, "demo": case.demo.name, "question_id": case.question_id,
            "question": case.question, "error": error, "stop_reason": "error"}


def done_case_ids(results_file: Path) -> set[str]:
    if not results_file.exists():
        return set()
    done = set()
    for line in results_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if not rec.get("error"):
                done.add(rec["case_id"])
    return done


async def run_eval(
    cases: list[Case],
    cfg: RunConfig,
    run_dir: Path,
    llm: LLM,
    server_factory: Callable[[], Any],
    on_progress: Callable[[str, dict[str, Any]], None] = lambda case_id, rec: None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(exist_ok=True)
    results_file = run_dir / "results.jsonl"
    config_file = run_dir / "config.json"
    if not config_file.exists():
        config_file.write_text(json.dumps({**asdict(cfg), "agent_prompt": AGENT_PROMPT_VERSION,
                                           "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                           "cases": [c.id for c in cases]}, indent=2))

    done = done_case_ids(results_file)
    todo = [c for c in cases if c.id not in done]
    if not todo:
        return run_dir

    # Ingest each demo once, sequentially: parsing is CPU-heavy and the same demo
    # must not be parsed by several cases at the same time.
    demos: dict[str, dict[str, Any] | str] = {}
    async with Toolbox.connect(server_factory()) as tb:
        listing, _ = await tb.call("list_demos", {})
        cached = {d["demo_id"]: d for d in (listing.get("demos", []) if isinstance(listing, dict) else [])}
        for case in todo:
            if case.demo.name in demos:
                continue
            if case.demo.file in cached:  # `file` may also be an already-ingested demo_id
                demos[case.demo.name] = cached[case.demo.file]
                continue
            result, is_error = await tb.call("ingest_demo", {"path": case.demo.file})
            demos[case.demo.name] = str(result) if is_error else result

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, cfg.concurrency))

    async def write(rec: dict[str, Any]) -> None:
        async with lock:
            with results_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        on_progress(rec["case_id"], rec)

    async def one(case: Case) -> None:
        demo = demos[case.demo.name]
        if isinstance(demo, str):
            await write(error_record(case, f"demo failed to load: {demo}"))
            return
        async with sem:
            trace = Trace(run_id=case.id.replace("/", "__"))
            agent_cfg = AgentConfig(model=cfg.model, judge_model=cfg.judge_model, max_steps=cfg.max_steps,
                                    max_revisions=cfg.max_revisions, player=case.demo.player,
                                    judge_prompt=cfg.judge_prompt)
            try:
                async with Toolbox.connect(server_factory()) as tb:
                    res = await run_agent(case.question, demo, llm, tb, agent_cfg, trace)
                rec = result_record(case, res)
            except Exception as e:  # noqa: BLE001 - one bad case must not kill the run
                cause = root_cause(e)
                rec = error_record(case, f"{type(cause).__name__}: {cause}")
                trace.log("error", error=traceback.format_exc())
            trace.write(run_dir / "traces")
            await write(rec)

    await asyncio.gather(*(one(c) for c in todo))
    return run_dir
