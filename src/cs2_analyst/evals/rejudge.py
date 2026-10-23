"""Re-run only the judge on a finished eval run.

To compare two judge prompts fairly they must rule on the SAME interpretations, the ones
already labelled. A fresh `eval run` would have the agent write new interpretations, which
need new labels. Rejudging rebuilds every answer from results.jsonl (claims, verified facts
and their values), resets the judge's verdicts, and asks the judge again. Facts and the
verifier's own rejections are left untouched: only the judge's decisions change.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..agent.findings import Check, Claim, Report
from ..agent.judge import judge
from ..agent.llm import LLM
from ..agent.trace import Trace
from ..agent.verifier import CheckResult, ClaimResult, Verification
from .metrics import load_results


def rebuild(rec: dict[str, Any]) -> tuple[Report, Verification]:
    """An answer's report and verification, as they stood before the judge ruled."""
    claims, results = [], []
    for c in rec["claims"]:
        claim = Claim(id=c["id"], text=c["text"], kind=c["kind"], supports=c.get("supports", []),
                      checks=[Check(call_id=k["call_id"], path=k["path"], expected=k["expected"])
                              for k in c.get("checks", [])])
        checks = [CheckResult(call_id=k["call_id"], path=k["path"], expected=k["expected"], ok=k["ok"],
                              actual=k.get("actual")) for k in c.get("checks", [])]
        status, note = c["status"], c.get("note")
        if c["kind"] == "interpretation" and (c.get("judged") or status == "not_judged"):
            status, note = "not_judged", None  # the judge's call is what we're redoing
        claims.append(claim)
        results.append(ClaimResult(claim, status, checks, note))
    return Report(summary=rec.get("summary") or "", claims=claims), Verification(results)


async def rejudge_run(src: Path, dst: Path, llm: LLM, model: str, prompt: str, concurrency: int = 5,
                      on_progress: Callable[[str, dict[str, Any]], None] = lambda case_id, rec: None) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    src_cfg = json.loads((src / "config.json").read_text()) if (src / "config.json").exists() else {}
    (dst / "config.json").write_text(json.dumps({
        **src_cfg, "rejudged_from": src.name, "judge_model": model, "judge_prompt": prompt,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Agent answers and costs are copied from the source run; only the judge was re-run.",
    }, indent=2))

    sem = asyncio.Semaphore(max(1, concurrency))
    out: dict[str, dict[str, Any]] = {}

    async def one(rec: dict[str, Any]) -> None:
        new = copy.deepcopy(rec)
        if not rec.get("error") and rec.get("claims") is not None and rec.get("stop_reason") == "submitted":
            report, verification = rebuild(rec)
            trace = Trace(run_id=rec["case_id"].replace("/", "__") + "__rejudge")
            async with sem:
                await judge(llm, model, report, verification, trace, prompt)
            by_id = {r.claim.id: r for r in verification.results}
            for c in new["claims"]:
                r = by_id[c["id"]]
                c.update(status=r.status, judged=r.judged, note=r.note)
            new["summary_issues"] = verification.summary_issues
            new["summary_contradictions"] = verification.summary_contradictions
            new["final_counts"] = verification.counts()
            new["judge_usage"] = trace.summary()
        out[rec["case_id"]] = new
        on_progress(rec["case_id"], new)

    await asyncio.gather(*(one(r) for r in load_results(src)))
    with (dst / "results.jsonl").open("w", encoding="utf-8") as f:
        for case_id in sorted(out):
            f.write(json.dumps(out[case_id], default=str) + "\n")
    return dst
