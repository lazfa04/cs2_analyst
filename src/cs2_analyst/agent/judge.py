"""Judge: does the verified evidence support the conclusions, and is the summary backed?

Facts are checked deterministically by the verifier. Two things can't be:
  * interpretations ("you lost because..."): does the cited evidence actually support them?
  * the summary: does it slip in specifics (a ranking, a name, a number) that no claim backs?
A second, cheaper model reviews both. It only sees the summary, the verified claims and each
interpretation with its supporting facts, not the whole conversation, so it judges the
argument, not the agent's confidence.
"""

from __future__ import annotations

import json
import time

from .findings import Report
from .llm import LLM
from .prompts import DEFAULT_JUDGE_PROMPT, JUDGE_PROMPTS
from .trace import Trace
from .verifier import Verification

SUBMIT_VERDICTS = "submit_verdicts"
VERDICT_TOOL = {
    "name": SUBMIT_VERDICTS,
    "description": "Submit one verdict per interpretation, plus any unbacked statements in the summary.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "supported": {"type": "boolean"},
                        "reason": {"type": "string", "description": "One sentence."},
                    },
                    "required": ["claim_id", "supported", "reason"],
                    "additionalProperties": False,
                },
            },
            "unbacked_summary_statements": {
                "type": "array",
                "description": "Specific statements from the summary that no verified claim backs. Empty if none.",
                "items": {"type": "string"},
            },
        },
        "required": ["verdicts", "unbacked_summary_statements"],
        "additionalProperties": False,
    },
}


async def judge(llm: LLM, model: str, report: Report, verification: Verification, trace: Trace,
                prompt: str = DEFAULT_JUDGE_PROMPT) -> None:
    """Upgrade `not_judged` interpretations to supported/unsupported and fill summary_issues, in place."""
    if not report.summary.strip():
        return
    pending = [r for r in verification.results if r.status == "not_judged"]
    facts = {r.claim.id: r for r in verification.results if r.status == "verified"}
    payload = {
        "summary": report.summary,
        "verified_claims": [r.claim.text for r in facts.values()],
        "interpretations": [
            {
                "claim_id": r.claim.id,
                "interpretation": r.claim.text,
                "facts": [
                    {"text": facts[s].claim.text,
                     "verified_values": [{"path": c.path, "value": c.actual} for c in facts[s].checks]}
                    for s in r.claim.supports if s in facts
                ],
            }
            for r in pending
        ],
    }
    t0 = time.perf_counter()
    resp = await llm.create(
        model=model, system=JUDGE_PROMPTS[prompt], tools=[VERDICT_TOOL], max_tokens=4000,
        messages=[{"role": "user", "content": json.dumps(payload, default=str, indent=1)}],
    )
    trace.llm_call(step=-1, model=model, purpose=f"judge-{prompt}", usage=resp.usage,
                   latency_ms=round((time.perf_counter() - t0) * 1000), stop_reason=resp.stop_reason)

    call = next((b for b in resp.tool_uses if b.name == SUBMIT_VERDICTS), None)
    if call is None:
        trace.log("judge_no_verdict", text=resp.text[:500])
        return
    data = call.input or {}
    verdicts = {v["claim_id"]: v for v in data.get("verdicts", [])}
    for r in pending:
        v = verdicts.get(r.claim.id)
        if v is not None:
            r.status = "supported" if v["supported"] else "unsupported"
            r.note = v.get("reason")
            r.judged = True
    verification.summary_issues = [s for s in data.get("unbacked_summary_statements", []) if s.strip()]
    trace.log("judge", verdicts=list(verdicts.values()), summary_issues=verification.summary_issues)
