"""The agent loop: model decides, tools run, results go back, repeat.

    question ──> model ──tool_use──> run tool (via MCP) ──tool_result──> model ──> ...
                   └──submit_report──> verifier ──all good──> judge ──> done
                                          └──failures──> feedback to model (max_revisions times)

Guardrails, so a run can't go off the rails or burn money:
  * max_steps     model turns; on the last one it's told to submit with what it has
  * token_budget  total tokens across the run; hitting it also triggers the final step
  * max_revisions how many times failed claims are sent back for fixing
  * a plain-text answer is never accepted silently: the model is asked to submit a report,
    and if it still won't, the text is kept but marked as an unverified answer
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .findings import SUBMIT_REPORT, SUBMIT_REPORT_TOOL, Report
from .judge import judge
from .llm import LLM
from .prompts import AGENT_PROMPT_VERSION, DEFAULT_JUDGE_PROMPT, FINAL_STEP, NO_REPORT, SYSTEM, first_message
from .toolbox import Toolbox
from .trace import Trace
from .verifier import Verification, feedback, verify


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-5-5"
    judge_model: str | None = "claude-haiku-5-5"
    max_steps: int = 12
    max_revisions: int = 1
    token_budget: int = 400_000
    max_tokens_per_turn: int = 16_000
    player: str | None = None
    judge_prompt: str = DEFAULT_JUDGE_PROMPT


@dataclass
class AgentResult:
    question: str
    demo: dict[str, Any]
    report: Report | None
    verification: Verification
    stop_reason: str
    revisions: int
    trace: Trace
    ledger: dict[str, Any] = field(default_factory=dict)
    # Claim statuses of the FIRST report, before any revision. It's the no-verifier baseline:
    # what the user would have seen if nothing checked the agent's work.
    first_draft: dict[str, int] | None = None


def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}


def _compact(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, separators=(",", ":"), default=str)


async def run_agent(question: str, demo: dict[str, Any], llm: LLM, toolbox: Toolbox,
                    config: AgentConfig | None = None, trace: Trace | None = None) -> AgentResult:
    cfg = config or AgentConfig()
    trace = trace or Trace()
    trace.log("run_start", question=question, demo_id=demo["demo_id"], model=cfg.model,
              judge_model=cfg.judge_model, judge_prompt=cfg.judge_prompt, agent_prompt=AGENT_PROMPT_VERSION,
              max_steps=cfg.max_steps)

    tools = [*await toolbox.schemas(), SUBMIT_REPORT_TOOL]
    messages: list[dict[str, Any]] = [{"role": "user", "content": first_message(question, demo, cfg.player)}]
    ledger: dict[str, Any] = {}
    report: Report | None = None
    verification = Verification([])
    revisions, stop_reason, final_turn_sent, nudged = 0, "step_limit", False, False
    first_draft: dict[str, int] | None = None

    step = 0
    while step < cfg.max_steps + 1:  # +1: one grace turn after the "submit now" message
        step += 1
        t0 = time.perf_counter()
        resp = await llm.create(model=cfg.model, system=SYSTEM, tools=tools, messages=messages,
                                max_tokens=cfg.max_tokens_per_turn)
        trace.llm_call(step=step, model=cfg.model, purpose="agent", usage=resp.usage,
                       latency_ms=round((time.perf_counter() - t0) * 1000), stop_reason=resp.stop_reason)
        messages.append({"role": "assistant", "content": resp.raw_content})
        if resp.text:
            trace.log("assistant_text", step=step, text=resp.text[:2000])

        if resp.stop_reason == "refusal":
            stop_reason = "refusal"
            break

        if not resp.tool_uses:
            if final_turn_sent or nudged:
                report = Report(summary=resp.text, claims=[])
                stop_reason = "text_answer_unverified"
                break
            nudged = True
            messages.append({"role": "user", "content": NO_REPORT})
            continue

        results: list[dict[str, Any]] = []
        submitted: list[Any] = []
        for tu in resp.tool_uses:
            if tu.name == SUBMIT_REPORT:
                submitted.append(tu)
                continue
            if final_turn_sent:
                results.append(_tool_result(tu.id, "Step limit reached: call submit_report instead.", True))
                continue
            call_id = f"t{len(ledger) + 1}"
            t1 = time.perf_counter()
            result, is_error = await toolbox.call(tu.name, tu.input or {})
            ledger[call_id] = {"name": tu.name, "args": tu.input, "result": result, "is_error": is_error}
            trace.tool_call(step=step, call_id=call_id, name=tu.name, args=tu.input or {}, result=result,
                            is_error=is_error, latency_ms=round((time.perf_counter() - t1) * 1000))
            results.append(_tool_result(tu.id, f"[call_id={call_id}]\n{_compact(result)}", is_error))

        done = False
        for tu in submitted:
            try:
                candidate = Report.model_validate(tu.input or {})
            except ValidationError as e:
                results.append(_tool_result(tu.id, f"Report doesn't match the schema: {e}", True))
                continue
            verification = verify(candidate, ledger)
            report = candidate
            if first_draft is None:
                first_draft = verification.counts()
            trace.log("report_submitted", step=step, revision=revisions, counts=verification.counts())
            if verification.failed and revisions < cfg.max_revisions and not final_turn_sent:
                revisions += 1
                results.append(_tool_result(tu.id, feedback(verification)))
            else:
                done = True
                stop_reason = "submitted"

        if done:
            break

        out_of_budget = (trace.totals.input_tokens + trace.totals.output_tokens + trace.totals.cache_read_tokens
                         + trace.totals.cache_write_tokens) >= cfg.token_budget
        content: list[dict[str, Any]] = list(results)
        if not final_turn_sent and (step >= cfg.max_steps or out_of_budget):
            final_turn_sent = True
            trace.log("final_step", step=step, reason="token_budget" if out_of_budget else "max_steps")
            content.append({"type": "text", "text": FINAL_STEP})
        elif final_turn_sent:
            break  # grace turn used up without a report
        messages.append({"role": "user", "content": content})

    if report is not None and cfg.judge_model and stop_reason == "submitted":
        await judge(llm, cfg.judge_model, report, verification, trace, cfg.judge_prompt)

    trace.log("run_end", stop_reason=stop_reason, revisions=revisions,
              counts=verification.counts(), **trace.summary())
    return AgentResult(question, demo, report, verification, stop_reason, revisions, trace, ledger, first_draft)
