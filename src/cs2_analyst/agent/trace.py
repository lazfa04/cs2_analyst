"""Tracing: every model call and tool call of a run, written to a JSONL file.

One line per event, so a run can be replayed, debugged ("why did it call
that tool?") and summed up (tokens, cost, latency). Phase 3 evals read these
files too.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pricing import cost_usd


@dataclass
class Totals:
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True


@dataclass
class Trace:
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    events: list[dict[str, Any]] = field(default_factory=list)
    totals: Totals = field(default_factory=Totals)
    started: float = field(default_factory=time.perf_counter)

    def log(self, event: str, **data: Any) -> dict[str, Any]:
        entry = {"event": event, "t_ms": round((time.perf_counter() - self.started) * 1000), **data}
        self.events.append(entry)
        return entry

    def llm_call(self, *, step: int, model: str, purpose: str, usage: dict[str, int], latency_ms: int,
                 stop_reason: str | None) -> None:
        t = self.totals
        inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        cr, cw = usage.get("cache_read_input_tokens", 0), usage.get("cache_creation_input_tokens", 0)
        cost = cost_usd(model, inp, out, cr, cw)
        t.llm_calls += 1
        t.input_tokens += inp
        t.output_tokens += out
        t.cache_read_tokens += cr
        t.cache_write_tokens += cw
        if cost is None:
            t.cost_known = False
        else:
            t.cost_usd = round(t.cost_usd + cost, 6)
        self.log("llm_call", step=step, model=model, purpose=purpose, latency_ms=latency_ms,
                 stop_reason=stop_reason, usage=usage, cost_usd=cost)

    def tool_call(self, *, step: int, call_id: str, name: str, args: dict[str, Any], result: Any,
                  is_error: bool, latency_ms: int) -> None:
        self.totals.tool_calls += 1
        self.totals.tool_errors += int(is_error)
        self.log("tool_call", step=step, call_id=call_id, name=name, args=args, is_error=is_error,
                 latency_ms=latency_ms, result=result)

    @property
    def elapsed_s(self) -> float:
        return round(time.perf_counter() - self.started, 2)

    def summary(self) -> dict[str, Any]:
        t = self.totals
        return {
            "run_id": self.run_id,
            "elapsed_s": self.elapsed_s,
            "llm_calls": t.llm_calls,
            "tool_calls": t.tool_calls,
            "tool_errors": t.tool_errors,
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
            "cache_read_tokens": t.cache_read_tokens,
            "cache_write_tokens": t.cache_write_tokens,
            "cost_usd": t.cost_usd if t.cost_known else None,
        }

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for e in [*self.events, {"event": "summary", **self.summary()}]:
                f.write(json.dumps(e, default=str) + "\n")
        return path
