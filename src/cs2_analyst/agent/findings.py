"""Findings schema: how the agent states its conclusions.

Instead of a free-text answer, the agent submits a report of atomic claims:

  fact            "babaya0310 won 6 of 7 opening duels"
                  + checks: [{call_id: "t2", path: "overall.won", expected: "6"}]
                  Each check points at ONE value inside a tool result the agent
                  actually received. The verifier looks it up and compares, so a
                  fact is either provably right, provably wrong, or unchecked.

  interpretation  "He wins the round when he wins the opening duel"
                  + supports: ["c1", "c3"]   (ids of fact claims it rests on)
                  Can't be checked mechanically; it's supported only if every
                  fact it rests on verified, and (optionally) a judge model agrees.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Check(BaseModel):
    call_id: str = Field(description="Id of a tool call from this run, e.g. 't2'.")
    path: str = Field(description="Path to one value in that tool result.")
    expected: str = Field(description="The value you are claiming, as text.")


class Claim(BaseModel):
    id: str
    text: str
    kind: Literal["fact", "interpretation"]
    checks: list[Check] = Field(default_factory=list)
    supports: list[str] = Field(default_factory=list)


class Report(BaseModel):
    summary: str
    claims: list[Claim]


SUBMIT_REPORT = "submit_report"

# Hand-written JSON schema (not generated) so it stays inside the subset that
# strict tool use accepts: every object closed, every field required.
SUBMIT_REPORT_TOOL = {
    "name": SUBMIT_REPORT,
    "description": (
        "Submit your final answer. Call this exactly once when you have enough evidence. "
        "`summary` is a short answer to the user's question in plain language. `claims` lists every "
        "specific statement the summary relies on. A fact claim must carry at least one check pointing "
        "at the value in a tool result that proves it. An interpretation claim must list the ids of the "
        "fact claims it rests on in `supports`. Your claims are verified automatically; wrong ones are "
        "sent back to you to fix."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Direct answer to the question, 2-6 sentences."},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "c1, c2, ..."},
                        "text": {"type": "string", "description": "One atomic statement."},
                        "kind": {"type": "string", "enum": ["fact", "interpretation"]},
                        "checks": {
                            "type": "array",
                            "description": "Facts: one or more. Interpretations: empty.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "call_id": {"type": "string"},
                                    "path": {"type": "string"},
                                    "expected": {"type": "string"},
                                },
                                "required": ["call_id", "path", "expected"],
                                "additionalProperties": False,
                            },
                        },
                        "supports": {
                            "type": "array",
                            "description": "Interpretations: ids of supporting fact claims. Facts: empty.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["id", "text", "kind", "checks", "supports"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "claims"],
        "additionalProperties": False,
    },
}
