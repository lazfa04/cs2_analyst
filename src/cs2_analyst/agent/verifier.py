"""Verifier: checks every claim against the tool results the agent actually received.

Facts are checked deterministically: resolve the cited path inside the
recorded tool result and compare with the claimed value. No model is involved,
so this step can't hallucinate. The tools are deterministic over cached demo
data, so the recorded result IS the ground truth for that query.

Path syntax (dots separate steps, brackets select from lists):
    overall.won                      dict keys
    duels.0.opponent                 list index
    duels[round=4].traded            first list item whose `round` equals 4
    scoreboard[name=organner.pl free trial].kills    dots/spaces are fine inside brackets
    kills[round=7,victim=cntoz].weapon               several conditions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from .findings import Claim, Report

Status = Literal["verified", "failed", "unchecked", "supported", "unsupported", "not_judged"]


class PathError(ValueError):
    pass


# ---------------------------------------------------------------- path resolution

def parse_path(path: str) -> list[str | int | dict[str, str]]:
    tokens: list[str | int | dict[str, str]] = []
    buf, i = "", 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if buf:
                tokens.append(buf)
            buf, i = "", i + 1
        elif c == "[":
            if buf:
                tokens.append(buf)
                buf = ""
            end = path.find("]", i)
            if end < 0:
                raise PathError(f"unclosed '[' in path '{path}'")
            inner = path[i + 1:end].strip()
            if inner.isdigit():
                tokens.append(int(inner))
            else:
                conds = {}
                for part in inner.split(","):
                    if "=" not in part:
                        raise PathError(f"filter '[{inner}]' must look like [key=value]")
                    k, v = part.split("=", 1)
                    conds[k.strip()] = v.strip()
                tokens.append(conds)
            i = end + 1
        else:
            buf += c
            i += 1
    if buf:
        tokens.append(buf)
    if not tokens:
        raise PathError("empty path")
    return tokens


def resolve(data: Any, path: str) -> Any:
    cur = data
    for tok in parse_path(path):
        if isinstance(tok, dict):
            if not isinstance(cur, list):
                raise PathError(f"filter {tok} applied to a {type(cur).__name__}, not a list")
            match = next((it for it in cur if isinstance(it, dict)
                          and all(k in it and values_match(it[k], v) for k, v in tok.items())), None)
            if match is None:
                raise PathError(f"no item matches {tok}")
            cur = match
        elif isinstance(cur, list):
            idx = tok if isinstance(tok, int) else int(tok) if str(tok).isdigit() else None
            if idx is None:
                raise PathError(f"'{tok}' used on a list; use an index or a [key=value] filter")
            if idx >= len(cur):
                raise PathError(f"index {idx} out of range (list has {len(cur)} items)")
            cur = cur[idx]
        elif isinstance(cur, dict):
            if tok not in cur:
                raise PathError(f"key '{tok}' not found; available: {list(cur)[:15]}")
            cur = cur[tok]
        else:
            raise PathError(f"can't go into '{tok}': reached a plain value")
    return cur


# ---------------------------------------------------------------- comparison

_TRUE, _FALSE = {"true", "yes", "1"}, {"false", "no", "0"}


def values_match(actual: Any, expected: str) -> bool:
    ok, _ = compare(actual, expected)
    return ok


def compare(actual: Any, expected: str) -> tuple[bool, str | None]:
    exp = str(expected).strip()
    if isinstance(actual, (dict, list)):
        return False, "path points to a list/object; point it at a single value"
    if actual is None:
        return exp.lower() in {"null", "none", ""}, None
    if isinstance(actual, bool):
        low = exp.lower()
        if low not in _TRUE | _FALSE:
            return False, f"expected a boolean, got '{exp}'"
        return actual == (low in _TRUE), None
    if isinstance(actual, (int, float)):
        num = exp.rstrip("%").replace(",", "").strip()
        try:
            e = float(num)
        except ValueError:
            return False, f"expected a number, got '{exp}'"
        if isinstance(actual, int):
            return float(actual) == e, None  # counts must match exactly
        # Precision-aware: the claim is right if the real value, rounded to the decimals the
        # agent wrote, equals the claim. 89.83 -> "90" and "89.8" pass; "89" and "89.9" fail.
        decimals = len(num.split(".")[1]) if "." in num else 0
        quantum = Decimal(1).scaleb(-decimals)  # 1, 0.1, 0.01, ...
        # Half-up rounding like a human would do (Python's round() rounds 2.5 to 2).
        rounded = Decimal(str(actual)).quantize(quantum, rounding=ROUND_HALF_UP)
        return rounded == Decimal(num).quantize(quantum, rounding=ROUND_HALF_UP), None
    return str(actual).strip().casefold() == exp.casefold(), None


# ---------------------------------------------------------------- verification

@dataclass
class CheckResult:
    call_id: str
    path: str
    expected: str
    ok: bool
    actual: Any = None
    reason: str | None = None


@dataclass
class ClaimResult:
    claim: Claim
    status: Status
    checks: list[CheckResult] = field(default_factory=list)
    note: str | None = None


@dataclass
class Verification:
    results: list[ClaimResult]
    # Specific statements in the summary that no verified claim backs (filled in by the judge).
    summary_issues: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[ClaimResult]:
        return [r for r in self.results if r.status in ("failed", "unsupported")]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out


def check_one(check, ledger: dict[str, Any]) -> CheckResult:
    base = dict(call_id=check.call_id, path=check.path, expected=check.expected)
    if check.call_id not in ledger:
        return CheckResult(**base, ok=False, reason=f"no tool call '{check.call_id}' in this run; known: {list(ledger)}")
    record = ledger[check.call_id]
    if record.get("is_error"):
        return CheckResult(**base, ok=False, reason=f"{check.call_id} was a failed tool call")
    try:
        actual = resolve(record["result"], check.path)
    except PathError as e:
        return CheckResult(**base, ok=False, reason=f"bad path: {e}")
    ok, reason = compare(actual, check.expected)
    return CheckResult(**base, ok=ok, actual=actual, reason=reason or (None if ok else "value differs"))


def verify(report: Report, ledger: dict[str, Any]) -> Verification:
    """ledger maps call_id -> {"name", "args", "result" (parsed JSON), "is_error"}."""
    results: dict[str, ClaimResult] = {}
    for claim in report.claims:
        if claim.kind != "fact":
            continue
        if not claim.checks:
            results[claim.id] = ClaimResult(claim, "unchecked", note="fact without a check")
            continue
        checks = [check_one(c, ledger) for c in claim.checks]
        results[claim.id] = ClaimResult(claim, "verified" if all(c.ok for c in checks) else "failed", checks)

    for claim in report.claims:
        if claim.kind != "interpretation":
            continue
        if not claim.supports:
            results[claim.id] = ClaimResult(claim, "unsupported", note="cites no supporting facts")
            continue
        bad = [s for s in claim.supports if results.get(s) is None or results[s].status != "verified"]
        if bad:
            results[claim.id] = ClaimResult(claim, "unsupported", note=f"supporting facts not verified: {bad}")
        else:
            results[claim.id] = ClaimResult(claim, "not_judged")

    return Verification([results[c.id] for c in report.claims if c.id in results])


def feedback(v: Verification) -> str:
    """What the agent is told when claims fail, so it can fix them."""
    lines = ["Some claims failed verification. Fix or remove them, then call submit_report again with the full report."]
    for r in v.failed:
        lines.append(f"- {r.claim.id} ({r.claim.kind}): {r.claim.text}")
        for c in r.checks:
            if not c.ok:
                got = f" actual value: {c.actual!r}." if c.actual is not None else ""
                lines.append(f"    check {c.call_id} {c.path} expected {c.expected!r}: {c.reason}.{got}")
        if r.note:
            lines.append(f"    {r.note}")
    return "\n".join(lines)
