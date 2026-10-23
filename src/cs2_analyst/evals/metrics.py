"""Turn a run's results.jsonl into numbers.

The headline metrics, and what each one answers:
  fact accuracy        of the facts shown to the user, how many verified?
  first-draft errors   how often the agent's FIRST answer had a wrong fact, i.e. what users
                       would have seen with no verifier. The gap to the final answer is what
                       verification buys.
  interpretations      how many conclusions survived the judge
  summary issues       answers that slipped in specifics no claim backs
  judge agreement      how often the judge agrees with a human expert (needs labels)
  cost / latency       what a question costs and how long it takes
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .labels import judged_items


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    f = run_dir / "results.jsonl"
    if not f.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["case_id"]] = rec  # a resumed case's newest attempt wins
    return list(latest.values())


def _pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None


def _flagged(rec: dict[str, Any]) -> bool:
    return bool(rec.get("summary_issues") or rec.get("summary_contradictions"))


def _critic(ok: list[dict[str, Any]]) -> dict[str, Any]:
    """Before vs after the critic revision: the judge's first review is the no-critic baseline."""
    reviewed = [r for r in ok if r.get("first_review") is not None]
    if not reviewed:
        return {"reviewed": 0}
    first_flagged = sum(1 for r in reviewed if r["first_review"]["summary_issues"] + r["first_review"]["summary_contradictions"])
    first_contra = sum(1 for r in reviewed if r["first_review"]["summary_contradictions"])
    return {
        "reviewed": len(reviewed),
        "revision_rate_pct": _pct(sum(1 for r in reviewed if r.get("critic_revisions")), len(reviewed)),
        "first_review_flagged_pct": _pct(first_flagged, len(reviewed)),
        "first_review_contradicting_pct": _pct(first_contra, len(reviewed)),
    }


def _claims(rec: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [c for c in rec.get("claims") or [] if c["kind"] == kind]


def compute(results: list[dict[str, Any]], labels: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    ok = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]
    submitted = [r for r in ok if r["stop_reason"] == "submitted"]

    facts = [c for r in ok for c in _claims(r, "fact")]
    fact_status = Counter(c["status"] for c in facts)
    interps = [c for r in ok for c in _claims(r, "interpretation")]
    interp_status = Counter(
        "rejected_by_verifier" if c["status"] == "unsupported" and not c.get("judged") else c["status"]
        for c in interps
    )

    first_wrong = [r for r in submitted if (r.get("first_draft") or {}).get("failed", 0) > 0]
    final_wrong = [r for r in submitted if r["final_counts"].get("failed", 0) > 0]
    usage = [r["usage"] for r in ok]
    costs = [u["cost_usd"] for u in usage if u.get("cost_usd") is not None]
    times = [u["elapsed_s"] for u in usage]
    tokens_in = sum(u["input_tokens"] + u["cache_read_tokens"] + u["cache_write_tokens"] for u in usage)

    metrics: dict[str, Any] = {
        "cases": len(results),
        "errors": len(errors),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in results)),
        "facts": {
            "total": len(facts),
            "verified_pct": _pct(fact_status["verified"], len(facts)),
            "failed_pct": _pct(fact_status["failed"], len(facts)),
            "unchecked_pct": _pct(fact_status["unchecked"], len(facts)),
            "per_answer": round(len(facts) / len(ok), 1) if ok else None,
        },
        "first_draft": {
            "answers_with_wrong_fact_pct": _pct(len(first_wrong), len(submitted)),
            "wrong_facts": sum((r.get("first_draft") or {}).get("failed", 0) for r in submitted),
            "final_answers_with_wrong_fact_pct": _pct(len(final_wrong), len(submitted)),
            "revision_rate_pct": _pct(sum(1 for r in submitted if r["revisions"] > 0), len(submitted)),
        },
        "interpretations": {
            "total": len(interps),
            "supported_pct": _pct(interp_status["supported"], len(interps)),
            "rejected_by_judge_pct": _pct(interp_status["unsupported"], len(interps)),
            "rejected_by_verifier_pct": _pct(interp_status["rejected_by_verifier"], len(interps)),
        },
        "summary_issues": {
            # flagged = anything unbacked OR contradicted (older runs put contradictions under unbacked)
            "answers_flagged_pct": _pct(sum(1 for r in ok if _flagged(r)), len(ok)),
            "answers_contradicting_pct": _pct(sum(1 for r in ok if r.get("summary_contradictions")), len(ok)),
            "total": sum(len(r.get("summary_issues") or []) + len(r.get("summary_contradictions") or []) for r in ok),
        },
        "critic": _critic(ok),
        "ops": {
            "total_cost_usd": round(sum(costs), 4) if costs else None,
            "avg_cost_usd": round(statistics.mean(costs), 4) if costs else None,
            "p50_seconds": round(statistics.median(times), 1) if times else None,
            "p95_seconds": round(sorted(times)[max(0, int(len(times) * 0.95) - 1)], 1) if times else None,
            "avg_tool_calls": round(statistics.mean(u["tool_calls"] for u in usage), 1) if usage else None,
            "avg_model_calls": round(statistics.mean(u["llm_calls"] for u in usage), 1) if usage else None,
            "cache_hit_pct": _pct(sum(u["cache_read_tokens"] for u in usage), tokens_in),
        },
        "per_question": {},
        "judge_agreement": judge_agreement(results, labels or {}),
    }

    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        by_q[r["question_id"]].append(r)
    for qid, recs in sorted(by_q.items()):
        qf = [c for r in recs for c in _claims(r, "fact")]
        qi = [c for r in recs for c in _claims(r, "interpretation")]
        qc = [r["usage"]["cost_usd"] for r in recs if r["usage"].get("cost_usd") is not None]
        metrics["per_question"][qid] = {
            "cases": len(recs),
            "facts_verified_pct": _pct(sum(c["status"] == "verified" for c in qf), len(qf)),
            "interpretations_supported_pct": _pct(sum(c["status"] == "supported" for c in qi), len(qi)),
            "summary_flagged": sum(1 for r in recs if _flagged(r)),
            "avg_cost_usd": round(statistics.mean(qc), 4) if qc else None,
        }
    return metrics


def judge_agreement(results: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Judge verdicts vs human labels. 'Positive' = the conclusion IS supported."""
    pairs = [(it["judge_supported"], labels[it["key"]]["supported"])
             for it in judged_items(results) if it["key"] in labels]
    tp = sum(j and h for j, h in pairs)
    tn = sum((not j) and (not h) for j, h in pairs)
    fp = sum(j and not h for j, h in pairs)   # judge let through something the expert rejects
    fn = sum((not j) and h for j, h in pairs)  # judge rejected something the expert accepts
    n = len(pairs)
    human_yes = sum(h for _, h in pairs)
    return {
        "labelled": n,
        "unlabelled": len(judged_items(results)) - n,
        "agreement_pct": _pct(tp + tn, n),
        # What a judge that always says "supported" (or always "not supported") would score.
        # A real judge has to beat the better of the two to be adding anything.
        "trivial_baseline_pct": _pct(max(human_yes, n - human_yes), n),
        "too_lenient": fp,
        "too_strict": fn,
        "both_accept": tp,
        "both_reject": tn,
    }
