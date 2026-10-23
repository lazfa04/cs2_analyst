"""Red-team the judge: score it on conclusions and summaries whose right answer is known.

Human labels from real runs are mostly "fair", so they can't show whether the judge rejects
anything. evals/redteam.toml mixes deliberately bad conclusions (overreach, contradictions,
unmeasured roles or intent, patterns from one match, invented causes) with fair ones, and plants
mistakes in some summaries. Each case goes through the real judge code once; no agent runs, so a
full pass costs about a cent with Haiku.

What it measures:
  bad conclusions rejected   recall on the conclusions the judge must reject, overall and per category
  fair conclusions accepted  the other side: a judge that rejects everything also catches every bad one
  planted mistakes caught    contradicted / unbacked statements planted in summaries, found anywhere in the
                             judge's flags; and whether contradictions were filed as contradictions
  clean summaries flagged    false alarms on summaries with nothing wrong
"""

from __future__ import annotations

import asyncio
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent.findings import Claim, Report
from ..agent.judge import judge
from ..agent.llm import LLM
from ..agent.trace import Trace
from ..agent.verifier import ClaimResult, Verification

CATEGORIES = ("overreach", "contradicts", "unmeasured", "one_event", "causal", "rating_only")


class RedTeamError(ValueError):
    pass


@dataclass
class Interp:
    text: str
    supports: list[str]
    expected: bool
    category: str
    why: str = ""


@dataclass
class Planted:
    kind: str  # "contradicted" | "unbacked"
    match: list[str]
    why: str = ""


@dataclass
class RedTeamCase:
    id: str
    facts: list[str]
    summary: str
    interpretations: list[Interp]
    planted: list[Planted] = field(default_factory=list)
    tier: str = "easy"


def load_redteam(path: Path) -> list[RedTeamCase]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RedTeamError(f"No red-team file at {path}.") from None
    except tomllib.TOMLDecodeError as e:
        raise RedTeamError(f"{path} isn't valid TOML: {e}") from None
    cases = []
    for raw in data.get("case", []):
        cid = raw.get("id", "?")
        facts = raw.get("facts") or []
        fact_ids = {f"f{i}" for i in range(1, len(facts) + 1)}
        interps = []
        for it in raw.get("interpretation", []):
            bad = [s for s in it.get("supports", []) if s not in fact_ids]
            if bad:
                raise RedTeamError(f"case '{cid}': supports {bad} don't exist (facts are f1-f{len(facts)})")
            cat = it.get("category", "fair")
            if cat != "fair" and cat not in CATEGORIES:
                raise RedTeamError(f"case '{cid}': unknown category '{cat}'. Use fair or one of {CATEGORIES}.")
            if (cat == "fair") != bool(it["expected"]):
                raise RedTeamError(f"case '{cid}': '{it['text'][:40]}...' is {cat} but expected={it['expected']}")
            interps.append(Interp(it["text"], it.get("supports", []), bool(it["expected"]), cat, it.get("why", "")))
        planted = []
        for p in raw.get("planted", []):
            if p.get("kind") not in ("contradicted", "unbacked"):
                raise RedTeamError(f"case '{cid}': planted kind must be contradicted or unbacked")
            planted.append(Planted(p["kind"], [m.lower() for m in p["match"]], p.get("why", "")))
        tier = raw.get("tier", "easy")
        if tier not in ("easy", "hard"):
            raise RedTeamError(f"case '{cid}': tier must be easy or hard")
        cases.append(RedTeamCase(cid, facts, raw["summary"], interps, planted, tier))
    if not cases:
        raise RedTeamError(f"{path} has no [[case]] entries.")
    return cases


def build(case: RedTeamCase) -> tuple[Report, Verification]:
    """The case as a report whose facts all verified, with every conclusion waiting for the judge."""
    claims, results = [], []
    for i, text in enumerate(case.facts, 1):
        c = Claim(id=f"f{i}", text=text, kind="fact")
        claims.append(c)
        results.append(ClaimResult(c, "verified"))
    for i, it in enumerate(case.interpretations, 1):
        c = Claim(id=f"i{i}", text=it.text, kind="interpretation", supports=it.supports)
        claims.append(c)
        results.append(ClaimResult(c, "not_judged"))
    return Report(summary=case.summary, claims=claims), Verification(results)


async def run_redteam(cases: list[RedTeamCase], llm: LLM, model: str, prompt: str,
                      concurrency: int = 5) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(case: RedTeamCase) -> dict[str, Any]:
        report, verification = build(case)
        trace = Trace(run_id=f"redteam__{case.id}")
        async with sem:
            await judge(llm, model, report, verification, trace, prompt)
        by_id = {r.claim.id: r for r in verification.results}
        interps = []
        for i, it in enumerate(case.interpretations, 1):
            r = by_id[f"i{i}"]
            got = None if not r.judged else r.status == "supported"
            interps.append({"text": it.text, "category": it.category, "expected": it.expected,
                            "judge": got, "reason": r.note, "correct": got == it.expected})
        flagged_u = [s.lower() for s in verification.summary_issues]
        flagged_c = [s.lower() for s in verification.summary_contradictions]
        planted = []
        for p in case.planted:
            in_c = any(m in s for s in flagged_c for m in p.match)
            in_u = any(m in s for s in flagged_u for m in p.match)
            planted.append({"kind": p.kind, "match": p.match, "caught": in_c or in_u,
                            "filed_as": "contradicted" if in_c else ("unbacked" if in_u else None)})
        return {
            "case_id": case.id,
            "tier": case.tier,
            "interpretations": interps,
            "summary_clean": not case.planted,
            "flagged_unbacked": verification.summary_issues,
            "flagged_contradicted": verification.summary_contradictions,
            "planted": planted,
            "usage": trace.summary(),
        }

    return list(await asyncio.gather(*(one(c) for c in cases)))


def _pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None


def _balanced(interps: list[dict[str, Any]]) -> float | None:
    bad = [i for i in interps if not i["expected"]]
    fair = [i for i in interps if i["expected"]]
    if not bad or not fair:
        return None
    return round((sum(i["judge"] is False for i in bad) / len(bad) + sum(i["judge"] is True for i in fair) / len(fair)) * 50, 1)


def score(results: list[dict[str, Any]]) -> dict[str, Any]:
    interps = [i for r in results for i in r["interpretations"]]
    bad = [i for i in interps if not i["expected"]]
    fair = [i for i in interps if i["expected"]]
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i in bad:
        by_cat[i["category"]].append(i)
    planted = [p for r in results for p in r["planted"]]
    contra = [p for p in planted if p["kind"] == "contradicted"]
    unbacked = [p for p in planted if p["kind"] == "unbacked"]
    clean = [r for r in results if r["summary_clean"]]
    rejected_bad = sum(i["judge"] is False for i in bad)
    accepted_fair = sum(i["judge"] is True for i in fair)
    costs = [r["usage"].get("cost_usd") for r in results if r["usage"].get("cost_usd") is not None]
    return {
        "cases": len(results),
        "conclusions": {
            "bad": len(bad),
            "fair": len(fair),
            "bad_rejected_pct": _pct(rejected_bad, len(bad)),
            "fair_accepted_pct": _pct(accepted_fair, len(fair)),
            # mean of the two: 50 for a judge that always says yes (or always no)
            "balanced_accuracy_pct": (round((rejected_bad / len(bad) + accepted_fair / len(fair)) * 50, 1)
                                      if bad and fair else None),
            "unanswered": sum(i["judge"] is None for i in interps),
            "balanced_accuracy_by_tier_pct": {
                t: _balanced([i for r in results if r.get("tier", "easy") == t for i in r["interpretations"]])
                for t in sorted({r.get("tier", "easy") for r in results})
            },
            "bad_rejected_by_category": {
                c: {"rejected": sum(i["judge"] is False for i in items), "total": len(items)}
                for c, items in sorted(by_cat.items())
            },
        },
        "summaries": {
            "planted_caught_pct": _pct(sum(p["caught"] for p in planted), len(planted)),
            "contradictions_caught_pct": _pct(sum(p["caught"] for p in contra), len(contra)),
            "contradictions_filed_correctly_pct": _pct(sum(p["filed_as"] == "contradicted" for p in contra), len(contra)),
            "unbacked_caught_pct": _pct(sum(p["caught"] for p in unbacked), len(unbacked)),
            "clean_summaries_flagged_pct": _pct(
                sum(bool(r["flagged_unbacked"] or r["flagged_contradicted"]) for r in clean), len(clean)),
            "planted": len(planted),
            "clean": len(clean),
        },
        "total_cost_usd": round(sum(costs), 4) if costs else None,
    }


def misses(results: list[dict[str, Any]]) -> list[str]:
    """One line per thing the judge got wrong, for reading after a run."""
    out = []
    for r in results:
        for i in r["interpretations"]:
            if not i["correct"]:
                verdict = {True: "accepted", False: "rejected", None: "skipped"}[i["judge"]]
                out.append(f"{r['case_id']}: {verdict} [{i['category']}, should be "
                           f"{'accepted' if i['expected'] else 'rejected'}] {i['text']}  (judge: {i['reason']})")
        for p in r["planted"]:
            if not p["caught"]:
                out.append(f"{r['case_id']}: missed planted {p['kind']} statement ({' / '.join(p['match'])})")
        if r["summary_clean"] and (r["flagged_unbacked"] or r["flagged_contradicted"]):
            out.append(f"{r['case_id']}: false alarm on a clean summary: "
                       f"{r['flagged_contradicted'] + r['flagged_unbacked']}")
    return out


def stability(runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """How much the judge's answers move between identical runs: the noise floor for comparing prompts.

    The judge samples, so the same prompt on the same items can give different verdicts. A prompt change
    that moves the score by less than the spread between repeats hasn't been shown to do anything.
    """
    balanced = [score(r)["conclusions"]["balanced_accuracy_pct"] for r in runs]
    verdicts: dict[tuple[str, str], list[bool | None]] = defaultdict(list)
    expected: dict[tuple[str, str], bool] = {}
    for results in runs:
        for r in results:
            for i in r["interpretations"]:
                key = (r["case_id"], i["text"])
                verdicts[key].append(i["judge"])
                expected[key] = i["expected"]
    unstable = [k for k, v in verdicts.items() if len(set(v)) > 1]
    wrong = {k: sum(x != expected[k] for x in v) for k, v in verdicts.items()}
    return {
        "repeats": len(runs),
        "balanced_accuracy_each_pct": balanced,
        "balanced_accuracy_mean_pct": round(sum(b for b in balanced if b is not None) / len(balanced), 1),
        "balanced_accuracy_spread_pct": round(max(balanced) - min(balanced), 1) if balanced else None,
        "unstable_conclusions": len(unstable),
        "conclusions": len(verdicts),
        "wrong_counts": {f"{c}: {t}": n for (c, t), n in wrong.items() if n},
    }
