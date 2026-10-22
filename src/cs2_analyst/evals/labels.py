"""Human labels for judge agreement.

You see an interpretation and the verified facts it rests on (NOT the judge's verdict,
so you aren't anchored by it) and answer: do these facts support this conclusion?
Labels live in evals/labels.jsonl, keyed by a hash of the interpretation and its facts,
so they're reused by every later run that produces the same item. That file is the
human-labelled dataset; commit it.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def item_key(interpretation: str, facts: list[str]) -> str:
    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    blob = norm(interpretation) + "||" + "|".join(sorted(norm(f) for f in facts))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def judged_items(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every interpretation the judge ruled on, with the facts it rests on."""
    items = []
    for rec in results:
        claims = {c["id"]: c for c in rec.get("claims") or []}
        for c in claims.values():
            if c["kind"] != "interpretation" or not c.get("judged"):
                continue
            facts = [claims[s] for s in c["supports"] if s in claims]
            items.append({
                "key": item_key(c["text"], [f["text"] for f in facts]),
                "case_id": rec["case_id"],
                "question": rec["question"],
                "interpretation": c["text"],
                "facts": [{"text": f["text"], "values": [ch.get("actual") for ch in f.get("checks", [])]} for f in facts],
                "judge_supported": c["status"] == "supported",
                "judge_reason": c.get("note"),
            })
    return items


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                labels[rec["key"]] = rec  # later lines override earlier ones (relabelling)
    return labels


def save_label(path: Path, item: dict[str, Any], agree: bool, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"key": item["key"], "supported": agree, "interpretation": item["interpretation"],
           "facts": [f["text"] for f in item["facts"]], "note": note,
           "labelled_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
