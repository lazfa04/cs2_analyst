"""The eval set: demos x questions, loaded from a TOML file (evals/cases.toml)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Demo:
    name: str
    file: str
    player: str | None = None


@dataclass(frozen=True)
class Case:
    id: str  # "<demo>/<question>", stable across runs so results can be compared
    demo: Demo
    question_id: str
    question: str


class CasesError(ValueError):
    pass


def load_cases(path: Path, demos: list[str] | None = None, questions: list[str] | None = None) -> list[Case]:
    if not path.is_file():
        raise CasesError(f"No eval set at {path}.")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    all_demos = [Demo(d["name"], d["file"], d.get("player")) for d in data.get("demo", [])]
    all_questions = [(q["id"], q["text"]) for q in data.get("question", [])]
    for kind, names in (("demo", [d.name for d in all_demos]), ("question", [q for q, _ in all_questions])):
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise CasesError(f"Duplicate {kind} names in {path}: {sorted(dupes)}")
    for wanted, have, kind in ((demos, [d.name for d in all_demos], "demo"),
                               (questions, [q for q, _ in all_questions], "question")):
        unknown = set(wanted or []) - set(have)
        if unknown:
            raise CasesError(f"Unknown {kind}(s) {sorted(unknown)}. Known: {have}")

    cases = []
    for demo in all_demos:
        if demos and demo.name not in demos:
            continue
        for qid, text in all_questions:
            if questions and qid not in questions:
                continue
            if "{player}" in text and not demo.player:
                continue  # question needs a player this demo doesn't name
            cases.append(Case(f"{demo.name}/{qid}", demo, qid, text.format(player=demo.player or "")))
    if not cases:
        raise CasesError("The filters left no cases to run.")
    return cases
