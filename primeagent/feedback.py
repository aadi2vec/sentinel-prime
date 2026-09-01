from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CriterionResult:
    id: str
    passed: bool
    reason: str


@dataclass
class Feedback:
    task_id: str
    score: float
    criteria: list[CriterionResult]

    @property
    def failures(self) -> list[CriterionResult]:
        return [c for c in self.criteria if c.passed is False]

    def as_text(self) -> str:
        if not self.failures:
            return "All rubric criteria passed."
        lines = [f"- [{c.id}] {c.reason}" for c in self.failures]
        return "Failed rubric criteria:\n" + "\n".join(lines)


def parse_lab_result(raw: dict) -> Feedback:
    criteria = [
        CriterionResult(id=c["id"], passed=bool(c["passed"]), reason=c.get("reason", ""))
        for c in raw.get("criteria", [])
    ]
    score = (sum(1 for c in criteria if c.passed) / len(criteria)) if criteria else 0.0
    return Feedback(task_id=raw["task_id"], score=score, criteria=criteria)
