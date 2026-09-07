"""Cost-ladder planner: asymmetric verification, ordered by the audit ledger.

Verifying an edit is asymmetric — a deterministic probe ("does the cited span
exist? does the citation resolve?") is orders of magnitude cheaper than the
generative verifier, and most rejections can be caught by a cheap probe. So the
right policy is a *conjunctive short-circuit ladder*: run the checks that catch
the most failures per unit cost first, and stop at the first failure.

The ordering is not static. The audit log is the statistics catalog (which
criteria fail, how often), so `from_audit_log` derives each check's empirical
failure probability and the planner orders by cost / P(fail) ascending — the
classic predicate-ordering result for a short-circuited conjunction: the check
with the best detection-per-dollar goes first. This is strictly cheaper in
expectation than an arbitrary (e.g. expensive-verifier-first) static order.

The planner decides *ordering and stopping only*; each Check owns its own logic,
so a deterministic probe and the generative verifier compose in the same ladder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Check:
    name: str
    cost: float
    run: Callable[[dict], bool]  # returns True if the edit/state passes this check


@dataclass
class LadderResult:
    passed: bool
    failed_at: str | None      # name of the first check that failed, or None
    cost_spent: float          # cost actually incurred (short-circuited)
    order: list[str]           # the executed order, cheapest-detection first


_CRITERION_RE = re.compile(r"\[([^\]]+)\]")


def criterion_ids(text: str) -> set[str]:
    """The rubric criterion ids named in a failure text (``- [c1] missed …`` -> {"c1"})."""
    return set(_CRITERION_RE.findall(text or ""))


def criterion_failure_counts(audit_log) -> dict[str, int]:
    """Count how often each rubric criterion id appears in the ledger's failures.

    `cause_failures` is rendered as lines like ``- [c1] missed clause``; we
    extract the bracketed criterion ids. One count per record a criterion
    appears in (not per mention), so a record is one observed failure event.
    """
    counts: dict[str, int] = {}
    for rec in audit_log.records():
        seen = criterion_ids(rec.cause_failures)
        for cid in seen:
            counts[cid] = counts.get(cid, 0) + 1
    return counts


class CostLadderPlanner:
    def __init__(self, stats: dict[str, float] | None = None,
                 default_fail_prob: float = 0.5) -> None:
        # stats maps check.name -> empirical P(this check fails / catches a failure).
        self.stats = stats or {}
        self.default_fail_prob = default_fail_prob

    def fail_prob(self, check: Check) -> float:
        return self.stats.get(check.name, self.default_fail_prob)

    def order(self, checks: list[Check]) -> list[Check]:
        # Ascending cost / P(fail): best detection-per-dollar runs first. A near-zero
        # fail_prob is floored so a check that never catches anything sinks to the end
        # rather than dividing by zero.
        return sorted(checks, key=lambda c: c.cost / max(self.fail_prob(c), 1e-9))

    def expected_cost(self, checks: list[Check]) -> float:
        # Short-circuited conjunction: pay check i only if every earlier check passed.
        # E[cost] = sum_i cost_i * prod_{j<i} (1 - P(fail_j)).
        reach = 1.0
        total = 0.0
        for c in checks:
            total += c.cost * reach
            reach *= (1.0 - self.fail_prob(c))
        return total

    def run(self, checks: list[Check], context: dict) -> LadderResult:
        ordered = self.order(checks)
        spent = 0.0
        for c in ordered:
            spent += c.cost
            if not c.run(context):
                return LadderResult(False, c.name, spent, [x.name for x in ordered])
        return LadderResult(True, None, spent, [x.name for x in ordered])

    @classmethod
    def from_audit_log(cls, audit_log, name_to_criterion: dict[str, str],
                       default_fail_prob: float = 0.5) -> "CostLadderPlanner":
        counts = criterion_failure_counts(audit_log)
        total = max(len(audit_log.records()), 1)
        stats = {
            name: counts.get(cid, 0) / total
            for name, cid in name_to_criterion.items()
        }
        return cls(stats=stats, default_fail_prob=default_fail_prob)
