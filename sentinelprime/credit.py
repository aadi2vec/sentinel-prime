"""Credit assignment: keep a lesson only while the failures it targeted keep clearing.

`refine()` writes a lesson because specific rubric criteria failed, and then — without
this — keeps it forever, whether or not it ever helped. That is the difference between a
ledger that learns and a ledger that only accumulates. A polluted ledger is worse than an
empty one: it costs prompt budget on every run and can actively mislead.

The attribution is label-free, which is what lets it run online. Two facts the system
already records are enough:

  1. **What the lesson was for.** The `AuditRecord` written at creation carries the verbatim
     failure text, so the criterion ids it targeted (`- [c1] …`) are recoverable.
  2. **Whether it was in the room.** `harness.admissible_items()` is the exact set of
     guidance surfaced for a task, so a lesson is only judged on runs where it could
     actually have had an effect.

After each task, every exposed lesson is scored against *its own* targeted criteria:
exposure + pass = credit, exposure + fail = blame. A lesson whose criteria keep failing
while it sits in the prompt is not earning its place and becomes retirable.

**This is correlational, not causal, and the distinction matters.** When two lessons target
the same criterion they share its outcome, and nothing here runs the counterfactual where
the lesson was withheld. `min_exposures` is the only guard against retiring on noise. The
honest upgrade is an A/B: withhold a lesson on a matched task and compare — cheap to
express (the ledger is already reversible) but it doubles eval cost, so it is parked until
there is a real benchmark to spend that budget on.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from sentinelprime.planner import criterion_ids


@dataclass
class CreditRecord:
    """Recent outcomes for one ledger item on the criteria it was written to fix.

    A bounded window, not a lifetime tally. The environment is non-stationary — another
    lesson can poison a criterion and later be retired — so a lifetime average would
    convict a lesson for a period that has already been fixed, and it could never recover.
    """

    outcomes: deque = field(default_factory=lambda: deque(maxlen=10))

    @property
    def exposures(self) -> int:
        return len(self.outcomes)

    @property
    def successes(self) -> int:
        return sum(self.outcomes)

    @property
    def success_rate(self) -> float:
        return self.successes / self.exposures if self.exposures else 0.0


class CreditAssigner:
    def __init__(self, audit_log, min_exposures: int = 3,
                 min_success_rate: float = 0.5, window: int = 10) -> None:
        self.audit_log = audit_log
        # Retire only on evidence: below the rate, and only after enough exposures that
        # the rate means something. Raising min_exposures trades reaction speed for noise.
        self.min_exposures = min_exposures
        self.min_success_rate = min_success_rate
        # How many recent exposures a verdict rests on. Short enough to react when the
        # ledger changes around a lesson; long enough that one bad task is not a verdict.
        self.window = window
        self._records: dict[str, CreditRecord] = {}

    def targets(self, item_id: str) -> set[str]:
        """The criterion ids this ledger item was written to fix, per its audit record.

        Prefers what the proposer *declared* (`meta.targets`). The fallback — every
        criterion that failed in the round that produced the edit — is coarse and
        cross-contaminating: it blames a lesson answering c1 whenever c2 also fails, which
        is enough to retire perfectly good guidance. Declare targets.
        """
        record = self.audit_log.latest_for(item_id)
        if record is None:
            return set()  # unattributable (e.g. externally seeded) -> never judged
        if record.targets:
            return set(record.targets)
        return criterion_ids(record.cause_failures)

    def observe(self, exposed_ids: list[str], feedback) -> None:
        """Score every exposed lesson against its own targeted criteria for this task."""
        outcome = {c.id: c.passed for c in feedback.criteria}
        for item_id in exposed_ids:
            for target in self.targets(item_id):
                if target not in outcome:
                    continue  # this task's rubric never exercised that criterion
                rec = self._records.get(item_id)
                if rec is None:
                    rec = CreditRecord(outcomes=deque(maxlen=self.window))
                    self._records[item_id] = rec
                rec.outcomes.append(bool(outcome[target]))

    def forget(self, item_ids: list[str]) -> None:
        """Drop observations for retired items.

        A lesson the proposer writes again after a retirement is a *new* attempt at the
        same problem. Carrying the old record forward would retire it on sight, and the
        ledger would oscillate — write, retire, rewrite — instead of converging.
        """
        for item_id in item_ids:
            self._records.pop(item_id, None)

    def stats(self) -> dict[str, CreditRecord]:
        return dict(self._records)

    def success_rate(self, item_id: str) -> float | None:
        """None when nothing attributable has been observed yet — not zero."""
        rec = self._records.get(item_id)
        return rec.success_rate if rec and rec.exposures else None

    def retirable(self) -> list[str]:
        return sorted(
            item_id for item_id, rec in self._records.items()
            if rec.exposures >= self.min_exposures
            and rec.success_rate < self.min_success_rate
        )

    def explain(self, item_id: str) -> str:
        rec = self._records.get(item_id)
        if rec is None or not rec.exposures:
            return "no attributable observations"
        return (f"targeted criteria passed on {rec.successes}/{rec.exposures} exposures "
                f"(rate {rec.success_rate:.2f} < {self.min_success_rate:.2f})")
