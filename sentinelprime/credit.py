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

One outcome is worse than noise: a run that reached the right answer *without reading the
source* credits whichever lesson was in the prompt, so a useless lesson keeps a passing
rate and never retires. `observe(..., grounded=...)` takes a per-criterion verdict from
`grounding.grounding_map` and drops those passes. It is opt-in and it only ever *removes*
credit, so with the argument omitted the behavior is exactly what it was.

**This is correlational, not causal, and the distinction matters.** When two lessons target
the same criterion they share its outcome, and nothing here runs the counterfactual where
the lesson was withheld. `min_exposures` is the only guard against retiring on noise. The
honest upgrade is an A/B: withhold a lesson on a matched task and compare — cheap to
express (the ledger is already reversible) but it doubles eval cost, so it is parked until
there is a real benchmark to spend that budget on.
"""
from __future__ import annotations

import re
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

    def observe(self, exposed_ids: list[str], feedback,
                grounded: dict[str, bool | None] | None = None) -> None:
        """Score every exposed lesson against its own targeted criteria for this task.

        `grounded` is the optional lucky-guess filter: `{criterion_id: did the run
        actually read the evidence}`, as produced by `grounding.grounding_map`. A criterion
        that *passed* while the run never saw the facts it names is dropped from the
        record rather than counted as a success — a right answer the run could not have
        derived says nothing about the guidance that was in the prompt, and letting it
        count is how a useless lesson keeps a passing rate.

        Dropped, not blamed: grounding is a lexical proxy, so "cannot prove the run saw
        it" has to mean *no evidence*, not evidence against. Only passes are filtered;
        a failure is a failure however the run reached it. A criterion absent from the
        map, or mapped to None (nothing checkable in its text), is scored normally —
        unknown must never be read as ungrounded.
        """
        outcome = {c.id: c.passed for c in feedback.criteria}
        for item_id in exposed_ids:
            for target in self.targets(item_id):
                if target not in outcome:
                    continue  # this task's rubric never exercised that criterion
                if outcome[target] and grounded is not None \
                        and grounded.get(target) is False:
                    continue  # passed without reading the evidence: not attributable
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


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _facts(text: str) -> set[str]:
    """Checkable literals in a rubric criterion: amounts, section numbers, dates.

    Digits only. Prose ("the memo is well organized") has nothing a trajectory can be
    checked against, which is why such criteria are reported as unjudgeable rather than
    ungrounded.
    """
    out = set()
    for match in _NUMBER_RE.findall(text or ""):
        norm = match.replace(",", "").rstrip(".")
        if len(norm.replace(".", "")) >= 2:   # skip bare single digits — too collidable
            out.add(norm)
    return out


def grounding_from_trajectory(trajectory: list[dict],
                              criterion_texts: dict[str, str]) -> dict[str, bool | None]:
    """Per-criterion: did the *environment* show this fact, or did the model just say it?

    Only the `output` of each REPL turn counts — that is what the interpreter actually
    returned. `reasoning` and `code` are the model's own words, and a number the model
    wrote is precisely what we are trying not to accept as evidence.

    One matching fact is enough. Withholding credit is the strong action, so the bar for
    *not* withholding is deliberately low; the target is the run that cited a figure it
    never looked up, not the run that was merely terse.
    """
    observed = " ".join(str(step.get("output", "")) for step in (trajectory or []))
    observed_norm = observed.replace(",", "")
    verdicts: dict[str, bool | None] = {}
    for cid, text in (criterion_texts or {}).items():
        facts = _facts(text)
        if not facts:
            verdicts[cid] = None          # nothing checkable — not a grounding failure
        else:
            verdicts[cid] = any(f in observed_norm for f in facts)
    return verdicts
