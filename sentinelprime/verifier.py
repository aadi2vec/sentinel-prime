"""Verifier admission-control: an edit must be *justified* before it enters the ledger.

refine() proposes edits label-free from the failure text alone, which means a proposer
hallucination would otherwise be persisted as a durable "lesson." The verifier is the
gate that stops that: it re-reads each proposed edit against the same failure/trajectory
and returns a scored, *reasoned* verdict. Only admitted edits are applied and audited.

Two properties matter:

  1. GEPA-optimizable: the verifier is a real dspy.Predict wrapped in a dspy.Module, so
     its judging prompt shows up in named_predictors() and can be tuned offline — the
     online admission bar improves with the same machinery that tunes the proposer.
  2. Auditable: the verdict's justification is threaded into the AuditRecord, so a
     compliance officer reading explain() sees *why* an edit was allowed in, not just that
     it was.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import dspy

from sentinelprime.planner import Check, CostLadderPlanner


@dataclass
class VerifierVerdict:
    admitted: bool
    score: float
    justification: str


class VerifyLedgerEdit(dspy.Signature):
    """Judge whether a proposed supplemental-ledger edit is justified by the observed
    rubric failures. Admit only edits that are directly grounded in a failure and that do
    not override the base task. Reject speculative, redundant, or task-contradicting edits.
    Return admit ("yes"/"no"), a 0..1 confidence score, and a one-sentence justification.
    """

    proposed_edit: str = dspy.InputField(desc="JSON of a single edit op")
    rubric_failures: str = dspy.InputField()
    trajectory_summary: str = dspy.InputField()
    admit: str = dspy.OutputField(desc='"yes" or "no"')
    score: str = dspy.OutputField(desc="confidence in [0,1]")
    justification: str = dspy.OutputField(desc="one sentence")


def _to_float(raw: str) -> float:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return 0.0


class PredictVerifier(dspy.Module):
    def __init__(self, min_score: float = 0.0) -> None:
        super().__init__()
        self.min_score = min_score
        self.verify_predict = dspy.Predict(VerifyLedgerEdit)

    def verify(self, op: dict, feedback, trajectory: list[dict]) -> VerifierVerdict:
        import json

        pred = self.verify_predict(
            proposed_edit=json.dumps(op),
            rubric_failures=feedback.as_text(),
            trajectory_summary=json.dumps(trajectory)[:4000],
        )
        score = _to_float(pred.score)
        admit = str(pred.admit).strip().lower() in ("yes", "true", "1")
        admitted = admit and score >= self.min_score
        return VerifierVerdict(
            admitted=admitted, score=score, justification=pred.justification
        )


@dataclass
class VerifierLevel:
    """One rung of the ladder: a named, priced check with a `.verify` method.

    `verifier` is anything exposing `verify(op, feedback, trajectory) -> VerifierVerdict`
    — a deterministic probe or the generative `PredictVerifier` compose identically.
    """

    name: str
    cost: float
    verifier: Any


class LadderVerifier(dspy.Module):
    """Admission control as a short-circuited conjunctive cost ladder.

    `refine()` calls a single `verifier.verify(...)` per edit, so this presents the
    *same* interface and fans out internally — the harness's semantics are unchanged.
    What it adds is ordering: verification is asymmetric (a deterministic span probe is
    orders of magnitude cheaper than a generative judge and catches most rejections), so
    `CostLadderPlanner` runs the levels by `cost / P(fail)` ascending and stops at the
    first rejection. An edit is admitted only if *every* level admits it.

    The executed order, the level that rejected, and the cost actually spent are written
    into the verdict's justification, so `refine()` threads them into the `AuditRecord` and
    `explain()` shows a compliance officer which checks ran and which one stopped the edit.

    Ordering statistics come from the planner. Three sources, in increasing order of
    honesty about this system:

      - the default planner has no stats, so it orders by raw cost ascending;
      - `CostLadderPlanner.from_audit_log(...)` derives P(fail) per *rubric criterion*,
        which fits a ladder whose rungs are criterion-specific probes;
      - `adaptive=True` derives P(fail) from the ladder's **own** observed rejections.
        This is the one that needs no mapping and no extra bookkeeping: rejected edits
        deliberately leave no audit record, so the ladder is the only component that sees
        them, which makes it the statistics catalog for its own ordering.

    Known limitation of the adaptive path: a rung that reaches the front and rejects
    short-circuits the rungs behind it, so their rates freeze at whatever was observed
    while they still ran. That is the standard exploration cost of a greedy ladder; with
    two or three rungs it is not worth a bandit.
    """

    def __init__(self, levels: list[VerifierLevel], planner: CostLadderPlanner | None = None,
                 adaptive: bool = False):
        super().__init__()
        self.levels = list(levels)
        # A plain list attribute so DSPy's module traversal reaches nested predictors —
        # a PredictVerifier rung stays GEPA-optimizable through the ladder.
        self.verifiers = [lv.verifier for lv in self.levels]
        self.planner = planner or CostLadderPlanner()
        self.adaptive = adaptive
        self._runs: dict[str, int] = {}
        self._rejections: dict[str, int] = {}

    def observed_stats(self) -> dict[str, float]:
        """Per-rung rejection rate over the edits that rung has actually judged."""
        return {name: self._rejections.get(name, 0) / runs
                for name, runs in self._runs.items() if runs}

    def _planner_for_round(self) -> CostLadderPlanner:
        if not self.adaptive:
            return self.planner
        # Observed rates override the configured ones; unobserved rungs keep whatever the
        # planner was given (or its default_fail_prob).
        stats = dict(self.planner.stats)
        stats.update(self.observed_stats())
        return CostLadderPlanner(stats=stats,
                                 default_fail_prob=self.planner.default_fail_prob)

    def _check_for(self, level: VerifierLevel, verdicts: dict[str, VerifierVerdict],
                   op: dict, feedback, trajectory: list[dict]) -> Check:
        def run(_context: dict) -> bool:
            verdict = level.verifier.verify(op, feedback, trajectory)
            verdicts[level.name] = verdict
            self._runs[level.name] = self._runs.get(level.name, 0) + 1
            if not verdict.admitted:
                self._rejections[level.name] = self._rejections.get(level.name, 0) + 1
            return verdict.admitted

        return Check(name=level.name, cost=level.cost, run=run)

    def verify(self, op: dict, feedback, trajectory: list[dict]) -> VerifierVerdict:
        verdicts: dict[str, VerifierVerdict] = {}
        checks = [self._check_for(lv, verdicts, op, feedback, trajectory)
                  for lv in self.levels]
        result = self._planner_for_round().run(checks, {"op": op})
        # Only the levels that actually ran, in the order the ladder ran them.
        trail = " -> ".join(name for name in result.order if name in verdicts)

        if not result.passed:
            rejected = verdicts[result.failed_at]
            return VerifierVerdict(
                admitted=False,
                score=rejected.score,
                justification=(f"ladder[{trail}] rejected at {result.failed_at} "
                               f"(cost {result.cost_spent:g}): {rejected.justification}"),
            )
        # Conjunctive admission: the weakest rung sets the score.
        score = min((v.score for v in verdicts.values()), default=0.0)
        why = "; ".join(f"{name}: {verdicts[name].justification}"
                        for name in result.order if name in verdicts)
        return VerifierVerdict(
            admitted=True,
            score=score,
            justification=f"ladder[{trail}] admitted (cost {result.cost_spent:g}): {why}",
        )


_TOKEN_RE = re.compile(r"\w+")


class GroundingProbe:
    """A deterministic, LM-free rung: is this edit even *about* an observed failure?

    The proposer runs label-free, so its two cheap failure modes are structural (an edit
    with no usable text) and topical (a plausible-sounding lesson that no observed failure
    supports). Both are catchable without a model: check well-formedness, then measure what
    fraction of the edit's vocabulary appears in the rubric-failure text.

    Lexical overlap is a coarse proxy for grounding — deliberately so. It is the *cheap*
    rung of the ladder, priced to run first and short-circuit the expensive generative
    judge on the obvious rejections; anything it admits still has to clear that judge.
    Delete ops carry no text to ground, so they pass through to the next rung.
    """

    def __init__(self, min_overlap: float = 0.3) -> None:
        self.min_overlap = min_overlap

    def named_predictors(self):
        return []  # no LM call: nothing for GEPA to tune here

    def verify(self, op: dict, feedback, trajectory: list[dict]) -> VerifierVerdict:
        if op.get("op") == "delete":
            return VerifierVerdict(True, 1.0, "grounding: delete op, deferred to next rung")

        text = (op.get("text") or "").strip()
        if not text:
            return VerifierVerdict(False, 0.0, "grounding: edit carries no text")

        edit_tokens = set(_TOKEN_RE.findall(text.lower()))
        failure_tokens = set(_TOKEN_RE.findall(feedback.as_text().lower()))
        overlap = len(edit_tokens & failure_tokens) / len(edit_tokens) if edit_tokens else 0.0
        if overlap < self.min_overlap:
            return VerifierVerdict(
                False, overlap,
                f"grounding: {overlap:.2f} of the edit's vocabulary appears in the "
                f"observed failures, below {self.min_overlap:.2f}",
            )
        return VerifierVerdict(
            True, overlap,
            f"grounding: {overlap:.2f} vocabulary overlap with the observed failures",
        )
