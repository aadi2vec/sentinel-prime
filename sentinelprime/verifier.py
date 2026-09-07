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

from dataclasses import dataclass

import dspy


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
