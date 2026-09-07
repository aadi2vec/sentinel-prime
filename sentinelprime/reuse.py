"""ReuseController: an auditable semantic cache — a controller, not a lookup.

Naive semantic caching (reuse on cosine > threshold) is reckless in a regulated
setting: a near-match can be wrong, and nothing invalidates it when the world
moves. This controller decides whether reusing a prior conclusion is
*admissible*, running three gates in order:

  1. scope validity  — intrinsic conclusions depend only on document invariants
     (reusable across sessions); external ones depend on mutable state and are
     valid only while every declared dependency still holds.
  2. causal currency — (stub, returns True) a conclusion is stale iff a source it
     depended on has advanced past its derivation point. Real implementation waits
     until multiple mutable versioned sources exist (see plan: parked vector clocks).
  3. verifier admission — the conclusion must have cleared the verifier. Until the
     generative verifier lands, the audit record's score against `min_score` is the
     proxy.

The same gates that make reuse safe are what a compliance officer inspects, so
`should_reuse` returns a reasoned decision, not a bare bool.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sentinelprime.audit import AuditRecord


@dataclass
class ReuseDecision:
    reuse: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.reuse


class ReuseController:
    def __init__(self, min_score: float = 0.0) -> None:
        self.min_score = min_score

    def scope_valid(self, record: AuditRecord, context: dict) -> tuple[bool, str]:
        if record.scope == "intrinsic":
            return True, "scope=intrinsic (document-invariant, reusable)"
        # external: every declared dependency must still match the present state.
        if not record.depends_on:
            return False, "scope=external with no declared dependencies — currency unprovable"
        for source, derived_value in record.depends_on.items():
            if context.get(source) != derived_value:
                return False, (
                    f"scope=external source '{source}' moved "
                    f"({derived_value!r} -> {context.get(source)!r})"
                )
        return True, "scope=external, all dependencies current"

    def causally_current(self, record: AuditRecord, context: dict) -> tuple[bool, str]:
        # Stub: real causal-currency check waits on version vectors over multiple
        # sources. Today there is one sequential writer, so nothing is causally stale.
        return True, "causal currency: not yet enforced (single writer)"

    def verifier_admitted(self, record: AuditRecord) -> tuple[bool, str]:
        if record.score < self.min_score:
            return False, f"verifier: score {record.score} < min_score {self.min_score}"
        return True, f"verifier: score {record.score} >= min_score {self.min_score}"

    def should_reuse(self, record: AuditRecord, current_context: dict) -> ReuseDecision:
        reasons: list[str] = []
        for gate in (
            lambda: self.scope_valid(record, current_context),
            lambda: self.causally_current(record, current_context),
            lambda: self.verifier_admitted(record),
        ):
            ok, why = gate()
            reasons.append(why)
            if not ok:
                return ReuseDecision(reuse=False, reasons=reasons)
        return ReuseDecision(reuse=True, reasons=reasons)
