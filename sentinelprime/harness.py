"""ContinualHarness: online, label-free, reversible self-improvement for a DSPy program.

The three properties that make this more than "an LLM editing a JSON file" — and that the
rest of the code exists to guarantee — are:

  1. Online + label-free: refine() runs after any live task using only the trajectory and the
     rubric-failure *text*. It never needs gold answers, so it can improve on unlabeled traffic
     (this is the gap vs. dspy.GEPA, which is offline and needs a labeled trainset).
  2. Reversible: every refine() brackets its edits between two backend snapshots, so any change
     can be undone with rollback(). GEPA cannot offer this — it rewrites prompts in place.
  3. Supplemental-only: the base task prompt is immutable. read() emits an *additional* block;
     it is prepended to instructions, never a substitute for them.

Known limitations (deliberately deferred — see docs/superpowers/specs): there is no credit
assignment yet (edits are applied whether or not they later help the rubric), no relevance
retrieval in read() (it serializes the whole ledger), and no decay/dedup. Those are what turn
this from a minimal core into the real mechanism.
"""
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import dspy

from sentinelprime.audit import AuditLog, AuditRecord, content_hash, digest
from sentinelprime.memory import MemoryBackend, MemoryItem
from sentinelprime.feedback import Feedback
from sentinelprime.reuse import ReuseController


@dataclass
class RefineResult:
    # created/updated/deleted are the ledger item ids touched this round.
    # from_version..to_version is the reversibility window: rollback(from_version) undoes the round.
    created: list[str]
    updated: list[str]
    deleted: list[str]
    from_version: int
    to_version: int
    # Edit ids the verifier refused. They never touched the ledger and produced no audit
    # record, so this is the only place the gate's work is countable — which is what makes
    # admission control ablatable in a benchmark. Empty when no verifier is configured.
    rejected: list[str] = field(default_factory=list)
    # Ledger items retired this round because the failures they targeted kept recurring
    # while they were in the prompt. Retirement happens inside the same reversibility
    # window, so rollback(from_version) restores them along with everything else.
    retired: list[str] = field(default_factory=list)


# This Signature is a real dspy.Predict predictor, so the harness's own edit-proposing prompt
# is itself GEPA-optimizable via named_predictors() — the online loop can be tuned offline.
class ProposeLedgerEdits(dspy.Signature):
    """Propose small, additive edits to a supplemental memory ledger so a future run
    avoids the rubric failures just observed. Never rewrite the base task; only add,
    refine, or remove supplemental notes / memories / reusable sub-agent specs.
    Return a JSON list of edit ops. Each op is one of:
      {"op":"create","id":<str>,"kind":"note|memory|sub_agent_spec","text":<str>,"scope":"session|global","meta":{"targets":["<criterion id>",...]}}
    Always set meta.targets to the rubric criterion ids the edit is meant to fix (e.g.
    ["c1"]). That is what lets a later round tell whether this specific edit helped.
      {"op":"update","id":<existing id>,"kind":...,"text":<str>,"scope":...}
      {"op":"delete","id":<existing id>}
    """
    trajectory_summary: str = dspy.InputField()
    rubric_failures: str = dspy.InputField()
    current_ledger: str = dspy.InputField()
    edits: str = dspy.OutputField(desc="JSON list of edit ops")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContinualHarness(dspy.Module):
    def __init__(self, backend: MemoryBackend, audit_log: AuditLog | None = None,
                 reuse_controller: ReuseController | None = None,
                 verifier=None, credit_assigner=None):
        super().__init__()
        self.backend = backend
        self.audit_log = audit_log
        self.reuse_controller = reuse_controller
        # Optional credit assignment: scores exposed lessons against the criteria they
        # were written to fix and retires the ones that stopped earning their place.
        # None -> lessons are kept unconditionally (the base behavior).
        self.credit_assigner = credit_assigner
        # Optional admission-control gate: when set, each proposed edit must clear the
        # verifier before it is applied and audited. None -> unconditional admission
        # (the base behavior; rollback() remains the safety net).
        self.verifier = verifier
        self.propose = dspy.Predict(ProposeLedgerEdits)

    def admissible_items(self, scope: str | None = None,
                         context: dict | None = None) -> list[MemoryItem]:
        """The ledger items `read()` would surface in this context.

        Split out from `read()` so callers can *count* what reuse gating withheld —
        the rendered prompt block alone makes that unmeasurable, and an ablation of the
        gate needs the number, not the prose.
        """
        items = self.backend.read(scope=scope)
        if context is None:
            # No context -> ungated (base behavior preserved).
            return items
        # Reuse gating: drop items whose provenance says they are inadmissible in the
        # current context (e.g. an external note whose source has moved). Items with no
        # audit record are not blocked.
        return [it for it in items if self._admissible(it, context)]

    def read(self, scope: str | None = None, context: dict | None = None) -> str:
        # Serialize the ledger into a supplemental prompt block. Empty ledger -> "" so that
        # nothing is prepended and the base prompt is used verbatim (the immutability invariant).
        # NOTE: this emits *every* admissible item; there is no relevance ranking yet, so it does
        # not scale to large ledgers. Relevance-ranked recall is the planned TraceMind-backend job.
        items = self.admissible_items(scope=scope, context=context)
        if not items:
            return ""
        # Prefix-cache guardrail: emit in a deterministic (kind, id) order so the block is
        # a byte-stable prompt prefix across reads. The backend does not promise an order,
        # so pin it here — otherwise provider prompt caching silently invalidates.
        items = sorted(items, key=lambda i: i.id)
        notes = [i for i in items if i.kind == "note"]
        memories = [i for i in items if i.kind == "memory"]
        specs = [i for i in items if i.kind == "sub_agent_spec"]
        out: list[str] = ["## Learned guidance (supplemental — do not override the task)"]
        if notes:
            out.append("### Notes")
            out += [f"- {n.text}" for n in notes]
        if memories:
            out.append("### Memories")
            out += [f"- {m.text}" for m in memories]
        if specs:
            out.append("### Reusable sub-agents")
            for s in specs:
                name = s.meta.get("name", s.id)
                when = s.meta.get("when_to_use", "")
                out.append(f"- {name}: {when} — {s.text}")
        return "\n".join(out)

    def _admissible(self, item: MemoryItem, context: dict) -> bool:
        # Gate a ledger item by its latest audit record's reuse decision. Absent a
        # record (e.g. externally seeded items) or a configured audit log/controller,
        # the item is retained — gating only ever *removes* provably-stale reuse.
        if self.audit_log is None:
            return True
        record = self.audit_log.latest_for(item.id)
        if record is None:
            return True
        controller = self.reuse_controller or ReuseController()
        return controller.should_reuse(record, context).reuse

    def _apply_edits(self, edits: list[dict]) -> tuple[list[str], list[str], list[str]]:
        # Deterministic, no LM call — this is the part we can unit-test exhaustively. refine()
        # keeps the (nondeterministic) LM call in propose() and hands the parsed ops to this.
        created: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []
        # Snapshot the id set up front so a create vs. update is classified against the pre-batch
        # state (backend.write is an upsert, so it can't distinguish the two on its own).
        existing = {i.id for i in self.backend.read()}
        for op in edits:
            kind = op.get("op")
            if kind == "delete":
                self.backend.delete([op["id"]])
                deleted.append(op["id"])
                continue
            item_id = op.get("id") or str(uuid.uuid4())
            item = MemoryItem(
                id=item_id,
                scope=op.get("scope", "global"),
                kind=op["kind"],
                text=op["text"],
                created_at=_now(),
                meta=op.get("meta", {}),
            )
            self.backend.write([item])
            (updated if item_id in existing else created).append(item_id)
        return created, updated, deleted

    def credit(self, exposed_ids: list[str], feedback: Feedback,
               grounded: dict[str, bool | None] | None = None) -> None:
        """Record this task's outcome against the guidance that was actually surfaced.

        Separate from refine() because exposure and outcome are known at *task* time,
        while retirement is a ledger edit that belongs in refine()'s snapshot window.

        `grounded` is passed straight to the assigner's lucky-guess filter; None leaves
        credit exactly as it was before that filter existed.
        """
        if self.credit_assigner is not None:
            self.credit_assigner.observe(exposed_ids, feedback, grounded=grounded)

    def _retire(self, feedback: Feedback, from_version: int, to_version: int) -> list[str]:
        # Only retire what is actually in the ledger; the assigner may still hold
        # observations for items removed by some earlier round.
        present = {i.id for i in self.backend.read()}
        retired = [i for i in self.credit_assigner.retirable() if i in present]
        if not retired:
            return []
        self.backend.delete(retired)
        if self.audit_log is not None:
            for item_id in retired:
                why = self.credit_assigner.explain(item_id)
                self.audit_log.append(AuditRecord(
                    edit_id=item_id,
                    op="retire",
                    scope="intrinsic",
                    from_version=from_version,
                    to_version=to_version,
                    cause_task_id=feedback.task_id,
                    cause_failures=why,
                    trajectory_digest="",
                    score=feedback.score,
                    created_at=_now(),
                    content_hash=content_hash("retire", item_id, why, feedback.task_id),
                ))
        # A rewrite of a retired lesson deserves a fresh trial, not inherited blame.
        self.credit_assigner.forget(retired)
        return retired

    def refine(self, trajectory: list[dict], feedback: Feedback) -> RefineResult:
        # The reversibility protocol: snapshot BEFORE and AFTER the edits, so `before.number`
        # names the exact restore point that undoes this whole round.
        before = self.backend.snapshot()
        # Label-free seam: the only signals are the trajectory and the rubric-failure *text*
        # (feedback.as_text()). No gold labels enter here — that is what lets this run online.
        pred = self.propose(
            trajectory_summary=json.dumps(trajectory)[:4000],  # bound prompt size on long runs
            rubric_failures=feedback.as_text(),
            current_ledger=self.read() or "(empty)",
        )
        # NOTE: edits are applied unconditionally. There is no check that they improved anything —
        # credit assignment (keep a lesson only if it later raises the pass-rate) is the key
        # mechanism still to build. Until then, rollback() is the manual safety net.
        edits = json.loads(pred.edits)
        # Pre-assign ids for create ops that omit one, so every edit is traceable to an
        # AuditRecord (otherwise _apply_edits would generate an id we couldn't observe).
        for op in edits:
            if op.get("op") != "delete" and not op.get("id"):
                op["id"] = str(uuid.uuid4())
        # Admission control: a configured verifier gates each edit before it is applied.
        # Rejected edits never touch the ledger and never produce an audit record — the
        # proposer's suggestion simply did not clear the bar. edit_id -> justification for
        # admitted edits threads the verifier's reasoning into the audit trail.
        verifications: dict[str, str] = {}
        rejected: list[str] = []
        if self.verifier is not None:
            admitted: list[dict] = []
            for op in edits:
                verdict = self.verifier.verify(op, feedback, trajectory)
                if verdict.admitted:
                    verifications[op["id"]] = verdict.justification
                    admitted.append(op)
                else:
                    rejected.append(op["id"])
            edits = admitted
        created, updated, deleted = self._apply_edits(edits)
        # Retire inside the same window as the additions, so one rollback undoes the whole
        # round — a lesson removed on weak evidence is as recoverable as one added on it.
        retired = (self._retire(feedback, before.number, before.number + 1)
                   if self.credit_assigner is not None else [])
        after = self.backend.snapshot()
        if self.audit_log is not None:
            self._emit_audit(edits, feedback, trajectory, before.number, after.number,
                             verifications)
        return RefineResult(created, updated, deleted, before.number, after.number,
                            rejected, retired)

    @staticmethod
    def _provenance_scope(op: dict) -> str:
        # Explicit override wins; otherwise map ledger scope -> validity boundary.
        # global guidance depends on document invariants (reusable across sessions =
        # intrinsic); session guidance depends on mutable state (external).
        override = op.get("meta", {}).get("provenance")
        if override in ("intrinsic", "external"):
            return override
        return "intrinsic" if op.get("scope") == "global" else "external"

    def _emit_audit(self, edits: list[dict], feedback: Feedback, trajectory: list[dict],
                    from_version: int, to_version: int,
                    verifications: dict[str, str] | None = None) -> None:
        traj_digest = digest(json.dumps(trajectory)[:4000])
        failures = feedback.as_text()
        verifications = verifications or {}
        for op in edits:
            kind = op.get("op")
            edit_id = op["id"]
            text = op.get("text", "")
            self.audit_log.append(AuditRecord(
                edit_id=edit_id,
                op=kind,
                scope=self._provenance_scope(op),
                from_version=from_version,
                to_version=to_version,
                cause_task_id=feedback.task_id,
                cause_failures=failures,
                trajectory_digest=traj_digest,
                score=feedback.score,
                created_at=_now(),
                content_hash=content_hash(kind, edit_id, text, feedback.task_id),
                depends_on=op.get("meta", {}).get("depends_on", {}),
                verification=verifications.get(edit_id, ""),
                targets=list(op.get("meta", {}).get("targets", []) or []),
            ))

    def rollback(self, version: int) -> None:
        # Restore the ledger to a snapshot taken by refine() (typically RefineResult.from_version).
        self.backend.rollback(version)

    def explain(self, version: int) -> str:
        # The compliance UI: reconstruct why the ledger changed at `version`.
        if self.audit_log is None:
            return "(no audit log configured)"
        return self.audit_log.explain(version)
