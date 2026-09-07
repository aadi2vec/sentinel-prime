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
from dataclasses import dataclass
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


# This Signature is a real dspy.Predict predictor, so the harness's own edit-proposing prompt
# is itself GEPA-optimizable via named_predictors() — the online loop can be tuned offline.
class ProposeLedgerEdits(dspy.Signature):
    """Propose small, additive edits to a supplemental memory ledger so a future run
    avoids the rubric failures just observed. Never rewrite the base task; only add,
    refine, or remove supplemental notes / memories / reusable sub-agent specs.
    Return a JSON list of edit ops. Each op is one of:
      {"op":"create","id":<str>,"kind":"note|memory|sub_agent_spec","text":<str>,"scope":"session|global","meta":{...}}
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
                 verifier=None):
        super().__init__()
        self.backend = backend
        self.audit_log = audit_log
        self.reuse_controller = reuse_controller
        # Optional admission-control gate: when set, each proposed edit must clear the
        # verifier before it is applied and audited. None -> unconditional admission
        # (the base behavior; rollback() remains the safety net).
        self.verifier = verifier
        self.propose = dspy.Predict(ProposeLedgerEdits)

    def read(self, scope: str | None = None, context: dict | None = None) -> str:
        # Serialize the ledger into a supplemental prompt block. Empty ledger -> "" so that
        # nothing is prepended and the base prompt is used verbatim (the immutability invariant).
        # NOTE: this emits *every* admissible item; there is no relevance ranking yet, so it does
        # not scale to large ledgers. Relevance-ranked recall is the planned TraceMind-backend job.
        items = self.backend.read(scope=scope)
        if context is not None:
            # Reuse gating: drop items whose provenance says they are inadmissible in the
            # current context (e.g. an external note whose source has moved). Items with no
            # audit record are not blocked. No context -> ungated (base behavior preserved).
            items = [it for it in items if self._admissible(it, context)]
        if not items:
            return ""
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
        if self.verifier is not None:
            admitted: list[dict] = []
            for op in edits:
                verdict = self.verifier.verify(op, feedback, trajectory)
                if verdict.admitted:
                    verifications[op["id"]] = verdict.justification
                    admitted.append(op)
            edits = admitted
        created, updated, deleted = self._apply_edits(edits)
        after = self.backend.snapshot()
        if self.audit_log is not None:
            self._emit_audit(edits, feedback, trajectory, before.number, after.number,
                             verifications)
        return RefineResult(created, updated, deleted, before.number, after.number)

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
            ))

    def rollback(self, version: int) -> None:
        # Restore the ledger to a snapshot taken by refine() (typically RefineResult.from_version).
        self.backend.rollback(version)

    def explain(self, version: int) -> str:
        # The compliance UI: reconstruct why the ledger changed at `version`.
        if self.audit_log is None:
            return "(no audit log configured)"
        return self.audit_log.explain(version)
