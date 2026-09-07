"""Auditable reasoning ledger: the product's atom.

Every self-improvement edit the ContinualHarness makes becomes a fully
reconstructable, provenance-scoped fact recorded here. The harness already
computes everything an AuditRecord needs — the reversibility window
(before/after snapshot versions), the cause (feedback + trajectory), and the
touched ledger ids — so this layer only *captures* that; it never changes what
refine() does.

The log is append-only: a rolled-back edit is still a fact that happened, so
records are never edited or deleted. Identical edits collapse by content_hash
(common-subexpression elimination / memoization) so the log does not re-append
duplicates.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class AuditRecord:
    edit_id: str            # the ledger item id touched
    op: str                 # "create" | "update" | "delete"
    scope: str              # provenance scope: "intrinsic" | "external"
    from_version: int       # reversibility window start (the rollback point)
    to_version: int         # reversibility window end
    cause_task_id: str      # feedback.task_id that triggered the edit
    cause_failures: str     # verbatim rubric-failure text (feedback.as_text())
    trajectory_digest: str  # sha256 of the trajectory (provenance, not the whole trace)
    score: float            # feedback score (later: verifier logit-expectation)
    created_at: str         # ISO-8601 UTC
    content_hash: str       # sha256(op, edit_id, text, cause_task_id) — dedup key
    # For external-scope records: the mutable sources this edit depended on, mapped
    # to their value at derivation time. Lets the ReuseController detect a moved
    # source. Empty for intrinsic records (they depend only on document invariants).
    depends_on: dict = field(default_factory=dict)
    # The verifier's one-line justification for admitting this edit. Empty when no
    # verifier gated the refine() (admission was unconditional).
    verification: str = ""


def digest(text: str) -> str:
    """Stable short-ish sha256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(op: str, edit_id: str, text: str, cause_task_id: str) -> str:
    """Content-addressed id for an edit (Merkle-style). Identical edits collapse."""
    return digest("\x1f".join([op, edit_id, text, cause_task_id]))


class AuditLog:
    """Append-only, content-addressed audit log persisted as JSON."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._records: list[AuditRecord] = []
        self._hashes: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            raw = json.load(f)
        self._records = [AuditRecord(**r) for r in raw.get("records", [])]
        self._hashes = {r.content_hash for r in self._records}

    def _persist(self) -> None:
        with open(self.path, "w") as f:
            json.dump({"records": [asdict(r) for r in self._records]}, f, indent=2)

    def seen(self, content_hash: str) -> bool:
        return content_hash in self._hashes

    def append(self, record: AuditRecord) -> None:
        # CSE/memoization: an edit already recorded is not re-appended.
        if record.content_hash in self._hashes:
            return
        self._records.append(record)
        self._hashes.add(record.content_hash)
        self._persist()

    def records(self) -> list[AuditRecord]:
        return list(self._records)

    def by_version(self, version: int) -> list[AuditRecord]:
        """Records whose reversibility window ends at `version`."""
        return [r for r in self._records if r.to_version == version]

    def latest_for(self, edit_id: str) -> AuditRecord | None:
        """The most recent record for a ledger id (highest to_version), or None."""
        matches = [r for r in self._records if r.edit_id == edit_id]
        if not matches:
            return None
        return max(matches, key=lambda r: r.to_version)

    def explain(self, version: int) -> str:
        """Render the derivation of the edit(s) at `version` as a compliance chain.

        The chain a compliance officer reads:
            failure -> (verifier justification) -> ledger edit -> reversibility point -> reuse

        `[verification]` and `[reuse]` lines are added by later components; this renders
        whatever fields are present, so it degrades gracefully today.
        """
        recs = self.by_version(version)
        if not recs:
            return f"(no audit records at version {version})"
        blocks: list[str] = []
        for r in recs:
            lines = [
                f"[failure       ] task {r.cause_task_id}: {r.cause_failures}",
            ]
            if r.verification:
                lines.append(f"[verification  ] {r.verification}")
            lines += [
                f"[ledger_edit   ] {r.op} '{r.edit_id}' "
                f"(v{r.from_version}->v{r.to_version}, scope={r.scope}, id={r.content_hash[:8]})",
                f"[reversibility ] rollback(from_version={r.from_version}) "
                f"restores prior ledger state exactly",
            ]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
