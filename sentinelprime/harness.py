from __future__ import annotations
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import dspy

from sentinelprime.memory import MemoryBackend, MemoryItem
from sentinelprime.feedback import Feedback


@dataclass
class RefineResult:
    created: list[str]
    updated: list[str]
    deleted: list[str]
    from_version: int
    to_version: int


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
    def __init__(self, backend: MemoryBackend):
        super().__init__()
        self.backend = backend
        self.propose = dspy.Predict(ProposeLedgerEdits)

    def read(self, scope: str | None = None) -> str:
        items = self.backend.read(scope=scope)
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

    def _apply_edits(self, edits: list[dict]) -> tuple[list[str], list[str], list[str]]:
        created: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []
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
        before = self.backend.snapshot()
        pred = self.propose(
            trajectory_summary=json.dumps(trajectory)[:4000],
            rubric_failures=feedback.as_text(),
            current_ledger=self.read() or "(empty)",
        )
        edits = json.loads(pred.edits)
        created, updated, deleted = self._apply_edits(edits)
        after = self.backend.snapshot()
        return RefineResult(created, updated, deleted, before.number, after.number)

    def rollback(self, version: int) -> None:
        self.backend.rollback(version)
