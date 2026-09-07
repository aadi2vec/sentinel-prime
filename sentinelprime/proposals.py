"""The proposal log: the training record the compliance ledger refuses to be.

`AuditLog` is deliberately incomplete as a dataset. A rejected edit never entered the
ledger, so it is not a fact *about* the ledger, and recording it there would dilute the
one thing the audit trail is for — reconstructing why the guidance a run actually used was
there. That invariant is load-bearing and stays.

It also makes the audit log useless for tuning the gate. Every audit record is an edit the
verifier admitted, so a trainset drawn from it has no negatives, and a classifier fit to it
learns "admit everything." The rejections exist only as a count on `LadderVerifier._runs`
and as bare ids on `RefineResult.rejected` — neither carries the edit text, so neither can
be replayed. That is the gap this module closes, and it is why the "GEPA can tune the
online machinery" affordance was never exercised.

So: two logs, two purposes, mirroring the two-caches split in `subcache.py` / `reuse.py`.

    AuditLog      what entered the ledger        compliance     read by a human
    ProposalLog   what the verifier was asked    training       read by an optimizer

The record is everything needed to *replay* a verification: the full op (text included),
the failure text, and the trajectory summary that was actually in the verifier's prompt.
Labels are deliberately **not** stored — `GroundingProbe` is deterministic, so a label
recomputed at training time is always current, while a stored one would silently encode
whichever probe threshold was configured on the day of the run.

Format is JSONL rather than the audit log's single JSON document, for one reason learned
from `telemetry.py`: this is appended during a live multi-hour sweep, and a run killed
mid-write must not cost the whole file. A truncated trailing line is dropped on load.

Outcome rows (`kind: "outcome"`) are how a proposal learns whether it *helped*, which is
the only honest metric for tuning the proposer. They are appended, never edited — a later
row for the same proposal supersedes an earlier one, so the file stays append-only while
`outcomes()` reports the current view.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from sentinelprime.audit import digest


@dataclass
class ProposalRecord:
    """One edit as the verifier saw it, and what the verifier said back."""

    proposal_id: str        # content-addressed over (op, task_id) — the dedup key
    task_id: str
    op: dict                # the full proposed edit, text included, so it can be replayed
    rubric_failures: str    # feedback.as_text() — the verifier's second input
    trajectory_digest: str  # provenance link back to the AuditRecord for the same round
    trajectory_summary: str # the bounded slice that was actually in the prompt
    admitted: bool          # the ladder's final verdict
    justification: str      # the ladder's rendered trail
    current_ledger: str = "" # the proposer's third input, so a round can be replayed whole
    rungs: dict = field(default_factory=dict)   # per-rung verdicts, for rung-vs-rung metrics
    created_at: str = ""


@dataclass
class OutcomeRecord:
    """What happened to an admitted proposal once it was in the prompt.

    `exposures`/`successes` come from `CreditAssigner`: the tasks where this lesson was
    actually surfaced, and how often the criteria it declared went on to pass. This is the
    downstream signal the proposer metric needs, and nothing else persists it.
    """

    proposal_id: str
    edit_id: str
    targets: list[str] = field(default_factory=list)
    exposures: int = 0
    successes: int = 0
    created_at: str = ""

    @property
    def success_rate(self) -> float:
        return self.successes / self.exposures if self.exposures else 0.0


def proposal_id(op: dict, task_id: str) -> str:
    """Content-addressed id for a proposal. Same edit on a different task is a new row —
    the failure text differs, so it is a genuinely different training example."""
    return digest("\x1f".join([json.dumps(op, sort_keys=True), task_id]))


class ProposalLog:
    """Append-only JSONL log of every proposal a verifier judged."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._records: list[ProposalRecord] = []
        self._outcomes: dict[str, OutcomeRecord] = {}
        self._ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write leaves a partial trailing line. One truncated
                    # row must not cost the whole trainset.
                    continue
                kind = row.pop("kind", "proposal")
                if kind == "outcome":
                    # Later rows supersede earlier ones; the file itself stays append-only.
                    self._outcomes[row["proposal_id"]] = OutcomeRecord(**row)
                elif row["proposal_id"] not in self._ids:
                    self._records.append(ProposalRecord(**row))
                    self._ids.add(row["proposal_id"])

    def _write(self, row: dict) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()

    def append(self, record: ProposalRecord) -> None:
        if record.proposal_id in self._ids:
            return
        self._records.append(record)
        self._ids.add(record.proposal_id)
        self._write({"kind": "proposal", **asdict(record)})

    def append_outcome(self, proposal_id: str, edit_id: str, targets: list[str],
                       exposures: int, successes: int, created_at: str = "") -> None:
        record = OutcomeRecord(proposal_id=proposal_id, edit_id=edit_id,
                               targets=list(targets), exposures=exposures,
                               successes=successes, created_at=created_at)
        self._outcomes[proposal_id] = record
        self._write({"kind": "outcome", **asdict(record)})

    def records(self) -> list[ProposalRecord]:
        return list(self._records)

    def outcomes(self) -> dict[str, OutcomeRecord]:
        return dict(self._outcomes)

    def for_edit(self, edit_id: str) -> ProposalRecord | None:
        """The most recent proposal that produced this ledger id, or None.

        The bridge from credit assignment (which keys by ledger id) back to the training
        record (which keys by proposal). Most recent wins: a lesson rewritten after a
        retirement is a new attempt, and the outcome belongs to the attempt in force.
        """
        matches = [r for r in self._records if r.op.get("id") == edit_id]
        return matches[-1] if matches else None

    def admitted(self) -> list[ProposalRecord]:
        return [r for r in self._records if r.admitted]

    def rejected(self) -> list[ProposalRecord]:
        return [r for r in self._records if not r.admitted]
