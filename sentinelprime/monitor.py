"""ProgressMonitor: the first slice of stop-and-rethink — measurement before machinery.

The shipped `dspy.RLM` loop is greedy single-path: it appends to a monotone history with
no progress signal and no backtrack, so a wrong strategy gets *elaborated* until max_iters.
Before investing in trajectory checkpoints or frontier search, this monitor quantifies the
problem from signals the run *already emits*:

  1. loop/novelty — a spiking SubQueryCache hit-rate means the agent is re-asking questions
     it already answered (spinning in place).
  2. reasoning stall — consecutive `reasoning` steps that are near-identical mean the agent is
     restating rather than advancing.

Both surrogates are lexical (Jaccard token overlap), deliberately dependency-free; an
embedding-similarity upgrade is a later refinement. On a trigger the monitor records a
`replan` event onto the same append-only audit log, so "at iteration N we detected thrash"
becomes first-class, explainable audit content — not a hidden heuristic.

Scope: detection + recording only. It does not itself abandon a subtree or switch strategy
(that needs trajectory-level checkpoints — see the plan's design notes); it is the honest
progress meter those later mechanisms will trigger on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sentinelprime.audit import AuditLog, AuditRecord, content_hash, digest


@dataclass
class MonitorDecision:
    replan: bool
    reasons: list[str] = field(default_factory=list)


_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class ProgressMonitor:
    def __init__(self, similarity_threshold: float = 0.8, window: int = 3,
                 cache_hit_threshold: float = 0.5, min_calls: int = 4) -> None:
        # A stall fires when the last `window` reasoning steps are pairwise
        # >= similarity_threshold similar; a loop fires when the sub-query cache
        # hit-rate >= cache_hit_threshold over at least `min_calls` calls.
        self.similarity_threshold = similarity_threshold
        self.window = window
        self.cache_hit_threshold = cache_hit_threshold
        self.min_calls = min_calls

    def _reasoning(self, trajectory: list[dict]) -> list[str]:
        return [s["reasoning"] for s in trajectory if s.get("reasoning")]

    def _reasoning_stalled(self, trajectory: list[dict]) -> bool:
        steps = self._reasoning(trajectory)
        if len(steps) < self.window:
            return False
        recent = steps[-self.window:]
        # every consecutive pair in the window must be near-identical
        return all(
            _jaccard(recent[i], recent[i + 1]) >= self.similarity_threshold
            for i in range(len(recent) - 1)
        )

    def _cache_looping(self, cache_stats: dict | None) -> bool:
        if not cache_stats:
            return False
        calls = cache_stats.get("calls", 0)
        if calls < self.min_calls:
            return False
        hits = cache_stats.get("hits", 0)
        return (hits / calls) >= self.cache_hit_threshold

    def assess(self, trajectory: list[dict], cache_stats: dict | None = None) -> MonitorDecision:
        reasons: list[str] = []
        if self._reasoning_stalled(trajectory):
            reasons.append(
                f"reasoning stall: last {self.window} steps pairwise similar "
                f">= {self.similarity_threshold}"
            )
        if self._cache_looping(cache_stats):
            rate = cache_stats["hits"] / cache_stats["calls"]
            reasons.append(
                f"sub-query hit-rate spike: {rate:.2f} >= {self.cache_hit_threshold} "
                "(re-asking answered questions)"
            )
        return MonitorDecision(replan=bool(reasons), reasons=reasons)

    def check_and_record(self, trajectory: list[dict], cache_stats: dict | None,
                         audit_log: AuditLog, version: int, task_id: str) -> MonitorDecision:
        # Assess and, on a trigger, append a `replan` event to the audit log so the
        # decision to abandon the current path is itself recorded and explainable.
        decision = self.assess(trajectory, cache_stats)
        if decision.replan and audit_log is not None:
            failures = "; ".join(decision.reasons)
            edit_id = f"replan@v{version}"
            audit_log.append(AuditRecord(
                edit_id=edit_id,
                op="replan",
                scope="external",  # a trajectory-state signal, not a document invariant
                from_version=version,
                to_version=version,
                cause_task_id=task_id,
                cause_failures=failures,
                trajectory_digest=digest(str(trajectory)[:4000]),
                score=0.0,
                created_at=datetime.now(timezone.utc).isoformat(),
                content_hash=content_hash("replan", edit_id, failures, task_id),
            ))
        return decision
