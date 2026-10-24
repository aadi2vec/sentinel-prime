"""Error-budget gating: self-modification is allowed while the SLIs hold, and frozen when they don't.

Every other safety mechanism in this repo is per-edit. `LadderVerifier` asks whether *this*
edit is grounded; `CreditAssigner` asks whether *this* lesson helped. Neither can see the
case where admission and attribution are both working correctly and the system is still
getting worse — a drift that no single edit is responsible for. That is what an error budget
is for, and it is the SRE formulation of the champion–challenger gate the plan already runs
offline: define the service levels, let edits flow freely inside them, and **stop** on breach.

This is the first component in the repo permitted to *act*. `ProgressMonitor` deliberately
stops at detection ("it does not itself abandon a subtree or switch strategy"), so until now
every signal was an annotation. `allows_refine()` is consulted by the caller before
`refine()`, and a breach halts learning rather than reporting it.

Which SLIs, and why not task success: §5 records a pooled criterion rate spanning 0.400–0.691
across five runs of one *identical* configuration, so a budget defined on task success would
trip on sampling noise. The metric the plan nominates instead is a property of the check set
over a fixed corpus — precision and recall at a cost ceiling — which agent variance does not
enter. `replay.CheckScore` computes exactly that, and `UsageMeter` supplies the cost side.

Three properties make the freeze safe rather than merely dramatic, and each mirrors a
decision made elsewhere in the repo:

  1. **Unknown is never a breach.** An objective with no observations is untested, not
     failing — the same reason `credit.success_rate` returns None rather than 0.0.
  2. **A breach needs enough observations to mean something**, over a bounded window rather
     than a lifetime tally. Identical reasoning to `CreditRecord`: the environment is
     non-stationary, and a lifetime average convicts the loop for a period already fixed.
  3. **The freeze is sticky and audited.** An auto-thaw on the next good task would resume
     without anyone having looked at why it stopped, and an un-audited freeze is a silent
     outage — a run that reads as having had nothing to learn rather than one that was
     halted. `thaw()` is deliberately manual.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sentinelprime.audit import AuditLog, AuditRecord, content_hash


@dataclass
class Objective:
    """One service level: a named metric with a one-sided bound.

    One-sided on purpose. A two-sided bound cannot say which side tripped, and the freeze
    record has to name the reason — a halt whose cause is ambiguous is a halt nobody can
    act on.
    """

    name: str
    floor: float | None = None      # breach when the windowed mean falls below this
    ceiling: float | None = None    # breach when the windowed mean rises above this
    min_observations: int = 3
    window: int = 10

    def __post_init__(self):
        if (self.floor is None) == (self.ceiling is None):
            raise ValueError(
                f"objective {self.name!r} needs exactly one of floor or ceiling: a metric "
                f"with neither has no bound to breach, and one with both cannot report "
                f"which side tripped")
        if self.min_observations < 1 or self.window < 1:
            raise ValueError("min_observations and window must be >= 1")

    def violated(self, mean: float) -> bool:
        if self.floor is not None:
            return mean < self.floor
        return mean > self.ceiling

    def bound(self) -> str:
        return (f">= {self.floor:g}" if self.floor is not None
                else f"<= {self.ceiling:g}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# The names the online loop feeds. Constants rather than literals at the call sites,
# because the whole point of `observe()` raising on an unknown name is that a budget must
# never end up watching a typo — and a *default* objective fed under a misspelled name
# would reproduce exactly that failure one level up: registered, never observed, therefore
# never in breach, and reported as guarding a metric it is blind to.
TURN_LIMIT_RATE = "turn_limit_rate"
EDIT_ADMISSION_RATE = "edit_admission_rate"


def online_objectives() -> list[Objective]:
    """The default service levels for the live loop — properties of the *machinery*.

    Deliberately not task success. §5 records a pooled criterion rate spanning 0.400-0.691
    across five runs of one *identical* configuration, so a budget defined on it would
    freeze the loop on sampling noise and the freeze would carry no information. Both
    objectives here are rates over the harness's own behavior, which agent sampling
    variance does not enter:

      - `turn_limit_rate` — the share of recent tasks that ran to the RLM's iteration
        limit. `monitor.py` names the failure this aggregates: the live context "appends to
        a monotone history with no progress signal and no backtrack, so a wrong strategy
        gets *elaborated* until max_iters". One task doing that is noise; half of them
        doing it is a degraded system.
      - `edit_admission_rate` — the share of proposed edits the ladder admitted. Rejected
        edits produce no audit record, so this is a signal nothing else in the repo can
        see, and a collapse in it means the proposer is paying the ladder's cost on every
        round to produce almost nothing.

    §5's primary metric — check-set precision and recall at a cost ceiling — is **not**
    here, because it is a property of a check set over a graded corpus and `assemble()` has
    no corpus. The replay path registers it explicitly; see `check_set_objectives()`.
    """
    return [
        Objective(TURN_LIMIT_RATE, ceiling=0.5, min_observations=5, window=20),
        Objective(EDIT_ADMISSION_RATE, floor=0.2, min_observations=5, window=20),
    ]


def check_set_objectives(min_precision: float = 0.9,
                         max_cost_per_task: float = 2.50) -> list[Objective]:
    """§5's primary metric, for a caller that has a graded corpus and a price table.

    Separate from `online_objectives()` because the two are observed from different places.
    Registering these in `assemble()` would produce a budget with objectives nobody can
    feed — the failure `online_objectives()`'s docstring exists to avoid.
    """
    return [
        Objective("check_precision", floor=min_precision, min_observations=3, window=10),
        Objective("cost_per_task", ceiling=max_cost_per_task,
                  min_observations=3, window=10),
    ]


class ErrorBudget:
    def __init__(self, objectives: list[Objective], audit_log: AuditLog | None = None,
                 task_id: str = "", version: int = 0) -> None:
        self.objectives = {o.name: o for o in objectives}
        self.audit_log = audit_log
        self.task_id = task_id
        self.version = version
        self._observations: dict[str, deque] = {
            name: deque(maxlen=o.window) for name, o in self.objectives.items()}
        self._frozen = False
        self._freeze_recorded = False

    def observe(self, name: str, value: float) -> None:
        """Record one measurement. An unregistered name raises rather than being created.

        Silently accepting an unknown metric is how a budget ends up watching a typo: the
        objective it was meant to guard receives nothing and reports itself healthy.
        """
        if name not in self.objectives:
            raise KeyError(f"unregistered objective {name!r}; "
                           f"registered: {sorted(self.objectives)}")
        self._observations[name].append(float(value))

    def mean(self, name: str) -> float | None:
        """Windowed mean, or None when nothing has been observed yet."""
        values = self._observations[name]
        return sum(values) / len(values) if values else None

    def breaches(self) -> list[str]:
        """Objectives whose windowed mean violates their bound, on enough evidence."""
        out = []
        for name, objective in self.objectives.items():
            values = self._observations[name]
            if len(values) < objective.min_observations:
                continue  # untested, not failing
            mean = sum(values) / len(values)
            if objective.violated(mean):
                out.append(name)
        return sorted(out)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def allows_refine(self) -> bool:
        """The gate the caller consults before `refine()`. Trips the freeze on first breach."""
        if self._frozen:
            return False
        breached = self.breaches()
        if breached:
            self._freeze(breached)
            return False
        return True

    def _freeze(self, breached: list[str]) -> None:
        self._frozen = True
        if self.audit_log is None or self._freeze_recorded:
            return
        reasons = "; ".join(
            f"[{name}] windowed mean {self.mean(name):.4g} violates "
            f"{self.objectives[name].bound()}" for name in breached)
        edit_id = f"freeze@v{self.version}"
        self.audit_log.append(AuditRecord(
            edit_id=edit_id,
            op="freeze",
            # An operational state, not a document invariant — same call ProgressMonitor
            # makes for a replan event.
            scope="external",
            from_version=self.version,
            to_version=self.version,
            cause_task_id=self.task_id,
            cause_failures=reasons,
            trajectory_digest="",
            score=0.0,
            created_at=_now(),
            content_hash=content_hash("freeze", edit_id, reasons, self.task_id),
        ))
        self._freeze_recorded = True

    def thaw(self) -> None:
        """Resume learning, and drop the evidence that stopped it.

        Clearing the window is not a convenience: resuming while the breaching
        observations are still in scope would re-trip on the same evidence on the next
        call, which would make a manual thaw useless. A thaw asserts that a human looked
        at the breach, so the measurement starts again from there.
        """
        self._frozen = False
        self._freeze_recorded = False
        for values in self._observations.values():
            values.clear()

    def describe(self) -> str:
        """One line for the run log, naming every objective and the current verdict."""
        breached = self.breaches()
        parts = []
        for name, objective in self.objectives.items():
            mean = self.mean(name)
            shown = "n/a" if mean is None else f"{mean:.4g}"
            parts.append(f"{name}={shown}({objective.bound()})")
        return (f"frozen={'yes' if self._frozen else 'no'} "
                f"breached={breached or 'none'} " + " ".join(parts))
