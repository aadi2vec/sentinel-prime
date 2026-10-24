"""Error-budget gating: self-modification flows while the SLIs hold, and freezes when they don't.

`ProgressMonitor` detects and records but deliberately never acts. This is the first
component that *acts* — it is allowed to stop `refine()`. These tests pin the three
properties that make that safe: unknown is never a breach, a breach needs enough
observations to mean something, and the freeze is sticky and audited.
"""
import json

from sentinelprime.audit import AuditLog
from sentinelprime.errorbudget import ErrorBudget, Objective


def _budget(tmp_path, **kw):
    objectives = [Objective("check_precision", floor=0.9, min_observations=3),
                  Objective("cost_per_task", ceiling=2.50, min_observations=3)]
    return ErrorBudget(objectives, **kw)


def test_a_fresh_budget_allows_refinement(tmp_path):
    assert _budget(tmp_path).allows_refine()


def test_an_objective_with_no_observations_is_not_breached(tmp_path):
    """Unknown must never read as a breach. `credit.success_rate` returns None rather than
    zero for the same reason: nothing observed is untested, not failing."""
    budget = _budget(tmp_path)
    assert budget.breaches() == []


def test_one_bad_observation_does_not_breach(tmp_path):
    """Below min_observations the mean means nothing, and freezing the loop on a single
    bad task would stop learning over noise."""
    budget = _budget(tmp_path)
    budget.observe("check_precision", 0.1)
    assert budget.breaches() == []
    assert budget.allows_refine()


def test_a_sustained_floor_violation_breaches(tmp_path):
    budget = _budget(tmp_path)
    for _ in range(3):
        budget.observe("check_precision", 0.1)
    assert budget.breaches() == ["check_precision"]


def test_a_sustained_ceiling_violation_breaches(tmp_path):
    budget = _budget(tmp_path)
    for _ in range(3):
        budget.observe("cost_per_task", 9.0)
    assert budget.breaches() == ["cost_per_task"]


def test_an_objective_comfortably_inside_its_bound_does_not_breach(tmp_path):
    budget = _budget(tmp_path)
    for _ in range(5):
        budget.observe("check_precision", 0.97)
        budget.observe("cost_per_task", 0.40)
    assert budget.breaches() == []
    assert budget.allows_refine()


def test_a_breach_stops_refinement(tmp_path):
    budget = _budget(tmp_path)
    for _ in range(3):
        budget.observe("cost_per_task", 9.0)
    assert not budget.allows_refine()


def test_the_freeze_is_sticky_once_tripped(tmp_path):
    """An auto-thaw on the next good task defeats the mechanism: the loop would resume
    without anyone having looked at why it stopped."""
    budget = _budget(tmp_path)
    for _ in range(3):
        budget.observe("check_precision", 0.1)
    assert not budget.allows_refine()
    for _ in range(10):
        budget.observe("check_precision", 1.0)
    assert not budget.allows_refine()


def test_an_explicit_thaw_resumes_refinement(tmp_path):
    budget = _budget(tmp_path)
    for _ in range(3):
        budget.observe("check_precision", 0.1)
    budget.thaw()
    assert budget.allows_refine()


def test_a_freeze_is_recorded_on_the_audit_log(tmp_path):
    """An un-audited freeze is a silent outage — a run that looks like it had nothing to
    learn rather than one that was stopped."""
    log = AuditLog(str(tmp_path / "audit.json"))
    budget = _budget(tmp_path, audit_log=log, task_id="task-7", version=4)
    for _ in range(3):
        budget.observe("cost_per_task", 9.0)
    budget.allows_refine()
    records = [r for r in log.records() if r.op == "freeze"]
    assert len(records) == 1
    assert "cost_per_task" in records[0].cause_failures


def test_a_freeze_is_recorded_once_not_on_every_check(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    budget = _budget(tmp_path, audit_log=log, task_id="task-7", version=4)
    for _ in range(3):
        budget.observe("cost_per_task", 9.0)
    for _ in range(5):
        budget.allows_refine()
    assert len([r for r in log.records() if r.op == "freeze"]) == 1


def test_an_objective_needs_a_floor_or_a_ceiling(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        Objective("nameless")


def test_an_objective_cannot_have_both_a_floor_and_a_ceiling(tmp_path):
    """Two-sided bounds hide which side tripped, and the freeze record has to say."""
    import pytest
    with pytest.raises(ValueError):
        Objective("both", floor=0.1, ceiling=0.9)


def test_observing_an_unregistered_objective_is_refused(tmp_path):
    import pytest
    budget = _budget(tmp_path)
    with pytest.raises(KeyError):
        budget.observe("invented_metric", 1.0)


def test_the_window_bounds_what_a_verdict_rests_on(tmp_path):
    """A lifetime tally convicts the loop for a period already fixed — the same reason
    `CreditRecord` uses a bounded deque."""
    budget = ErrorBudget([Objective("check_precision", floor=0.9,
                                    min_observations=3, window=3)])
    for _ in range(3):
        budget.observe("check_precision", 0.1)
    budget.thaw()
    for _ in range(3):
        budget.observe("check_precision", 1.0)
    assert budget.breaches() == []


def test_the_state_renders_for_a_run_log(tmp_path):
    budget = _budget(tmp_path)
    for _ in range(3):
        budget.observe("cost_per_task", 9.0)
    described = budget.describe()
    assert "cost_per_task" in described
    assert "frozen" in described
