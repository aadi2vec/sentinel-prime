from sentinelprime.monitor import ProgressMonitor
from sentinelprime.audit import AuditLog


def _looping_trajectory():
    step = {"reasoning": "re-read section 3 to find the change-of-control clause"}
    return [dict(step) for _ in range(4)]


def _healthy_trajectory():
    return [
        {"reasoning": "list files in the working directory"},
        {"reasoning": "open contract.txt and locate the term section"},
        {"reasoning": "extract the governing-law clause from section 7"},
        {"reasoning": "draft the one-sentence deliverable"},
    ]


def test_monitor_fires_on_repeated_reasoning():
    mon = ProgressMonitor(similarity_threshold=0.8, window=3)
    decision = mon.assess(_looping_trajectory(), cache_stats={"hits": 0, "calls": 0})
    assert decision.replan is True
    assert any("similar" in r for r in decision.reasons)


def test_monitor_fires_on_cache_hit_rate_spike():
    mon = ProgressMonitor(cache_hit_threshold=0.5, min_calls=4)
    decision = mon.assess(_healthy_trajectory(),
                          cache_stats={"hits": 6, "calls": 8})
    assert decision.replan is True
    assert any("hit-rate" in r for r in decision.reasons)


def test_monitor_quiet_on_healthy_trajectory():
    mon = ProgressMonitor(similarity_threshold=0.8, cache_hit_threshold=0.5, min_calls=4)
    decision = mon.assess(_healthy_trajectory(),
                          cache_stats={"hits": 1, "calls": 8})
    assert decision.replan is False
    assert decision.reasons == []


def test_check_and_record_emits_replan_audit_record(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    mon = ProgressMonitor(similarity_threshold=0.8, window=3)
    decision = mon.check_and_record(
        _looping_trajectory(), cache_stats={"hits": 0, "calls": 0},
        audit_log=log, version=7, task_id="ma-009",
    )
    assert decision.replan is True
    recs = log.records()
    assert len(recs) == 1
    assert recs[0].op == "replan"
    assert recs[0].to_version == 7
    assert recs[0].cause_task_id == "ma-009"
    assert "similar" in recs[0].cause_failures


def test_check_and_record_noop_when_healthy(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    mon = ProgressMonitor(similarity_threshold=0.8, cache_hit_threshold=0.5, min_calls=4)
    decision = mon.check_and_record(
        _healthy_trajectory(), cache_stats={"hits": 1, "calls": 8},
        audit_log=log, version=3, task_id="t",
    )
    assert decision.replan is False
    assert log.records() == []
