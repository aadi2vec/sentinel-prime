from sentinelprime.planner import Check, CostLadderPlanner
from sentinelprime.audit import AuditLog, AuditRecord


def _checks():
    # cheap check catches failures often; expensive check rarely does.
    cheap = Check(name="span_exists", cost=1.0, run=lambda ctx: ctx.get("span_ok", True))
    expensive = Check(name="semantic", cost=10.0, run=lambda ctx: ctx.get("sem_ok", True))
    return cheap, expensive


def test_order_puts_cheapest_per_detection_first():
    cheap, expensive = _checks()
    planner = CostLadderPlanner(stats={"span_exists": 0.9, "semantic": 0.1})
    # static order deliberately expensive-first
    static = [expensive, cheap]
    ordered = planner.order(static)
    assert [c.name for c in ordered] == ["span_exists", "semantic"]


def test_planner_order_is_strictly_cheaper_than_static():
    cheap, expensive = _checks()
    planner = CostLadderPlanner(stats={"span_exists": 0.9, "semantic": 0.1})
    static = [expensive, cheap]
    assert planner.expected_cost(planner.order(static)) < planner.expected_cost(static)


def test_run_short_circuits_on_first_failure():
    cheap, expensive = _checks()
    planner = CostLadderPlanner(stats={"span_exists": 0.9, "semantic": 0.1})
    # cheap check fails -> expensive never runs, cost = 1.0 only
    result = planner.run([expensive, cheap], context={"span_ok": False, "sem_ok": True})
    assert result.passed is False
    assert result.failed_at == "span_exists"
    assert result.cost_spent == 1.0
    assert result.order == ["span_exists", "semantic"]


def test_run_all_pass_pays_full_cost():
    cheap, expensive = _checks()
    planner = CostLadderPlanner(stats={"span_exists": 0.9, "semantic": 0.1})
    result = planner.run([cheap, expensive], context={"span_ok": True, "sem_ok": True})
    assert result.passed is True
    assert result.failed_at is None
    assert result.cost_spent == 11.0


def _rec(cause_failures, content_hash):
    return AuditRecord(
        edit_id="e", op="create", scope="intrinsic", from_version=1, to_version=2,
        cause_task_id="t", cause_failures=cause_failures, trajectory_digest="d",
        score=1.0, created_at="t", content_hash=content_hash,
    )


def test_from_audit_log_derives_failure_rates(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    log.append(_rec("- [c1] missed clause", "h1"))
    log.append(_rec("- [c1] missed clause again", "h2"))
    log.append(_rec("- [c2] wrong date", "h3"))
    planner = CostLadderPlanner.from_audit_log(
        log, name_to_criterion={"span_exists": "c1", "semantic": "c2"}
    )
    # c1 failed in 2/3 records, c2 in 1/3
    assert planner.fail_prob(Check("span_exists", 1.0, lambda c: True)) == 2 / 3
    assert planner.fail_prob(Check("semantic", 10.0, lambda c: True)) == 1 / 3
