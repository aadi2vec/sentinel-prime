from sentinelprime.audit import AuditRecord
from sentinelprime.reuse import ReuseController, ReuseDecision


def _rec(scope="intrinsic", depends_on=None, score=1.0):
    return AuditRecord(
        edit_id="checklist.change_of_control", op="create", scope=scope,
        from_version=1, to_version=2, cause_task_id="ma-001", cause_failures="",
        trajectory_digest="d", score=score, created_at="t", content_hash="h",
        depends_on=depends_on or {},
    )


def test_intrinsic_record_is_reused():
    dec = ReuseController().should_reuse(_rec(scope="intrinsic"), {"playbook_version": 9})
    assert isinstance(dec, ReuseDecision)
    assert dec.reuse is True


def test_external_record_whose_source_moved_is_rejected():
    rec = _rec(scope="external", depends_on={"playbook_version": 5})
    dec = ReuseController().should_reuse(rec, {"playbook_version": 6})
    assert dec.reuse is False
    assert any("scope" in r.lower() or "source" in r.lower() for r in dec.reasons)


def test_external_record_with_current_source_is_reused():
    rec = _rec(scope="external", depends_on={"playbook_version": 5})
    dec = ReuseController().should_reuse(rec, {"playbook_version": 5})
    assert dec.reuse is True


def test_external_without_declared_dependencies_is_rejected():
    # external means "depends on mutable state"; if we can't name it, we can't
    # prove currency, so recompute rather than risk a stale reuse.
    dec = ReuseController().should_reuse(_rec(scope="external", depends_on={}), {})
    assert dec.reuse is False


def test_verifier_gate_rejects_low_score():
    rec = _rec(scope="intrinsic", score=0.3)
    dec = ReuseController(min_score=0.5).should_reuse(rec, {})
    assert dec.reuse is False
    assert any("verif" in r.lower() or "score" in r.lower() for r in dec.reasons)


def test_decision_is_truthy_and_records_gate_trail():
    dec = ReuseController().should_reuse(_rec(scope="intrinsic"), {})
    assert bool(dec) is True
    assert dec.reasons  # non-empty trail
