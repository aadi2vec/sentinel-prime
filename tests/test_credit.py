"""Credit assignment: a lesson survives only while the failures it targeted keep clearing."""
from sentinelprime.audit import AuditLog, AuditRecord
from sentinelprime.feedback import parse_lab_result


def _log(tmp_path):
    return AuditLog(str(tmp_path / "audit.json"))


def _record(log, edit_id, failures, version=1):
    log.append(AuditRecord(
        edit_id=edit_id, op="create", scope="intrinsic", from_version=version - 1,
        to_version=version, cause_task_id="t", cause_failures=failures,
        trajectory_digest="d", score=1.0, created_at="t",
        content_hash=f"h-{edit_id}-{version}",
    ))


def _fb(**criteria):
    return parse_lab_result({
        "task_id": "t",
        "criteria": [{"id": cid, "passed": passed, "reason": ""}
                     for cid, passed in criteria.items()],
    })


def _assigner(log, **kw):
    from sentinelprime.credit import CreditAssigner
    return CreditAssigner(log, **kw)


def test_lesson_is_credited_when_its_target_criterion_passes(tmp_path):
    log = _log(tmp_path)
    _record(log, "lesson.c1", "- [c1] missed change of control")
    ca = _assigner(log)
    ca.observe(["lesson.c1"], _fb(c1=True))
    assert ca.success_rate("lesson.c1") == 1.0


def test_lesson_is_blamed_when_its_target_criterion_still_fails(tmp_path):
    log = _log(tmp_path)
    _record(log, "lesson.c1", "- [c1] missed change of control")
    ca = _assigner(log)
    ca.observe(["lesson.c1"], _fb(c1=False))
    assert ca.success_rate("lesson.c1") == 0.0


def test_criteria_the_lesson_never_targeted_are_not_attributed_to_it(tmp_path):
    """A lesson written for c1 gets no credit for c2 passing."""
    log = _log(tmp_path)
    _record(log, "lesson.c1", "- [c1] missed change of control")
    ca = _assigner(log)
    ca.observe(["lesson.c1"], _fb(c2=True))
    assert ca.success_rate("lesson.c1") is None  # no attributable observation


def test_an_unexposed_lesson_accrues_no_observations(tmp_path):
    """Only guidance actually surfaced in the prompt can be credited or blamed."""
    log = _log(tmp_path)
    _record(log, "lesson.c1", "- [c1] missed change of control")
    ca = _assigner(log)
    ca.observe([], _fb(c1=True))
    assert ca.success_rate("lesson.c1") is None


def test_lesson_with_no_audit_record_is_never_retired(tmp_path):
    """Externally seeded items have no stated target, so they are unattributable."""
    log = _log(tmp_path)
    ca = _assigner(log, min_exposures=1)
    ca.observe(["seeded"], _fb(c1=False))
    assert ca.retirable() == []


def test_underperforming_lesson_becomes_retirable_after_enough_exposures(tmp_path):
    log = _log(tmp_path)
    _record(log, "lesson.c1", "- [c1] missed change of control")
    ca = _assigner(log, min_exposures=3, min_success_rate=0.5)
    for _ in range(2):
        ca.observe(["lesson.c1"], _fb(c1=False))
    assert ca.retirable() == []          # not enough evidence yet
    ca.observe(["lesson.c1"], _fb(c1=False))
    assert ca.retirable() == ["lesson.c1"]


def test_a_lesson_that_keeps_working_is_never_retirable(tmp_path):
    log = _log(tmp_path)
    _record(log, "lesson.c1", "- [c1] missed change of control")
    ca = _assigner(log, min_exposures=2, min_success_rate=0.5)
    for _ in range(5):
        ca.observe(["lesson.c1"], _fb(c1=True))
    assert ca.retirable() == []


def test_a_lesson_that_helps_more_often_than_not_survives(tmp_path):
    log = _log(tmp_path)
    _record(log, "lesson.c1", "- [c1] missed change of control")
    ca = _assigner(log, min_exposures=3, min_success_rate=0.5)
    ca.observe(["lesson.c1"], _fb(c1=True))
    ca.observe(["lesson.c1"], _fb(c1=True))
    ca.observe(["lesson.c1"], _fb(c1=False))
    assert ca.success_rate("lesson.c1") > 0.5
    assert ca.retirable() == []


# ---- Wiring: retirement happens inside refine()'s reversibility window ---------------

def _harness(tmp_path, **kw):
    from sentinelprime.harness import ContinualHarness
    from sentinelprime.memory import JsonMemoryBackend
    return ContinualHarness(JsonMemoryBackend(str(tmp_path / "state.json")), **kw)


class _NoEdits:
    def __call__(self, **kw):
        import dspy
        return dspy.Prediction(edits="[]")


def test_refine_retires_a_lesson_that_stopped_earning_its_place(tmp_path):
    from sentinelprime.memory import MemoryItem

    log = _log(tmp_path)
    ca = _assigner(log, min_exposures=2, min_success_rate=0.5)
    h = _harness(tmp_path, audit_log=log, credit_assigner=ca)
    h.propose = _NoEdits()
    h.backend.write([MemoryItem(id="lesson.c1", scope="global", kind="note",
                                text="always extract change of control", created_at="t")])
    _record(log, "lesson.c1", "- [c1] missed change of control")

    for _ in range(2):
        h.credit(["lesson.c1"], _fb(c1=False))
    result = h.refine(trajectory=[], feedback=_fb(c1=False))

    assert result.retired == ["lesson.c1"]
    assert [i.id for i in h.backend.read()] == []


def test_retirement_is_reversible_like_any_other_round(tmp_path):
    from sentinelprime.memory import MemoryItem

    log = _log(tmp_path)
    ca = _assigner(log, min_exposures=1, min_success_rate=0.5)
    h = _harness(tmp_path, audit_log=log, credit_assigner=ca)
    h.propose = _NoEdits()
    h.backend.write([MemoryItem(id="lesson.c1", scope="global", kind="note",
                                text="lesson body", created_at="t")])
    _record(log, "lesson.c1", "- [c1] missed change of control")
    h.credit(["lesson.c1"], _fb(c1=False))

    result = h.refine(trajectory=[], feedback=_fb(c1=False))
    assert [i.id for i in h.backend.read()] == []
    h.rollback(result.from_version)
    assert [i.id for i in h.backend.read()] == ["lesson.c1"]


def test_retirement_is_audited_with_the_evidence_that_justified_it(tmp_path):
    from sentinelprime.memory import MemoryItem

    log = _log(tmp_path)
    ca = _assigner(log, min_exposures=1, min_success_rate=0.5)
    h = _harness(tmp_path, audit_log=log, credit_assigner=ca)
    h.propose = _NoEdits()
    h.backend.write([MemoryItem(id="lesson.c1", scope="global", kind="note",
                                text="lesson body", created_at="t")])
    _record(log, "lesson.c1", "- [c1] missed change of control")
    h.credit(["lesson.c1"], _fb(c1=False))
    result = h.refine(trajectory=[], feedback=_fb(c1=False))

    retirements = [r for r in log.records() if r.op == "retire"]
    assert [r.edit_id for r in retirements] == ["lesson.c1"]
    assert "0/1" in retirements[0].cause_failures
    assert "retire" in h.explain(result.to_version)


def test_refine_retires_nothing_when_no_credit_assigner_is_configured(tmp_path):
    from sentinelprime.memory import MemoryItem

    h = _harness(tmp_path)
    h.propose = _NoEdits()
    h.backend.write([MemoryItem(id="lesson.c1", scope="global", kind="note",
                                text="lesson body", created_at="t")])
    result = h.refine(trajectory=[], feedback=_fb(c1=False))
    assert result.retired == []
    assert [i.id for i in h.backend.read()] == ["lesson.c1"]


def test_agent_records_which_guidance_was_exposed_for_the_task(tmp_path):
    """Credit needs the exact set surfaced in the prompt, not the whole ledger."""
    import dspy
    from sentinelprime.agent import PrimeAgent
    from sentinelprime.memory import MemoryItem

    h = _harness(tmp_path)
    h.backend.write([MemoryItem(id="n1", scope="global", kind="note",
                                text="body", created_at="t")])

    class _RLM:
        def __call__(self, **kw):
            return dspy.Prediction(deliverable="X", trajectory=[])

    agent = PrimeAgent(harness=h, root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=_RLM())
    agent.run_task("t1", workdir=str(tmp_path))
    assert agent.last_exposed_ids == ["n1"]


def test_agent_learn_credits_the_exposed_guidance_before_refining(tmp_path):
    import dspy
    from sentinelprime.agent import PrimeAgent
    from sentinelprime.memory import MemoryItem

    log = _log(tmp_path)
    ca = _assigner(log, min_exposures=1, min_success_rate=0.5)
    h = _harness(tmp_path, audit_log=log, credit_assigner=ca)
    h.propose = _NoEdits()
    h.backend.write([MemoryItem(id="lesson.c1", scope="global", kind="note",
                                text="body", created_at="t")])
    _record(log, "lesson.c1", "- [c1] missed change of control")

    class _RLM:
        def __call__(self, **kw):
            return dspy.Prediction(deliverable="X", trajectory=[])

    agent = PrimeAgent(harness=h, root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=_RLM())
    agent.run_task("t1", workdir=str(tmp_path))
    result = agent.learn(agent.last_trajectory, _fb(c1=False))

    # Retirement is only reachable if learn() credited the exposed guidance first;
    # the record is then cleared so a rewrite gets a fair trial.
    assert result.retired == ["lesson.c1"]
    assert ca.success_rate("lesson.c1") is None


# ---- Attribution precision: a lesson answers one failure, not the whole round ---------

def test_explicit_targets_beat_the_whole_rounds_failure_text(tmp_path):
    """A lesson written for c1 must not be blamed when c2 fails."""
    log = _log(tmp_path)
    log.append(AuditRecord(
        edit_id="lesson.c1", op="create", scope="intrinsic", from_version=0, to_version=1,
        cause_task_id="t",
        cause_failures="- [c1] missed change of control\n- [c2] missed governing law",
        trajectory_digest="d", score=1.0, created_at="t", content_hash="h",
        targets=["c1"],
    ))
    ca = _assigner(log, min_exposures=1, min_success_rate=0.5)
    ca.observe(["lesson.c1"], _fb(c1=True, c2=False))
    assert ca.success_rate("lesson.c1") == 1.0
    assert ca.retirable() == []


def test_untagged_lesson_falls_back_to_the_rounds_failures(tmp_path):
    """Coarser and cross-contaminating, but better than no attribution at all."""
    log = _log(tmp_path)
    _record(log, "lesson.old", "- [c1] missed change of control\n- [c2] missed governing law")
    ca = _assigner(log, min_exposures=1)
    ca.observe(["lesson.old"], _fb(c1=True, c2=False))
    assert ca.success_rate("lesson.old") == 0.5


def test_refine_records_the_targets_the_proposer_declared(tmp_path):
    from sentinelprime.harness import ContinualHarness
    from sentinelprime.memory import JsonMemoryBackend
    import dspy

    log = _log(tmp_path)
    h = ContinualHarness(JsonMemoryBackend(str(tmp_path / "s.json")), audit_log=log)
    h.propose = lambda **kw: dspy.Prediction(edits=(
        '[{"op":"create","id":"n1","kind":"note","text":"t","scope":"global",'
        '"meta":{"targets":["c1"]}}]'))
    h.refine(trajectory=[], feedback=_fb(c1=False, c2=False))
    assert log.latest_for("n1").targets == ["c1"]


# ---- Non-stationarity: the environment changes, so lifetime averages are the wrong stat --

def test_a_lesson_recovers_once_its_criterion_starts_passing_again(tmp_path):
    """Another lesson may have been poisoning the criterion; judge on recent evidence."""
    log = _log(tmp_path)
    _record(log, "lesson.c1", "- [c1] missed change of control")
    ca = _assigner(log, min_exposures=3, min_success_rate=0.6, window=4)

    for _ in range(4):
        ca.observe(["lesson.c1"], _fb(c1=False))
    assert ca.retirable() == ["lesson.c1"]          # earned its retirement

    for _ in range(4):
        ca.observe(["lesson.c1"], _fb(c1=True))     # environment fixed
    assert ca.success_rate("lesson.c1") == 1.0
    assert ca.retirable() == []                     # and it recovers


def test_retiring_a_lesson_clears_its_record_so_a_rewrite_gets_a_fair_trial(tmp_path):
    """A recreated lesson is a new lesson; inheriting old blame retires it on sight."""
    log = _log(tmp_path)
    _record(log, "lesson.c1", "- [c1] missed change of control")
    ca = _assigner(log, min_exposures=2, min_success_rate=0.6)
    for _ in range(3):
        ca.observe(["lesson.c1"], _fb(c1=False))
    assert ca.retirable() == ["lesson.c1"]

    ca.forget(["lesson.c1"])
    assert ca.success_rate("lesson.c1") is None
    assert ca.retirable() == []


def test_refine_forgets_the_observations_of_what_it_retired(tmp_path):
    from sentinelprime.memory import MemoryItem

    log = _log(tmp_path)
    ca = _assigner(log, min_exposures=1, min_success_rate=0.6)
    h = _harness(tmp_path, audit_log=log, credit_assigner=ca)
    h.propose = _NoEdits()
    h.backend.write([MemoryItem(id="lesson.c1", scope="global", kind="note",
                                text="body", created_at="t")])
    _record(log, "lesson.c1", "- [c1] missed change of control")
    h.credit(["lesson.c1"], _fb(c1=False))
    h.refine(trajectory=[], feedback=_fb(c1=False))

    assert ca.success_rate("lesson.c1") is None
