"""One assembly point: the wiring is a single reviewable fact, not six copies."""
import dspy
import pytest

from sentinelprime.assembly import Assembly, assemble

LM = dspy.LM("openai/gpt-4o-mini")   # never called; construction only


def _a(tmp_path, **kw):
    return assemble(lm=LM, root=tmp_path, **kw)


# --- the whole stack, wired --------------------------------------------------------

def test_assemble_returns_an_agent_wired_to_its_harness(tmp_path):
    a = _a(tmp_path)
    assert isinstance(a, Assembly)
    assert a.agent.harness is a.harness


def test_the_full_stack_is_on_by_default(tmp_path):
    # The default is the configuration the project actually claims. A thinner default is
    # how five of the six call sites ended up under-wired.
    a = _a(tmp_path)
    assert a.harness.verifier is not None
    assert a.harness.audit_log is not None
    assert a.harness.proposal_log is not None
    assert a.harness.credit_assigner is not None
    assert a.agent.monitor is not None
    assert a.error_budget is not None
    assert a.agent.error_budget is a.error_budget


def test_state_lands_under_the_given_root(tmp_path):
    a = _a(tmp_path)
    a.harness.backend.write([])
    assert (tmp_path / "ledger.json").exists()
    assert a.proposal_log.path.startswith(str(tmp_path))


# --- every collaborator is still individually ablatable ------------------------------

def test_gate_none_leaves_the_harness_ungated(tmp_path):
    assert _a(tmp_path, gate=None).harness.verifier is None


def test_gate_probe_is_hermetic(tmp_path):
    a = _a(tmp_path, gate="probe")
    assert [lv.name for lv in a.verifier.levels] == ["grounding_probe"]


def test_credit_off_keeps_lessons_unconditionally(tmp_path):
    assert _a(tmp_path, credit=False).harness.credit_assigner is None


def test_record_off_writes_no_proposal_log(tmp_path):
    a = _a(tmp_path, record=False)
    assert a.harness.proposal_log is None and a.proposal_log is None


def test_monitor_off(tmp_path):
    assert _a(tmp_path, monitor=False).agent.monitor is None


def test_audit_off_disables_the_audit_log(tmp_path):
    # Credit and the error budget both read or write the audit log, so an audit ablation
    # has to drop them with it rather than be refused.
    a = _a(tmp_path, audit=False, credit=False, error_budget=False)
    assert a.harness.audit_log is None


def test_credit_without_audit_is_refused(tmp_path):
    # CreditAssigner reads targets off the audit record. Silently disabling credit here
    # would make an ablation report a mechanism as on while it did nothing.
    with pytest.raises(ValueError, match="audit"):
        _a(tmp_path, audit=False, credit=True)


def test_error_budget_off_leaves_refinement_ungoverned(tmp_path):
    a = _a(tmp_path, error_budget=False)
    assert a.error_budget is None and a.agent.error_budget is None


def test_an_error_budget_without_audit_is_refused(tmp_path):
    """A freeze nobody records is a silent outage: the run reads as having had nothing to
    learn rather than as one that was halted. Same class of refusal as credit/audit."""
    with pytest.raises(ValueError, match="audit"):
        _a(tmp_path, audit=False, credit=False, error_budget=True)


# --- the budget actually guards something -------------------------------------------

def test_every_default_objective_is_one_the_agent_observes(tmp_path):
    """The `credit=True, audit=False` hazard, one level up. An objective nobody feeds never
    reaches `min_observations`, so it never breaches — and the budget then reports itself
    as guarding a metric it is structurally blind to. So every default objective must come
    back with an observation after a single real round."""
    from sentinelprime.feedback import parse_lab_result
    from sentinelprime.harness import RefineResult

    a = _a(tmp_path)
    a.harness.refine = lambda trajectory, feedback: RefineResult(
        created=["a"], updated=[], deleted=[], from_version=0, to_version=1,
        rejected=["b"])
    feedback = parse_lab_result(
        {"task_id": "t1", "criteria": [{"id": "c1", "passed": False, "reason": "x"}]})
    a.agent.learn(trajectory=[{"output": "x"}], feedback=feedback)
    unfed = [name for name in a.error_budget.objectives
             if a.error_budget.mean(name) is None]
    assert unfed == []


def test_the_default_objectives_avoid_task_success(tmp_path):
    """§5 records a pooled criterion rate spanning 0.400-0.691 across five runs of one
    *identical* configuration. A budget defined on task success would freeze the loop on
    sampling noise, so every default objective is a property of the machinery instead."""
    a = _a(tmp_path)
    assert "score" not in a.error_budget.objectives
    assert "criterion_rate" not in a.error_budget.objectives


# --- the compiled program re-enters here --------------------------------------------

def test_a_compiled_program_changes_the_machinery_of_the_assembled_harness(tmp_path):
    from sentinelprime.audit import machinery_fingerprint
    from sentinelprime.verifier import PredictVerifier
    program = PredictVerifier()
    program.verify_predict.signature = \
        program.verify_predict.signature.with_instructions("tuned by GEPA")
    path = str(tmp_path / "v.json")
    program.save(path)
    tuned = assemble(lm=LM, root=tmp_path / "t", program=path)
    untuned = assemble(lm=LM, root=tmp_path / "u")
    assert (machinery_fingerprint(tuned.harness)
            != machinery_fingerprint(untuned.harness))


def test_exploration_reaches_the_assembled_ladder(tmp_path):
    assert _a(tmp_path, explore=0.25).verifier.explore == 0.25


# --- what it reports ------------------------------------------------------------------

def test_describe_names_what_is_on_and_off(tmp_path):
    line = _a(tmp_path, credit=False).describe()
    assert "gate=" in line and "credit=off" in line


def test_describe_carries_the_machinery_fingerprint(tmp_path):
    from sentinelprime.audit import machinery_fingerprint
    a = _a(tmp_path)
    assert machinery_fingerprint(a.harness)[:12] in a.describe()


def test_describe_of_an_ungated_assembly_says_so(tmp_path):
    assert "gate=none" in _a(tmp_path, gate=None).describe()


def test_describe_names_the_budget_state(tmp_path):
    assert "budget=on" in _a(tmp_path).describe()
    assert "budget=off" in _a(tmp_path / "b", error_budget=False).describe()


def test_describe_distinguishes_a_frozen_run_from_a_governed_one(tmp_path):
    """A run that was halted and a run that was free are not the same experiment, and
    `budget=on` alone does not say which this was."""
    a = _a(tmp_path)
    for _ in range(5):
        a.error_budget.observe("turn_limit_rate", 1.0)
    a.error_budget.allows_refine()
    assert "budget=frozen" in a.describe()


def test_credit_thresholds_are_part_of_the_assembly_surface(tmp_path):
    # run_lab deliberately retires below 0.6, not 0.5 — a lesson that helps half the time
    # is not earning the prompt budget it costs on every run. A factory that cannot express
    # that sends the caller back to hand-wiring, which is what this module exists to stop.
    a = _a(tmp_path, credit_min_success_rate=0.6, credit_min_exposures=5)
    assert a.harness.credit_assigner.min_success_rate == 0.6
    assert a.harness.credit_assigner.min_exposures == 5
