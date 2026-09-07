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
    assert _a(tmp_path, audit=False, credit=False).harness.audit_log is None


def test_credit_without_audit_is_refused(tmp_path):
    # CreditAssigner reads targets off the audit record. Silently disabling credit here
    # would make an ablation report a mechanism as on while it did nothing.
    with pytest.raises(ValueError, match="audit"):
        _a(tmp_path, audit=False, credit=True)


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


def test_credit_thresholds_are_part_of_the_assembly_surface(tmp_path):
    # run_lab deliberately retires below 0.6, not 0.5 — a lesson that helps half the time
    # is not earning the prompt budget it costs on every run. A factory that cannot express
    # that sends the caller back to hand-wiring, which is what this module exists to stop.
    a = _a(tmp_path, credit_min_success_rate=0.6, credit_min_exposures=5)
    assert a.harness.credit_assigner.min_success_rate == 0.6
    assert a.harness.credit_assigner.min_exposures == 5
