"""Tuning the online machinery offline, from the record the online loop already writes."""
import json

import dspy

from sentinelprime.optimize import (LabeledProposal, admission_metric, label_proposals,
                                    proposer_metric, proposer_trainset, rollout_metric,
                                    verifier_trainset)
from sentinelprime.proposals import ProposalLog, ProposalRecord
from sentinelprime.verifier import GroundingProbe

FAILURES = "Failed rubric criteria:\n- [c1] missed the change of control clause"


def _rec(pid, text, admitted=True, targets=None):
    return ProposalRecord(
        proposal_id=pid, task_id="ma-001",
        op={"op": "create", "id": pid, "kind": "note", "text": text, "scope": "global",
            "meta": {"targets": targets or ["c1"]}},
        rubric_failures=FAILURES, trajectory_digest="d", trajectory_summary="[]",
        admitted=admitted, justification="j", rungs={}, created_at="t")


def _log(tmp_path, records, outcomes=()):
    log = ProposalLog(str(tmp_path / "p.jsonl"))
    for r in records:
        log.append(r)
    for pid, exposures, successes in outcomes:
        log.append_outcome(pid, edit_id=pid, targets=["c1"], exposures=exposures,
                           successes=successes)
    return log


# --- labelling ------------------------------------------------------------------------

def test_probe_rejection_is_a_gold_negative(tmp_path):
    # An edit whose vocabulary is disjoint from the observed failures is ungrounded by
    # construction — the one place the deterministic rung is authoritative.
    log = _log(tmp_path, [_rec("p1", "bake a sponge cake at 180 degrees")])
    labels = {l.record.proposal_id: l for l in label_proposals(log)}
    assert labels["p1"].admit is False
    assert labels["p1"].source == "probe"


def test_probe_admission_alone_is_not_a_label(tmp_path):
    # Training the judge to agree with the probe where the probe has no real opinion
    # collapses the judge into an expensive copy of it, and the ladder loses its point.
    log = _log(tmp_path, [_rec("p1", "missed the change of control clause")])
    assert label_proposals(log) == []


def test_downstream_success_labels_an_admitted_proposal_positive(tmp_path):
    log = _log(tmp_path, [_rec("p1", "missed the change of control clause")],
               outcomes=[("p1", 4, 4)])
    label = label_proposals(log)[0]
    assert label.admit is True
    assert label.source == "outcome"


def test_downstream_failure_labels_an_admitted_proposal_negative(tmp_path):
    log = _log(tmp_path, [_rec("p1", "missed the change of control clause")],
               outcomes=[("p1", 4, 0)])
    assert label_proposals(log)[0].admit is False


def test_an_outcome_below_min_exposures_is_not_a_label(tmp_path):
    # Same guard as CreditAssigner: do not convict a lesson on one observation.
    log = _log(tmp_path, [_rec("p1", "missed the change of control clause")],
               outcomes=[("p1", 1, 0)])
    assert label_proposals(log, min_exposures=3) == []


def test_probe_verdict_wins_over_an_outcome(tmp_path):
    # A lesson that scored well downstream while being ungrounded is the lucky-guess case
    # the whole grounding story exists to refuse. The deterministic signal ranks higher.
    log = _log(tmp_path, [_rec("p1", "bake a sponge cake at 180 degrees")],
               outcomes=[("p1", 9, 9)])
    label = label_proposals(log)[0]
    assert label.admit is False and label.source == "probe"


# --- trainsets ------------------------------------------------------------------------

def test_verifier_trainset_matches_the_signature_input_fields(tmp_path):
    log = _log(tmp_path, [_rec("p1", "bake a sponge cake")])
    ex = verifier_trainset(log)[0]
    assert set(ex.inputs().keys()) == {"proposed_edit", "rubric_failures",
                                       "trajectory_summary"}
    assert ex.admit == "no"
    assert json.loads(ex.proposed_edit)["text"] == "bake a sponge cake"


def test_verifier_trainset_is_empty_when_nothing_is_labelled(tmp_path):
    log = _log(tmp_path, [_rec("p1", "missed the change of control clause")])
    assert verifier_trainset(log) == []


def test_proposer_trainset_groups_one_example_per_round(tmp_path):
    # The proposer emits a whole list per round, so two proposals from one task are one
    # training example, not two.
    log = _log(tmp_path, [_rec("p1", "a"), _rec("p2", "b")])
    assert len(proposer_trainset(log)) == 1
    assert set(proposer_trainset(log)[0].inputs()) == {"trajectory_summary",
                                                       "rubric_failures", "current_ledger"}


# --- metrics --------------------------------------------------------------------------

def _gold(admit="no"):
    return dspy.Example(proposed_edit=json.dumps({"op": "create", "text": "cake"}),
                        rubric_failures=FAILURES, trajectory_summary="[]",
                        admit=admit).with_inputs("proposed_edit", "rubric_failures",
                                                 "trajectory_summary")


def test_admission_metric_rewards_agreement_with_the_gold_label():
    m = admission_metric()
    out = m(_gold("no"), dspy.Prediction(admit="no", score="0.1", justification="j"),
            None, None, None)
    assert out.score == 1.0


def test_admission_metric_punishes_disagreement_and_says_why():
    m = admission_metric()
    out = m(_gold("no"), dspy.Prediction(admit="yes", score="0.9", justification="j"),
            None, None, None)
    assert out.score == 0.0
    assert "no" in out.feedback and "yes" in out.feedback


def test_admission_metric_returns_feedback_gepa_can_reflect_on():
    m = admission_metric()
    out = m(_gold("no"), dspy.Prediction(admit="yes", score="0.9", justification="j"),
            None, None, None)
    assert isinstance(out.feedback, str) and len(out.feedback) > 20


def test_proposer_metric_scores_unparseable_edits_zero_and_names_the_problem():
    m = proposer_metric()
    gold = dspy.Example(trajectory_summary="[]", rubric_failures=FAILURES,
                        current_ledger="(empty)").with_inputs(
        "trajectory_summary", "rubric_failures", "current_ledger")
    out = m(gold, dspy.Prediction(edits="not json at all"), None, None, None)
    assert out.score == 0.0
    assert "pars" in out.feedback.lower()


def test_proposer_metric_prefers_a_grounded_edit_that_declares_its_targets():
    m = proposer_metric()
    gold = dspy.Example(trajectory_summary="[]", rubric_failures=FAILURES,
                        current_ledger="(empty)").with_inputs(
        "trajectory_summary", "rubric_failures", "current_ledger")
    good = json.dumps([{"op": "create", "id": "a", "kind": "note",
                        "text": "always extract the change of control clause",
                        "meta": {"targets": ["c1"]}}])
    bad = json.dumps([{"op": "create", "id": "b", "kind": "note",
                       "text": "bake a sponge cake at 180 degrees"}])
    assert (m(gold, dspy.Prediction(edits=good), None, None, None).score
            > m(gold, dspy.Prediction(edits=bad), None, None, None).score)


def test_proposer_metric_penalises_an_edit_with_no_declared_targets():
    # meta.targets is load-bearing for credit assignment; an edit without it is unscorable
    # downstream, so the optimizer must not be indifferent to dropping it.
    m = proposer_metric()
    gold = dspy.Example(trajectory_summary="[]", rubric_failures=FAILURES,
                        current_ledger="(empty)").with_inputs(
        "trajectory_summary", "rubric_failures", "current_ledger")
    text = "always extract the change of control clause"
    with_t = json.dumps([{"op": "create", "id": "a", "kind": "note", "text": text,
                          "meta": {"targets": ["c1"]}}])
    without = json.dumps([{"op": "create", "id": "a", "kind": "note", "text": text}])
    assert (m(gold, dspy.Prediction(edits=with_t), None, None, None).score
            > m(gold, dspy.Prediction(edits=without), None, None, None).score)


def test_rollout_metric_delegates_to_the_injected_rollout(tmp_path):
    # The honest downstream measure is a re-run, which needs an LM and a workspace. It is
    # injected so the expensive path is opt-in and still testable.
    seen = {}

    def fake_rollout(edits, gold):
        seen["edits"] = edits
        return 0.75

    gold = dspy.Example(trajectory_summary="[]", rubric_failures=FAILURES,
                        current_ledger="(empty)").with_inputs(
        "trajectory_summary", "rubric_failures", "current_ledger")
    edits = json.dumps([{"op": "create", "id": "a", "kind": "note", "text": "t"}])
    out = rollout_metric(fake_rollout)(gold, dspy.Prediction(edits=edits), None, None, None)
    assert out.score == 0.75
    assert seen["edits"][0]["id"] == "a"


def test_rollout_metric_scores_unparseable_edits_zero_without_paying_for_a_rollout():
    def boom(edits, gold):
        raise AssertionError("must not run a rollout for edits that do not parse")

    gold = dspy.Example(trajectory_summary="[]", rubric_failures=FAILURES,
                        current_ledger="(empty)").with_inputs(
        "trajectory_summary", "rubric_failures", "current_ledger")
    assert rollout_metric(boom)(gold, dspy.Prediction(edits="{"), None, None, None).score == 0.0


# --- can this trainset support a compile at all? --------------------------------------

def test_report_counts_labels_by_stratum(tmp_path):
    from sentinelprime.optimize import trainset_report
    log = _log(tmp_path,
               [_rec("p1", "bake a sponge cake"),                      # probe negative
                _rec("p2", "missed the change of control clause"),      # outcome positive
                _rec("p3", "missed the change of control clause")],     # unlabelled
               outcomes=[("p2", 4, 4)])
    r = trainset_report(log)
    assert r.proposals == 3
    assert r.labeled == 2
    assert r.by_source == {"probe": 1, "outcome": 1}
    assert (r.positives, r.negatives) == (1, 1)


def test_a_trainset_with_one_class_is_not_viable(tmp_path):
    # All-positive is exactly the failure the audit log would have given us. Fitting to it
    # teaches "admit everything", which is the absence of a gate.
    from sentinelprime.optimize import trainset_report
    log = _log(tmp_path, [_rec("p1", "missed the change of control clause")],
               outcomes=[("p1", 4, 4)])
    r = trainset_report(log)
    assert r.viable is False
    assert "one class" in r.why.lower() or "negative" in r.why.lower()


def test_a_two_class_trainset_over_the_minimum_is_viable(tmp_path):
    from sentinelprime.optimize import trainset_report
    records, outcomes = [], []
    for i in range(4):
        records.append(_rec(f"n{i}", "bake a sponge cake"))
        records.append(_rec(f"y{i}", "missed the change of control clause"))
        outcomes.append((f"y{i}", 4, 4))
    log = _log(tmp_path, records, outcomes)
    assert trainset_report(log, min_examples=4).viable is True


def test_split_is_deterministic_and_disjoint():
    from sentinelprime.optimize import split
    examples = [dspy.Example(proposed_edit=str(i), admit="no").with_inputs("proposed_edit")
                for i in range(20)]
    train_a, val_a = split(examples, holdout=0.25)
    train_b, val_b = split(examples, holdout=0.25)
    assert [e.proposed_edit for e in val_a] == [e.proposed_edit for e in val_b]
    assert not ({e.proposed_edit for e in train_a} & {e.proposed_edit for e in val_a})
    assert len(train_a) + len(val_a) == 20
    assert len(val_a) == 5


def test_split_never_empties_the_trainset():
    # A tiny log must still produce a runnable compile rather than a zero-length trainset.
    from sentinelprime.optimize import split
    examples = [dspy.Example(proposed_edit="a", admit="no").with_inputs("proposed_edit")]
    train, val = split(examples, holdout=0.5)
    assert len(train) == 1 and val == []


# --- the circuit: the online loop manufactures the offline optimizer's trainset --------

def test_running_the_online_loop_produces_a_two_class_trainset(tmp_path):
    """End-to-end, no LM: refine + credit alone yield labelled examples in both strata.

    This is the claim the module exists to support. Nothing here labels anything by hand:
    the rejection is deterministic, and the positive comes from the lesson still clearing
    its criteria three tasks later.
    """
    from sentinelprime.audit import AuditLog
    from sentinelprime.credit import CreditAssigner
    from sentinelprime.feedback import parse_lab_result
    from sentinelprime.harness import ContinualHarness
    from sentinelprime.memory import JsonMemoryBackend
    from sentinelprime.optimize import trainset_report, verifier_trainset
    from sentinelprime.verifier import LadderVerifier, VerifierLevel

    audit = AuditLog(str(tmp_path / "a.json"))
    plog = ProposalLog(str(tmp_path / "p.jsonl"))
    ladder = LadderVerifier([VerifierLevel("grounding_probe", 1, GroundingProbe())])
    harness = ContinualHarness(
        JsonMemoryBackend(str(tmp_path / "led.json")), audit_log=audit,
        credit_assigner=CreditAssigner(audit), verifier=ladder, proposal_log=plog)

    # One grounded lesson and one the proposer invented — the real label-free failure mode.
    harness.propose.forward = lambda **kw: type("P", (), {"edits": json.dumps([
        {"op": "create", "id": "good", "kind": "note", "scope": "global",
         "text": "always extract the change of control clause",
         "meta": {"targets": ["c1"]}},
        {"op": "create", "id": "cake", "kind": "note", "scope": "global",
         "text": "bake a sponge cake at 180 degrees", "meta": {"targets": ["c1"]}},
    ])})()

    failed = parse_lab_result({"task_id": "ma-001", "criteria": [
        {"id": "c1", "passed": False, "reason": "missed the change of control clause"}]})
    result = harness.refine([{"output": "..."}], failed)
    assert result.created == ["good"] and result.rejected == ["cake"]

    # the lesson keeps clearing its criterion on later tasks
    for n in range(3):
        harness.credit(["good"], parse_lab_result({"task_id": f"ma-00{n + 2}", "criteria": [
            {"id": "c1", "passed": True, "reason": "found it"}]}))

    report = trainset_report(plog, min_examples=2)
    assert report.by_source == {"probe": 1, "outcome": 1}
    assert (report.positives, report.negatives) == (1, 1)
    assert report.viable is True

    labels = {json.loads(e.proposed_edit)["id"]: e.admit for e in verifier_trainset(plog)}
    assert labels == {"good": "yes", "cake": "no"}


# --- the counterfactual rollout: what invariant 2 is actually for ---------------------

def _rollout_harness(tmp_path):
    from sentinelprime.harness import ContinualHarness
    from sentinelprime.memory import JsonMemoryBackend
    return ContinualHarness(JsonMemoryBackend(str(tmp_path / "led.json")))


def _edits():
    return [{"op": "create", "id": "trial", "kind": "note", "scope": "global",
             "text": "always extract the change of control clause"}]


def test_rollout_scores_the_run_it_performed(tmp_path):
    from sentinelprime.optimize import LedgerRollout
    rollout = LedgerRollout(_rollout_harness(tmp_path), lambda task_id: 0.8)
    assert rollout(_edits(), dspy.Example(task_id="ma-001")) == 0.8


def test_the_proposed_edits_are_actually_in_the_prompt_during_the_run(tmp_path):
    # A rollout that scores a run the edits were not visible to measures nothing.
    from sentinelprime.optimize import LedgerRollout
    harness = _rollout_harness(tmp_path)
    seen = {}

    def run_and_score(task_id):
        seen["guidance"] = harness.read()
        return 1.0

    LedgerRollout(harness, run_and_score)(_edits(), dspy.Example(task_id="t"))
    assert "change of control" in seen["guidance"]


def test_the_ledger_is_restored_after_the_rollout(tmp_path):
    # Reversibility is the mechanism: a trial leaves no trace, so candidates are
    # independent and the ledger under test is never the one the optimizer polluted.
    from sentinelprime.optimize import LedgerRollout
    harness = _rollout_harness(tmp_path)
    LedgerRollout(harness, lambda task_id: 1.0)(_edits(), dspy.Example(task_id="t"))
    assert harness.read() == ""


def test_the_ledger_is_restored_even_when_the_run_raises(tmp_path):
    from sentinelprime.optimize import LedgerRollout
    harness = _rollout_harness(tmp_path)

    def boom(task_id):
        raise RuntimeError("the agent died mid-task")

    try:
        LedgerRollout(harness, boom)(_edits(), dspy.Example(task_id="t"))
    except RuntimeError:
        pass
    assert harness.read() == ""


def test_rollout_preserves_guidance_that_was_already_in_the_ledger(tmp_path):
    from sentinelprime.memory import MemoryItem
    from sentinelprime.optimize import LedgerRollout
    harness = _rollout_harness(tmp_path)
    harness.backend.write([MemoryItem(id="kept", scope="global", kind="note",
                                      text="prior guidance", created_at="t", meta={})])
    LedgerRollout(harness, lambda task_id: 1.0)(_edits(), dspy.Example(task_id="t"))
    assert "prior guidance" in harness.read()


def test_rollout_metric_over_a_real_ledger_rollout(tmp_path):
    from sentinelprime.optimize import LedgerRollout, rollout_metric
    harness = _rollout_harness(tmp_path)
    metric = rollout_metric(LedgerRollout(harness, lambda task_id: 0.5))
    gold = dspy.Example(trajectory_summary="[]", rubric_failures=FAILURES,
                        current_ledger="(empty)", task_id="ma-001").with_inputs(
        "trajectory_summary", "rubric_failures", "current_ledger")
    out = metric(gold, dspy.Prediction(edits=json.dumps(_edits())), None, None, None)
    assert out.score == 0.5 and harness.read() == ""


# --- making the distribution gap measurable rather than a caveat ----------------------

def _rec_rungs(pid, text, rungs, admitted=False):
    r = _rec(pid, text, admitted=admitted)
    r.rungs = rungs
    return r


PROBE_ONLY = {"grounding_probe": {"admitted": False, "score": 0.0, "justification": "j",
                                  "shadow": False}}
PROBE_PLUS_SHADOW = {
    "grounding_probe": {"admitted": False, "score": 0.0, "justification": "j",
                        "shadow": False},
    "llm_verifier": {"admitted": True, "score": 0.9, "justification": "j",
                     "shadow": True},
}


def test_a_probe_rejection_the_judge_never_saw_is_off_distribution(tmp_path):
    log = _log(tmp_path, [_rec_rungs("p1", "bake a sponge cake", PROBE_ONLY)])
    assert label_proposals(log)[0].on_distribution is False


def test_exploration_makes_a_probe_rejection_on_distribution(tmp_path):
    # The judge actually ran on this row, so tuning against it is not extrapolation.
    log = _log(tmp_path, [_rec_rungs("p1", "bake a sponge cake", PROBE_PLUS_SHADOW)])
    assert label_proposals(log)[0].on_distribution is True


def test_an_outcome_label_is_always_on_distribution(tmp_path):
    # It was admitted, so every rung ran on it by construction.
    log = _log(tmp_path, [_rec("p1", "missed the change of control clause")],
               outcomes=[("p1", 4, 4)])
    assert label_proposals(log)[0].on_distribution is True


def test_report_measures_the_on_distribution_fraction(tmp_path):
    from sentinelprime.optimize import trainset_report
    log = _log(tmp_path, [_rec_rungs("p1", "bake a sponge cake", PROBE_ONLY),
                          _rec_rungs("p2", "whisk three eggs", PROBE_PLUS_SHADOW)])
    r = trainset_report(log)
    assert r.on_distribution == 1
    assert 0.49 < r.on_distribution_rate < 0.51


def test_report_warns_when_most_rows_are_off_distribution(tmp_path):
    from sentinelprime.optimize import trainset_report
    records = [_rec_rungs(f"n{i}", "bake a sponge cake", PROBE_ONLY) for i in range(6)]
    records += [_rec(f"y{i}", "missed the change of control clause") for i in range(2)]
    log = _log(tmp_path, records, outcomes=[(f"y{i}", 4, 4) for i in range(2)])
    r = trainset_report(log, min_examples=4)
    assert r.viable is True          # trainable, but
    assert "distribution" in r.caveat.lower()   # not without saying this


def test_no_caveat_when_every_row_is_on_distribution(tmp_path):
    from sentinelprime.optimize import trainset_report
    records = [_rec_rungs(f"n{i}", "bake a sponge cake", PROBE_PLUS_SHADOW)
               for i in range(4)]
    records += [_rec(f"y{i}", "missed the change of control clause") for i in range(4)]
    log = _log(tmp_path, records, outcomes=[(f"y{i}", 4, 4) for i in range(4)])
    assert trainset_report(log, min_examples=4).caveat == ""


# --- stratum balance and stratified scoring -------------------------------------------

def test_balance_caps_the_majority_stratum(tmp_path):
    from sentinelprime.optimize import balance
    examples = ([dspy.Example(proposed_edit=f"probe{i}", admit="no", label_source="probe",
                              on_distribution=False).with_inputs("proposed_edit")
                 for i in range(20)]
                + [dspy.Example(proposed_edit=f"out{i}", admit="yes",
                                label_source="outcome",
                                on_distribution=True).with_inputs("proposed_edit")
                   for i in range(3)])
    kept = balance(examples, max_ratio=2.0)
    sources = [e.label_source for e in kept]
    assert sources.count("outcome") == 3
    assert sources.count("probe") == 6


def test_balance_keeps_everything_when_already_balanced():
    from sentinelprime.optimize import balance
    examples = [dspy.Example(proposed_edit=str(i), admit="no",
                             label_source="probe" if i % 2 else "outcome",
                             on_distribution=True).with_inputs("proposed_edit")
                for i in range(10)]
    assert len(balance(examples, max_ratio=2.0)) == 10


def test_balance_is_deterministic():
    from sentinelprime.optimize import balance
    examples = [dspy.Example(proposed_edit=f"p{i}", admit="no", label_source="probe",
                             on_distribution=False).with_inputs("proposed_edit")
                for i in range(9)]
    examples += [dspy.Example(proposed_edit="o", admit="yes", label_source="outcome",
                              on_distribution=True).with_inputs("proposed_edit")]
    assert ([e.proposed_edit for e in balance(examples, max_ratio=2.0)]
            == [e.proposed_edit for e in balance(examples, max_ratio=2.0)])


def test_stratified_scores_split_a_gain_by_where_it_came_from():
    # The number that matters: a gain confined to the off-distribution stratum is not
    # evidence about the live ladder, and a pooled mean hides exactly that.
    from sentinelprime.optimize import stratified_scores
    examples = [
        dspy.Example(admit="no", label_source="probe", on_distribution=False),
        dspy.Example(admit="no", label_source="probe", on_distribution=False),
        dspy.Example(admit="yes", label_source="outcome", on_distribution=True),
    ]
    out = stratified_scores(examples, [1.0, 1.0, 0.0])
    assert out["probe"] == 1.0
    assert out["outcome"] == 0.0
    assert out["on_distribution"] == 0.0
    assert out["overall"] > out["on_distribution"]
