"""A policy may change execution, never its evaluator or resource ceiling."""
import json

import pytest

from sentinelprime.policy_search import (
    Budget, CheckResult, Case, Policy, PolicyArchive, PolicySearch, PolicyRunner,
)


def case(name, family=None):
    return Case(name, family or name, "solve", "private gold")


def runner(solve=None, check=None, budget=None):
    return PolicyRunner(solve or (lambda task, feedback: "fixed" if feedback else "wrong"),
                        {"inspect": check or (lambda task, answer: CheckResult(
                            answer == "fixed", "inspect the source"))},
                        budget or Budget())


def test_policy_changes_real_execution_and_rechecks_revision():
    r = runner()
    base = r.run(Policy(), "solve")
    revised = r.run(Policy(("inspect",), 1), "solve")
    assert base.answer == "wrong" and base.solves == 1
    assert revised.answer == "fixed" and revised.solves == 2 and revised.checks == 2
    assert [e["passed"] for e in revised.trace if e["kind"] == "check"] == [False, True]


def test_check_failure_without_revision_is_not_silently_fixed():
    result = runner().run(Policy(("inspect",), 0), "solve")
    assert result.answer == "wrong" and result.solves == 1


@pytest.mark.parametrize("raw", [
    '{"checks": ["inspect"], "max_revisions": true}',
    '{"checks": ["inspect", "inspect"], "max_revisions": 1}',
    '{"checks": [], "max_revisions": -1}',
    '{"checks": [], "max_revisions": 1, "evaluator": "always pass"}',
])
def test_policy_schema_rejects_unbounded_or_unknown_fields(raw):
    with pytest.raises(ValueError):
        Policy.parse(raw)


def test_budget_and_unknown_checks_refused_before_solver_runs():
    def boom(*args):
        pytest.fail("invalid candidate executed")
    r = runner(solve=boom, budget=Budget(max_solves=1, max_checks=1))
    for policy in (Policy(("missing",), 0), Policy(("inspect",), 1)):
        with pytest.raises(ValueError):
            r.run(policy, "solve")


def test_checker_exception_is_an_execution_error_not_a_failed_check():
    def broken(*args):
        raise RuntimeError("offline")
    result = runner(check=broken).run(Policy(("inspect",), 1), "solve")
    assert result.error and "offline" in result.error
    assert result.solves == 1


def test_fresh_holdout_promotes_and_final_test_never_reaches_proposer(tmp_path):
    seen = []
    archive = PolicyArchive(tmp_path / "archive.json")
    def propose(policy, feedback):
        seen.append(json.dumps(feedback))
        return Policy(("inspect",), 1)
    search = PolicySearch(runner(), lambda c, answer: float(answer == "fixed"),
                          propose, archive)
    result = search.run([case("dev")], [[case("validation")]], [case("secret-test")])
    assert result["rounds"][0]["promoted"]
    assert result["test"]["champion"]["score"] == 1
    assert "secret-test" not in "".join(seen) and "private gold" not in "".join(seen)
    assert archive.current == Policy(("inspect",), 1)
    assert PolicyArchive(tmp_path / "archive.json").current == archive.current
    archive.rollback(0)
    assert archive.current == Policy() and len(archive.events) == 2


def test_regression_on_validation_is_rejected(tmp_path):
    r = runner(solve=lambda task, feedback: "fixed" if feedback else "wrong")
    def evaluate(c, answer):
        return float(answer == ("fixed" if c.id == "dev" else "wrong"))
    archive = PolicyArchive(tmp_path / "a.json")
    result = PolicySearch(r, evaluate, lambda *args: Policy(("inspect",), 1), archive).run(
        [case("dev")], [[case("val")]], [case("test")])
    assert not result["rounds"][0]["promoted"]
    assert archive.current == Policy()


def test_evaluator_error_prevents_promotion(tmp_path):
    def evaluate(c, answer):
        if c.id == "val":
            raise RuntimeError("judge unavailable")
        return float(answer == "fixed")
    result = PolicySearch(runner(), evaluate, lambda *args: Policy(("inspect",), 1),
                          PolicyArchive(tmp_path / "a.json")).run(
        [case("dev")], [[case("val")]], [case("test")])
    assert not result["rounds"][0]["promoted"]
    assert result["rounds"][0]["candidate"]["score"] is None


def test_family_leakage_and_repeated_validation_are_refused(tmp_path):
    search = PolicySearch(runner(), lambda *args: 1.0, lambda *args: Policy(),
                          PolicyArchive(tmp_path / "a.json"))
    with pytest.raises(ValueError, match="family"):
        search.run([case("a", "same")], [[case("b", "same")]], [case("c")])
    with pytest.raises(ValueError):
        search.run([case("a")], [[case("b")], [case("b")]], [case("c")])


def test_invalid_candidate_is_recorded_without_replacing_champion(tmp_path):
    result = PolicySearch(runner(), lambda *args: 0.0,
                          lambda *args: Policy(("invented",), 0),
                          PolicyArchive(tmp_path / "a.json")).run(
        [case("a")], [[case("b")]], [case("c")])
    assert not result["rounds"][0]["promoted"]
    assert "unknown" in result["rounds"][0]["error"]


def test_restart_cannot_reuse_reserved_evaluation_families(tmp_path):
    path = tmp_path / "a.json"
    def run():
        return PolicySearch(runner(), lambda *args: 0.0, lambda *args: Policy(),
                            PolicyArchive(path)).run(
            [case("a")], [[case("b")]], [case("c")])
    run()
    with pytest.raises(ValueError, match="already used"):
        run()


def test_mean_gain_cannot_hide_a_case_regression(tmp_path):
    def evaluate(c, answer):
        if c.id == "v1":
            return float(answer == "fixed")
        if c.id == "v2":
            return 0.25 if answer == "wrong" else 0.0
        return 0.0
    result = PolicySearch(runner(), evaluate, lambda *args: Policy(("inspect",), 1),
                          PolicyArchive(tmp_path / "a.json")).run(
        [case("dev")], [[case("v1"), case("v2")]], [case("test")])
    assert result["rounds"][0]["gain"] > 0
    assert not result["rounds"][0]["promoted"]


@pytest.mark.integration
def test_real_rlm_uses_full_assembler_and_check_feedback_to_repair(tmp_path, monkeypatch):
    from sentinelprime import policy_experiment as experiment
    observed = []
    real_assemble = experiment.assemble
    def assembled(**kwargs):
        system = real_assemble(**kwargs)
        observed.append(system)
        def no_learning(*args, **kwargs):
            pytest.fail("holdout feedback must not enter supplemental memory")
        system.agent.learn = no_learning
        return system
    monkeypatch.setattr(experiment, "assemble", assembled)
    solve = experiment.AssembledSolver(tmp_path, lm_factory=experiment.scripted_lm)
    task = experiment.make_cases("test", 37, 1)[0]
    result = PolicyRunner(solve, experiment.CHECKS, Budget()).run(
        Policy(tuple(experiment.CHECKS), 1), task.task)
    assert result.error is None
    assert experiment.evaluate(task, result.answer) == 1.0
    assert len(observed) == 2
    for system in observed:
        assert system.gate == "ladder"
        assert system.audit_log and system.proposal_log and system.agent.monitor
        assert system.harness.credit_assigner
    assert observed[0].backend is not observed[1].backend
    assert all(r["trajectory"] for r in solve.runs)


def test_scripted_experiment_completes_and_preserves_outputs(tmp_path):
    from scripts.policy_lab import main
    out = tmp_path / "experiment"
    assert main(["--out", str(out), "--cases", "1"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["test"]["initial"]["score"] == 0.0
    assert report["test"]["champion"]["score"] == 1.0
    assert "SCRIPTED" in report["mode"]
    assert all("gate=ladder" in r["system"] for r in report["solver_runs"])
    with pytest.raises(SystemExit):
        main(["--out", str(out)])
