"""A gain bought with more solver calls is not a gain from better policy.

The mechanism demo reports `initial: solves=3 checks=0` against
`champion: solves=6 checks=12`. The champion is better *and* spends twice the solver
budget, and nothing in that pair separates the two explanations. These tests pin the
control that does: same solver attempts, same sight of the previous answer, no
diagnosis of what was wrong. What remains between the champion and that control is
what checking bought.
"""
import pytest

from sentinelprime.policy_search import (
    CONTROL_CHECK, Budget, Case, CheckResult, Policy, PolicyArchive, PolicyRunner,
    PolicySearch, matched_compute_control,
)


def case(name, family=None):
    return Case(name, family or name, "solve", "gold")


def runner(solve=None, budget=None):
    return PolicyRunner(solve or (lambda task, feedback: "fixed" if feedback else "wrong"),
                        {"inspect": lambda task, answer: CheckResult(
                            answer == "fixed", "inspect the source")},
                        budget or Budget())


def test_the_control_spends_the_same_solver_attempts_as_the_policy_it_matches():
    """The whole point: equal solver budget, so a difference is not bought with compute."""
    candidate = Policy(("inspect",), 1)
    control = matched_compute_control(candidate)

    r = runner()
    spent = r.run(candidate, "solve")
    matched = r.control_runner().run(control, "solve")

    assert matched.solves == spent.solves == 2


def test_the_control_carries_no_diagnosis_only_the_previous_answer():
    """An uninformed retry. If its feedback named the fault it would be a check."""
    seen = []

    def solve(task, feedback):
        seen.append(feedback)
        return "wrong"

    runner(solve=solve).control_runner().run(matched_compute_control(Policy((), 1)), "solve")

    assert seen[0] == ""                       # first attempt has no feedback at all
    assert "wrong" in seen[1]                  # the retry sees its previous answer
    assert "inspect the source" not in seen[1]  # and no diagnosis of what was wrong


def test_the_control_check_never_passes_so_every_revision_is_spent():
    """A control that could pass early would silently under-spend the matched budget."""
    # The control is bound by the same ceiling as any candidate, so the budget has to
    # allow the attempts being matched — that constraint is the point, not a workaround.
    always_right = runner(solve=lambda task, feedback: "fixed",
                          budget=Budget(max_solves=3, max_checks=3))
    result = always_right.control_runner().run(matched_compute_control(Policy((), 2)), "solve")
    assert result.solves == 3


def test_a_candidate_can_never_name_the_control():
    """Otherwise a candidate 'wins' by burning solver calls, which is the confound itself."""
    with pytest.raises(ValueError):
        runner().run(Policy((CONTROL_CHECK,), 1), "solve")


def test_a_gain_that_is_only_more_compute_reports_an_adjusted_gain_of_zero(tmp_path):
    """The case the headline number cannot distinguish, and this one can.

    The solver here improves on *any* retry — it needs no diagnosis, only a second go.
    So the champion beats the initial policy by a full point while having discovered
    nothing about checking, and `champion - initial` says 1.0. The control spends the
    same solver attempts and reaches the same answer, so the adjusted gain is 0.
    """
    archive = PolicyArchive(tmp_path / "a.json")
    retries_alone_suffice = lambda task, feedback: "fixed" if feedback else "wrong"
    search = PolicySearch(runner(solve=retries_alone_suffice),
                          lambda c, answer: float(answer == "fixed"),
                          lambda policy, feedback: Policy(("inspect",), 1), archive)
    report = search.run([case("d", "fd")], [[case("v", "fv")]], [case("t", "ft")])

    test = report["test"]
    assert test["champion"]["score"] - test["initial"]["score"] == 1.0  # the flattering number
    assert test["matched_compute"]["solves"] == test["champion"]["solves"]
    assert report["compute_adjusted_gain"] == 0.0                       # the honest one


def test_a_gain_that_came_from_the_diagnosis_survives_the_control(tmp_path):
    """And the case where checking genuinely earned it, so the control must not erase it."""
    archive = PolicyArchive(tmp_path / "a.json")
    needs_the_diagnosis = lambda task, feedback: (
        "fixed" if "inspect the source" in (feedback or "") else "wrong")
    search = PolicySearch(runner(solve=needs_the_diagnosis),
                          lambda c, answer: float(answer == "fixed"),
                          lambda policy, feedback: Policy(("inspect",), 1), archive)
    report = search.run([case("d", "fd")], [[case("v", "fv")]], [case("t", "ft")])

    assert report["test"]["champion"]["score"] == 1.0
    assert report["test"]["matched_compute"]["score"] == 0.0
    assert report["compute_adjusted_gain"] == 1.0


def test_the_adjusted_gain_is_none_when_the_control_could_not_be_scored(tmp_path):
    """An incomplete control licenses no claim, exactly as an incomplete arm does not.

    Reporting `champion - initial` while quietly dropping a failed control would put the
    confounded number back in the headline slot with nothing marking it.
    """
    def breaks_only_under_the_control(task, feedback):
        feedback = feedback or ""
        if CONTROL_CHECK in feedback:
            raise RuntimeError("offline")
        return "fixed" if "inspect the source" in feedback else "wrong"

    archive = PolicyArchive(tmp_path / "a.json")
    search = PolicySearch(runner(solve=breaks_only_under_the_control),
                          lambda c, answer: float(answer == "fixed"),
                          lambda policy, feedback: Policy(("inspect",), 1), archive)
    report = search.run([case("d", "fd")], [[case("v", "fv")]], [case("t", "ft")])

    assert report["test"]["champion"]["score"] == 1.0     # the champion scored fine
    assert report["test"]["matched_compute"]["score"] is None
    assert report["compute_adjusted_gain"] is None        # and still claims nothing
