"""The parity family — the kill gate for the write path (PLAN §3a, Phase 0).

This family exists to answer one question before Phases 1-3 are paid for: does the
baseline RLM already handle surgical in-place context edits competently? If it does, the
write path is not this system's bottleneck and the phases are cancelled.

So the thing these tests pin hardest is the *separation of two measurements*. A full
rewrite that happens to contain the right new value is correct and not surgical, and a
scorer that collapses those into one number cannot answer the gate's question. Correctness
and footprint are therefore scored independently, and footprint over an empty preserve set
is None — unknown, not perfect, the same convention `CheckScore.precision` and
`credit.success_rate` already use.
"""
import pytest

from sentinelprime.parity import (
    FAMILY_KINDS, ParitySpec, build_cases, evaluate, gate_verdict, kind_of, parity_report,
    score,
)


# --- the scoring boundary -----------------------------------------------------------

def test_an_answer_with_every_required_string_is_correct():
    verdict = score(ParitySpec(require=("Monterrey",)), "The Monterrey lease.")
    assert verdict.correct


def test_a_missing_required_string_is_not_correct_and_is_named():
    verdict = score(ParitySpec(require=("Monterrey", "consent")), "The lease needs consent.")
    assert not verdict.correct
    assert verdict.missing == ("Monterrey",)


def test_a_leaked_forbidden_string_is_not_correct():
    """The pre-edit value left behind. The new value being present does not mean the old
    one was removed, and an edit that only appended did not edit anything."""
    verdict = score(ParitySpec(require=("1,450,000",), forbid=("1,250,000",)),
                    "cap raised from 1,250,000 to 1,450,000")
    assert not verdict.correct
    assert verdict.leaked == ("1,250,000",)


def test_matching_is_case_insensitive():
    assert score(ParitySpec(require=("Section 14.2",)), "see section 14.2").correct


# --- footprint: what the task said to keep, and whether it survived ------------------

def test_footprint_is_unknown_when_nothing_was_marked_for_preservation():
    """Unknown must never read as a flawless edit. A retrieval task preserves nothing, so
    it has no opinion on surgicality and must not contribute a 0.0 to the gate's mean."""
    verdict = score(ParitySpec(require=("x",)), "x")
    assert verdict.footprint is None


def test_footprint_is_zero_when_every_preserved_line_survives():
    spec = ParitySpec(require=("new",), preserve=("line A", "line B"))
    assert score(spec, "line A\nnew\nline B").footprint == 0.0


def test_footprint_is_one_when_nothing_preserved_survives():
    spec = ParitySpec(require=("new",), preserve=("line A", "line B"))
    assert score(spec, "new").footprint == 1.0


def test_footprint_is_partial_and_names_what_was_disturbed():
    spec = ParitySpec(require=("new",), preserve=("line A", "line B"))
    verdict = score(spec, "line A\nnew")
    assert verdict.footprint == 0.5
    assert verdict.disturbed == ("line B",)


def test_a_correct_full_rewrite_is_distinguished_only_by_its_footprint():
    """The gate's whole purpose. Both answers carry the new value, so both are correct;
    only footprint separates the surgical edit from the rewrite that lost the rest."""
    spec = ParitySpec(require=("new",), preserve=("line A", "line B"))
    surgical = score(spec, "line A\nnew\nline B")
    rewrite = score(spec, "new")
    assert surgical.correct and rewrite.correct
    assert surgical.footprint == 0.0 and rewrite.footprint == 1.0


# --- the evaluator the policy search consumes ---------------------------------------

def test_the_evaluator_returns_a_fraction_in_range():
    case = build_cases(seed=0)[0]
    assert 0.0 <= evaluate(case, "") <= 1.0


def test_the_evaluator_scores_correctness_only_not_footprint():
    """Folding footprint into the search objective would make surgicality a thing a
    candidate policy is selected for — a different claim from the gate's measurement, and
    one §6 reserves for the evaluator rather than the search."""
    case = next(c for c in build_cases(seed=0) if kind_of(c.family) == "sketchpad")
    full_rewrite = _answer_satisfying(case, surgical=False)
    assert evaluate(case, full_rewrite) == 1.0
    assert score(case.gold, full_rewrite).footprint == 1.0


def test_a_leak_zeroes_the_evaluator_score():
    spec = ParitySpec(require=("new",), forbid=("old",))
    assert evaluate(_case_with(spec), "old and new") == 0.0


# --- the generated families ----------------------------------------------------------

def test_every_advertised_kind_is_generated():
    kinds = {kind_of(c.family) for c in build_cases(seed=0)}
    assert kinds == set(FAMILY_KINDS)


def test_case_ids_are_unique():
    cases = build_cases(seed=0, per_kind=3)
    assert len({c.id for c in cases}) == len(cases)


def test_families_are_disjoint_so_splits_can_be_reserved():
    """`PolicySearch.run` refuses a family that spans two groups and `PolicyArchive.reserve`
    refuses a family twice, so every case instance needs its own family — one family per
    kind would make dev/validation/test unsplittable."""
    cases = build_cases(seed=0, per_kind=3)
    assert len({c.family for c in cases}) == len(cases)


def test_generation_is_deterministic_for_a_seed():
    assert ([c.task for c in build_cases(seed=7, per_kind=2)]
            == [c.task for c in build_cases(seed=7, per_kind=2)])


def test_a_different_seed_produces_different_tasks():
    """§3 names "one generated template, shared across all splits" as a defect of the
    current substrate. A held-out split that is the training split with other numbers is
    not held out."""
    assert ([c.task for c in build_cases(seed=1, per_kind=2)]
            != [c.task for c in build_cases(seed=2, per_kind=2)])


def test_the_needle_task_actually_contains_the_fact_it_asks_for():
    case = next(c for c in build_cases(seed=0) if kind_of(c.family) == "needle")
    assert all(r.lower() in case.task.lower() for r in case.gold.require)


def test_the_sketchpad_task_preserves_everything_except_its_one_target():
    """A surgical edit is only measurable if the task names exactly one thing to change
    and the untouched remainder is scored."""
    case = next(c for c in build_cases(seed=0) if kind_of(c.family) == "sketchpad")
    assert len(case.gold.require) == 1
    assert len(case.gold.forbid) == 1
    assert len(case.gold.preserve) >= 2
    assert case.gold.forbid[0] not in case.gold.preserve


def test_the_kvstore_task_states_a_capacity_below_its_item_count():
    """With capacity at or above the item count nothing has to be evicted, and the task
    silently stops testing retention under pressure."""
    case = next(c for c in build_cases(seed=0) if kind_of(c.family) == "kvstore")
    assert case.gold.capacity is not None
    assert case.gold.capacity < case.gold.item_count


def test_the_triage_task_forbids_the_noise_it_must_drop():
    case = next(c for c in build_cases(seed=0) if kind_of(c.family) == "triage")
    assert case.gold.require and case.gold.forbid


def test_echoing_the_prompt_back_is_refused_on_every_kind():
    """Keeping everything is the degenerate strategy on all four kinds, and three of them
    are retrieval-shaped, so the answer is sitting in the prompt. Precision is the only
    thing that refuses it — the same argument `synthesis.admissible` makes for why check
    admission can never be recall alone. Every kind therefore carries a forbid set."""
    for case in build_cases(seed=0, per_kind=2):
        assert evaluate(case, case.task) < 1.0, kind_of(case.family)


def test_the_grading_key_is_never_serialized_into_a_task_prompt():
    """Invariant 3, in the form that is actually true here. A retrieval task necessarily
    contains its answer inside the haystack — that is what makes it a retrieval task — so
    what must be absent from the prompt is the *key*: the spec naming which strings are
    graded and which must not survive."""
    for case in build_cases(seed=0, per_kind=2):
        assert "require" not in case.task
        assert "forbid" not in case.task
        assert "preserve" not in case.task


def test_the_sketchpad_target_value_must_be_derived_not_copied():
    """An in-place edit whose new value is quoted in the instruction is satisfiable by
    appending that literal, which would score a copy as an edit. The instruction states a
    delta instead, so the graded binding has to be computed from the entry being edited."""
    for case in build_cases(seed=0, per_kind=3):
        if kind_of(case.family) != "sketchpad":
            continue
        assert all(r not in case.task for r in case.gold.require)


# --- the gate report -----------------------------------------------------------------

def test_the_report_separates_correctness_from_footprint_per_kind():
    case = next(c for c in build_cases(seed=0) if kind_of(c.family) == "sketchpad")
    rows = [(case, _answer_satisfying(case, surgical=False))]
    report = parity_report(rows)
    assert report["sketchpad"]["correct_rate"] == 1.0
    assert report["sketchpad"]["mean_footprint"] == 1.0


def test_the_report_refuses_a_footprint_verdict_without_observations():
    case = next(c for c in build_cases(seed=0) if kind_of(c.family) == "needle")
    report = parity_report([(case, "")])
    assert report["needle"]["mean_footprint"] is None


def test_the_report_renders_a_line_naming_the_gate_verdict():
    cases = build_cases(seed=0)
    report = parity_report([(c, "") for c in cases])
    assert "sketchpad" in report["describe"]


# --- the kill condition, decided in advance ------------------------------------------

def _sketchpad_cases(n=4, seed=0):
    return [c for c in build_cases(seed=seed, per_kind=n)
            if kind_of(c.family) == "sketchpad"]


def test_a_competent_surgical_baseline_cancels_the_write_path():
    """PLAN §3a states the kill condition in advance: if the baseline already edits in
    place competently, Phases 1-3 are cancelled."""
    rows = [(c, _answer_satisfying(c, surgical=True)) for c in _sketchpad_cases()]
    assert gate_verdict(parity_report(rows))["decision"] == "cancel"


def test_a_baseline_that_is_correct_only_by_rewriting_justifies_the_write_path():
    rows = [(c, _answer_satisfying(c, surgical=False)) for c in _sketchpad_cases()]
    assert gate_verdict(parity_report(rows))["decision"] == "build"


def test_a_baseline_that_cannot_do_the_task_at_all_is_inconclusive():
    """Footprint on a wrong answer measures nothing about editing. A baseline that fails
    the task may simply be bad at the task, which is not evidence about the write path."""
    rows = [(c, "") for c in _sketchpad_cases()]
    assert gate_verdict(parity_report(rows))["decision"] == "inconclusive"


def test_a_gate_with_no_surgicality_observations_is_inconclusive():
    rows = [(c, "") for c in build_cases(seed=0) if kind_of(c.family) == "needle"]
    assert gate_verdict(parity_report(rows))["decision"] == "inconclusive"


def test_the_verdict_states_the_numbers_it_rested_on():
    """A decision to cancel two phases of work has to be re-checkable later against the
    evidence that produced it."""
    rows = [(c, _answer_satisfying(c, surgical=True)) for c in _sketchpad_cases()]
    verdict = gate_verdict(parity_report(rows))
    assert verdict["correct_rate"] == 1.0
    assert verdict["mean_footprint"] == 0.0
    assert "sketchpad" in verdict["why"]


# --- helpers -------------------------------------------------------------------------

def _case_with(spec):
    from sentinelprime.policy_search import Case
    return Case(id="x", family="parity-test-0", task="t", gold=spec)


def _answer_satisfying(case, *, surgical: bool) -> str:
    """An answer that satisfies every requirement, optionally keeping the preserved lines."""
    parts = list(case.gold.require)
    if surgical:
        parts = list(case.gold.preserve) + parts
    return "\n".join(parts)
