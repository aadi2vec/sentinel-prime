"""Check synthesis: criterion text -> a parameterised, deterministic check.

The boundary under test is the same one `Policy.parse` draws. A synthesized check is a
*filled template*, never emitted Python — `interpreter.py` is explicitly not a sandbox, so
the synthesizer selects a kind and supplies parameters, and anything it did not understand
is refused rather than coerced. These tests pin that refusal, then the deterministic
compile, then the admission bar that decides whether a synthesized check earns the catalog.
"""
import pytest

from sentinelprime.replay import GradedCriterion, GradedRun
from sentinelprime.synthesis import (
    CheckSpec, admissible, compile_check, synthesized_catalog,
)


# --- the parse boundary -------------------------------------------------------------

def test_an_unknown_check_kind_is_refused():
    """The synthesizer picks from the catalog of kinds the taxonomy commissioned. A kind
    nobody wrote a compiler for cannot be run, and coercing it to the nearest match would
    silently score a different check than the one proposed."""
    with pytest.raises(ValueError):
        CheckSpec.parse('{"kind": "sentiment", "params": {"required": ["x"]}}')


def test_an_unknown_param_field_is_refused():
    with pytest.raises(ValueError):
        CheckSpec.parse('{"kind": "span", "params": {"required": ["x"], "weight": 3}}')


def test_a_span_check_needs_at_least_one_term():
    """An empty term list compiles to a check that passes everything, which would score as
    a flawless silent check and pollute the catalog."""
    with pytest.raises(ValueError):
        CheckSpec.parse('{"kind": "span", "params": {"required": []}}')


def test_a_well_formed_span_spec_parses():
    spec = CheckSpec.parse('{"kind": "span", "params": {"required": ["Section 14.2"]}}')
    assert spec.kind == "span"
    assert spec.params["required"] == ["Section 14.2"]


# --- the deterministic compile ------------------------------------------------------

def test_a_span_check_passes_when_every_required_term_is_present():
    check = compile_check(CheckSpec("span", {"required": ["Monterrey", "consent"]}))
    verdict = check("task", "The Monterrey lease requires landlord consent.")
    assert verdict.passed


def test_a_span_check_fails_and_names_the_term_it_could_not_find():
    check = compile_check(CheckSpec("span", {"required": ["Monterrey", "consent"]}))
    verdict = check("task", "The lease requires landlord consent.")
    assert not verdict.passed
    assert "Monterrey" in verdict.feedback


def test_a_span_check_is_case_insensitive():
    check = compile_check(CheckSpec("span", {"required": ["Section 14.2"]}))
    assert check("task", "see section 14.2 of the agreement").passed


def test_a_lexical_check_fails_when_the_deliverable_declined_to_analyse():
    """The abandonment mode: 29 omissions dominate the corpus, but abandonment is the one
    mode stated in the deliverable's own words, so a phrase test reaches it."""
    check = compile_check(
        CheckSpec("lexical", {"forbidden": ["no substantive analysis"]}))
    verdict = check("task", "No substantive analysis was supplied for the lease.")
    assert not verdict.passed
    assert "no substantive analysis" in verdict.feedback.lower()


def test_a_lexical_check_passes_a_deliverable_that_did_the_work():
    check = compile_check(
        CheckSpec("lexical", {"forbidden": ["no substantive analysis"]}))
    assert check("task", "The lease was reviewed in full.").passed


def test_an_arithmetic_check_matches_an_amount_regardless_of_separators():
    check = compile_check(CheckSpec("arithmetic", {"amounts": ["1250000"]}))
    assert check("task", "The cap is $1,250,000 per claim.").passed


def test_an_arithmetic_check_fails_when_the_figure_is_absent():
    check = compile_check(CheckSpec("arithmetic", {"amounts": ["1250000"]}))
    verdict = check("task", "The cap is $900,000 per claim.")
    assert not verdict.passed


# --- the admission bar --------------------------------------------------------------

def _run(run_id, deliverable, verdict, cid="C-1"):
    return GradedRun(
        run_id=run_id, task_id="t", task="task", deliverable=deliverable,
        criteria=(GradedCriterion(id=cid, title="t", match_criteria="PASS if Monterrey",
                                  verdict=verdict, reasoning="r"),))


def test_a_check_that_reproduces_the_judge_on_a_split_criterion_is_admissible():
    corpus = [_run("a", "The Monterrey lease is flagged.", "pass"),
              _run("b", "Nothing about that facility.", "fail")]
    spec = CheckSpec("span", {"required": ["Monterrey"]})
    assert admissible(spec, corpus, "C-1", min_precision=0.9, min_recall=0.9)


def test_a_check_that_flags_everything_is_refused_however_good_its_recall():
    """The degenerate strategy. Perfect recall, and precision is the only thing that
    refuses it — which is why admission can never be recall alone."""
    corpus = [_run("a", "The Monterrey lease is flagged.", "pass"),
              _run("b", "Nothing about that facility.", "fail")]
    spec = CheckSpec("span", {"required": ["a-term-in-no-deliverable"]})
    assert not admissible(spec, corpus, "C-1", min_precision=0.9, min_recall=0.9)


def test_a_check_is_refused_on_a_criterion_the_judge_never_split():
    """A criterion that always passed is reproduced perfectly by a constant, so agreement
    there says nothing about whether the check reads the deliverable at all."""
    corpus = [_run("a", "The Monterrey lease is flagged.", "pass"),
              _run("b", "The Monterrey lease is flagged again.", "pass")]
    spec = CheckSpec("span", {"required": ["Monterrey"]})
    assert not admissible(spec, corpus, "C-1", min_precision=0.0, min_recall=0.0)


def test_the_catalog_only_contains_checks_that_cleared_the_bar():
    corpus = [_run("a", "The Monterrey lease is flagged.", "pass"),
              _run("b", "Nothing about that facility.", "fail")]
    proposed = {
        "C-1_good": CheckSpec("span", {"required": ["Monterrey"]}),
        "C-1_always_flags": CheckSpec("span", {"required": ["absent-term"]}),
    }
    catalog = synthesized_catalog(proposed, corpus, {"C-1_good": "C-1",
                                                     "C-1_always_flags": "C-1"})
    assert set(catalog) == {"C-1_good"}


# --- the seam onto the live path -----------------------------------------------------
#
# A synthesized check that nothing can select is a measurement, not a mechanism. These
# pin the path from an admitted check to a catalog a candidate `Policy` can name.

def test_an_admitted_check_round_trips_through_a_file(tmp_path):
    """The replay stage and the live stage are separate processes — replay admits a check
    at ~0 cost, a live run spends an agent on it — so the catalog has to survive on disk.
    `CheckSpec` is the only format that crosses, because it is the only one `parse`
    refuses malformed input in."""
    import json
    from sentinelprime.synthesis import dump_specs, load_specs
    specs = {"C-1_monterrey": CheckSpec("span", {"required": ["Monterrey"]})}
    path = tmp_path / "checks.json"
    path.write_text(json.dumps(dump_specs(specs), indent=2))
    assert load_specs(path) == specs


def test_a_malformed_check_file_is_refused_rather_than_partly_loaded(tmp_path):
    from sentinelprime.synthesis import load_specs
    path = tmp_path / "checks.json"
    path.write_text('{"ok": {"kind": "span", "params": {"required": ["x"]}},'
                    ' "bad": {"kind": "sentiment", "params": {}}}')
    with pytest.raises(ValueError):
        load_specs(path)


def test_a_synthesized_check_becomes_selectable_by_a_candidate_policy():
    """`PolicyProposer` can only name checks the *description* catalog lists, and
    `PolicyRunner` can only run ones the *callable* catalog holds. A check in one and not
    the other is either unreachable or unrunnable, so the two are built together."""
    from sentinelprime.policy_experiment import build_catalogs
    descriptions, checks = build_catalogs(
        {"C-1_monterrey": CheckSpec("span", {"required": ["Monterrey"]})})
    assert "C-1_monterrey" in descriptions and "C-1_monterrey" in checks
    assert set(descriptions) == set(checks)


def test_the_hand_written_checks_survive_synthesis():
    """Synthesis adds vocabulary; it does not replace the baseline. If the hand-written
    checks vanished, a candidate's gain would be measured against a different catalog than
    the one §5 reports on."""
    from sentinelprime.policy_experiment import CHECKS, build_catalogs
    descriptions, checks = build_catalogs(
        {"C-1_monterrey": CheckSpec("span", {"required": ["Monterrey"]})})
    assert set(CHECKS) <= set(checks)


def test_a_synthesized_name_colliding_with_a_hand_written_check_is_refused():
    """Shadowing `arithmetic` would silently change what the baseline arm measured while
    still reporting itself as the same catalog — the `credit=True, audit=False` failure in
    the search's vocabulary."""
    from sentinelprime.policy_experiment import build_catalogs
    with pytest.raises(ValueError, match="arithmetic"):
        build_catalogs({"arithmetic": CheckSpec("span", {"required": ["x"]})})


def test_no_synthesized_checks_leaves_the_hand_written_catalog_exactly_as_it_was():
    from sentinelprime.policy_experiment import CATALOG, CHECKS, build_catalogs
    descriptions, checks = build_catalogs(None)
    assert descriptions == CATALOG and checks == CHECKS


def test_a_synthesized_description_names_the_terms_it_checks():
    """The proposer selects from text alone. A description that says only "synthesized
    check" gives it nothing to select *on*, so the catalog degenerates to a list of
    opaque names."""
    from sentinelprime.policy_experiment import build_catalogs
    descriptions, _ = build_catalogs(
        {"C-1_monterrey": CheckSpec("span", {"required": ["Monterrey"]})})
    assert "Monterrey" in descriptions["C-1_monterrey"]
