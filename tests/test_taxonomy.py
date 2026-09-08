"""How a deliverable failed, and whether a deterministic check could have caught it.

Synthesis needs this first. Asked to invent checks with no account of what actually goes
wrong, an LM produces checks for the failures it imagines — and the imagined distribution
is nothing like the real one. Across 50 failed LAB criteria the split was 29 omissions,
17 partials and roughly zero fabrications, which means a check that verifies the claims a
deliverable *made* addresses almost none of them. That is the finding that decides what
synthesis should target.

The classifier reads the judge's own explanation, not the deliverable. That is a proxy and
its limits are real: it inherits the judge's phrasing conventions, and a rewording of the
rubric prompt would shift the counts. It is instrumentation for choosing what to build,
never a label anything is trained against.
"""
import pytest

from sentinelprime.replay import GradedCriterion
from sentinelprime.taxonomy import (
    ABANDONMENT, DETECTABILITY, FABRICATION, OMISSION, PARTIAL, UNCLASSIFIED, WRONG_VALUE,
    classify, taxonomy_report,
)


def _criterion(reasoning, verdict="fail", cid="C-1"):
    return GradedCriterion(id=cid, title="t", match_criteria="PASS if x",
                           verdict=verdict, reasoning=reasoning)


def test_a_passing_criterion_has_no_failure_mode():
    assert classify(_criterion("The report identifies Section 14.2.", verdict="pass")) is None


def test_an_omission_is_a_claim_the_deliverable_never_made():
    assert classify(_criterion(
        "The memo does not identify the Monterrey facility lease or its consent "
        "requirement.")) == OMISSION


def test_an_abandonment_is_a_source_the_deliverable_declined_to_analyse():
    assert classify(_criterion(
        "The report states that no substantive analysis was supplied for the Summit "
        "credit agreement.")) == ABANDONMENT


def test_abandonment_outranks_omission_when_the_explanation_says_both():
    """Most abandonment explanations also say the claim is missing — of course it is, the
    document was never read. Classifying that as omission would hide the larger fault and
    point synthesis at a span check when the real problem is coverage."""
    assert classify(_criterion(
        "The report states no substantive analysis was supplied for the JV agreement and "
        "does not identify the Section 12.3(b) carve-out.")) == ABANDONMENT


def test_a_partial_made_the_claim_but_left_a_component_out():
    assert classify(_criterion(
        "The memo identifies two omitted patents but does not specify their patent "
        "numbers.")) == PARTIAL


def test_a_wrong_value_asserted_something_the_source_contradicts():
    assert classify(_criterion(
        "The memo references an implied value of approximately $250 million instead of "
        "the $287 million aggregate purchase price.")) == WRONG_VALUE


def test_an_unrecognised_explanation_is_unclassified_rather_than_guessed():
    """A forced bucket would inflate whichever mode owns the loosest pattern, and the
    counts are the whole reason this exists."""
    assert classify(_criterion("The report is unsatisfactory.")) == UNCLASSIFIED


# --- what the report is for -----------------------------------------------------------

def test_the_report_counts_modes_over_the_corpus(tmp_path):
    from sentinelprime.replay import GradedRun

    corpus = [GradedRun("r1", "t1", "task", "deliverable", (
        _criterion("does not identify the lease", cid="C-1"),
        _criterion("does not mention the penalty", cid="C-2"),
        _criterion("no substantive analysis was supplied", cid="C-3"),
        _criterion("identifies it but does not quantify the exposure", cid="C-4"),
        _criterion("The report is fine.", verdict="pass", cid="C-5"),
    ))]

    report = taxonomy_report(corpus)

    assert report.counts[OMISSION] == 2
    assert report.counts[ABANDONMENT] == 1
    assert report.counts[PARTIAL] == 1
    assert report.graded == 5 and report.failures == 4
    assert OMISSION in report.dominant


def test_the_report_says_what_a_deterministic_check_could_reach():
    """The number synthesis is actually commissioned from: how much of the real failure
    mass is reachable by a check at all, and by which kind."""
    assert DETECTABILITY[OMISSION] == "span"
    assert DETECTABILITY[ABANDONMENT] == "lexical"
    assert DETECTABILITY[WRONG_VALUE] == "arithmetic"
    assert DETECTABILITY[FABRICATION] == "grounding"
    assert DETECTABILITY[UNCLASSIFIED] is None


def test_reachable_mass_excludes_what_no_check_kind_covers():
    from sentinelprime.replay import GradedRun

    corpus = [GradedRun("r1", "t1", "task", "d", (
        _criterion("does not identify the lease", cid="C-1"),      # span-reachable
        _criterion("The report is unsatisfactory.", cid="C-2"),    # unclassified
    ))]

    report = taxonomy_report(corpus)

    assert report.reachable == 1
    assert report.reachable_fraction == 0.5


# --- gaps found by reading the unclassified bucket against the real corpus -------------
# Adding verbs the judge uses that the first pass missed is legitimate; adding patterns
# until nothing is unclassified would be fitting the classifier to this corpus. The line
# drawn here: same *kind* of phrasing, different word. A value contradiction stated as
# "is rated Medium, not Critical" is left unclassified rather than matched by a pattern
# loose enough to catch it, because that pattern would also catch ordinary prose.

@pytest.mark.parametrize("reasoning", [
    "The report does not reference UCC section 2-210 or the delegation rules.",
    "It does not flag the Crestline ERP license as a change-of-control risk.",
    "The report does not calculate the JV buy-out price of $22.95 million.",
    "The report does not elevate the rating despite the indirect assignment risk.",
    "The memo does not provide the patent numbers for the omitted patents.",
])
def test_omission_covers_the_judge_verbs_the_first_pass_missed(reasoning):
    assert classify(_criterion(reasoning)) == OMISSION


@pytest.mark.parametrize("reasoning", [
    "The report identifies the 50% and 35% triggers and describes their consequences, "
    "but it never expressly analyzes the interaction between them.",
    "The report mentions reverse mergers generally, but it never expressly states that "
    "the provision covers triangular mergers.",
])
def test_partial_covers_the_never_expressly_hinge(reasoning):
    """`never expressly X` is the same shape as `does not X` after a made claim."""
    assert classify(_criterion(reasoning)) == PARTIAL


def test_a_value_contradiction_without_a_marker_stays_unclassified():
    """'is rated Medium, not Critical' is a wrong value, and no pattern tight enough to
    be safe will catch it. Left unclassified deliberately: under-claiming the reachable
    fraction is the honest direction to be wrong in."""
    assert classify(_criterion(
        "The Northland MSA is expressly rated Medium, not Critical or High.")) == UNCLASSIFIED
