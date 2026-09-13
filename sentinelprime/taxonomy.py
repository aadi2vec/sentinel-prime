"""How a deliverable failed, and whether a deterministic check could have caught it.

This exists to make check synthesis well-posed. Asked to invent checks with no account of
what actually goes wrong, a model produces checks for the failures it imagines — and the
imagined distribution is nothing like the observed one. Across 50 failed LAB criteria the
split was 29 omissions, 17 partials and roughly zero fabrications. A check that verifies the
claims a deliverable *made* therefore addresses almost none of the real failures, which is
the opposite of where citation-style verification points.

Five modes, ordered by what a check would have to do to catch each:

  abandonment  the deliverable declined to analyse a source at all
  omission     the claim was never made
  partial      the claim was made but a component is missing
  wrong value  a claim was made and contradicts the source
  fabrication  a claim was made with no support in the source

Precedence is deliberate and abandonment leads it. Most abandonment explanations *also* say
the claim is missing — of course they do, the document was never read — so classifying on
first match would file them under omission and point synthesis at a span check when the real
fault is coverage. The larger fault wins.

**This reads the judge's explanation, not the deliverable.** That is a proxy and its limits
are real: it inherits the judge's phrasing conventions, and rewording the rubric prompt would
move the counts. It is instrumentation for deciding what to build. Nothing is trained against
these labels, and an explanation that matches no pattern stays `UNCLASSIFIED` rather than
being forced into the bucket with the loosest regex — a forced bucket would inflate exactly
the number this module exists to report.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sentinelprime.replay import GradedRun

ABANDONMENT = "abandonment"
OMISSION = "omission"
PARTIAL = "partial"
WRONG_VALUE = "wrong_value"
FABRICATION = "fabrication"
UNCLASSIFIED = "unclassified"

MODES = (ABANDONMENT, OMISSION, PARTIAL, WRONG_VALUE, FABRICATION, UNCLASSIFIED)

# What kind of deterministic check could reach each mode. This is the table that
# commissions synthesis: it says which check kinds are worth generating and, by omission,
# which failures no check of any kind will find.
DETECTABILITY: dict[str, str | None] = {
    # "the document was never read" is stated in the deliverable's own words.
    ABANDONMENT: "lexical",
    # a required entity or section reference simply is not present.
    OMISSION: "span",
    # the claim is present, a named component of it is not — still a span test, but on the
    # component rather than the claim, which is why synthesis needs the criterion's wording.
    PARTIAL: "span",
    # a stated figure disagrees with one derivable from the source.
    WRONG_VALUE: "arithmetic",
    # a claim resolves to nothing in the source at all.
    FABRICATION: "grounding",
    # no check kind applies, because we could not say what went wrong.
    UNCLASSIFIED: None,
}

# Ordered. The first pattern that matches wins, so the list order *is* the precedence rule.
_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    (ABANDONMENT, re.compile(
        r"no substantive analysis|not supplied|could not be performed|was not provided"
        r"|no analysis was|declines? to analy[sz]e|leaves the issue undetermined", re.I)),
    (WRONG_VALUE, re.compile(
        r"\binstead of\b|\brather than the\b|incorrectly (?:states|reports|identifies)"
        r"|misstates|contradicts the", re.I)),
    (FABRICATION, re.compile(
        r"\bfabricat|\bno support in the (?:source|record)|not found in (?:any|the) "
        r"(?:document|source)|invented", re.I)),
    # Partial before omission: "identifies X but does not specify Y" contains an omission
    # phrase, and the claim was in fact made. The contrastive hinge is what separates them.
    # `never expressly X` is the same shape as `does not X` once a claim has been made,
    # and the judge uses both freely.
    (PARTIAL, re.compile(
        r"(?:identifies|states|notes|discusses|mentions|includes|describes|quantifies)\b"
        r".{0,400}?\b(?:but|however|although)\b.{0,200}?"
        r"(?:\bdoes not\b|\bnever\b|\bfails to\b)", re.I | re.S)),
    (OMISSION, re.compile(
        r"(?:does not|did not|fails to|failed to)\s+"
        r"(?:identify|mention|quantify|state|discuss|analy[sz]e|address|specify|expressly"
        r"|include|report|name|reference|flag|calculate|elevate|provide|note|cite|list)"
        r"|omits|never (?:identifies|mentions|states|references)", re.I)),
)


def classify(criterion) -> str | None:
    """The failure mode of one graded criterion, or None if it passed.

    Precedence follows `_PATTERNS` order, not text position: an explanation naming both an
    abandoned source and a missing claim is abandonment, and one naming both a made claim
    and a missing component is partial.
    """
    if str(criterion.verdict).strip().lower() != "fail":
        return None
    reasoning = criterion.reasoning or ""
    for mode, pattern in _PATTERNS:
        if pattern.search(reasoning):
            return mode
    return UNCLASSIFIED


@dataclass(frozen=True)
class TaxonomyReport:
    counts: dict[str, int] = field(default_factory=dict)
    graded: int = 0
    failures: int = 0
    by_criterion: dict[str, str] = field(default_factory=dict)

    @property
    def dominant(self) -> list[str]:
        """Modes ordered by how much failure mass they carry. Synthesis targets the top."""
        return sorted((m for m in self.counts if self.counts[m]),
                      key=lambda m: -self.counts[m])

    @property
    def reachable(self) -> int:
        """Failures a deterministic check of *some* kind could detect."""
        return sum(n for mode, n in self.counts.items() if DETECTABILITY.get(mode))

    @property
    def reachable_fraction(self) -> float | None:
        """The number that commissions synthesis, or None when nothing failed.

        A low fraction is a result, not a setback: it says the ceiling on deterministic
        checking is set by how much of the real failure mass any check kind can touch.
        """
        return self.reachable / self.failures if self.failures else None

    def by_check_kind(self) -> dict[str, int]:
        """Failure mass per check kind — what to generate, in priority order."""
        kinds: dict[str, int] = {}
        for mode, n in self.counts.items():
            kind = DETECTABILITY.get(mode)
            if kind and n:
                kinds[kind] = kinds.get(kind, 0) + n
        return dict(sorted(kinds.items(), key=lambda kv: -kv[1]))


def taxonomy_report(corpus: list[GradedRun]) -> TaxonomyReport:
    """Classify every graded criterion in the corpus.

    Counted per (run, criterion) rather than per criterion: the same criterion fails
    differently on different deliverables, and collapsing those would discard exactly the
    variation that makes the corpus discriminative.
    """
    counts = {mode: 0 for mode in MODES}
    by_criterion: dict[str, str] = {}
    graded = failures = 0
    for run in corpus:
        for criterion in run.criteria:
            graded += 1
            mode = classify(criterion)
            if mode is None:
                continue
            failures += 1
            counts[mode] += 1
            by_criterion[f"{run.run_id}:{criterion.id}"] = mode
    return TaxonomyReport(counts=counts, graded=graded, failures=failures,
                          by_criterion=by_criterion)
