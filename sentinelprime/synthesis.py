"""Check synthesis: a rubric criterion's wording in, a deterministic check out.

This is the open item on the plan's critical path (§3 item 3), and `taxonomy.py` is what
makes it well-posed. Asked to invent checks with no account of what actually goes wrong, a
model writes checks for the failures it imagines — and the imagined distribution is nothing
like the observed one. `taxonomy.by_check_kind()` reports the real failure mass per check
kind, so the kinds compiled here are the ones that mass commissions: `span` for the
omissions and partials that dominate, `lexical` for abandonment (the one mode a deliverable
states in its own words), `arithmetic` for wrong values. There is deliberately no synthesized
grounding check — fabrication is ~0 in the corpus, so citation-style verification is where
the intuition points and almost none of the failures are.

**A synthesized check is a filled template, never emitted code.** `interpreter.py` is
explicitly not a sandbox, so the synthesizer chooses a `kind` and supplies parameters, and
`CheckSpec.parse` refuses anything it does not recognise — the same boundary `Policy.parse`
draws, as a type rather than a convention. Coercing an unrecognised field to the nearest
match would score a different check than the one proposed.

**Agreement with the judge is the admission bar, and it is deliberately asymmetric.** A
check earns the catalog only on criteria the judge *split* — passed on one deliverable and
failed on another — because a criterion that always passed is reproduced perfectly by a
constant, so agreement there says nothing about whether the check reads the deliverable at
all. Precision is weighted above recall because the costs differ: a false alarm spends a
solver attempt on a deliverable that was already fine, while a miss merely leaves the
failure where it already was. Recall alone would admit the degenerate always-flag check.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import dspy

from sentinelprime.policy_search import CheckResult
from sentinelprime.replay import GradedRun, discriminative_criteria, score_check

# kind -> the exact parameter keys that kind accepts. This table *is* the synthesizer's
# vocabulary: a kind absent here has no compiler, and a key absent here is refused.
KINDS: dict[str, frozenset[str]] = {
    "span": frozenset({"required"}),
    "lexical": frozenset({"forbidden"}),
    "arithmetic": frozenset({"amounts"}),
}

# The parameter each kind reads. Every one is a non-empty list of non-empty strings.
_TERMS_KEY = {"span": "required", "lexical": "forbidden", "arithmetic": "amounts"}


@dataclass(frozen=True)
class CheckSpec:
    kind: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        allowed = KINDS.get(self.kind)
        if allowed is None:
            raise ValueError(f"unknown check kind {self.kind!r}; expected one of "
                             f"{sorted(KINDS)}")
        unknown = set(self.params) - allowed
        if unknown:
            raise ValueError(f"check kind {self.kind!r} does not accept "
                             f"{sorted(unknown)}")
        terms = self.params.get(_TERMS_KEY[self.kind])
        # An empty term list compiles to a check that passes (or flags) everything, which
        # scores as a flawless silent check and pollutes the catalog with a constant.
        if (not isinstance(terms, list) or not terms
                or any(not isinstance(t, str) or not t.strip() for t in terms)):
            raise ValueError(f"check kind {self.kind!r} needs a non-empty "
                             f"{_TERMS_KEY[self.kind]!r} list of non-empty strings")

    @classmethod
    def parse(cls, raw: str) -> "CheckSpec":
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) != {"kind", "params"}:
            raise ValueError("check spec fields must be exactly kind (str), params (dict)")
        if not isinstance(data["params"], dict):
            raise ValueError("params must be an object")
        return cls(data["kind"], data["params"])


_SEPARATORS = re.compile(r"[,\s]")


def _normalize(text: str) -> str:
    """Lowercase, and strip the separators that make an equal figure compare unequal."""
    return _SEPARATORS.sub("", (text or "").lower())


def compile_check(spec: CheckSpec) -> Callable[[str, str], CheckResult]:
    """A `(task, answer) -> CheckResult` predicate, deterministic and LM-free.

    The shape is the live runner's own, so a check that scores well under replay drops into
    the catalog without being rewritten. The compiled closure never sees the verdict it
    will be scored against.
    """
    if spec.kind == "span":
        required = spec.params["required"]

        def span(_task: str, answer: str) -> CheckResult:
            haystack = (answer or "").lower()
            missing = [t for t in required if t.lower() not in haystack]
            if missing:
                return CheckResult(False, f"missing required span(s): {missing}")
            return CheckResult(True, "all required spans present")
        return span

    if spec.kind == "lexical":
        forbidden = spec.params["forbidden"]

        def lexical(_task: str, answer: str) -> CheckResult:
            haystack = (answer or "").lower()
            found = [p for p in forbidden if p.lower() in haystack]
            if found:
                return CheckResult(False, f"declined-to-analyse language present: {found}")
            return CheckResult(True, "no declined-to-analyse language")
        return lexical

    # arithmetic: a required figure must appear, separator-insensitively.
    amounts = spec.params["amounts"]

    def arithmetic(_task: str, answer: str) -> CheckResult:
        haystack = _normalize(answer)
        missing = [a for a in amounts if _normalize(a) not in haystack]
        if missing:
            return CheckResult(False, f"expected figure(s) not found: {missing}")
        return CheckResult(True, "all expected figures present")
    return arithmetic


def admissible(spec: CheckSpec, corpus: list[GradedRun], criterion_id: str,
               min_precision: float = 0.9, min_recall: float = 0.5) -> bool:
    """Does this check reproduce the judge well enough to join the catalog?

    Two gates, and the first one is the one that is easy to forget. The criterion must be
    *discriminative* — the judge both passed and failed it somewhere in the corpus —
    because a criterion with one verdict is reproduced by a constant and proves nothing.
    Only then is agreement scored, precision first.
    """
    if criterion_id not in discriminative_criteria(corpus):
        return False
    score = score_check(compile_check(spec), corpus, criterion_id)
    if score.precision is None or score.recall is None:
        # Flagged nothing, or nothing was wrong: untested rather than good.
        return False
    return score.precision >= min_precision and score.recall >= min_recall


def synthesized_catalog(proposed: dict[str, CheckSpec], corpus: list[GradedRun],
                        criterion_for: dict[str, str], min_precision: float = 0.9,
                        min_recall: float = 0.5) -> dict[str, Callable[[str, str], CheckResult]]:
    """Compile only the proposals that cleared the bar, keyed by check name.

    A rejected proposal leaves nothing behind here on purpose: the catalog is what the
    policy search is allowed to select from, so an unproven check must not be reachable.
    What was proposed and refused belongs in a log, not in the vocabulary.
    """
    catalog: dict[str, Callable[[str, str], CheckResult]] = {}
    for name, spec in proposed.items():
        criterion_id = criterion_for.get(name)
        if criterion_id is None:
            continue
        if admissible(spec, corpus, criterion_id, min_precision, min_recall):
            catalog[name] = compile_check(spec)
    return catalog


def dump_specs(specs: dict[str, CheckSpec]) -> dict:
    """Specs as plain JSON, keyed by check name — the shape `load_specs` reads back."""
    return {name: {"kind": s.kind, "params": s.params} for name, s in specs.items()}


def load_specs(path) -> dict[str, CheckSpec]:
    """Read a catalog of admitted checks back, refusing the whole file on any bad entry.

    Replay and the live run are separate processes — replay admits a check at ~0 cost, a
    live run spends an agent attempt on it — so an admitted catalog has to survive on disk,
    and `CheckSpec.parse` is the only format with a refusal in it. Loading the good entries
    and dropping the bad ones would run a *different* check set than the one that cleared
    the bar, under the name of the one that did.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("a check catalog is a JSON object of name -> {kind, params}")
    return {name: CheckSpec.parse(json.dumps(raw)) for name, raw in data.items()}


def describe_spec(spec: CheckSpec) -> str:
    """One line for the proposer's catalog, naming the terms the check actually reads.

    `PolicyProposer` selects from descriptions alone, so a description that says only
    "synthesized check" gives it nothing to select *on* and the catalog degenerates into a
    list of opaque names.
    """
    terms = ", ".join(repr(t) for t in spec.params[_TERMS_KEY[spec.kind]])
    verb = {"span": "Check that the answer contains",
            "lexical": "Check that the answer avoids",
            "arithmetic": "Check that the answer states the figure(s)"}[spec.kind]
    return f"{verb}: {terms}. (synthesized {spec.kind} check)"


class SynthesizeCheck(dspy.Signature):
    """Write a deterministic check that decides whether a deliverable satisfies one rubric
    criterion.

    Return ONLY a JSON object with exactly two fields, kind and params:
      {"kind":"span","params":{"required":["<literal text that must appear>", ...]}}
      {"kind":"lexical","params":{"forbidden":["<declined-to-analyse phrasing>", ...]}}
      {"kind":"arithmetic","params":{"amounts":["<figure that must appear>", ...]}}

    Choose `span` for a claim or component that must be present (this is the common case),
    `lexical` when the failure is that a source was never analysed, `arithmetic` when a
    specific figure must match. You cannot supply code, and any other field is refused.
    Prefer few, highly specific terms: a term that appears in every deliverable makes the
    check a constant, and a check that flags everything is rejected on precision.
    """

    criterion_title: str = dspy.InputField()
    criterion_text: str = dspy.InputField(desc="the rubric's own match_criteria wording")
    failure_mode: str = dspy.InputField(desc="observed mode: omission, partial, abandonment, wrong_value")
    check_json: str = dspy.OutputField(desc="JSON object with exactly kind and params")
