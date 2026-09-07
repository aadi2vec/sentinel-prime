"""Grounding checks: did the run actually *read* the evidence, or did it guess well?

Every admission gate in this repo points at the ledger — `LadderVerifier` decides what
enters memory, `CreditAssigner` decides what stays. Nothing looks at whether the run that
produced an outcome had any contact with the source documents. Two deterministic, LM-free
checks close part of that gap, both computed from artifacts the loop already records.

**Document coverage.** In due diligence the dominant failure is not faulty reasoning, it is
never opening a document. That is measurable without a judge and without a token: which of
the workspace's documents does the trajectory ever name?

**Criterion grounding — the lucky-guess filter.** `CreditAssigner` scores a lesson purely by
whether its targeted criteria passed. A run that produced the right dollar amount from the
model's prior, without reading it, therefore *credits* the lesson that was in the prompt.
That is how a ledger is poisoned while the curve goes up.

The signal separating the two is an asymmetry already present in a `dspy.RLM` trajectory:

    output          <- came back from the interpreter. Data the run observed.
    reasoning/code  <- came out of the model. Text the run generated.

So: extract the checkable facts named by a rubric criterion (amounts, section references,
durations, dates) and ask whether they appear on the *observed* side. A criterion whose
anchors only ever appear in model-generated text was not grounded in the documents.

Three honest limitations, because this is a proxy and not a proof:

  1. Anchor matching is lexical after normalization. "$12.5M" and "twelve and a half
     million" do not match, so a genuinely grounded run can be reported ungrounded.
  2. It shows the fact was *in view*, not that it was used. Reading the right page and
     reasoning badly from it still reads as grounded.
  3. A criterion with no extractable anchors is unjudgeable, and reports `None`. Unknown
     must never be treated as ungrounded — the whole point is to withhold credit on
     evidence, not on silence.

Limitation 1 makes false "ungrounded" verdicts possible, which is why the credit path
*skips* an ungrounded pass rather than blaming it (see `CreditAssigner.observe`): the
conservative reading of "I cannot prove this run saw the evidence" is no information about
the lesson, not evidence against it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Anchor patterns, applied in this order. Each match is consumed out of the working text
# before the next pattern runs, so "$12,500,000" yields one money anchor and not also a
# bare-number anchor for its digits — otherwise every amount would be double-counted and
# `missing` would name the same fact twice.
_MONEY = re.compile(r"\$\s*\d[\d,]*(?:\.\d+)?\s*(?:[kmb]\b|thousand|million|billion)?",
                    re.I)
_SECTION = re.compile(r"(?:§|sections?|articles?|art\.)\s*(\d+(?:\.\d+)*)", re.I)
_DURATION = re.compile(r"\b(\d+)[-\s]?(business\s+day|day|week|month|year|hour)s?\b", re.I)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Three or more digits: small bare integers ("2 parties", "section 7") are too common to
# be evidence of anything, and they would make almost every criterion look grounded.
_NUMBER = re.compile(r"\b\d[\d,]{2,}(?:\.\d+)?\b")

_SEPARATORS = re.compile(r"[,\s]")


def _norm(text: str) -> str:
    return _SEPARATORS.sub("", text).lower()


def anchors(text: str) -> set[str]:
    """The checkable facts named by `text`, normalized so formatting differences match.

    Normalization folds the separators that vary freely between a rubric's phrasing and a
    document's: "$1,200,000"/"$1200000", "Section 7.3"/"§ 7.3", "30-day"/"30 days".
    """
    found: set[str] = set()
    working = text or ""

    def consume(pattern: re.Pattern, render) -> None:
        nonlocal working
        for match in pattern.finditer(working):
            found.add(render(match))
        working = pattern.sub(" ", working)

    consume(_MONEY, lambda m: _norm(m.group(0)))
    consume(_SECTION, lambda m: "§" + _norm(m.group(1)))
    consume(_DURATION, lambda m: _norm(m.group(1)) + _norm(m.group(2)))
    consume(_DATE, lambda m: _norm(m.group(0)))
    consume(_NUMBER, lambda m: _norm(m.group(0)))
    return found


def observed_text(trajectory: list[dict]) -> str:
    """What came back from the interpreter — the only part of a trajectory the run *read*."""
    return "\n".join(str(step.get("output") or "") for step in trajectory)


def generated_text(trajectory: list[dict]) -> str:
    """What the model wrote. Present for contrast; never counts as evidence."""
    return "\n".join(f"{step.get('reasoning') or ''}\n{step.get('code') or ''}"
                     for step in trajectory)


@dataclass
class GroundingReport:
    anchors: list[str]
    observed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool | None:
        """True/False, or None when the criterion named nothing checkable."""
        if not self.anchors:
            return None
        return not self.missing


def criterion_grounding(trajectory: list[dict], criterion_text: str) -> GroundingReport:
    """Were the facts this criterion names present in what the run actually read?"""
    wanted = anchors(criterion_text)
    if not wanted:
        return GroundingReport(anchors=[])
    seen = anchors(observed_text(trajectory))
    return GroundingReport(
        anchors=sorted(wanted),
        observed=sorted(wanted & seen),
        missing=sorted(wanted - seen),
    )


def grounding_map(trajectory: list[dict],
                  criterion_texts: dict[str, str]) -> dict[str, bool | None]:
    """`{criterion_id: grounded}` — the shape `CreditAssigner.observe` consumes.

    Kept here rather than in `credit.py` so credit assignment stays about credit and takes
    the verdict as data; the two modules do not import each other.
    """
    return {cid: criterion_grounding(trajectory, text).grounded
            for cid, text in criterion_texts.items()}


def criterion_texts(criteria: Iterable[dict]) -> dict[str, str]:
    """`{id: title + match_criteria}` from LAB criterion dicts — the anchor source."""
    return {c.get("id", ""): f"{c.get('title', '')} {c.get('match_criteria', '')}".strip()
            for c in criteria if c.get("id")}


@dataclass
class CoverageReport:
    documents: list[str]
    touched: list[str]
    untouched: list[str]

    @property
    def rate(self) -> float:
        # No documents is complete coverage, not a division error: a task with nothing to
        # read cannot have missed anything.
        return len(self.touched) / len(self.documents) if self.documents else 1.0


def document_names(documents_dir: str | Path) -> list[str]:
    """Filenames directly under `documents_dir`; empty when it does not exist."""
    path = Path(documents_dir)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_file())


def document_coverage(trajectory: list[dict],
                      documents: Iterable[str]) -> CoverageReport:
    """Which documents the trajectory ever named, in the order given.

    A name mentioned anywhere in the run's code or interpreter output counts, so a
    directory listing marks every file it prints as touched. That is deliberately
    permissive: coverage is a diagnostic, and over-reporting missed documents would bury
    the real ones in noise. It bounds the failure from one side — a document whose name
    never appears was certainly not read.
    """
    haystack = "\n".join(
        f"{step.get('code') or ''}\n{step.get('output') or ''}" for step in trajectory
    )
    docs = list(documents)
    touched = [name for name in docs if name and name in haystack]
    return CoverageReport(
        documents=docs,
        touched=touched,
        untouched=[name for name in docs if name not in touched],
    )
