"""The parity family: context-management tasks, scored deterministically. PLAN §3a, Phase 0.

This is a **gate, not a feature.** §3a admits the write path — an editable live context the
model edits in place — in phases, and states the kill condition in advance: if the baseline
RLM already handles surgical in-place edits competently, the write path is not this system's
bottleneck and Phases 1-3 are cancelled. This module is the instrument that settles it.

Why a new family at all, when §5 already has 368 graded LAB criterion rows: the LAB judge's
own verdict instability is the measurement problem, not an inconvenience. A pooled criterion
rate spanning 0.400-0.691 across five runs of one *identical* configuration cannot answer a
yes/no question about a mechanism. So every task here is scored by substring presence over a
generated spec — no judge, no LM, no sampling variance — which is the same reason §5 nominates
a property of the check set as the primary metric rather than task success.

**Two measurements, deliberately not collapsed.** A full rewrite that happens to contain the
right new value is *correct* and *not surgical*, and the gate's question is about the second
one. So `ParityScore` reports `correct` and `footprint` independently, and `evaluate()` — the
[0,1] float `PolicySearch` consumes — carries correctness only. Folding footprint into the
search objective would make surgicality something a candidate policy is *selected for*, which
is a different claim from the baseline measurement the gate needs, and §6 reserves what counts
as success to the evaluator rather than the search.

**Footprint over an empty preserve set is None.** A retrieval task preserves nothing, so it
has no opinion on surgicality; scoring it 0.0 would let retrieval tasks dilute the mean into
a flawless-edit verdict nobody measured. Same convention as `replay.CheckScore.precision` and
`credit.success_rate`: unknown is untested, not good.

The four kinds, and what each one can and cannot see:

  - `needle`     retention — one fact buried in filler. Pure retrieval; no footprint.
  - `sketchpad`  in-place edit — change one register entry, leave the rest alone. **This is
                 the gate's primary instrument**, and the only kind whose required value must
                 be derived rather than copied out of the prompt.
  - `kvstore`    retention under pressure — more items than the stated capacity, some pinned.
                 Required: the pinned values survive. Preserved: their *bindings* survive,
                 which an answer that dumps values without labels loses.
  - `triage`     selection — report the ERROR lines and carry none of the noise. No footprint.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from sentinelprime.policy_search import Case

FAMILY_KINDS = ("needle", "sketchpad", "kvstore", "triage")

_FAMILY_PREFIX = "parity"


@dataclass(frozen=True)
class ParitySpec:
    """The grading key for one parity case. Reaches the evaluator only, never the prompt.

    `capacity` and `item_count` are descriptive — they record the shape of the generated
    task so a test can assert the task still forces an eviction, and they are not scored.
    A capacity that drifted up to the item count would leave a task that reads as testing
    retention under pressure while applying none.
    """

    require: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    preserve: tuple[str, ...] = ()
    capacity: int | None = None
    item_count: int = 0

    def __post_init__(self):
        # A spec with nothing required cannot be failed, and `evaluate` would have to
        # return a number for it — which would be 1.0, scoring an unmeasured task as a
        # pass. Raising instead surfaces as `evaluation_error` in PolicySearch, and
        # incomplete evaluation cannot promote.
        if not self.require:
            raise ValueError("a parity spec needs at least one required string")


@dataclass(frozen=True)
class ParityScore:
    correct: bool
    missing: tuple[str, ...]
    leaked: tuple[str, ...]
    disturbed: tuple[str, ...]
    preserved_total: int

    @property
    def footprint(self) -> float | None:
        """Fraction of what the task said to keep that did not survive, or None.

        0.0 is a surgical edit; 1.0 destroyed everything it was told to leave alone. None
        when the task marked nothing for preservation — unknown, not perfect.
        """
        if self.preserved_total == 0:
            return None
        return len(self.disturbed) / self.preserved_total


def _present(needle: str, haystack: str) -> bool:
    return needle.lower() in haystack.lower()


def score(spec: ParitySpec, answer: str) -> ParityScore:
    answer = answer or ""
    missing = tuple(r for r in spec.require if not _present(r, answer))
    leaked = tuple(f for f in spec.forbid if _present(f, answer))
    disturbed = tuple(p for p in spec.preserve if not _present(p, answer))
    return ParityScore(
        # A leak is a failure even when everything required is present: the new value
        # appearing does not mean the old one was removed, and an edit that only appended
        # did not edit anything.
        correct=not missing and not leaked,
        missing=missing, leaked=leaked, disturbed=disturbed,
        preserved_total=len(spec.preserve),
    )


def evaluate(case: Case, answer: str) -> float:
    """The [0,1] evaluator `PolicySearch` consumes. Correctness only — see module docstring."""
    spec: ParitySpec = case.gold
    verdict = score(spec, answer)
    if verdict.leaked:
        return 0.0
    return (len(spec.require) - len(verdict.missing)) / len(spec.require)


def kind_of(family: str) -> str:
    """Which of the four kinds a generated family belongs to.

    Every case instance gets its own family — `PolicySearch.run` refuses a family spanning
    two groups and `PolicyArchive.reserve` refuses a family twice, so one family per *kind*
    would make dev/validation/test unsplittable.
    """
    parts = family.split("-")
    return parts[1] if len(parts) > 2 and parts[0] == _FAMILY_PREFIX else ""


# --- generators ---------------------------------------------------------------------
#
# Deterministic from a seed, and *different* across seeds. §3 names "one generated
# template, shared across all splits" as a defect of the current substrate: a held-out
# split that is the training split with other numbers is not held out.

_FILLER = (
    "Counterparty disclosed no pending litigation for the quarter.",
    "Transfer pricing memorandum circulated to the steering committee.",
    "Vendor master data reconciliation closed with no exceptions.",
    "Site inspection deferred pending access arrangements.",
    "Insurance certificates collected for all leased premises.",
    "Payroll accrual tied to the general ledger without variance.",
    "Board minutes confirm the delegation of authority matrix.",
    "Supplier concentration remains below the disclosure threshold.",
)

_LABELS = (
    "cap-ex-reserve", "tenant-estoppel-fee", "environmental-indemnity",
    "working-capital-peg", "escrow-holdback", "transfer-tax-accrual",
    "deferred-maintenance", "title-insurance-premium",
)

_COMPONENTS = ("ingest", "scheduler", "ledger-writer", "judge-pool", "doc-parser")


def _needle(rng: random.Random, n: int) -> tuple[str, ParitySpec]:
    value = f"{rng.randrange(100, 999)}-{rng.randrange(1000, 9999)}"
    lines = list(_FILLER) * 3
    rng.shuffle(lines)
    buried = f"Allocation reference for the holdback account is {value}."
    lines.insert(rng.randrange(len(lines) // 3, len(lines)), buried)
    task = ("Read the diligence log below and report the allocation reference for the "
            "holdback account. Report the reference and nothing else.\n\n"
            + "\n".join(f"- {line}" for line in lines))
    # Forbidding filler is what refuses the echo. Without it the answer is sitting in the
    # prompt and handing the whole log back scores a flawless retrieval.
    return task, ParitySpec(require=(value,), forbid=tuple(rng.sample(_FILLER, 2)),
                            item_count=len(lines))


def _sketchpad(rng: random.Random, n: int) -> tuple[str, ParitySpec]:
    labels = rng.sample(_LABELS, 5)
    amounts = [rng.randrange(2, 90) * 10_000 for _ in labels]
    target = rng.randrange(len(labels))
    delta = rng.randrange(2, 20) * 10_000
    entries = [f"{label} = {amount:,}" for label, amount in zip(labels, amounts)]
    old = entries[target]
    new = f"{labels[target]} = {amounts[target] + delta:,}"
    # The instruction states a delta, never the resulting figure: an edit whose new value
    # is quoted in the prompt is satisfiable by appending that literal, which would score
    # a copy as an edit.
    task = ("Below is a working register. Increase the "
            f"{labels[target]} entry by {delta:,} and leave every other entry exactly as "
            "it stands. Return the complete register.\n\n" + "\n".join(entries))
    return task, ParitySpec(
        require=(new,), forbid=(old,),
        preserve=tuple(e for i, e in enumerate(entries) if i != target),
        item_count=len(entries))


def _kvstore(rng: random.Random, n: int) -> tuple[str, ParitySpec]:
    labels = rng.sample(_LABELS, 7)
    values = [f"{rng.randrange(1000, 9999)}" for _ in labels]
    entries = [f"{label} = {value}" for label, value in zip(labels, values)]
    capacity = 3
    pinned = rng.sample(range(len(labels)), capacity)
    pinned_labels = [labels[i] for i in pinned]
    evicted = [i for i in range(len(labels)) if i not in pinned]
    task = (f"You may retain at most {capacity} of the entries below in working memory; "
            f"evict the rest. The entries for {', '.join(pinned_labels)} are pinned and "
            "must survive. Return the retained table.\n\n" + "\n".join(entries))
    return task, ParitySpec(
        # Required: the pinned information survived at all. Preserved: its *binding*
        # survived — an answer that lists values without labels loses the association,
        # which is a distinguishable failure and exactly what footprint is for.
        require=tuple(values[i] for i in pinned),
        # An answer over capacity evicted nothing, which is the degenerate strategy here:
        # retention is trivial if you keep everything.
        forbid=tuple(entries[i] for i in rng.sample(evicted, 2)),
        preserve=tuple(entries[i] for i in pinned),
        capacity=capacity, item_count=len(entries))


def _triage(rng: random.Random, n: int) -> tuple[str, ParitySpec]:
    errors, noise, lines = [], [], []
    for _ in range(12):
        component = rng.choice(_COMPONENTS)
        code = rng.randrange(1000, 9999)
        if rng.random() < 0.3:
            line = f"ERROR {component} failed with code {code}"
            errors.append(line)
        else:
            line = f"INFO {component} completed batch {code}"
            noise.append(line)
        lines.append(line)
    # Degenerate draws make the task unscoreable, so force at least one of each.
    if not errors:
        line = f"ERROR {rng.choice(_COMPONENTS)} failed with code {rng.randrange(1000, 9999)}"
        errors.append(line)
        lines.append(line)
    if not noise:
        line = f"INFO {rng.choice(_COMPONENTS)} completed batch {rng.randrange(1000, 9999)}"
        noise.append(line)
        lines.append(line)
    task = ("Report every ERROR line from the run log below, and nothing else.\n\n"
            + "\n".join(lines))
    return task, ParitySpec(require=tuple(errors), forbid=tuple(noise),
                            item_count=len(lines))


_GENERATORS = {"needle": _needle, "sketchpad": _sketchpad,
               "kvstore": _kvstore, "triage": _triage}


def build_cases(seed: int = 0, per_kind: int = 1) -> list[Case]:
    """Generate `per_kind` cases for each of the four kinds, one family per case."""
    if per_kind < 1:
        raise ValueError("per_kind must be >= 1")
    rng = random.Random(seed)
    cases = []
    for kind in FAMILY_KINDS:
        for n in range(per_kind):
            task, spec = _GENERATORS[kind](rng, n)
            family = f"{_FAMILY_PREFIX}-{kind}-{seed}-{n}"
            cases.append(Case(id=family, family=family, task=task, gold=spec))
    return cases


def parity_report(rows: list[tuple[Case, str]]) -> dict:
    """Per-kind correctness and mean footprint — the two numbers the gate is read from.

    Reported separately because the kill condition turns on the second one. A baseline that
    is fully correct at footprint ~1.0 solves these tasks by rewriting, which is the case
    for building the write path; fully correct at footprint ~0.0 is the case for cancelling
    Phases 1-3.
    """
    buckets: dict[str, list[ParityScore]] = {k: [] for k in FAMILY_KINDS}
    for case, answer in rows:
        kind = kind_of(case.family)
        if kind in buckets:
            buckets[kind].append(score(case.gold, answer))

    report: dict = {}
    for kind, scores in buckets.items():
        footprints = [s.footprint for s in scores if s.footprint is not None]
        report[kind] = {
            "n": len(scores),
            "correct_rate": (sum(1 for s in scores if s.correct) / len(scores)
                             if scores else None),
            # None, not 0.0: a kind that preserves nothing has no opinion on surgicality,
            # and averaging it in would report a flawless edit nobody measured.
            "mean_footprint": (sum(footprints) / len(footprints) if footprints else None),
            "footprint_n": len(footprints),
        }

    def _fmt(value) -> str:
        return "n/a" if value is None else f"{value:.3g}"

    report["describe"] = " ".join(
        f"{kind}[n={report[kind]['n']} correct={_fmt(report[kind]['correct_rate'])} "
        f"footprint={_fmt(report[kind]['mean_footprint'])}]"
        for kind in FAMILY_KINDS)
    return report


# Stated here rather than decided when the numbers come in. A threshold chosen after
# seeing the result is not a gate, and §7's discipline — "written in advance, so a null
# result is a result" — applies to the decision rule as much as to the failure modes.
COMPETENT_CORRECT_RATE = 0.8
SURGICAL_FOOTPRINT = 0.2


def gate_verdict(report: dict, kind: str = "sketchpad") -> dict:
    """Apply §3a's kill condition to a parity report.

    > if Phase 0 shows the baseline RLM already handles surgical in-place edits
    > competently, the write path is not this system's bottleneck and Phases 1-3 are
    > cancelled.

    Read off the `sketchpad` kind, because that is the one that measures an in-place edit;
    the other three establish surrounding context-management competence and do not bear on
    this decision. Three outcomes, and the third one matters:

      - **cancel**   correct and surgical. The baseline already does this.
      - **build**    correct only by rewriting. The outcome is reached, the mechanism is
                     not, which is the case §3a admits the write path on.
      - **inconclusive**  the baseline cannot do the task, or nothing measured
                     surgicality. Footprint on a wrong answer says nothing about editing,
                     and a baseline that is simply bad at the task is not evidence about
                     the write path either way. This is the honest third answer, and
                     collapsing it into `build` is how a gate becomes a formality.
    """
    row = report.get(kind) or {}
    correct_rate, footprint = row.get("correct_rate"), row.get("mean_footprint")
    numbers = {"kind": kind, "n": row.get("n", 0), "correct_rate": correct_rate,
               "mean_footprint": footprint,
               "footprint_n": row.get("footprint_n", 0)}
    if correct_rate is None or footprint is None:
        return {**numbers, "decision": "inconclusive",
                "why": f"no {kind} correctness or surgicality observations to read"}
    if correct_rate < COMPETENT_CORRECT_RATE:
        return {**numbers, "decision": "inconclusive",
                "why": (f"{kind} correct_rate {correct_rate:.3g} < "
                        f"{COMPETENT_CORRECT_RATE:g}: the baseline does not reach the "
                        f"outcome, so its footprint does not describe how it edits")}
    if footprint <= SURGICAL_FOOTPRINT:
        return {**numbers, "decision": "cancel",
                "why": (f"{kind} correct_rate {correct_rate:.3g} at footprint "
                        f"{footprint:.3g} <= {SURGICAL_FOOTPRINT:g}: the baseline already "
                        f"edits in place, so the write path is not the bottleneck")}
    return {**numbers, "decision": "build",
            "why": (f"{kind} correct_rate {correct_rate:.3g} at footprint "
                    f"{footprint:.3g} > {SURGICAL_FOOTPRINT:g}: the baseline reaches the "
                    f"outcome by rewriting rather than editing")}
