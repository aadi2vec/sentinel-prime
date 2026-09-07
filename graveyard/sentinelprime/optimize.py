"""Tuning the online machinery offline, from the record the online loop already writes.

This is the half of the DSPy-native claim that was only ever an *affordance*. The proposer
and the verifier are real `dspy.Predict`s, so `named_predictors()` exposes them and GEPA
can rewrite their instructions — but nothing here ever called `.compile()`, because a
compile needs a trainset and a metric, and the honest versions of both are not obvious.

The claim this module makes concrete is narrower and, we think, more interesting than
"our predictors are visible to GEPA", which any DSPy module can say:

    **The online loop manufactures the offline optimizer's training set, without labels.**

Every round of `refine()` writes a `ProposalRecord`; every exposed lesson writes an
`OutcomeRecord`. Together those are supervised examples that nobody labelled — the loop
that consumes the optimizer's output also produces its input. That closes the two loops in
`docs/ARCHITECTURE.md` into one circuit.

## Where the labels come from, and why not from agreement

The obvious metric — train the generative judge to agree with `GroundingProbe` — is wrong,
and it is worth saying why, because it is the first thing anyone reaches for. The ladder's
value is that the rungs *disagree*: the probe is cheap and shallow, the judge is expensive
and deep, and the judge exists to catch what lexical overlap cannot. Fit the judge to the
probe and you have paid 100x for a copy of the probe, and the ladder's second rung becomes
dead weight. Agreement is a measure of collapse, not of quality.

So labels come from the two places where a verdict is actually *earned*:

  1. **Probe rejections.** An edit whose vocabulary is disjoint from the observed failures
     is ungrounded by construction, not by opinion. This is the only region where the
     deterministic rung is authoritative, and it is a gold negative. Probe *admissions*
     carry no such authority and are deliberately left unlabelled.
  2. **Downstream outcomes.** For an edit that did enter the ledger, `CreditAssigner`
     eventually knows whether the criteria it declared kept clearing while it was in the
     prompt. Past `min_exposures`, that is a gold label on the judge's own distribution.

Precedence is (1) over (2), matching the verification hierarchy in the 2026-09-07 related-work
notes: a deterministic signal outranks an outcome, because a lesson that scored well while
ungrounded is exactly the lucky-guess case the grounding filter exists to refuse.

## Two honest limitations

**Distribution shift — mitigated, measured, not eliminated.** Stratum (1) is drawn from
proposals the ladder short-circuits, so by default the judge never sees them and tuning
against them is extrapolation. Three things address it, in order of how much they buy:

  - `LadderVerifier(explore=eps)` runs the rungs behind a rejection on that fraction of
    rejected rounds, recording **shadow** verdicts that never touch the decision. Those rows
    are then part of the judge's real input distribution at rate eps rather than never. This
    is the actual fix; the rest is bookkeeping around it.
  - `LabeledProposal.on_distribution` and `TrainsetReport.on_distribution_rate` *measure*
    what fraction of the trainset the judge has genuinely run on, and `.caveat` says so in
    the runner's output when it is below half. The limitation stops being a docstring
    caveat and becomes a number printed next to the compile.
  - `balance()` caps the majority stratum, because probe rejections are cheap and plentiful
    while outcome rows require a lesson to survive several tasks; left alone the trainset
    drifts to almost-all off-distribution rows.

What remains: eps < 1 means the gap is narrowed, not closed, and `stratified_scores()` is
the honest read — a gain confined to stratum (1) is still not evidence about the live
ladder, and the pooled mean will happily hide that.

**Correlational, like credit assignment.** Stratum (2) inherits every caveat in `credit.py`:
two lessons targeting one criterion share its outcome, and nothing runs the counterfactual.
`rollout_metric` is the escape hatch — the real downstream measure, priced at one agent run
per candidate, injected so the expensive path is opt-in rather than assumed.

## Provenance

Compiling rewrites the instructions of the predictors that admit ledger edits, which moves
the admission bar. `audit.machinery_fingerprint` stamps every `AuditRecord` with the prompt
version in force, so a compile is as auditable as the edits it goes on to admit — otherwise
"prove the guidance was valid at the time" holds for the ledger and quietly fails one level
up, at the machinery.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import dspy

from sentinelprime.harness import ProposeLedgerEdits
from sentinelprime.proposals import ProposalLog, ProposalRecord
from sentinelprime.verifier import GroundingProbe, PredictVerifier

VERIFIER_INPUTS = ("proposed_edit", "rubric_failures", "trajectory_summary")
PROPOSER_INPUTS = ("trajectory_summary", "rubric_failures", "current_ledger")


@dataclass
class _TextFeedback:
    """The `.as_text()` seam a probe expects, rebuilt from a stored failure string.

    `GroundingProbe.verify` wants a Feedback, but a replay only has the rendered text —
    and the text is all the probe reads. Reconstructing a full `Feedback` would mean
    inventing criterion pass/fail states that were never recorded.
    """

    text: str

    def as_text(self) -> str:
        return self.text


@dataclass
class LabeledProposal:
    record: ProposalRecord
    admit: bool
    source: str   # "probe" (deterministic, by construction) | "outcome" (downstream)
    why: str
    # Did the generative judge actually run on this row? A probe rejection short-circuits
    # the ladder, so by default the judge never sees it and tuning against it is
    # extrapolation. `LadderVerifier(explore=...)` runs the deeper rungs anyway on a
    # fraction of rejected rounds, which is what moves a row into distribution. Outcome
    # rows are on-distribution by construction: they were admitted, so every rung ran.
    on_distribution: bool = True


def label_proposals(log: ProposalLog, probe: GroundingProbe | None = None,
                    min_exposures: int = 3,
                    min_success_rate: float = 0.5) -> list[LabeledProposal]:
    """Gold labels for the proposals a verifier can actually be scored on.

    Unlabelled proposals are *dropped*, not scored neutrally. A neutral score is a claim
    that the verifier's answer did not matter, and GEPA would optimize against it.
    """
    probe = probe or GroundingProbe()
    outcomes = log.outcomes()
    labeled: list[LabeledProposal] = []
    for record in log.records():
        feedback = _TextFeedback(record.rubric_failures)
        verdict = probe.verify(record.op, feedback, [])
        if not verdict.admitted:
            # The one region where the deterministic rung is authoritative. More than one
            # rung in the record means a deeper rung ran here (gated or shadow), so the
            # judge has genuinely seen this kind of input.
            labeled.append(LabeledProposal(record, False, "probe", verdict.justification,
                                           on_distribution=len(record.rungs) > 1))
            continue
        outcome = outcomes.get(record.proposal_id)
        if outcome is None or outcome.exposures < min_exposures:
            # Probe admission alone is not evidence of quality; saying otherwise is what
            # collapses the judge into the probe.
            continue
        admit = outcome.success_rate >= min_success_rate
        labeled.append(LabeledProposal(
            record, admit, "outcome",
            f"targeted criteria cleared on {outcome.successes}/{outcome.exposures} "
            f"exposures ({outcome.success_rate:.0%})"))
    return labeled


def verifier_trainset(log: ProposalLog, probe: GroundingProbe | None = None,
                      min_exposures: int = 3,
                      min_success_rate: float = 0.5) -> list[dspy.Example]:
    """Labelled examples shaped exactly like `VerifyLedgerEdit`'s input fields."""
    return [
        dspy.Example(
            proposed_edit=json.dumps(l.record.op),
            rubric_failures=l.record.rubric_failures,
            trajectory_summary=l.record.trajectory_summary,
            admit="yes" if l.admit else "no",
            label_source=l.source,
            label_why=l.why,
            on_distribution=l.on_distribution,
        ).with_inputs(*VERIFIER_INPUTS)
        for l in label_proposals(log, probe, min_exposures, min_success_rate)
    ]


def proposer_trainset(log: ProposalLog) -> list[dspy.Example]:
    """One example per *round*, not per edit.

    The proposer emits a whole JSON list in one call, so two proposals from one task are
    one training example. Grouping by (task, trajectory) reconstructs the round.
    """
    rounds: dict[tuple[str, str], ProposalRecord] = {}
    for record in log.records():
        rounds.setdefault((record.task_id, record.trajectory_digest), record)
    return [
        dspy.Example(
            trajectory_summary=r.trajectory_summary,
            rubric_failures=r.rubric_failures,
            current_ledger=r.current_ledger or "(empty)",
            task_id=r.task_id,
        ).with_inputs(*PROPOSER_INPUTS)
        for r in rounds.values()
    ]


def _normalize_admit(raw) -> str:
    return "yes" if str(raw).strip().lower() in ("yes", "true", "1") else "no"


def admission_metric() -> Callable:
    """GEPA metric for `VerifyLedgerEdit`: agreement with an *earned* gold label.

    The feedback string is the part GEPA actually reflects on, so it names the edit, the
    failures it was judged against, and *why* the gold label is what it is — a rejection
    sourced from the probe is a different lesson for the judge than one sourced from a
    lesson that later stopped working.
    """
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        want = _normalize_admit(getattr(gold, "admit", "no"))
        got = _normalize_admit(getattr(pred, "admit", ""))
        source = getattr(gold, "label_source", "unknown")
        why = getattr(gold, "label_why", "")
        edit = str(getattr(gold, "proposed_edit", ""))[:300]
        if got == want:
            return dspy.Prediction(
                score=1.0,
                feedback=(f"Correct: '{want}' was right for this edit. Evidence "
                          f"({source}): {why}"))
        return dspy.Prediction(
            score=0.0,
            feedback=(f"Wrong verdict. You answered '{got}' but the correct answer is "
                      f"'{want}'. Edit: {edit}. Rubric failures it was judged against: "
                      f"{str(getattr(gold, 'rubric_failures', ''))[:300]}. Evidence for "
                      f"'{want}' ({source}): {why}. Admitting an ungrounded edit pollutes "
                      f"a durable ledger; rejecting a grounded one loses a real lesson."))
    return metric


def proposer_metric(probe: GroundingProbe | None = None,
                    grounding_weight: float = 0.6) -> Callable:
    """GEPA metric for `ProposeLedgerEdits`, priced at zero extra LM calls.

    Scores what can be checked deterministically about a proposed batch: it parses, every
    edit is grounded in the observed failures, and every edit declares `meta.targets`.
    Targets are weighted because they are load-bearing elsewhere — an edit without them is
    unscorable by `CreditAssigner`, so a proposer that drops them silently disables
    retirement (see the "declared targets, not the round's failure text" note in CLAUDE.md).

    This is a *proxy*. It cannot see whether a lesson helped; `rollout_metric` can, and
    costs an agent run per candidate.
    """
    probe = probe or GroundingProbe()
    target_weight = 1.0 - grounding_weight

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        raw = getattr(pred, "edits", "")
        try:
            edits = json.loads(raw)
        except (TypeError, ValueError) as exc:
            return dspy.Prediction(
                score=0.0,
                feedback=(f"The edits field did not parse as JSON ({exc}). It must be a "
                          f"JSON list of edit ops and nothing else — no prose, no code "
                          f"fence. Got: {str(raw)[:200]}"))
        if not isinstance(edits, list) or not edits:
            return dspy.Prediction(
                score=0.0,
                feedback="Expected a non-empty JSON list of edit ops; a round that "
                         "observed failures and proposed nothing cannot fix them.")

        feedback_text = str(getattr(gold, "rubric_failures", ""))
        shim = _TextFeedback(feedback_text)
        scores, notes = [], []
        for op in edits:
            if not isinstance(op, dict):
                scores.append(0.0)
                notes.append(f"not an object: {str(op)[:80]}")
                continue
            verdict = probe.verify(op, shim, [])
            has_targets = bool((op.get("meta") or {}).get("targets"))
            scores.append(grounding_weight * (1.0 if verdict.admitted else 0.0)
                          + target_weight * (1.0 if has_targets else 0.0))
            if not verdict.admitted:
                notes.append(f"ungrounded ({op.get('id')}): {verdict.justification}")
            if not has_targets:
                notes.append(f"no meta.targets on '{op.get('id')}' — credit assignment "
                             f"cannot score it, so it can never be retired")
        score = sum(scores) / len(scores)
        return dspy.Prediction(
            score=score,
            feedback=(f"Scored {score:.2f} over {len(edits)} edit(s) against: "
                      f"{feedback_text[:200]}. " +
                      ("Problems: " + "; ".join(notes[:6]) if notes
                       else "Every edit was grounded in an observed failure and declared "
                            "the criteria it targets.")))
    return metric


def rollout_metric(rollout: Callable[[list[dict], dspy.Example], float]) -> Callable:
    """The honest downstream metric: did applying these edits actually help?

    `rollout(edits, gold) -> float` re-runs the agent with the proposed edits in the ledger
    and returns a criterion pass rate. That needs an LM, a workspace, and real money — one
    agent run per candidate per example — so it is injected rather than assumed, and the
    cheap `proposer_metric` remains the default.

    Unparseable edits short-circuit to zero *before* the rollout: there is nothing to apply,
    and paying for an agent run to discover that is the one avoidable cost here.
    """
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        try:
            edits = json.loads(getattr(pred, "edits", ""))
        except (TypeError, ValueError) as exc:
            return dspy.Prediction(
                score=0.0,
                feedback=f"Edits did not parse as JSON ({exc}); no rollout was run.")
        if not isinstance(edits, list) or not edits:
            return dspy.Prediction(score=0.0,
                                   feedback="No edit ops to apply; no rollout was run.")
        score = float(rollout(edits, gold))
        return dspy.Prediction(
            score=score,
            feedback=(f"Applying these {len(edits)} edit(s) and re-running the task scored "
                      f"{score:.2f} on the rubric. This is the downstream effect of the "
                      f"guidance, not a judgement of how it reads."))
    return metric


@dataclass
class TrainsetReport:
    """Whether a compile is even worth starting, answered without spending a token.

    The failure this exists to catch is the one that made the affordance untestable in the
    first place: a single-class trainset. Fit a gate to all-positives and it learns to
    admit everything, which is indistinguishable from having no gate — and the score will
    look excellent while it happens.
    """

    proposals: int
    admitted: int
    rejected: int
    labeled: int
    positives: int
    negatives: int
    by_source: dict
    outcomes: int
    on_distribution: int = 0
    min_examples: int = 8

    @property
    def on_distribution_rate(self) -> float:
        return self.on_distribution / self.labeled if self.labeled else 0.0

    @property
    def caveat(self) -> str:
        """The distribution warning, empty when it does not apply.

        Separate from `viable` on purpose: an off-distribution trainset is still
        trainable, and refusing it would throw away the only data a short run produces.
        What must not happen is reporting a gain from it as evidence about the live
        ladder. Raise `LadderVerifier(explore=...)` to move rows into distribution.
        """
        if not self.labeled or self.on_distribution_rate >= 0.5:
            return ""
        return (f"only {self.on_distribution_rate:.0%} of labelled rows are ones the "
                f"generative judge actually ran on — the rest are proposals the ladder "
                f"short-circuits, so a gain on them is not evidence about the live "
                f"ladder. Raise LadderVerifier(explore=...) to close the distribution "
                f"gap, and read stratified_scores() rather than the pooled mean.")

    @property
    def viable(self) -> bool:
        return (self.labeled >= self.min_examples
                and self.positives > 0 and self.negatives > 0)

    @property
    def why(self) -> str:
        if self.positives == 0 or self.negatives == 0:
            missing = "negatives" if self.negatives == 0 else "positives"
            return (f"only one class present ({self.positives} positive / "
                    f"{self.negatives} negative): a gate fit to this learns to admit "
                    f"everything. Run more tasks until there are {missing}.")
        if self.labeled < self.min_examples:
            return (f"{self.labeled} labelled example(s), below the {self.min_examples} "
                    f"minimum: GEPA would be reflecting on noise.")
        return (f"{self.labeled} labelled examples across both classes "
                f"({self.by_source}) — enough to compile.")

    def summary(self) -> str:
        lines = [f"proposals {self.proposals} ({self.admitted} admitted / "
                 f"{self.rejected} rejected), outcomes {self.outcomes}",
                 f"labelled  {self.labeled} = {self.positives} admit / "
                 f"{self.negatives} reject, by source {self.by_source}",
                 f"in-dist   {self.on_distribution}/{self.labeled} "
                 f"({self.on_distribution_rate:.0%}) rows the judge actually ran on",
                 f"viable    {self.viable}: {self.why}"]
        if self.caveat:
            lines.append(f"CAVEAT    {self.caveat}")
        return "\n".join(lines)


def balance(examples: list[dspy.Example], max_ratio: float = 3.0) -> list[dspy.Example]:
    """Cap the majority label-source so one stratum cannot swamp the compile.

    Probe rejections are cheap and plentiful; outcome rows need a lesson to survive
    several tasks. Left alone the trainset drifts to almost-all probe rows, and GEPA
    optimizes for the stratum that matters least — the one furthest from what the live
    ladder sees. Trimming is deterministic (by content hash, like `split`) so a re-run on
    a grown log keeps the same rows rather than resampling.
    """
    import hashlib

    def key(example: dspy.Example) -> str:
        return hashlib.sha256(repr(sorted(example.toDict().items())).encode()).hexdigest()

    by_source: dict = {}
    for example in examples:
        by_source.setdefault(getattr(example, "label_source", "unknown"), []).append(example)
    if len(by_source) < 2:
        return list(examples)
    smallest = min(len(v) for v in by_source.values())
    cap = max(1, int(smallest * max_ratio))
    kept: list[dspy.Example] = []
    for rows in by_source.values():
        kept.extend(sorted(rows, key=key)[:cap] if len(rows) > cap else rows)
    return sorted(kept, key=key)


def stratified_scores(examples: list[dspy.Example],
                      scores: list[float]) -> dict[str, float]:
    """Mean score per label source, plus the on-distribution subset and the pooled mean.

    The pooled mean is the number that can lie. A judge tuned on plentiful probe
    rejections can post an excellent overall score while getting worse on the rows the
    live ladder actually routes to it — `on_distribution` is the one to read, and it is
    reported next to `overall` so the gap is visible rather than inferred.
    """
    buckets: dict = {}
    for example, score in zip(examples, scores):
        buckets.setdefault(getattr(example, "label_source", "unknown"), []).append(score)
        if getattr(example, "on_distribution", True):
            buckets.setdefault("on_distribution", []).append(score)
    out = {name: sum(v) / len(v) for name, v in buckets.items() if v}
    out["overall"] = sum(scores) / len(scores) if scores else 0.0
    return out


def trainset_report(log: ProposalLog, probe: GroundingProbe | None = None,
                    min_exposures: int = 3, min_success_rate: float = 0.5,
                    min_examples: int = 8) -> TrainsetReport:
    labels = label_proposals(log, probe, min_exposures, min_success_rate)
    by_source: dict = {}
    for l in labels:
        by_source[l.source] = by_source.get(l.source, 0) + 1
    return TrainsetReport(
        proposals=len(log.records()),
        admitted=len(log.admitted()),
        rejected=len(log.rejected()),
        labeled=len(labels),
        positives=sum(1 for l in labels if l.admit),
        negatives=sum(1 for l in labels if not l.admit),
        by_source=by_source,
        outcomes=len(log.outcomes()),
        on_distribution=sum(1 for l in labels if l.on_distribution),
        min_examples=min_examples,
    )


def split(examples: list[dspy.Example],
          holdout: float = 0.25) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Deterministic train/val split.

    Deterministic by content hash rather than by shuffle seed, so re-running a compile on a
    grown log keeps every previously-held-out example held out. A reshuffle would quietly
    move validation rows into training between runs and inflate the reported score.

    A holdout that would empty the trainset yields no validation set instead: reporting a
    training score is a known bias, and having nothing to train on is a broken run.
    """
    import hashlib

    def key(example: dspy.Example) -> str:
        return hashlib.sha256(repr(sorted(example.toDict().items())).encode()).hexdigest()

    ordered = sorted(examples, key=key)
    n_val = int(len(ordered) * holdout)
    if len(ordered) - n_val < 1:
        return ordered, []
    return ordered[n_val:], ordered[:n_val]


class LedgerRollout:
    """Apply proposed edits, run the task, score it, and put the ledger back.

    This is the honest answer to "did the guidance help?", and it is where invariant 2
    stops being a compliance feature and starts paying for itself. Measuring a proposed
    lesson's effect means running a task *with* it, which means mutating the ledger — and
    an optimizer that mutates the ledger it is optimizing against poisons its own baseline
    within a few candidates. Because every round is bracketed by snapshots, a trial is a
    snapshot, an apply, a run, and a rollback: candidates stay independent, and the ledger
    under test is never the one the optimizer wrote to.

    `run_and_score(task_id) -> float` is injected — it needs an agent, a workspace and a
    judge, and that is the caller's world, not this module's. `scripts/gepa_runner.py`
    builds one from the LAB harness.

    The restore runs in a `finally`: an agent that dies mid-task must not leave a trial
    lesson behind, or every subsequent candidate is scored against a contaminated ledger
    and the whole compile is quietly invalid.
    """

    def __init__(self, harness, run_and_score: Callable[[str], float]) -> None:
        self.harness = harness
        self.run_and_score = run_and_score

    def __call__(self, edits: list[dict], gold) -> float:
        before = self.harness.backend.snapshot()
        try:
            self.harness._apply_edits(edits)
            return float(self.run_and_score(str(getattr(gold, "task_id", ""))))
        finally:
            self.harness.rollback(before.number)


def compile_verifier(trainset: list[dspy.Example], reflection_lm: dspy.LM,
                     min_score: float = 0.0, valset: list[dspy.Example] | None = None,
                     auto: str = "light", **gepa_kwargs) -> PredictVerifier:
    """GEPA-tune the generative rung of the ladder. Returns a fresh compiled module."""
    from dspy.teleprompt import GEPA

    optimizer = GEPA(metric=admission_metric(), reflection_lm=reflection_lm, auto=auto,
                     **gepa_kwargs)
    return optimizer.compile(PredictVerifier(min_score=min_score),
                             trainset=trainset, valset=valset or trainset)


class _Proposer(dspy.Module):
    """The proposer alone, so GEPA optimizes it without dragging in the whole harness."""

    def __init__(self) -> None:
        super().__init__()
        self.propose = dspy.Predict(ProposeLedgerEdits)

    def forward(self, trajectory_summary: str, rubric_failures: str, current_ledger: str):
        return self.propose(trajectory_summary=trajectory_summary,
                            rubric_failures=rubric_failures,
                            current_ledger=current_ledger)


def compile_proposer(trainset: list[dspy.Example], reflection_lm: dspy.LM,
                     metric: Callable | None = None,
                     valset: list[dspy.Example] | None = None,
                     auto: str = "light", **gepa_kwargs) -> _Proposer:
    """GEPA-tune `ProposeLedgerEdits`. Pass `rollout_metric(...)` for the honest measure."""
    from dspy.teleprompt import GEPA

    optimizer = GEPA(metric=metric or proposer_metric(), reflection_lm=reflection_lm,
                     auto=auto, **gepa_kwargs)
    return optimizer.compile(_Proposer(), trainset=trainset, valset=valset or trainset)
