"""Admission control over the model's edits to its own live context.

This is the whole reason the write path belongs in *this* repo rather than being a port.
CLM hands the model unrestricted write access to its context; SentinelPrime's thesis is that a
label-free write path needs an admission gate, and `LadderVerifier` already is one. So the
context edits go through the same ladder the ledger edits go through — same interface, same
`CostLadderPlanner` ordering, same adaptive statistics, same compliance trail — rather than a
parallel mechanism that would drift from it.

**What changes is what an edit can be grounded in, and this is the load-bearing difference.**
`refine()` runs *after* a task is graded, so `GroundingProbe` can ask whether a proposed lesson
is about an observed rubric failure. A context edit happens *mid-task*: nothing is graded, there
are no failures, and `feedback.as_text()` would be the empty string. Lexical overlap against
nothing is 0.0 for every candidate, so the ledger's own probe would reject every context edit —
and overlap against a *default* of "" admits nothing while reporting itself as a grounding
check. The only thing available to ground a mid-task edit in is the task it is meant to be
serving, which is what `TaskStatement` carries.

Three rungs, priced the same hundred-to-one as the ledger's:

  - `RetentionProbe` (cost 1) refuses an edit that would drop the **last visible copy** of a
    figure. This is the one failure mode that is both cheap to detect and expensive to miss:
    §5's observed distribution is 29 omissions and ~0 fabrications, so the dominant error is
    already *not stating something*, and an edit that removes the only copy of a number
    manufactures exactly that error. It also does something a pure accounting layer cannot —
    it forces a compaction's note to carry the figures forward, which turns `compact_step`
    from "shorten this" into "summarise this losslessly in the dimension that matters".
  - `ReprefillProbe` (cost 2) refuses an edit that will not pay back the prefix it
    invalidates within a stated number of turns. §3a's first constraint says an
    editable-context win that is really a cache miss is the failure mode; Phase 1 made that
    measurable, and this is the rung that makes it *binding*.
  - `PredictContextVerifier` (cost 100) is the generative judge, and the only rung that can
    read whether a turn is still load-bearing for the task. A `dspy.Predict` over a Signature,
    so it reaches `named_predictors()` and GEPA tunes this bar with the same machinery that
    tunes the ledger's.

Two honest limits, both in the rung docstrings. `RetentionProbe`'s notion of a "figure" is a
regex over numbers and filenames, which over-protects (a loop counter looks like a figure) and
under-protects (a materially important *clause name* does not). And `ReprefillProbe` prices an
edit in characters and amortises it over an *assumed* number of remaining turns, because
nothing in the loop knows how many are left — so its bar is a stated threshold, not a
measurement.
"""
from __future__ import annotations

import json
import re

import dspy

from sentinelprime.verifier import LadderVerifier, VerifierLevel, VerifierVerdict


class TaskStatement:
    """A `feedback`-shaped carrier for the one thing a mid-task edit can be grounded in.

    `LadderVerifier.verify(op, feedback, trajectory)` is the interface `refine()` already
    uses, and reusing it verbatim is what lets the context gate be the *same* ladder rather
    than a lookalike. `feedback` there means graded rubric failures, which do not exist
    mid-task — so this supplies the task statement under the same `as_text()` name, and the
    rungs that read it are written knowing which of the two they are getting.

    `task_id` is present because `ErrorBudget` and `AuditLog` both key on it; an edit refused
    mid-task is attributable to the task that proposed it.
    """

    def __init__(self, task: str, task_id: str = "") -> None:
        self.task = task
        self.task_id = task_id

    def as_text(self) -> str:
        return self.task


# Numbers (money, percentages, dates, section numbers) and document filenames. Both are
# things a due-diligence deliverable has to *state*, which is what makes dropping the last
# copy of one a manufactured omission rather than a tidy-up.
_FIGURE_RE = re.compile(
    r"\d[\d,._/%-]*|\b[\w-]+\.(?:pdf|txt|md|docx?|csv|json|html?)\b", re.IGNORECASE)


def _figures(text: str) -> set[str]:
    return {m.group(0).rstrip(".,_-/").lower() for m in _FIGURE_RE.finditer(text or "")
            # A bare one- or two-digit token is a loop counter far more often than a
            # material figure, and refusing on those would make the rung reject
            # everything. Over-protecting is still a refusal the model has to argue past.
            if len(m.group(0).rstrip(".,_-/")) > 2}


class RetentionProbe:
    """Deterministic rung: would this edit remove the last visible copy of a figure?

    Measured against the rendered context *after* the edit, not against a list of what the
    edit touched, so a compaction whose note repeats the figures passes and one that
    paraphrases them away does not. That asymmetry is the point: the model is free to
    compact, and not free to lose a number while doing it.

    The honest limit: "figure" is `_FIGURE_RE`, which catches numbers and filenames and
    misses a materially important clause *name*. It over-refuses on long identifiers that
    happen to be numeric and under-refuses on prose. It is the cheap rung — anything it
    admits still has to clear the judge.
    """

    name = "context_retention"

    def named_predictors(self):
        return []  # deterministic: nothing for GEPA to tune

    def verify(self, op: dict, feedback, trajectory: list[dict]) -> VerifierVerdict:
        dropped = _figures(op.get("text", ""))
        if not dropped:
            return VerifierVerdict(True, 1.0, "retention: edit drops no figures")
        # Everything the prompt will still carry, including the replacement note and the
        # sketchpad — `after` is the real rendered context, so this is exact.
        surviving = _figures(op.get("after", ""))
        lost = sorted(dropped - surviving)
        if lost:
            return VerifierVerdict(
                False, 0.0,
                f"retention: this would leave no copy of {', '.join(lost[:5])} anywhere in "
                f"your context. Carry them into the note or the sketchpad first.")
        return VerifierVerdict(
            True, 1.0, f"retention: all {len(dropped)} figure(s) still present after the edit")


class ReprefillProbe:
    """Deterministic rung: does this edit cost more cache than it will ever save?

    Both quantities come from the rendered before/after, so this is Phase 1's accounting
    used as a decision rather than a log. An edit near the tail invalidates about what it
    saves; an edit at the head invalidates everything after it, which is how a 600-character
    "saving" costs 1000 characters of prefix.

    **The two quantities are not the same kind of number, and an earlier version of this rung
    compared them as if they were.** Invalidation is paid once, at the next turn. The saving
    recurs: a turn removed from the prompt is absent from *every* remaining turn's prefix.
    Comparing a one-off cost to a per-turn saving refuses every edit near the head — which is
    precisely backwards, because a head edit is the one with the most remaining turns to
    amortise over. Found by running a template over a three-turn context: evicting a 10-char
    failed attempt at step 1 invalidated 264 characters and was refused at 26.4x, even though
    it pays for itself inside thirty turns.

    So the comparison is `invalidated` against `saved * horizon_turns`, and `horizon_turns`
    is an assumption this module states rather than hides: *how many further turns the saving
    will be collected over*. It is not read from the run, because nothing in the loop knows
    how many turns are left — `max_iters` is a ceiling, not a forecast. 8 is a threshold.

    The second honest limit is unchanged: this compares characters to characters.
    Re-prefilling a cached prompt is cheaper per character than generating one, by a
    provider-specific factor this repo refuses to hardcode (the same reason `load_config`
    will not assume a provider). So the rung is conservative in a direction it cannot
    quantify.
    """

    name = "context_reprefill"

    def __init__(self, horizon_turns: float = 8.0) -> None:
        self.horizon_turns = horizon_turns

    def named_predictors(self):
        return []

    def verify(self, op: dict, feedback, trajectory: list[dict]) -> VerifierVerdict:
        before, after = op.get("before", ""), op.get("after", "")
        saved = len(before) - len(after)
        i = 0
        limit = min(len(before), len(after))
        while i < limit and before[i] == after[i]:
            i += 1
        invalidated = len(before) - i
        if saved <= 0:
            # An edit that grows the context has no saving to weigh the cost against, so the
            # horizon does not apply and it is judged on its re-prefill cost alone. The
            # sketchpad is the legitimate case: it buys retention, not size. `render()` puts
            # it last precisely so this stays cheap, and the bar here is what catches a
            # future layout change that would silently make it expensive again.
            if invalidated > len(after):
                return VerifierVerdict(
                    False, 0.0,
                    f"reprefill: this adds {-saved:,} chars and invalidates "
                    f"{invalidated:,}; edit later in the context instead")
            return VerifierVerdict(True, 1.0,
                                   f"reprefill: grows by {-saved:,} chars, "
                                   f"{invalidated:,} invalidated")
        # Turns, not a ratio of characters: "this edit pays for itself after N more turns".
        # Reported in those terms because that is the quantity the model can act on — it
        # knows whether it is near the start of a long task or the end of a short one, and a
        # bare multiple does not tell it which way to move.
        payback = invalidated / saved
        if payback > self.horizon_turns:
            return VerifierVerdict(
                False, self.horizon_turns / payback,
                f"reprefill: saves {saved:,} chars per turn but invalidates {invalidated:,} "
                f"once, so it only pays off after {payback:.0f} more turns (budget "
                f"{self.horizon_turns:.0f}). Everything after step {op.get('step')} would be "
                f"re-sent; edit a later step, or remove more at once.")
        return VerifierVerdict(True, min(1.0, self.horizon_turns / max(payback, 1e-9)),
                               f"reprefill: saves {saved:,} chars/turn, invalidates "
                               f"{invalidated:,} once (pays off in {payback:.0f} turns)")


class JudgeContextEdit(dspy.Signature):
    """Judge whether a model's edit to its own working context should be allowed.

    The edit is being made part-way through the task, so there is no grading to appeal to —
    judge it against the task statement and what the run has done so far. Admit an edit that
    removes bulk the task no longer needs. Reject one that discards evidence the deliverable
    will have to cite, that drops the only record of a decision already made, or that is
    clearing away a failed approach whose conclusion has not been written down anywhere.
    Return admit ("yes"/"no"), a 0..1 confidence score, and a one-sentence justification.
    """

    proposed_edit: str = dspy.InputField(desc="JSON of one context edit")
    task: str = dspy.InputField(desc="the task being worked on")
    context_after: str = dspy.InputField(desc="the context as the edit would leave it")
    admit: str = dspy.OutputField(desc='"yes" or "no"')
    score: str = dspy.OutputField(desc="confidence in [0,1]")
    justification: str = dspy.OutputField(desc="one sentence")


class PredictContextVerifier(dspy.Module):
    """The generative rung: is this turn still load-bearing for the task?

    Shown the context *after* the edit rather than the turn being removed, because the
    question is whether what remains still supports the task — a turn can be safely dropped
    precisely when its conclusion survives somewhere else, and that is only visible in the
    result.
    """

    name = "context_judge"

    def __init__(self, min_score: float = 0.0, max_context_chars: int = 6000) -> None:
        super().__init__()
        self.min_score = min_score
        self.max_context_chars = max_context_chars
        self.judge = dspy.Predict(JudgeContextEdit)

    def verify(self, op: dict, feedback, trajectory: list[dict]) -> VerifierVerdict:
        # The op's `before`/`after` renderings are the whole context and would dominate the
        # judge's prompt; `context_after` carries one of them deliberately and the op
        # itself is trimmed to the edit.
        edit = {k: v for k, v in op.items() if k not in ("before", "after")}
        pred = self.judge(
            proposed_edit=json.dumps(edit)[:2000],
            task=feedback.as_text(),
            # Tail, not head: the question is what the edit left behind, and the turns
            # nearest the edit are the ones whose support is in doubt.
            context_after=op.get("after", "")[-self.max_context_chars:],
        )
        try:
            score = float(str(pred.score).strip())
        except (TypeError, ValueError):
            score = 0.0
        admit = str(pred.admit).strip().lower() in ("yes", "true", "1")
        return VerifierVerdict(admitted=admit and score >= self.min_score, score=score,
                               justification=f"judge: {pred.justification}")


def build_context_ladder(*, generative: bool = True, min_score: float = 0.0,
                         horizon_turns: float = 8.0, adaptive: bool = True,
                         explore: float = 0.0, seed: int = 0) -> LadderVerifier:
    """The live context-edit ladder, built in one place so every arm shares it.

    Same `LadderVerifier` as the ledger's, so `CostLadderPlanner` orders these rungs by
    `cost / P(fail)` and short-circuits at the first refusal, and `adaptive=True` for the
    same reason it is there: a refused context edit leaves no reprefill event, so the ladder
    is the only component that observes its own refusals.

    `generative=False` is the hermetic rung set — the two deterministic probes, no LM call.
    That is what `gate="probe"` gets, and unlike the ledger ladder it is not a degraded
    version of the gate: retention and re-prefill are the two refusals this phase exists to
    make, and the judge is the rung that catches what they cannot express.
    """
    levels = [
        VerifierLevel("context_retention", 1, RetentionProbe()),
        VerifierLevel("context_reprefill", 2, ReprefillProbe(horizon_turns=horizon_turns)),
    ]
    if generative:
        levels.append(VerifierLevel("context_judge", 100,
                                    PredictContextVerifier(min_score=min_score)))
    return LadderVerifier(levels, adaptive=adaptive, explore=explore, seed=seed)
