"""Score a checking strategy against runs that already happened.

A live LAB task costs ~218s and ~270k tokens and its pooled score spans 29pp across
identical configurations. Paying one agent run per candidate policy makes the search
unaffordable and unreadable at the same time. Replay removes both problems at once: the
agent run becomes a fixed cost amortised over every candidate ever tested, and because the
deliverable is fixed, agent sampling variance does not enter the measurement at all.

What replay buys is **detection** — does this check flag the deliverables the judge failed
and leave alone the ones it passed. It cannot buy **correction**: a stored run cannot be
re-solved, so a revision policy is unmeasurable here. That is the whole reason promotion
keeps a live stage after this one, and why a candidate that wins on replay and loses live is
a finding rather than a bug.

The corpus is unusual in a way worth stating, because it is the reason this works. The same
criterion appears across several runs of the same task, and the agent is noisy enough that
the judge split many of them — passing a criterion on one deliverable and failing it on
another. Those split criteria are what a check can be *shown* to earn: a criterion the judge
never split is satisfied by a constant. The run-to-run variance that made the pooled rate
unreadable is what makes this corpus discriminative.

Checks take the live runner's own `(task, answer) -> CheckResult` shape, so one that scores
well here drops into the catalog without being rewritten.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sentinelprime.lab import document_text


@dataclass(frozen=True)
class GradedCriterion:
    """One rubric criterion, joined to the verdict a judge actually returned for it.

    `match_criteria` is the rubric's own wording — the text a check is synthesized *from*.
    `verdict` is what the judge said about this deliverable — the label a check is scored
    *against*. Both halves are required, and they live in different files.
    """

    id: str
    title: str
    match_criteria: str
    verdict: str
    reasoning: str


@dataclass(frozen=True)
class GradedRun:
    run_id: str
    task_id: str
    task: str
    deliverable: str
    criteria: tuple[GradedCriterion, ...]


@dataclass(frozen=True)
class CheckScore:
    """Agreement between a check and the judge, in the four cells that matter.

    Named for what they mean rather than tp/fp/tn/fn, because the asymmetry is the point: a
    false alarm costs a solver attempt on a deliverable that was already fine, while a miss
    costs nothing extra and simply leaves the failure where it was.
    """

    caught: int = 0         # judge failed it, the check flagged it
    false_alarms: int = 0   # judge passed it, the check flagged it anyway
    missed: int = 0         # judge failed it, the check let it through
    cleared: int = 0        # judge passed it, the check let it through

    @property
    def total(self) -> int:
        return self.caught + self.false_alarms + self.missed + self.cleared

    @property
    def flagged(self) -> int:
        return self.caught + self.false_alarms

    @property
    def precision(self) -> float | None:
        """Of what this check flagged, how much was really wrong.

        None when it flagged nothing. Reporting 0.0 would rank a silent check below a
        wrong one, and a silent check is not wrong — it simply has not been tested. The
        degenerate always-flag strategy has perfect recall, so precision is the only thing
        that refuses it, which is why admission can never be recall alone.
        """
        return self.caught / self.flagged if self.flagged else None

    @property
    def recall(self) -> float | None:
        """Of what was really wrong, how much this check found. None when nothing was."""
        failures = self.caught + self.missed
        return self.caught / failures if failures else None


def load_corpus(runs_root: str | Path, tasks_root: str | Path,
                skipped: list[dict] | None = None) -> list[GradedRun]:
    """Every stored run that has both a deliverable and graded criteria.

    A run's `*.json` carries the verdicts; the task directory carries the rubric wording;
    the workspace carries the deliverable. All three are needed, and a run missing any of
    them is skipped rather than half-loaded — a check scored against a partial row would be
    scored against a fiction.

    A run pairs with the workspace **named for it** — `rep0.json` with `ws-rep0/` — never
    with whatever `output/` happens to sit nearby. Runs share a directory, so searching
    downward from the run's parent silently concatenates a sibling's deliverable, and the
    check is then scored against text the judge never saw. That bug is invisible in a
    fixture with one run per directory, which is how it survived its first tests.

    Pass `skipped` to receive a reason per rejected file. An empty corpus and a corpus
    whose layout does not match look identical otherwise, and the silent version of that
    cost a debugging cycle.
    """
    rejected = skipped if skipped is not None else []
    tasks_root = Path(tasks_root)
    rubrics: dict[str, dict] = {}
    prompts: dict[str, str] = {}
    for task_json in sorted(tasks_root.glob("*/task.json")):
        spec = json.loads(task_json.read_text())
        task_id = task_json.parent.name
        rubrics[task_id] = {c.get("id", ""): c for c in spec.get("criteria", [])}
        prompts[task_id] = f"{spec.get('title', '')}\n\n{spec.get('instructions', '')}"

    runs_root = Path(runs_root)
    corpus: list[GradedRun] = []
    for result in sorted(runs_root.rglob("*.json")):
        try:
            data = json.loads(result.read_text())
        except (ValueError, OSError):
            continue
        # The corpus tree also holds ledgers, policy archives and traces. Anything that is
        # not shaped like a run record is passed over silently — it was never a candidate,
        # so reporting it as "skipped" would bury the real layout problems in noise.
        if not isinstance(data, dict):
            continue
        rows = data.get("criteria")
        if not isinstance(rows, list) or not rows:
            continue
        task_id = data.get("task_id")
        if task_id is None:
            rejected.append({"path": str(result), "reason": "no task_id: cannot join rubric text"})
            continue
        if task_id not in rubrics:
            rejected.append({"path": str(result), "reason": f"unknown task_id {task_id!r}"})
            continue
        deliverable = _deliverable(result.parent / f"ws-{result.stem}")
        if not deliverable:
            rejected.append({"path": str(result),
                             "reason": f"no deliverable under ws-{result.stem}/output"})
            continue
        rubric = rubrics[task_id]
        criteria = tuple(
            GradedCriterion(
                id=row["id"], title=row.get("title", ""),
                match_criteria=rubric.get(row["id"], {}).get("match_criteria", ""),
                verdict=row["verdict"], reasoning=row.get("reasoning", ""))
            for row in rows
            if isinstance(row, dict) and row.get("id") in rubric
            # `error` means the judge never rendered an opinion. Scoring a check against it
            # would measure agreement with a rate limiter.
            and str(row.get("verdict", "")).strip().lower() in ("pass", "fail")
            and not _is_legacy_judge_error(row)
        )
        if criteria:
            corpus.append(GradedRun(
                # Path-relative, not the bare filename: two arms both write `rep0.json`,
                # and keyed on the stem alone they collide.
                run_id=str(result.relative_to(runs_root).with_suffix("")),
                task_id=task_id, task=prompts[task_id],
                deliverable=deliverable, criteria=criteria))
    return corpus


def _is_legacy_judge_error(row: dict) -> bool:
    """A transport failure recorded before `error` was a verdict.

    Those runs stored the exception as verdict "fail" with the exception text as the
    reasoning, so a verdict filter admits them and 44 rate limits become 44 missed
    criteria. The judge no longer produces them, but the stored corpus cannot be re-graded
    — it would cost another full run and the deliverables would differ — so they are
    recognised here by the marker `RubricJudge` wrote.
    """
    return str(row.get("reasoning", "")).lstrip().startswith("judge error:")


def _deliverable(workspace: Path) -> str:
    """Everything this run wrote to its own `output/`, concatenated.

    Scoped to one named workspace rather than searched for, so a sibling run's deliverable
    can never leak in.
    """
    out = workspace / "output"
    if not out.is_dir():
        return ""
    return "\n\n".join(
        text for text in (document_text(p) for p in sorted(out.rglob("*")) if p.is_file())
        if text)


Check = Callable[[str, str], object]  # (task, answer) -> CheckResult


def score_check(check: Check, corpus: list[GradedRun], criterion_id: str) -> CheckScore:
    """How well `check` reproduces the judge's verdicts on one criterion.

    Scored only where the judge actually graded that criterion. A check is a *predicate on
    the deliverable*, so it sees the same two arguments the live runner would pass it and
    nothing else — never the verdict it is being scored against, and never the gold.
    """
    cells = {"caught": 0, "false_alarms": 0, "missed": 0, "cleared": 0}
    for run in corpus:
        for criterion in run.criteria:
            if criterion.id != criterion_id:
                continue
            satisfied = bool(check(run.task, run.deliverable).passed)
            judged_pass = criterion.verdict == "pass"
            if judged_pass:
                cells["cleared" if satisfied else "false_alarms"] += 1
            else:
                cells["missed" if satisfied else "caught"] += 1
    return CheckScore(**cells)


def discriminative_criteria(corpus: list[GradedRun]) -> list[str]:
    """Criteria the judge split — passed on one deliverable, failed on another.

    These are the only rows on which a check can be *shown* to earn its place. A criterion
    that always passed, or always failed, is reproduced perfectly by a constant, so
    agreement there says nothing about whether the check reads the deliverable at all.
    """
    seen: dict[str, set[str]] = {}
    for run in corpus:
        for criterion in run.criteria:
            seen.setdefault(criterion.id, set()).add(criterion.verdict)
    return sorted(cid for cid, verdicts in seen.items() if len(verdicts) > 1)
