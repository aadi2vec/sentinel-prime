"""Score a checking strategy against runs that already happened.

A live LAB task costs ~218s and ~270k tokens and carries 29pp of spread, so one agent run
per candidate makes the search unaffordable and unreadable at once. Replay makes that run a
fixed cost amortised over every candidate ever tested: the candidate's checks execute for
real, only the solver is served from storage.

What replay buys is *detection* — does this check flag the deliverables the judge failed,
and leave alone the ones it passed. It cannot buy correction, because a stored run cannot be
re-solved. That is why promotion needs a live stage after this one.

Checks here take the same `(task, answer) -> CheckResult` shape the live runner uses, so a
check that scores well on replay drops into the catalog without being rewritten.
"""
import json

import pytest

from sentinelprime.policy_search import CheckResult
from sentinelprime.replay import (
    CheckScore, GradedRun, discriminative_criteria, load_corpus, score_check,
)


def _corpus(tmp_path, runs):
    """runs: {run_id: (deliverable_text, {criterion_id: verdict})}"""
    tasks = tmp_path / "tasks" / "t1"
    (tasks / "documents").mkdir(parents=True)
    ids = sorted({cid for _, v in runs.values() for cid in v})
    (tasks / "task.json").write_text(json.dumps({
        "title": "T", "instructions": "do it", "work_type": "analyze",
        "deliverables": ["out.txt"],
        "criteria": [{"id": c, "title": c, "match_criteria": f"PASS if {c} present"}
                     for c in ids]}))
    # All runs share one directory, as the real corpus does. A run pairs with the
    # workspace named for it — `rep0.json` with `ws-rep0/` — never with a sibling's.
    d = tmp_path / "runs"
    d.mkdir(parents=True, exist_ok=True)
    for run_id, (text, verdicts) in runs.items():
        (d / f"ws-{run_id}" / "output").mkdir(parents=True)
        (d / f"ws-{run_id}" / "output" / "out.txt").write_text(text)
        (d / f"{run_id}.json").write_text(json.dumps({
            "task_id": "t1",
            "criteria": [{"id": c, "title": c, "verdict": v, "reasoning": "because"}
                         for c, v in verdicts.items()]}))
    return load_corpus(d, tmp_path / "tasks")


def test_a_run_joins_its_verdicts_to_the_rubric_text_that_produced_them(tmp_path):
    """The verdict lives in the run, the criterion wording in the task. A check is
    synthesized from the wording and scored against the verdict, so both must arrive."""
    corpus = _corpus(tmp_path, {"r1": ("alpha", {"C-1": "pass"})})

    assert len(corpus) == 1
    run = corpus[0]
    assert isinstance(run, GradedRun)
    assert run.task_id == "t1" and run.deliverable == "alpha"
    assert run.criteria[0].match_criteria == "PASS if C-1 present"
    assert run.criteria[0].verdict == "pass"


def test_an_ungraded_criterion_never_enters_the_corpus(tmp_path):
    """A rate-limited judge call said nothing about the deliverable. Scoring a check
    against it would measure agreement with a transport failure."""
    corpus = _corpus(tmp_path, {"r1": ("alpha", {"C-1": "pass", "C-2": "error"})})
    assert [c.id for c in corpus[0].criteria] == ["C-1"]


def test_a_run_that_produced_no_deliverable_is_skipped(tmp_path):
    _corpus(tmp_path, {"r1": ("alpha", {"C-1": "pass"})})
    (tmp_path / "runs" / "ws-r1" / "output" / "out.txt").unlink()
    assert load_corpus(tmp_path / "runs", tmp_path / "tasks") == []


def test_a_run_is_paired_with_its_own_workspace_never_a_sibling(tmp_path):
    """Two runs live in one directory. Scoring r1's check against r1+r2's text
    concatenated would measure a check against a deliverable the judge never saw."""
    corpus = _corpus(tmp_path, {"r1": ("alpha only", {"C-1": "pass"}),
                                "r2": ("beta only", {"C-1": "fail"})})

    by_id = {r.run_id: r.deliverable for r in corpus}
    assert by_id["r1"] == "alpha only"
    assert by_id["r2"] == "beta only"


def test_a_run_that_does_not_name_its_task_is_skipped_and_said_so(tmp_path):
    """The rubric wording is joined by task. A run that cannot be joined has no
    criterion text, so a check could be scored but never synthesized."""
    _corpus(tmp_path, {"r1": ("alpha", {"C-1": "pass"})})
    path = tmp_path / "runs" / "r1.json"
    data = json.loads(path.read_text()); del data["task_id"]
    path.write_text(json.dumps(data))

    skipped = []
    assert load_corpus(tmp_path / "runs", tmp_path / "tasks", skipped=skipped) == []
    assert len(skipped) == 1 and "task_id" in skipped[0]["reason"]


def test_an_empty_corpus_is_never_silent(tmp_path):
    """The failure that wasted a cycle: load_corpus returned [] and said nothing, so a
    layout mismatch looked identical to having no data."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "tasks").mkdir()
    skipped = []
    assert load_corpus(tmp_path / "runs", tmp_path / "tasks", skipped=skipped) == []
    assert skipped == []


# --- scoring --------------------------------------------------------------------------

def _mentions(word):
    """A check in the live runner's own shape: (task, answer) -> CheckResult."""
    return lambda task, answer: CheckResult(word in answer, f"say {word}")


def test_a_check_is_scored_on_whether_it_agrees_with_the_judge(tmp_path):
    """The judge failed r2 and passed r1. A check earns its place by reproducing that
    split from the deliverable alone."""
    corpus = _corpus(tmp_path, {"r1": ("has alpha", {"C-1": "pass"}),
                                "r2": ("empty",     {"C-1": "fail"})})

    score = score_check(_mentions("alpha"), corpus, "C-1")

    assert score == CheckScore(caught=1, false_alarms=0, missed=0, cleared=1)
    assert score.precision == 1.0 and score.recall == 1.0


def test_a_check_that_flags_everything_is_caught_by_precision(tmp_path):
    """The degenerate strategy. Perfect recall, and precision is the only thing that
    refuses it — which is why admission cannot be recall alone."""
    corpus = _corpus(tmp_path, {"r1": ("has alpha", {"C-1": "pass"}),
                                "r2": ("empty",     {"C-1": "fail"})})

    score = score_check(lambda task, answer: CheckResult(False, "always"), corpus, "C-1")

    assert score.recall == 1.0
    assert score.precision == 0.5


def test_a_check_is_scored_only_where_the_judge_graded_that_criterion(tmp_path):
    corpus = _corpus(tmp_path, {"r1": ("has alpha", {"C-1": "pass"}),
                                "r2": ("empty",     {"C-2": "fail"})})
    assert score_check(_mentions("alpha"), corpus, "C-1").total == 1


def test_precision_and_recall_are_undefined_rather_than_zero_when_nothing_applies(tmp_path):
    """A check that flagged nothing has no precision; reporting 0.0 would rank it below a
    check that was merely wrong, and it was not wrong — it was silent."""
    corpus = _corpus(tmp_path, {"r1": ("has alpha", {"C-1": "pass"})})
    score = score_check(_mentions("alpha"), corpus, "C-1")
    assert score.precision is None   # never flagged
    assert score.recall is None      # nothing to catch


def test_discriminative_criteria_are_the_ones_the_judge_split(tmp_path):
    """Same criterion, different deliverables, different verdicts. Those are the rows a
    check can be shown to earn; a criterion the judge never split is satisfied by a
    constant."""
    corpus = _corpus(tmp_path, {"r1": ("has alpha", {"C-1": "pass", "C-2": "pass"}),
                                "r2": ("empty",     {"C-1": "fail", "C-2": "pass"})})
    assert discriminative_criteria(corpus) == ["C-1"]


def test_json_that_is_not_a_run_record_is_ignored(tmp_path):
    """The corpus tree holds ledgers, archives and traces too. A top-level list, or any
    shape that is not a run record, must be passed over rather than crash the load."""
    corpus_dir = tmp_path / "runs"
    _corpus(tmp_path, {"r1": ("alpha", {"C-1": "pass"})})
    (corpus_dir / "ledger.json").write_text(json.dumps([{"id": "lesson.x"}]))
    (corpus_dir / "archive.json").write_text(json.dumps({"policy": {"checks": []}}))
    (corpus_dir / "broken.json").write_text("{not json")

    skipped = []
    assert len(load_corpus(corpus_dir, tmp_path / "tasks", skipped=skipped)) == 1
    assert skipped == []


def test_run_ids_are_unique_across_subdirectories(tmp_path):
    """Two arms each write `rep0.json`. Keyed on the filename alone they collide, and any
    per-run report then silently describes one of them twice."""
    tasks = tmp_path / "tasks" / "t1"
    (tasks / "documents").mkdir(parents=True)
    (tasks / "task.json").write_text(json.dumps({
        "title": "T", "instructions": "do it", "deliverables": ["out.txt"],
        "criteria": [{"id": "C-1", "title": "C-1", "match_criteria": "PASS if alpha"}]}))
    for arm in ("empty", "oracle"):
        d = tmp_path / "runs" / arm
        (d / "ws-rep0" / "output").mkdir(parents=True)
        (d / "ws-rep0" / "output" / "out.txt").write_text(f"{arm} text")
        (d / "rep0.json").write_text(json.dumps({
            "task_id": "t1",
            "criteria": [{"id": "C-1", "verdict": "pass", "reasoning": ""}]}))

    corpus = load_corpus(tmp_path / "runs", tmp_path / "tasks")

    assert len({r.run_id for r in corpus}) == 2, [r.run_id for r in corpus]
