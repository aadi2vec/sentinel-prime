"""arm_f.py — does a ledger help at all? The cheapest test that can answer it.

Every gate, retirement rule and optimizer in this repo protects a ledger. None of them is
worth anything if guidance carried into a task does not change what the agent does — and
that premise has never been measured. `paired_ab.py` measures whether the *automatic* loop
finds good guidance, which is a harder question asked later; this asks whether good
guidance helps when it is simply handed over. If the answer is no, no proposer can rescue
it and most of the stack is deletable.

So: the same task, twice, differing only in whether one hand-written procedural lesson sits
in the ledger. This is arm F of the eval plan, scoped down from "an oracle ledger" to a
single lesson aimed at the failure the run data actually shows.

**Why that lesson.** The 50 failed criteria in `lab_runs/paired/` are 29 omissions, 17
partials and ~0 hallucinations. On `extract-change-of-control-provisions` the agent wrote
"no substantive analysis was supplied" for three of eight documents and shipped it; those
documents scored 9 fail / 1 pass, while the ones it read scored 8 fail / 18 pass. The
dominant failure is abandonment, not reasoning — so the lesson targets abandonment.

**Why documents, not the pass rate, is the metric.** Agent spread on this benchmark is
12.7pp over three replicates (`scripts/noise_floor.py`), which swamps any effect two runs
could show on a pooled rate. Whether a named document was ever opened is a substring test
over the trajectory — `grounding.document_coverage`, no LM, no judge, essentially no
variance. The pooled rate is reported too, as a secondary that two replicates cannot
settle.

The ledger is frozen: seeded before the run, `learn()` never called. That isolates *does
guidance help* from *can the loop find it*, which is the whole point of the arm.

    .venv/bin/python scripts/arm_f.py --replicates 2
    .venv/bin/python scripts/arm_f.py --report
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import time
from datetime import datetime, timezone

from sentinelprime.assembly import assemble
from sentinelprime.grounding import document_coverage, document_names
from sentinelprime.lab import (RubricJudge, all_pass, judge_errors, load_task,
                               output_files, prepare_workspace, read_output, task_prompt,
                               to_feedback)
from sentinelprime.memory import MemoryItem
from sentinelprime.telemetry import RunLog, UsageMeter, load_prices

TASK_DIR = pathlib.Path("lab_tasks/corporate-ma/extract-change-of-control-provisions")
RUNS = pathlib.Path("lab_runs/arm_f")
LOG = RUNS / "run.jsonl"

# The one lesson under test. Written by hand from the observed failures, deliberately
# *procedural* rather than legal: it says nothing about change-of-control clauses, so a
# gain cannot be explained by leaking domain answers into the prompt. If it helps, what
# helped is task discipline — which is the kind of thing a ledger can plausibly learn.
LESSON = (
    "Before writing any deliverable, analyze EVERY file in the documents/ directory. "
    "List the directory first, then read each file in full and analyze it individually. "
    "Never write a deliverable that says analysis was 'not supplied', 'not available', "
    "'could not be performed', or that a document was not provided — every document in "
    "the workspace IS provided. If a file is hard to parse, extract what text you can and "
    "analyze that. A deliverable that skips a document scores zero on every criterion "
    "tied to it."
)

# Phrases the failed run actually used to declare a document unanalyzed. Counted in the
# deliverable as a direct, LM-free read on whether the lesson changed the behavior it names.
ABANDONMENT = ("no substantive analysis", "not supplied", "could not be performed",
               "was not provided", "no analysis was")


def _load_dotenv(path: str = ".env") -> None:
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        os.environ.setdefault(key.strip(), val)


def _token_total(usage: dict) -> int:
    return sum(int(v.get("prompt_tokens") or 0) + int(v.get("completion_tokens") or 0)
               for v in (usage or {}).values() if isinstance(v, dict))


async def _run_one(arm: str, rep: int, lm, judge, log, meter) -> dict:
    """One replicate of one arm. Returns the row; also writes it for --report."""
    import dspy

    task = load_task(TASK_DIR)
    dest = RUNS / arm / f"rep{rep}.json"
    if dest.exists():
        row = json.loads(dest.read_text())
        log.event("skip", arm=arm, rep=rep, pooled=row["pooled"])
        return row

    root = RUNS / arm / f"state-rep{rep}"
    # Through assemble(), like every other entry point. Gate and record are off because
    # nothing is admitted here — the ledger is seeded by hand and frozen, so there is no
    # proposal for a gate to judge and no round for a trainset to record. Audit stays on
    # so the seeded item's provenance is still on the record.
    system = assemble(lm=lm, root=root, sub_lm=lm, gate=None,
                      audit=True, credit=False, monitor=False, record=False)

    if arm == "oracle":
        system.backend.write([MemoryItem(
            id="lesson.analyze-every-document", scope="global", kind="note",
            text=LESSON, created_at=datetime.now(timezone.utc).isoformat(),
            meta={"source": "arm_f", "targets": [], "hand_written": True})])

    ws = prepare_workspace(task, RUNS / arm / f"ws-rep{rep}")
    docs = document_names(ws / "documents")
    log.event("task_start", arm=arm, rep=rep, criteria=len(task.criteria),
              documents=len(docs), ledger=len(system.backend.read()),
              system=system.describe())
    started = time.time()

    with dspy.track_usage() as tracker:
        await system.agent.arun_task(task_prompt(task, ws), workdir=str(ws),
                                     task_id=task.task_id)
        trajectory = system.agent.last_trajectory
        produced = output_files(ws)
        deliverable = read_output(ws)
        results = judge.judge(task, deliverable, available_files=produced)

    usage = tracker.get_total_tokens()
    meter.add(usage)
    fb = to_feedback(task.task_id, results)

    # The primary metric. Permissive by construction (a directory listing marks every file
    # it prints as touched), so it under-reports misses — which makes a *drop* in untouched
    # documents the conservative direction to read a result in.
    coverage = document_coverage(trajectory, docs)
    abandoned = [p for p in ABANDONMENT if p in deliverable.lower()]

    # The trajectory is the replay corpus: a candidate checking policy is scored against
    # stored runs rather than by re-running the agent, which is the only way to evaluate
    # policies at a sample size this task's ~29pp spread does not destroy. Writing it is
    # the difference between a run costing 300k tokens once and costing them again per
    # candidate.
    (RUNS / arm / f"trace-rep{rep}.json").write_text(
        json.dumps(trajectory, indent=2, default=str))

    row = {
        "arm": arm, "rep": rep,
        "untouched": coverage.untouched, "n_untouched": len(coverage.untouched),
        "coverage": round(coverage.rate, 4),
        "abandonment_phrases": abandoned, "n_abandonment": len(abandoned),
        "pooled": round(fb.score, 4), "all_pass": all_pass(fb, results),
        # A pooled rate over 37 of 55 criteria is a different number from one over 55, and
        # nothing in the rate says which it is. Record the denominator.
        "n_criteria": len(results), "n_graded": len(fb.criteria),
        "judge_errors": judge_errors(results), "turns": len(trajectory),
        "elapsed_s": round(time.time() - started), "tokens": _token_total(usage),
        "criteria": results,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row, indent=2))
    log.event("task_done", **{k: v for k, v in row.items() if k != "criteria"})
    return row


async def _main(replicates: int) -> None:
    import dspy

    _load_dotenv()
    model = os.environ.get("OPENAI_MODEL", "openai/gpt-5.6-luna")
    lm = dspy.LM(model, max_tokens=8000)
    dspy.configure(lm=lm)
    RUNS.mkdir(parents=True, exist_ok=True)
    log = RunLog(LOG)
    meter = UsageMeter(prices=load_prices())
    judge = RubricJudge(lm, parallel=8, run_log=log)

    log.event("arm_f_start", model=model, replicates=replicates, lesson=LESSON[:80])
    # Interleaved (empty, oracle, empty, oracle) rather than blocked, so any provider-side
    # drift over the run lands on both arms instead of only the one that ran second.
    for rep in range(replicates):
        for arm in ("empty", "oracle"):
            await _run_one(arm, rep, lm, judge, log, meter)
    log.event("arm_f_done", usage=meter.summary())
    report()


def report() -> None:
    rows = [json.loads(p.read_text())
            for p in sorted(RUNS.glob("*/rep*.json"))]
    if not rows:
        print("no runs yet")
        return
    print(f"\n{'arm':8s} {'rep':>3s} {'untouched':>9s} {'abandon':>7s} "
          f"{'pooled':>7s} {'graded':>7s} {'jerr':>5s} {'turns':>5s} {'tokens':>8s}")
    for r in sorted(rows, key=lambda r: (r["arm"], r["rep"])):
        print(f"{r['arm']:8s} {r['rep']:3d} {r['n_untouched']:9d} {r['n_abandonment']:7d} "
              f"{r['pooled']:7.3f} {r.get('n_graded', r['n_criteria']):7d} "
              f"{r.get('judge_errors', 0):5d} {r['turns']:5d} {r['tokens']:8d}")

    def mean(arm: str, key: str) -> float:
        vals = [r[key] for r in rows if r["arm"] == arm]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n--- the number this experiment exists to produce ---")
    for key, label in (("n_untouched", "documents never opened"),
                       ("n_abandonment", "abandonment phrases in deliverable"),
                       ("pooled", "pooled criterion rate")):
        e, o = mean("empty", key), mean("oracle", key)
        print(f"  {label:38s} empty={e:7.3f}  oracle={o:7.3f}  delta={o - e:+7.3f}")
    print("\n  Read: a drop in the first two means guidance changed behavior, and the\n"
          "  ledger has something worth gating. No drop means a completion check is the\n"
          "  product and the admission stack protects an asset that is not there.\n"
          "  The pooled rate cannot be settled by this few replicates (12.7pp noise floor).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        asyncio.run(_main(args.replicates))
