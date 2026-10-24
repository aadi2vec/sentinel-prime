"""Synthesize checks from rubric criteria and admit the ones that reproduce the judge.

This is §3 item 3 and the replay half of §4's two-stage evaluation: it ranks and filters at
~0 marginal cost, and the catalog it writes is what `policy_lab.py --checks` lets a candidate
policy select from. Nothing here runs an agent — the corpus is 368 graded criterion rows over
runs that already happened, so agent sampling variance (0.400-0.691 across five runs of one
identical config, §5) does not enter the measurement at all.

Three things are deliberate:

  - **Only discriminative criteria are attempted.** A criterion the judge never split is
    reproduced perfectly by a constant, so agreement there says nothing about whether the
    check reads the deliverable. `discriminative_criteria` is the filter, and a corpus with
    none of them is reported rather than silently producing an empty catalog.
  - **The error budget governs admission.** `check_set_objectives()` is §5's primary metric,
    and this is the one caller that has the graded corpus it needs. A synthesis run whose
    admitted checks keep landing under the precision floor is frozen and the freeze is
    audited — the same gate `learn()` consults, on the only objective `assemble()` cannot
    feed.
  - **Scripted mode proposes nothing.** A deterministic stand-in proposer would be a
    hand-written check wearing synthesis's name, which is exactly the limit §3 says this
    task exists to remove. Without `--live` the script loads the corpus, reports what is
    discriminative, and stops.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dspy

from sentinelprime.audit import AuditLog
from sentinelprime.errorbudget import ErrorBudget, check_set_objectives
from sentinelprime.replay import discriminative_criteria, load_corpus, score_check
from sentinelprime.synthesis import (
    CheckSpec, SynthesizeCheck, admissible, compile_check, dump_specs,
)


def _criterion_rows(corpus, criterion_ids):
    """One (criterion_id, title, text) per discriminative criterion, deduplicated.

    The same criterion recurs across runs of the same task — that recurrence is what makes
    the corpus discriminative — so synthesizing per *row* would pay for the same criterion
    several times and propose several checks under colliding names.
    """
    seen: dict[str, tuple[str, str]] = {}
    for run in corpus:
        for criterion in run.criteria:
            if criterion.id in criterion_ids and criterion.id not in seen:
                seen[criterion.id] = (criterion.title, criterion.match_criteria)
    return [(cid, *seen[cid]) for cid in sorted(seen)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="lab_runs", help="Stored graded runs")
    parser.add_argument("--tasks", default="lab_tasks/corporate-ma", help="Rubric source")
    parser.add_argument("--out", default="lab_runs/synthesis/checks.json")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", help="Explicit DSPy provider/model; credentials from environment")
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--min-recall", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Cap criteria attempted (0 = all)")
    args = parser.parse_args(argv)
    if args.live and not args.model:
        parser.error("--live requires an explicit --model")
    if not 0 < args.min_precision <= 1 or not 0 < args.min_recall <= 1:
        parser.error("min-precision and min-recall must be in (0, 1]")

    skipped: list[dict] = []
    corpus = load_corpus(args.runs, args.tasks, skipped=skipped)
    # An empty corpus and a corpus whose layout does not match look identical otherwise.
    print(f"corpus: {len(corpus)} graded runs, {len(skipped)} files skipped", flush=True)
    if not corpus:
        for row in skipped[:5]:
            print(f"  skipped: {row}", flush=True)
        return 1

    discriminative = discriminative_criteria(corpus)
    rows = _criterion_rows(corpus, set(discriminative))
    if args.limit:
        rows = rows[:args.limit]
    print(f"discriminative criteria: {len(discriminative)}; attempting {len(rows)}",
          flush=True)
    if not rows:
        print("no criterion was both passed and failed somewhere in the corpus; a check "
              "cannot be *shown* to earn the catalog here", flush=True)
        return 1

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # §5's primary metric, on the one caller that has the corpus it is defined over. The
    # freeze is an audit event, so it needs a log to land in.
    budget = ErrorBudget(
        check_set_objectives(min_precision=args.min_precision),
        audit_log=AuditLog(str(out.parent / "audit.json")),
        task_id="check-synthesis")

    if not args.live:
        print("SCRIPTED: no proposer. A deterministic stand-in would be a hand-written "
              "check wearing synthesis's name — the limit §3 says this task removes. "
              "Re-run with --live --model PROVIDER/MODEL.", flush=True)
        for cid, title, text in rows[:10]:
            print(f"  {cid}: {title} :: {text[:90]}", flush=True)
        return 1

    lm = dspy.LM(args.model, max_tokens=1200, cache=False)
    synthesize = dspy.Predict(SynthesizeCheck)
    admitted: dict[str, CheckSpec] = {}
    attempts: list[dict] = []
    for cid, title, text in rows:
        row = {"criterion_id": cid, "title": title}
        try:
            with dspy.context(lm=lm):
                pred = synthesize(criterion_title=title, criterion_text=text,
                                  # The corpus is 29 omissions / 17 partials / ~0
                                  # fabrications, so omission is the honest default prior.
                                  failure_mode="omission")
            spec = CheckSpec.parse(pred.check_json)
        except Exception as exc:
            # A refused or unparseable proposal is a recorded outcome, not a crash: the
            # refusal rate is itself the measurement of whether the signature works.
            row["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(row)
            print(f"  {cid}: refused ({row['error'][:70]})", flush=True)
            continue
        row["spec"] = {"kind": spec.kind, "params": spec.params}
        score = score_check(compile_check(spec), corpus, cid)
        row.update(precision=score.precision, recall=score.recall, flagged=score.flagged)
        if score.precision is not None:
            budget.observe("check_precision", score.precision)
        ok = admissible(spec, corpus, cid, args.min_precision, args.min_recall)
        row["admitted"] = ok
        if ok:
            admitted[f"{cid}_{spec.kind}"] = spec
        attempts.append(row)
        print(f"  {cid}: {spec.kind} precision={score.precision} "
              f"recall={score.recall} admitted={ok}", flush=True)
        if not budget.allows_refine():
            # The freeze governs synthesis exactly as it governs refine(): the admitted set
            # keeps landing under the precision floor, so stop producing more of it.
            print(f"FROZEN: {budget.describe()}", flush=True)
            break

    out.write_text(json.dumps(dump_specs(admitted), indent=2))
    (out.parent / "attempts.json").write_text(json.dumps(
        {"budget": budget.describe(), "frozen": budget.frozen,
         "min_precision": args.min_precision, "min_recall": args.min_recall,
         "model": args.model, "attempts": attempts}, indent=2, default=str))
    print(f"admitted {len(admitted)} of {len(attempts)} proposals -> {out}", flush=True)
    print(f"budget: {budget.describe()}", flush=True)
    print(f"next: .venv/bin/python scripts/policy_lab.py --checks {out} --out lab_runs/<new>",
          flush=True)
    return 0 if admitted and not budget.frozen else 1


if __name__ == "__main__":
    raise SystemExit(main())
