"""Read the Phase 0 kill gate: does the baseline RLM already edit its context in place?

PLAN §3a admits the write path (Phases 1-3) only if this says it should, and states the
condition in advance: "if Phase 0 shows the baseline RLM already handles surgical in-place
edits competently, the write path is not this system's bottleneck and Phases 1-3 are
cancelled." `parity.gate_verdict` holds the thresholds; this script supplies it numbers.

    .venv/bin/python scripts/parity_gate.py --live --model PROVIDER/MODEL \
        --per-kind 4 --out lab_runs/<new>

**--live is required for a verdict.** The gate's question is about a model's behavior, so a
scripted run can exercise this script but cannot answer it — without --live the report is
written and the verdict is withheld rather than printed from a stub.

`learn()` is never called here. The gate measures the *baseline*, and a ledger that grew
across the case sequence would mean later cases were run by a different system than earlier
ones — which is also why every case gets a fresh root.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dspy

from sentinelprime.assembly import assemble
from sentinelprime.parity import build_cases, gate_verdict, kind_of, parity_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="lab_runs/parity-gate")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", help="Explicit DSPy provider/model; credentials from environment")
    parser.add_argument("--per-kind", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 1 <= args.per_kind <= 20:
        parser.error("--per-kind must be 1..20")
    if args.live and not args.model:
        parser.error("--live requires an explicit --model")
    root = pathlib.Path(args.out)
    if root.exists():
        parser.error("output directory already exists; choose a fresh --out")
    root.mkdir(parents=True)

    cases = build_cases(seed=args.seed, per_kind=args.per_kind)
    print(f"parity gate: {len(cases)} cases, seed={args.seed}, "
          f"live={args.live}", flush=True)

    def lm_for(case) -> object:
        if args.live:
            return dspy.LM(args.model, max_tokens=3000, cache=False)
        # Scripted mode runs an **echo baseline** through the real RLM loop: one turn that
        # submits the task back verbatim. That is a useful degenerate control rather than
        # filler — an echo keeps every preserved line (footprint 0.0) while never deriving
        # the new value (correct_rate 0.0), so it demonstrates that the instrument
        # separates "left everything alone" from "made the edit", and that `gate_verdict`
        # reports `inconclusive` instead of mistaking a no-op for a surgical edit.
        from dspy.utils.dummies import DummyLM
        return DummyLM([{"reasoning": "scripted echo baseline; not learned reasoning",
                         "code": f"SUBMIT(deliverable={case.task!r})"}])

    rows, failures = [], []
    for case in cases:
        # One root per case: the gate is a baseline measurement, so nothing a case
        # produced may be visible to the next one.
        workdir = root / "work" / case.id
        (workdir / "documents").mkdir(parents=True)
        (workdir / "output").mkdir(parents=True)
        bundle = assemble(lm=lm_for(case), root=root / "state" / case.id)
        if not rows:
            print(bundle.describe(), flush=True)
        try:
            pred = bundle.agent.run_task(case.task, str(workdir), task_id=case.id)
            answer = str(getattr(pred, "deliverable", "") or "")
        except Exception as exc:
            # Recorded as an empty answer *and* as a failure. Silently scoring a crash as
            # a wrong answer would read as the model being bad at editing.
            answer, _ = "", failures.append({"id": case.id, "error": f"{type(exc).__name__}: {exc}"})
        rows.append((case, answer))

    report = parity_report(rows)
    verdict = gate_verdict(report)
    payload = {
        "seed": args.seed, "per_kind": args.per_kind,
        "model": args.model if args.live else "scripted",
        "live": args.live, "report": report, "verdict": verdict,
        "failures": failures,
        "answers": [{"id": c.id, "kind": kind_of(c.family), "answer": a}
                    for c, a in rows],
    }
    (root / "report.json").write_text(json.dumps(payload, indent=2, default=str))

    print(report["describe"], flush=True)
    if failures:
        print(f"{len(failures)} case(s) errored; see report.json", flush=True)
    if args.live:
        print(f"verdict: {verdict['decision']} — {verdict['why']}", flush=True)
    else:
        print("verdict withheld: a scripted run cannot answer a question about a model's "
              "editing behavior. Re-run with --live --model.", flush=True)
    print(f"Artifacts: {root / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
