"""noise_floor.py — measure how much a LAB score moves when nothing changes.

Before any A/B is interpretable, you need to know what a null result looks like. Two
independent sources of jitter, measured separately because they have different fixes:

  --judge   Re-grade the SAME deliverable N times. Isolates judge non-determinism; costs
            no agent runs. If this spread is 8%, a learning effect of 5% is unmeasurable.
  --agent   Re-run the SAME task N times end to end, grading each once. Includes judge
            noise plus the RLM taking a different path each run. This is the floor an
            experiment actually has to clear.

The number that gates the next experiment is the **minimum detectable effect**: an arm
difference smaller than the agent spread is indistinguishable from doing nothing.

    .venv/bin/python scripts/noise_floor.py --task <dir> --judge --n 3
    .venv/bin/python scripts/noise_floor.py --task <dir> --agent --n 3
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import time

from sentinelprime.lab import (RubricJudge, load_task, output_files, prepare_workspace,
                               read_output, verdict_map, verdict_stability)


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


def _report(label: str, stability: dict, elapsed: float, n_criteria: int) -> None:
    rates = stability["rates"]
    print(f"\n=== {label} noise floor ({stability['replicates']} replicates, "
          f"{n_criteria} criteria, {elapsed:.0f}s) ===")
    print("  pooled rates : " + "  ".join(f"{r:.1%}" for r in rates))
    print(f"  mean         : {stability['mean']:.1%}")
    print(f"  spread       : {stability['spread']:.1%}  (max - min)")
    if len(rates) > 2:
        print(f"  stdev        : {statistics.stdev(rates):.1%}")
    print(f"  flipped      : {len(stability['flipped'])}/{n_criteria} criteria "
          f"({stability['flip_rate']:.1%}) changed verdict at least once")
    if stability["flipped"]:
        print(f"                 {', '.join(stability['flipped'][:12])}"
              + (" ..." if len(stability["flipped"]) > 12 else ""))
    print(f"\n  => minimum detectable effect is ABOVE {stability['spread']:.1%} pooled "
          "rate. An arm difference smaller than that means nothing.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--judge", action="store_true", help="re-grade one fixed deliverable")
    ap.add_argument("--agent", action="store_true", help="re-run the task end to end")
    ap.add_argument("--workspace", default=None,
                    help="existing workspace to re-grade (--judge); defaults to e1")
    ap.add_argument("--temperature", type=float, default=None,
                    help="judge/agent temperature; leave unset for the provider default")
    ap.add_argument("--out", default="lab_runs")
    args = ap.parse_args()
    if not (args.judge or args.agent):
        raise SystemExit("pick --judge and/or --agent")

    _load_dotenv()
    import dspy
    from sentinelprime.agent import PrimeAgent
    from sentinelprime.audit import AuditLog
    from sentinelprime.harness import ContinualHarness
    from sentinelprime.memory import JsonMemoryBackend

    task = load_task(args.task)
    model = os.environ.get("OPENAI_MODEL", "openai/gpt-5.6-luna")
    # cache=False is load-bearing: with LiteLLM's cache on, replicates 2..N are served
    # from replicate 1 and the measured spread is exactly 0.0% — the cache's determinism,
    # not the judge's. (Observed: 165 judge calls in 2s on the first attempt.)
    lm_kwargs = {"max_tokens": 8000, "cache": False}
    if args.temperature is not None:
        lm_kwargs["temperature"] = args.temperature
    lm = dspy.LM(model, **lm_kwargs)
    dspy.configure(lm=lm)
    judge = RubricJudge(lm)
    run_root = pathlib.Path(args.out) / task.task_id
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[noise] task {task.task_id} | {len(task.criteria)} criteria | model {model}")

    if args.judge:
        ws = pathlib.Path(args.workspace or (run_root / "workspace-e1"))
        out, produced = read_output(ws), output_files(ws)
        if not produced:
            raise SystemExit(f"no deliverable in {ws}; run lab_eval.py first")
        print(f"[noise] re-grading {len(out):,} chars from {produced} x{args.n}")
        reps, t0 = [], time.time()
        for i in range(args.n):
            res = judge.judge(task, out, available_files=produced)
            reps.append(verdict_map(res))
            print(f"  judge replicate {i + 1}: "
                  f"{sum(1 for v in reps[-1].values() if v == 'pass')}/{len(reps[-1])}")
        st = verdict_stability(reps)
        _report("JUDGE", st, time.time() - t0, len(task.criteria))
        (run_root / "noise-judge.json").write_text(json.dumps(st, indent=2))

    if args.agent:
        print(f"[noise] re-running the agent end to end x{args.n}")
        reps, t0 = [], time.time()
        for i in range(args.n):
            # A fresh, empty ledger each run: this measures the *baseline* agent's spread,
            # with no learning carried between replicates.
            backend = JsonMemoryBackend(str(run_root / f"noise-ledger-{i}.json"))
            harness = ContinualHarness(backend, audit_log=AuditLog(str(run_root / f"noise-audit-{i}.json")))
            agent = PrimeAgent(harness, root_lm=lm, sub_lm=lm)
            ws = prepare_workspace(task, run_root / f"workspace-noise-{i}")
            started = time.time()
            agent.run_task(
                f"{task.instructions}\n\n"
                f"The source documents are the .txt files in {ws / 'documents'}. "
                f"You MUST write your deliverable(s) as files into {ws / 'output'} — "
                f"named {', '.join(task.deliverables)}. "
                f"Returning text without writing the file(s) scores zero.",
                workdir=str(ws), task_id=task.task_id,
            )
            out, produced = read_output(ws), output_files(ws)
            res = judge.judge(task, out, available_files=produced)
            reps.append(verdict_map(res))
            passed = sum(1 for v in reps[-1].values() if v == "pass")
            print(f"  agent replicate {i + 1}: {passed}/{len(reps[-1])} "
                  f"| wrote {produced or 'NOTHING'} | {time.time() - started:.0f}s")
        st = verdict_stability(reps)
        _report("AGENT+JUDGE", st, time.time() - t0, len(task.criteria))
        (run_root / "noise-agent.json").write_text(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
