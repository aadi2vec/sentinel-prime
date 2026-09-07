"""paired_ab.py — the experiment: does the ledger help, measured task-by-task.

Two arms over the SAME task sequence:

  control   : empty ledger, refine() disabled. The agent never learns anything.
  learning  : one ledger carried across the whole sequence — task 1's failures become
              task 2's guidance. This is the online setting the project claims.

Reported per task and then paired, because per-run agent variance on this benchmark is
~7% stdev (see scripts/noise_floor.py) — large enough to swamp any plausible effect if the
arms were compared as independent samples. Comparing arms on the same task cancels the
task-difficulty term.

Resumable by design: a full sweep is hours long and this machine has already had jobs
killed for memory. Every (task, arm) result is written as it completes and skipped on
re-run, so an interrupted sweep continues where it stopped.

    .venv/bin/python scripts/paired_ab.py --plan            # cost/time estimate, no calls
    .venv/bin/python scripts/paired_ab.py --arm control
    .venv/bin/python scripts/paired_ab.py --arm learning
    .venv/bin/python scripts/paired_ab.py --report          # analyse what has completed
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

from sentinelprime.audit import AuditLog
from sentinelprime.credit import CreditAssigner
from sentinelprime.harness import ContinualHarness
from sentinelprime.lab import (RubricJudge, all_pass, load_task, output_files,
                               paired_summary, prepare_workspace, read_output, to_feedback)
from sentinelprime.memory import JsonMemoryBackend
from sentinelprime.monitor import ProgressMonitor

TASKS_DIR = pathlib.Path("lab_tasks/corporate-ma")
RUNS = pathlib.Path("lab_runs/paired")


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


def task_dirs() -> list[pathlib.Path]:
    return sorted(d for d in TASKS_DIR.iterdir() if (d / "task.json").is_file())


def result_path(arm: str, task_id: str) -> pathlib.Path:
    return RUNS / arm / f"{task_id}.json"


def prompt_for(task, ws) -> str:
    return (f"{task.instructions}\n\n"
            f"The source documents are the .txt files in {ws / 'documents'}. "
            f"You MUST write your deliverable(s) as files into {ws / 'output'} — "
            f"named {', '.join(task.deliverables)}. "
            f"Returning text without writing the file(s) scores zero.")


def run_arm(arm: str, limit: int | None) -> None:
    import dspy
    from sentinelprime.agent import PrimeAgent

    _load_dotenv()
    model = os.environ.get("OPENAI_MODEL", "openai/gpt-5.6-luna")
    lm = dspy.LM(model, max_tokens=8000)
    dspy.configure(lm=lm)
    judge = RubricJudge(lm)
    (RUNS / arm).mkdir(parents=True, exist_ok=True)

    learning = arm == "learning"
    # ONE ledger for the whole sequence in the learning arm — carrying lessons between
    # tasks is the thing under test. The control arm never constructs one.
    audit_log = AuditLog(str(RUNS / arm / "_audit.json"))
    backend = JsonMemoryBackend(str(RUNS / arm / "_ledger.json"))
    harness = ContinualHarness(
        backend, audit_log=audit_log,
        credit_assigner=CreditAssigner(audit_log) if learning else None,
    )
    agent = PrimeAgent(harness, root_lm=lm, sub_lm=lm,
                       monitor=ProgressMonitor() if learning else None)

    tasks = task_dirs()[:limit] if limit else task_dirs()
    print(f"[{arm}] {len(tasks)} tasks | model {model}\n")
    for i, d in enumerate(tasks, 1):
        task = load_task(d)
        dest = result_path(arm, task.task_id)
        if dest.exists():
            prior = json.loads(dest.read_text())
            print(f"  [{i}/{len(tasks)}] {task.task_id:<48} skip (have "
                  f"{prior['pooled']:.1%})")
            continue

        ws = prepare_workspace(task, RUNS / arm / f"ws-{task.task_id}")
        started = time.time()
        agent.run_task(prompt_for(task, ws), workdir=str(ws),
                       context={"matter": task.task_id} if learning else None,
                       task_id=task.task_id)
        produced = output_files(ws)
        results = judge.judge(task, read_output(ws), available_files=produced)
        fb = to_feedback(task.task_id, results)
        elapsed = time.time() - started

        ledger_note = ""
        if learning:
            refine = agent.learn(agent.last_trajectory, fb)
            ledger_note = (f" | ledger {len(backend.read())} "
                           f"(+{len(refine.created)} -{len(refine.retired)} "
                           f"x{len(refine.rejected)})")
        dest.write_text(json.dumps({
            "task_id": task.task_id, "arm": arm, "pooled": fb.score,
            "all_pass": all_pass(fb), "n_criteria": len(results),
            "wrote": produced, "elapsed_s": elapsed, "criteria": results,
        }, indent=2))
        print(f"  [{i}/{len(tasks)}] {task.task_id:<48} {fb.score:>6.1%} "
              f"({sum(1 for r in results if r['verdict'] == 'pass')}/{len(results)}) "
              f"{elapsed:>4.0f}s{ledger_note}")


def report() -> None:
    rows = []
    for d in task_dirs():
        tid = d.name
        c, t = result_path("control", tid), result_path("learning", tid)
        if c.exists() and t.exists():
            rows.append((tid, json.loads(c.read_text())["pooled"],
                         json.loads(t.read_text())["pooled"]))
    if len(rows) < 2:
        print(f"only {len(rows)} paired task(s) complete — run both arms first")
        return

    print(f"{'task':<48} {'control':>8} {'learning':>9} {'delta':>8}")
    for tid, ctl, trt in rows:
        print(f"{tid:<48} {ctl:>7.1%} {trt:>8.1%} {trt - ctl:>+7.1%}")
    s = paired_summary(rows)
    print(f"\nmean control  : {sum(r[1] for r in rows) / len(rows):.1%}")
    print(f"mean learning : {sum(r[2] for r in rows) / len(rows):.1%}")
    print(f"mean delta    : {s['mean_delta']:+.1%}  (sd {s['sd']:.1%}, se {s['se']:.1%})")
    print(f"win/loss/tie  : {s['wins']}/{s['losses']}/{s['ties']}")
    print(f"detectable at : +-{s['detectable_at']:.1%} with n={s['n']}")
    print(f"\nVERDICT: {'effect exceeds its own error bar' if s['significant'] else 'INSIDE the noise — not a result'}")
    if not s["significant"]:
        print("An effect this size cannot be distinguished from run-to-run variance at "
              f"n={s['n']}. More tasks or more orders, or the effect is not there.")


def plan() -> None:
    tasks = [load_task(d) for d in task_dirs()]
    crit = sum(len(t.criteria) for t in tasks)
    # Measured on this machine: ~285s per agent run, ~2.2s per judge call.
    agent_s = len(tasks) * 2 * 285
    judge_s = crit * 2 * 2.2
    print(f"tasks              : {len(tasks)}")
    print(f"rubric criteria    : {crit}  (x2 arms = {crit * 2} judge calls)")
    print(f"agent runs         : {len(tasks) * 2}  (~285s each)")
    print(f"estimated wall time: {(agent_s + judge_s) / 3600:.1f}h "
          f"(agent {agent_s / 3600:.1f}h + judge {judge_s / 3600:.1f}h)")
    print("token cost         : NOT instrumented — this is the biggest unknown.")
    print(f"sensitivity        : with n={len(tasks)} pairs and ~10% delta sd, "
          f"detectable effect is about {2 * 10 / len(tasks) ** 0.5:.1f} points.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["control", "learning"])
    ap.add_argument("--limit", type=int, default=None, help="first N tasks only")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    if args.plan:
        plan()
    elif args.report:
        report()
    elif args.arm:
        run_arm(args.arm, args.limit)
    else:
        ap.error("pick --plan, --arm, or --report")


if __name__ == "__main__":
    main()
