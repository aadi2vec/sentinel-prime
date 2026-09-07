"""paired_ab.py — the experiment: does the ledger help, measured task-by-task.

Four arms over the SAME task sequence, each adding exactly one mechanism to the one before
it, so a delta is attributable to that mechanism and not to the stack:

  control   : empty ledger, refine() disabled. The agent never learns anything.
  learning  : one ledger carried across the whole sequence — task 1's failures become
              task 2's guidance. This is the online setting the project claims.
  gated     : learning + the untuned admission ladder (grounding probe -> generative
              judge). Isolates admission control: does refusing bad edits beat taking them?
  tuned     : gated, with the judge GEPA-compiled by scripts/gepa_runner.py. Isolates the
              *tuning*, which is the prediction worth testing — under the solver/verifier
              gap model the ladder's fidelity sets the ceiling on everything downstream,
              so a better judge should move the asymptote and not merely the mean.

`gated` is the arm `tuned` must be compared against. Comparing `tuned` to `learning` would
credit the tuning for the gate's effect, and the gate is the larger of the two.

Reported per task and then paired, because per-run agent variance on this benchmark is
~7% stdev (see scripts/noise_floor.py) — large enough to swamp any plausible effect if the
arms were compared as independent samples. Comparing arms on the same task cancels the
task-difficulty term.

Everything is logged as it happens to `lab_runs/paired/run.jsonl`: ledger edits, verifier
rejections, retirements, sub-query cache stats, replan events, judge verdicts, token
usage. Tail it live with `scripts/watch_run.py`.

    .venv/bin/python scripts/paired_ab.py --plan
    .venv/bin/python scripts/paired_ab.py --arm control --concurrency 3
    .venv/bin/python scripts/paired_ab.py --arm learning         # concurrency forced to 1
    .venv/bin/python scripts/paired_ab.py --arm gated
    .venv/bin/python scripts/paired_ab.py --arm tuned --verifier-program compiled/verifier.json
    .venv/bin/python scripts/paired_ab.py --arm control --terminal   # run in a new window
    .venv/bin/python scripts/paired_ab.py --report

The learning arms also write `_proposals.jsonl`, which is the trainset `gepa_runner.py`
compiles from — the online loop manufactures the offline optimizer's training data, so the
`tuned` arm is only runnable after a `gated` (or `learning`) arm has produced one.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time

from sentinelprime.assembly import assemble
from sentinelprime.credit import grounding_from_trajectory
from sentinelprime.lab import (RubricJudge, all_pass, judge_errors, load_task,
                               output_files,
                               paired_summary, prepare_workspace, read_output, task_prompt,
                               to_feedback)
from sentinelprime.telemetry import RunLog, UsageMeter, load_prices

# Every arm that carries a ledger across tasks. Ordered: each adds one mechanism to the
# previous, which is what makes a between-arm delta attributable.
ARMS = ("control", "learning", "gated", "tuned")
LEARNING_ARMS = ("learning", "gated", "tuned")
# What each arm should be measured against. `tuned` vs `control` would bundle the ledger,
# the gate and the tuning into one number.
BASELINE = {"learning": "control", "gated": "learning", "tuned": "gated"}

TASKS_DIR = pathlib.Path("lab_tasks/corporate-ma")
RUNS = pathlib.Path("lab_runs/paired")
LOG = RUNS / "run.jsonl"


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


# The prompt itself lives in lab.py: gepa_runner's rollout must send a byte-identical one,
# and two copies that drift by a word are two different tasks.
prompt_for = task_prompt


def _token_total(usage: dict) -> int:
    return sum(int(v.get("prompt_tokens") or 0) + int(v.get("completion_tokens") or 0)
               for v in (usage or {}).values() if isinstance(v, dict))


def _emit_ledger_events(log: RunLog, task_id: str, refine, harness, backend) -> None:
    """Surface everything the harness did this round — the layers we built on top."""
    for edit_id in refine.created:
        log.event("ledger_create", task_id=task_id, edit_id=edit_id)
    for edit_id in refine.updated:
        log.event("ledger_update", task_id=task_id, edit_id=edit_id)
    for edit_id in refine.rejected:
        log.event("verifier_reject", task_id=task_id, edit_id=edit_id)
    for edit_id in refine.retired:
        log.event("credit_retire", task_id=task_id, edit_id=edit_id)
    if harness.audit_log is not None:
        for rec in harness.audit_log.by_version(refine.to_version):
            log.event("audit", task_id=task_id, edit_id=rec.edit_id, op=rec.op,
                      scope=rec.scope, targets=rec.targets,
                      verification=rec.verification[:140])
    log.event("ledger_size", task_id=task_id, size=len(backend.read()),
              versions=f"{refine.from_version}->{refine.to_version}")


async def _run_one(task_dir, arm, agent, judge, harness, backend, log, meter, learning,
                   sem) -> None:
    import dspy

    task = load_task(task_dir)
    dest = result_path(arm, task.task_id)
    if dest.exists():
        log.event("skip", task_id=task.task_id, arm=arm,
                  pooled=json.loads(dest.read_text())["pooled"])
        return

    async with sem:
        ws = prepare_workspace(task, RUNS / arm / f"ws-{task.task_id}")
        log.event("task_start", task_id=task.task_id, arm=arm,
                  criteria=len(task.criteria), documents=len(task.documents))
        started = time.time()

        # One tracker spanning the agent AND the judge. Tracking only the agent understates
        # a task by roughly half — the judge makes one call per rubric criterion — and a
        # cost figure that silently omits half the calls is worse than no figure at all.
        # dspy.track_usage() records raw lm() calls too, which is how the judge calls.
        # contextvars are per-asyncio-task, so this stays correct under --concurrency.
        with dspy.track_usage() as tracker:
            pred = await agent.arun_task(
                prompt_for(task, ws), workdir=str(ws),
                context={"matter": task.task_id} if learning else None,
                task_id=task.task_id)
            agent_usage = _token_total(tracker.get_total_tokens())

            # The RLM trace records *how* the answer was reached — the only artifact that
            # explains a score after the fact.
            trace = agent.last_trajectory
            (RUNS / arm / f"trace-{task.task_id}.json").write_text(
                json.dumps(trace, indent=2, default=str))
            log.event("rlm_trace", task_id=task.task_id, turns=len(trace),
                      chars=sum(len(str(t)) for t in trace))
            if agent.last_cache_stats:
                log.event("subquery_cache", task_id=task.task_id, **agent.last_cache_stats)
            if agent.last_monitor_decision and agent.last_monitor_decision.replan:
                log.event("replan", task_id=task.task_id,
                          reasons="; ".join(agent.last_monitor_decision.reasons))

            produced = output_files(ws)
            results = judge.judge(task, read_output(ws), available_files=produced)

        usage = tracker.get_total_tokens()
        meter.add(usage)
        total_tok = _token_total(usage)
        # total_tok == 0 with a completed task means LiteLLM served every call from its
        # local cache: real result, zero spend. Worth labelling, because a silent 0 reads
        # as broken instrumentation (and a cached judge is how the first noise floor
        # measured a false 0.0% spread).
        log.event("tokens", task_id=task.task_id, agent=agent_usage,
                  judge=total_tok - agent_usage, total=total_tok,
                  cached=(total_tok == 0))

        fb = to_feedback(task.task_id, results)
        elapsed = time.time() - started

        if learning:
            # match_criteria is the rubric's own wording, so its literals (amounts, section
            # numbers) are what a grounded run should have surfaced from the documents.
            criterion_texts = {c.get("id", ""): c.get("match_criteria", "")
                               for c in task.criteria}
            grounded = grounding_from_trajectory(agent.last_trajectory, criterion_texts)
            ungrounded = [cid for cid, g in grounded.items()
                          if g is False and any(r["id"] == cid and r["verdict"] == "pass"
                                                for r in results)]
            if ungrounded:
                log.event("ungrounded_pass", task_id=task.task_id, n=len(ungrounded),
                          ids=",".join(ungrounded[:10]),
                          why="passed but the facts never came back from the REPL; "
                              "credit withheld")
            refine = agent.learn(agent.last_trajectory, fb,
                                 criterion_texts=criterion_texts)
            _emit_ledger_events(log, task.task_id, refine, harness, backend)

        dest.write_text(json.dumps({
            "task_id": task.task_id, "arm": arm, "pooled": fb.score,
            "all_pass": all_pass(fb, results), "n_criteria": len(results),
            "n_graded": len(fb.criteria), "judge_errors": judge_errors(results),
            "wrote": produced, "elapsed_s": elapsed, "tokens": total_tok,
            "criteria": results,
        }, indent=2))
        log.event("task_done", task_id=task.task_id, arm=arm, pooled=round(fb.score, 4),
                  passed=f"{sum(1 for r in results if r['verdict'] == 'pass')}/{len(fb.criteria)}",
                  judge_errors=judge_errors(results), all_pass=all_pass(fb, results), wrote=produced or "NOTHING",
                  elapsed_s=round(elapsed), tokens=total_tok,
                  cached=(total_tok == 0))


def run_arm(arm: str, limit: int | None, concurrency: int,
            verifier_program: str | None = None, explore: float = 0.0) -> None:
    import dspy

    _load_dotenv()
    model = os.environ.get("OPENAI_MODEL", "openai/gpt-5.6-luna")
    lm = dspy.LM(model, max_tokens=8000)
    dspy.configure(lm=lm)
    (RUNS / arm).mkdir(parents=True, exist_ok=True)
    log = RunLog(LOG)
    meter = UsageMeter(prices=load_prices())

    learning = arm in LEARNING_ARMS
    if learning and concurrency != 1:
        # The ledger is the dependency between tasks: task 2 must see what task 1 taught.
        # Running them concurrently would break exactly the causal chain under test, and
        # would also race a single-writer ledger from several coroutines.
        log.event("concurrency_forced", arm=arm, requested=concurrency, using=1,
                  why="the learning arm is sequential by construction")
        concurrency = 1

    # The gate, and whether it has been tuned, is the only difference between the three
    # learning arms. Everything else is held identical because they share one assembly.
    if arm == "tuned" and (not verifier_program
                           or not pathlib.Path(verifier_program).is_file()):
        raise SystemExit(
            f"--arm tuned needs a compiled judge; {verifier_program!r} is not a file.\n"
            f"Produce one from a completed learning arm:\n"
            f"    .venv/bin/python scripts/gepa_runner.py --inspect\n"
            f"    .venv/bin/python scripts/gepa_runner.py --target verifier "
            f"--out {verifier_program or 'compiled/verifier.json'}")
    system = assemble(
        lm=lm, root=RUNS / arm, sub_lm=lm,
        gate="ladder" if arm in ("gated", "tuned") else None,
        program=verifier_program if arm == "tuned" else None,
        explore=explore,
        # The trainset is a by-product of running the loop: every proposal this arm judges,
        # admitted or not, is a labelled-in-hindsight example for gepa_runner.py.
        credit=learning, monitor=learning, record=learning)
    harness, agent, backend = system.harness, system.agent, system.backend
    judge = RubricJudge(lm, parallel=8, run_log=log)

    tasks = task_dirs()[:limit] if limit else task_dirs()
    # The admission bar in force for this arm, stamped on every AuditRecord it writes.
    # Logging it here is what makes two arms' ledgers comparable after the fact.
    log.event("arm_start", arm=arm, tasks=len(tasks), model=model,
              concurrency=concurrency, log=str(LOG),
              system=system.describe())

    sem = asyncio.Semaphore(concurrency)

    async def sweep():
        jobs = [_run_one(d, arm, agent, judge, harness, backend, log, meter, learning, sem)
                for d in tasks]
        if concurrency == 1:
            for job in jobs:
                await job
        else:
            await asyncio.gather(*jobs)

    asyncio.run(sweep())
    log.event("arm_done", arm=arm, usage=meter.summary(),
              by_model=json.dumps(meter.by_model()))
    if meter.unpriced():
        print(f"\n[cost] token counts recorded; no price for {meter.unpriced()}. "
              f"Put a table in lm_prices.json to get dollars:")
        print('       {"%s": {"input": 1.25, "output": 10.0}}  # $ per 1M tokens'
              % meter.unpriced()[0])


def _pairs(baseline: str, arm: str) -> list[tuple[str, float, float]]:
    rows = []
    for d in task_dirs():
        tid = d.name
        b, a = result_path(baseline, tid), result_path(arm, tid)
        if b.exists() and a.exists():
            rows.append((tid, json.loads(b.read_text())["pooled"],
                         json.loads(a.read_text())["pooled"]))
    return rows


def _report_pair(baseline: str, arm: str) -> None:
    rows = _pairs(baseline, arm)
    print(f"\n{'=' * 78}\n{arm}  vs  {baseline}")
    if len(rows) < 2:
        print(f"  only {len(rows)} paired task(s) complete — run both arms first")
        return
    print(f"  {'task':<46} {baseline:>9} {arm:>9} {'delta':>8}")
    for tid, base, treat in rows:
        print(f"  {tid:<46} {base:>8.1%} {treat:>8.1%} {treat - base:>+7.1%}")
    s = paired_summary(rows)
    print(f"\n  mean {baseline:<9}: {sum(r[1] for r in rows) / len(rows):.1%}")
    print(f"  mean {arm:<9}: {sum(r[2] for r in rows) / len(rows):.1%}")
    print(f"  mean delta    : {s['mean_delta']:+.1%}  (sd {s['sd']:.1%}, se {s['se']:.1%})")
    print(f"  win/loss/tie  : {s['wins']}/{s['losses']}/{s['ties']}")
    print(f"  detectable at : +-{s['detectable_at']:.1%} with n={s['n']}")
    print(f"  VERDICT: {'effect exceeds its own error bar' if s['significant'] else 'INSIDE the noise — not a result'}")
    if not s["significant"]:
        print(f"  An effect this size cannot be distinguished from run-to-run variance "
              f"at n={s['n']}. More tasks or more orders, or the effect is not there.")


def report() -> None:
    """Each arm against the arm it adds one mechanism to — never against control.

    A `tuned` vs `control` number would bundle the ledger, the gate and the tuning into one
    figure and attribute all of it to whichever the reader had in mind.
    """
    present = [a for a in ARMS if any(result_path(a, d.name).exists() for d in task_dirs())]
    if not present:
        print("no completed tasks in any arm yet")
        return
    print(f"arms with results: {', '.join(present)}")
    for arm in ARMS[1:]:
        if arm in present and BASELINE[arm] in present:
            _report_pair(BASELINE[arm], arm)
    missing = [a for a in ARMS[1:] if a in present and BASELINE[a] not in present]
    for arm in missing:
        print(f"\n{arm}: cannot be scored — its baseline arm '{BASELINE[arm]}' has no "
              f"results. Run that arm; comparing it to control instead would credit this "
              f"mechanism for every mechanism below it.")


def plan() -> None:
    tasks = [load_task(d) for d in task_dirs()]
    crit = sum(len(t.criteria) for t in tasks)
    agent_s = len(tasks) * 2 * 285
    judge_s = crit * 2 * 2.2
    print(f"tasks              : {len(tasks)}")
    print(f"rubric criteria    : {crit}  (x2 arms = {crit * 2} judge calls)")
    print(f"agent runs         : {len(tasks) * 2}  (~285s each)")
    print(f"sequential estimate: {(agent_s + judge_s) / 3600:.1f}h "
          f"(agent {agent_s / 3600:.1f}h + judge {judge_s / 3600:.1f}h)")
    print(f"with judge x8      : {(agent_s + judge_s / 8) / 3600:.1f}h")
    print(f"  + control x3     : {(agent_s / 2 / 3 + agent_s / 2 + judge_s / 8) / 3600:.1f}h "
          "(learning arm stays sequential by construction)")
    print(f"sensitivity        : detectable effect about "
          f"{2 * 10 / len(tasks) ** 0.5:.1f} points at n={len(tasks)}")


def in_terminal(argv: list[str]) -> None:
    """Re-launch this sweep in a new Terminal window, so it is watchable and detached."""
    repo = pathlib.Path.cwd()
    inner = " ".join(["cd", str(repo), "&&", ".venv/bin/python", "-u", "scripts/paired_ab.py",
                      *[a for a in argv if a != "--terminal"]])
    script = f'tell application "Terminal" to do script "{inner}"'
    subprocess.run(["osascript", "-e", script, "-e",
                    'tell application "Terminal" to activate'], check=True)
    print(f"[launched] sweep running in a new Terminal window\n"
          f"[watch]    .venv/bin/python scripts/watch_run.py\n"
          f"[raw]      tail -f {LOG}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--verifier-program", default="compiled/verifier.json",
                    help="compiled judge for --arm tuned (from scripts/gepa_runner.py)")
    ap.add_argument("--explore", type=float, default=0.0,
                    help="fraction of rejected rounds that also run the rungs behind the "
                         "rejection, for observation only. Costs the expensive rung on "
                         "edits already refused; buys training rows the judge has actually "
                         "seen. Use ~0.2 on a run whose purpose is to produce a trainset.")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks only")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="overlap N tasks (control arm only; memory-hungry)")
    ap.add_argument("--terminal", action="store_true",
                    help="run the sweep in a new Terminal window")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    if args.terminal:
        in_terminal(sys.argv[1:])
    elif args.plan:
        plan()
    elif args.report:
        report()
    elif args.arm:
        run_arm(args.arm, args.limit, args.concurrency, args.verifier_program,
                args.explore)
    else:
        ap.error("pick --plan, --arm, or --report")


if __name__ == "__main__":
    main()
