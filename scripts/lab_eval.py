"""lab_eval.py — run PrimeAgent against a real Harvey LAB task and score it.

Distinct from `run_lab.py`, which is a mechanism demo over synthetic fixtures. This one
touches the real benchmark: real documents, real rubric, LAB's own judge prompt.

    .venv/bin/python scripts/lab_eval.py --task lab_tasks/corporate-ma/<task> --dry-run
    .venv/bin/python scripts/lab_eval.py --task lab_tasks/corporate-ma/<task>
    .venv/bin/python scripts/lab_eval.py --task ... --epochs 3          # learning arm
    .venv/bin/python scripts/lab_eval.py --task ... --epochs 3 --frozen # control arm

`--frozen` is the control arm from the eval plan: identical ledger reads, `refine()`
disabled. The result that means anything is the gap between the two, not the shape of
either curve on its own.

Scoring caveat, stated in the output too: this uses LAB's rubric prompt and per-criterion
contract, run through the configured LM. LAB's own harness uses its own judge models and
SDK path. Same prompt, different runner — so these are *indicative*, not official LAB scores.
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
                               prepare_workspace, read_output, to_feedback)
from sentinelprime.memory import JsonMemoryBackend
from sentinelprime.monitor import ProgressMonitor


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


def _pick_model() -> str:
    if os.environ.get("LAB_MODEL"):
        return os.environ["LAB_MODEL"]
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("OPENAI_MODEL", "openai/gpt-5.6-luna")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-5-20250929"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini/gemini-2.5-flash"
    raise SystemExit("No provider API key found in environment (.env).")


def _lm_kwargs() -> dict:
    kwargs = {"max_tokens": 8000}
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        kwargs["api_base"] = base
    return kwargs


def task_context(task, workspace: pathlib.Path) -> dict:
    """Live state reuse gating sees: which matter, and which document revisions."""
    from sentinelprime.audit import digest

    return {
        "matter": task.task_id,
        "documents": len(task.documents),
        "corpus_sha": digest("".join(sorted(p.name for p in task.documents)))[:12],
    }


def describe(task) -> None:
    print(f"[lab] task      : {task.task_id}")
    print(f"[lab] title     : {task.title}")
    print(f"[lab] work_type : {task.work_type}")
    print(f"[lab] documents : {len(task.documents)}")
    print(f"[lab] criteria  : {len(task.criteria)}  (all-pass needs every one)")
    print(f"[lab] deliverab.: {', '.join(task.deliverables)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="path to a LAB task directory")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--frozen", action="store_true",
                    help="control arm: read the ledger but never refine()")
    ap.add_argument("--max-criteria", type=int, default=0,
                    help="judge only the first N criteria (bounds cost while shaking out)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the workspace and stop — no LM calls, no cost")
    ap.add_argument("--out", default="lab_runs", help="where to write run artifacts")
    args = ap.parse_args()

    _load_dotenv()
    task = load_task(args.task)
    describe(task)

    run_root = pathlib.Path(args.out) / task.task_id
    run_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        ws = prepare_workspace(task, run_root / "workspace-dry")
        total = sum(len(p.read_text()) for p in (ws / "documents").iterdir())
        print(f"\n[lab] workspace : {ws}")
        print(f"[lab] extracted : {total:,} chars (~{total // 4:,} tokens) across "
              f"{len(list((ws / 'documents').iterdir()))} files")
        print("[lab] dry run — no LM calls made.")
        return

    import dspy
    from sentinelprime.agent import PrimeAgent

    model = _pick_model()
    lm = dspy.LM(model, **_lm_kwargs())
    dspy.configure(lm=lm)

    audit_log = AuditLog(str(run_root / "audit.json"))
    backend = JsonMemoryBackend(str(run_root / "ledger.json"))
    harness = ContinualHarness(backend, audit_log=audit_log,
                               credit_assigner=CreditAssigner(audit_log))
    agent = PrimeAgent(harness, root_lm=lm, sub_lm=lm, monitor=ProgressMonitor())
    judge = RubricJudge(lm)
    criteria = task.criteria[:args.max_criteria] if args.max_criteria else task.criteria

    arm = "frozen (control)" if args.frozen else "learning"
    print(f"\n[lab] model     : {model}")
    print(f"[lab] arm       : {arm}")
    print(f"[lab] judging   : {len(criteria)}/{len(task.criteria)} criteria "
          f"(LAB's rubric prompt, our runner — indicative, not an official LAB score)\n")

    curve = []
    for epoch in range(1, args.epochs + 1):
        ws = prepare_workspace(task, run_root / f"workspace-e{epoch}")
        started = time.time()
        pred = agent.run_task(
            f"{task.instructions}\n\n"
            f"The source documents are the .txt files in {ws / 'documents'} "
            f"(also reachable as ./documents). You MUST write your deliverable(s) as "
            f"files into {ws / 'output'} — named {', '.join(task.deliverables)} "
            f"(a .md or .txt body under that name is fine). "
            f"Returning text without writing the file(s) scores zero.",
            workdir=str(ws),
            context=task_context(task, ws),
            task_id=task.task_id,
        )
        elapsed = time.time() - started

        # Grade only what was actually written. No falling back to the submitted string:
        # the deliverable contract is a file, and a judge asked to grade a report that does
        # not exist will confabulate one (26/55 invented passes on the first warmup).
        output = read_output(ws)
        produced = output_files(ws)
        results = judge.judge(task, output, criteria=criteria, available_files=produced)
        fb = to_feedback(task.task_id, results)
        curve.append(fb.score)

        print(f"  epoch {epoch}: pooled criterion pass rate = {fb.score:.1%} "
              f"({sum(1 for r in results if r['verdict'] == 'pass')}/{len(results)}) | "
              f"all-pass = {all_pass(fb)} | {elapsed:.0f}s")
        print(f"    wrote {len(produced)} file(s): {produced or 'NOTHING'} "
              f"({len(output):,} chars graded)")
        if not produced:
            print("    submitted-but-unwritten text: "
                  f"{str(getattr(pred, 'deliverable', ''))[:120]!r}")
        if agent.last_monitor_decision and agent.last_monitor_decision.replan:
            print(f"    replan: {'; '.join(agent.last_monitor_decision.reasons)}")

        (run_root / f"scores-e{epoch}.json").write_text(json.dumps({
            "task_id": task.task_id, "epoch": epoch, "arm": arm,
            "pooled_criterion_pass_rate": fb.score, "all_pass": all_pass(fb),
            "elapsed_s": elapsed, "output_files": produced, "criteria": results,
        }, indent=2))

        if not args.frozen:
            refine = agent.learn(agent.last_trajectory, fb)
            print(f"    ledger: created={refine.created} rejected={refine.rejected} "
                  f"retired={refine.retired} (size={len(backend.read())})")

    print(f"\n[lab] curve     : {' -> '.join(f'{c:.1%}' for c in curve)}")
    print(f"[lab] artifacts : {run_root}")
    if args.epochs > 1 and not args.frozen:
        print("[lab] NOTE: a single arm proves nothing. Re-run with --frozen and compare.")


if __name__ == "__main__":
    main()
