"""gepa_runner.py — tune the online machinery offline, from what the online loop wrote.

The README has always claimed that the proposer and the verifier are ordinary
`dspy.Predict`s, so `named_predictors()` exposes them and GEPA can tune the machinery that
does the online learning. That was true as an *affordance* and never exercised, for a
reason worth stating: the audit log records only edits that were admitted, so a trainset
drawn from it has no negatives, and a gate fit to it learns to admit everything.

`ProposalLog` closes that gap, and this is the driver:

    # is a compile even possible? costs nothing, calls no LM
    .venv/bin/python scripts/gepa_runner.py --inspect

    # tune the generative rung of the ladder
    .venv/bin/python scripts/gepa_runner.py --target verifier --out compiled/verifier.json

    # tune the proposer against the cheap proxy metric
    .venv/bin/python scripts/gepa_runner.py --target proposer --out compiled/proposer.json

    # what prompt is inside a compiled artifact?
    .venv/bin/python scripts/gepa_runner.py --fingerprint compiled/verifier.json

Start with `--inspect`. It prints the label counts by stratum and refuses to pretend a
single-class log is trainable — which, on a short run, is the most likely state it is in.

The compiled artifact is state-only JSON (no pickle): small, diffable, and reviewable, so
a changed admission bar shows up in a code review rather than inside a binary.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sentinelprime.audit import machinery_fingerprint
from sentinelprime.assembly import assemble
from sentinelprime.optimize import (LedgerRollout, balance, compile_proposer,
                                    compile_verifier, proposer_metric, proposer_trainset,
                                    rollout_metric, split, trainset_report,
                                    verifier_trainset)
from sentinelprime.proposals import ProposalLog
from sentinelprime.verifier import PredictVerifier

# Any learning arm writes a proposal log; prefer the gated one, since its rejections come
# from the ladder rather than from nothing. Resolved at import so --help shows what is real.
PROPOSAL_CANDIDATES = ("lab_runs/paired/gated/_proposals.jsonl",
                       "lab_runs/paired/tuned/_proposals.jsonl",
                       "lab_runs/paired/learning/_proposals.jsonl")
DEFAULT_PROPOSALS = next((p for p in PROPOSAL_CANDIDATES if pathlib.Path(p).is_file()),
                         PROPOSAL_CANDIDATES[0])
TASKS_DIR = "lab_tasks/corporate-ma"
ROLLOUT_RUNS = "lab_runs/rollout"


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


def inspect(path: str, min_examples: int) -> int:
    log = ProposalLog(path)
    if not log.records():
        print(f"no proposals at {path}\n"
              f"Run a learning arm first — the trainset is a by-product of the online "
              f"loop, so it does not exist until the loop has run:\n"
              f"    .venv/bin/python scripts/paired_ab.py --arm gated\n"
              f"Looked for: {', '.join(PROPOSAL_CANDIDATES)}")
        return 1
    report = trainset_report(log, min_examples=min_examples)
    print(f"source    {path}")
    print(report.summary())
    if not report.viable:
        # Not an error the user can fix by trying again — it needs more tasks.
        return 1
    examples = verifier_trainset(log)
    balanced = balance(examples)
    train, val = split(balanced)
    if len(balanced) < len(examples):
        print(f"balance   {len(examples)} -> {len(balanced)} (majority stratum capped so "
              f"it cannot swamp the compile)")
    print(f"split     {len(train)} train / {len(val)} val (deterministic by content)")
    return 0


def fingerprint(path: str) -> int:
    program = PredictVerifier()
    program.load(path)
    print(f"{machinery_fingerprint(program)}  {path}")
    for name, predictor in program.named_predictors():
        instructions = getattr(predictor.signature, "instructions", "")
        print(f"\n--- {name} ---\n{instructions}")
    return 0


def build_rollout(lm, tasks_dir: str = TASKS_DIR, runs_dir: str = ROLLOUT_RUNS):
    """The honest downstream metric: apply the proposed edits, re-run the task, score it.

    Costs one full agent run *plus* one judge pass per candidate per example, which is why
    it is opt-in. Budget it before starting: `paired_ab.py --plan` prices a single arm, and
    a GEPA compile evaluates far more programs than an arm has tasks.

    The rollout runs against a scratch ledger holding **only** the proposed edits, not the
    arm's accumulated ledger. That measures the edits in isolation, which is the cleaner
    causal question but is not the online setting — online, a lesson is read alongside
    whatever else has accumulated, and interactions between lessons are invisible here.
    Reconstructing the exact contemporaneous ledger would need item-level snapshots that
    `current_ledger` (a rendered string) does not carry.
    """
    from sentinelprime.lab import (RubricJudge, load_task, output_files, prepare_workspace,
                                   read_output, task_prompt, to_feedback)

    scratch = pathlib.Path(runs_dir)
    # A deliberately bare system: the rollout measures the *edits*, so a gate that could
    # refuse them or a credit assigner that could retire them would confound the score.
    system = assemble(lm=lm, root=scratch / "_scratch", sub_lm=lm, gate=None, audit=False,
                      credit=False, monitor=False, record=False)
    harness, agent = system.harness, system.agent
    judge = RubricJudge(lm, parallel=8)
    counter = {"n": 0}

    def run_and_score(task_id: str) -> float:
        task_dir = pathlib.Path(tasks_dir) / task_id
        if not (task_dir / "task.json").is_file():
            # A proposal whose task is no longer on disk cannot be scored. Zero would read
            # as "the guidance was bad"; it means "not measurable", so say so and skip.
            print(f"  [rollout] no task at {task_dir}; scoring 0 and continuing")
            return 0.0
        task = load_task(task_dir)
        counter["n"] += 1
        ws = prepare_workspace(task, scratch / f"ws-{task_id}-{counter['n']}")
        agent.run_task(task_prompt(task, ws), workdir=str(ws), task_id=task_id)
        results = judge.judge(task, read_output(ws), available_files=output_files(ws))
        score = to_feedback(task_id, results).score
        print(f"  [rollout] {task_id} scored {score:.1%}")
        return score

    return LedgerRollout(harness, run_and_score)


def run(target: str, path: str, out: str, auto: str, holdout: float,
        metric_name: str, min_examples: int) -> int:
    import dspy

    _load_dotenv()
    model = os.environ.get("OPENAI_MODEL", "openai/gpt-5.6-luna")
    lm = dspy.LM(model, max_tokens=8000)
    dspy.configure(lm=lm)
    # GEPA reflects with a strong model over a handful of examples; it is called far less
    # often than the task LM, so the same model is a reasonable default here.
    reflection_lm = dspy.LM(os.environ.get("REFLECTION_MODEL", model), max_tokens=8000,
                            temperature=1.0)

    log = ProposalLog(path)
    report = trainset_report(log, min_examples=min_examples)
    print(report.summary())
    if not report.viable:
        print("\nrefusing to compile: see above. --inspect explains what is missing.")
        return 1

    if target == "verifier":
        train, val = split(balance(verifier_trainset(log)), holdout=holdout)
        before = machinery_fingerprint(PredictVerifier())
        print(f"\ncompiling verifier on {len(train)} train / {len(val)} val, auto={auto}")
        compiled = compile_verifier(train, reflection_lm=reflection_lm, valset=val or None,
                                    auto=auto)
    else:
        train, val = split(proposer_trainset(log), holdout=holdout)
        if metric_name == "rollout":
            print("metric    rollout — one agent run + judge pass per candidate per "
                  "example. This is the expensive, honest path.")
            metric = rollout_metric(build_rollout(lm))
        else:
            metric = proposer_metric()
            print("metric    proxy — parses, grounded, declares meta.targets. Cheap, and "
                  "blind to whether the guidance actually helped. Use --metric rollout "
                  "for the downstream measure.")
        from sentinelprime.optimize import _Proposer
        before = machinery_fingerprint(_Proposer())
        print(f"\ncompiling proposer on {len(train)} train / {len(val)} val, auto={auto}")
        compiled = compile_proposer(train, reflection_lm=reflection_lm, metric=metric,
                                    valset=val or None, auto=auto)

    after = machinery_fingerprint(compiled)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    compiled.save(out)
    print(f"\nsaved     {out}")
    print(f"machinery {before[:12]} -> {after[:12]}")
    print("Every AuditRecord written while this program is loaded carries the new "
          "fingerprint, so the admission bar in force is recoverable per edit.")
    if report.caveat:
        print(f"\nREAD THIS BEFORE QUOTING A GAIN: {report.caveat}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proposals", default=DEFAULT_PROPOSALS,
                    help=f"proposal log to build the trainset from "
                         f"(default {DEFAULT_PROPOSALS})")
    ap.add_argument("--inspect", action="store_true",
                    help="report trainset viability and exit; calls no LM")
    ap.add_argument("--fingerprint", metavar="PROGRAM",
                    help="print the machinery fingerprint and prompts of a saved program")
    ap.add_argument("--target", choices=["verifier", "proposer"])
    ap.add_argument("--out", default="compiled/verifier.json")
    ap.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--metric", choices=["proxy", "rollout"], default="proxy",
                    help="proposer only; rollout is the honest downstream measure and "
                         "costs one agent run per candidate")
    ap.add_argument("--min-examples", type=int, default=8)
    args = ap.parse_args()

    if args.fingerprint:
        sys.exit(fingerprint(args.fingerprint))
    if args.inspect:
        sys.exit(inspect(args.proposals, args.min_examples))
    if not args.target:
        ap.error("pick --inspect, --fingerprint, or --target")
    sys.exit(run(args.target, args.proposals, args.out, args.auto, args.holdout,
                 args.metric, args.min_examples))


if __name__ == "__main__":
    main()
