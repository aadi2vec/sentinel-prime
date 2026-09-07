"""Run bounded execution-policy search. Default: scripted, no network.

Every solver attempt uses sentinelprime.assembly.assemble with its full defaults.
Use --live --model PROVIDER/MODEL only for explicitly budgeted model experiments.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dspy

from sentinelprime.policy_experiment import (
    AssembledSolver, CATALOG, CHECKS, evaluate, make_cases, scripted_lm,
)
from sentinelprime.policy_search import Budget, PolicyArchive, PolicyProposer, PolicyRunner, PolicySearch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="lab_runs/policy-lab")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", help="Explicit DSPy provider/model; API credentials from environment")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cases", type=int, default=3, help="Cases per development/validation/test batch")
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.cases < 1 or args.rounds > 10 or args.cases > 20:
        parser.error("rounds must be 1..10 and cases must be 1..20")
    if args.live and not args.model:
        parser.error("--live requires an explicit --model")
    root = pathlib.Path(args.out)
    # No silent reuse of old results or a final test across independent invocations.
    if root.exists():
        parser.error("output directory already exists; choose a fresh --out")
    root.mkdir(parents=True)
    budget = Budget(max_solves=2, max_checks=4)
    if args.live:
        lm = dspy.LM(args.model, max_tokens=3000, cache=False)
        solver = AssembledSolver(root / "attempts", lm=lm)
        reflection_lm = lm
    else:
        from dspy.utils.dummies import DummyLM
        solver = AssembledSolver(root / "attempts", lm_factory=scripted_lm)
        reflection_lm = DummyLM([
            {"policy_json": json.dumps({"checks": ["arithmetic"], "max_revisions": 1})},
            *[{"policy_json": json.dumps({"checks": list(CATALOG), "max_revisions": 1})}
              for _ in range(args.rounds - 1)],
        ])
    proposer = PolicyProposer(CATALOG, budget)
    proposal_runs = []
    def propose(policy, feedback):
        from dataclasses import asdict
        with dspy.context(lm=reflection_lm), dspy.track_usage() as usage:
            candidate = proposer(policy, feedback)
        proposal_runs.append({"current": asdict(policy), "candidate": asdict(candidate),
                              "usage": usage.get_total_tokens()})
        return candidate
    dev = make_cases("development", 1, args.cases)
    validation = [make_cases(f"validation-{r}", 100 + 30 * r, args.cases)
                  for r in range(args.rounds)]
    test = make_cases("test", 1000, args.cases)
    search = PolicySearch(PolicyRunner(solver, CHECKS, budget), evaluate, propose,
                          PolicyArchive(root / "archive.json"))
    mode = "LIVE, generated fixture tasks" if args.live else "SCRIPTED MECHANISM DEMO (not model-improvement evidence)"
    print(mode, flush=True)
    print("Full assemble() stack; memory learning not invoked on evaluation data.", flush=True)
    print("Budget per task: <=2 RLM attempts, <=4 check calls; each RLM <=6 turns, <=4 subqueries.", flush=True)
    print("These are call ceilings, not a dollar cap. Final test is never sent to the proposer.", flush=True)
    report = search.run(dev, validation, test)
    report.update(mode=mode, caveat="Shared synthetic template; no claim of legal or domain transfer.",
                  solver_runs=solver.runs, proposal_runs=proposal_runs,
                  model=args.model if args.live else "scripted",
                  evaluator="fixed exact values/total v1", check_catalog=CATALOG)
    (root / "report.json").write_text(json.dumps(report, indent=2, default=str))
    for i, row in enumerate(report["rounds"], 1):
        print(f"round {i}: promoted={row['promoted']} gain={row.get('gain')} error={row.get('error')}")
    for name, result in report["test"].items():
        print(f"test {name}: score={result['score']} solves={result['solves']} checks={result['checks']}")
    # champion - initial moves when the champion merely runs the solver more often.
    # matched_compute holds solver attempts fixed and removes the diagnosis, so this is
    # the difference checking actually bought. Print it last: it is the headline.
    adjusted = report["compute_adjusted_gain"]
    print(f"compute-adjusted gain (champion - matched_compute): {adjusted}")
    if adjusted == 0:
        print("  -> the gain is explained by extra solver calls, not by better checking")
    print(f"Artifacts: {root / 'report.json'}")
    return 1 if any(r["score"] is None for r in report["test"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
