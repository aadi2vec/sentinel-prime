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
    AssembledSolver, CATALOG as HAND_WRITTEN, build_catalogs, evaluate, make_cases,
    scripted_lm,
)
from sentinelprime.policy_search import Budget, PolicyArchive, PolicyProposer, PolicyRunner, PolicySearch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="lab_runs/policy-lab")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", help="Explicit DSPy provider/model; API credentials from environment")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cases", type=int, default=3, help="Cases per development/validation/test batch")
    parser.add_argument("--checks", help="Synthesized check catalog from scripts/synthesize_checks.py; "
                                        "merged into the hand-written catalog the proposer selects from")
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
    # The candidate's whole vocabulary, built once. Without --checks this is the two
    # hand-written checks §3 names as the honest limit — which is the *baseline*, so the
    # run log has to say which of the two it was rather than leaving it to be inferred.
    synthesized = None
    if args.checks:
        from sentinelprime.synthesis import load_specs
        synthesized = load_specs(args.checks)
    CATALOG, CHECKS = build_catalogs(synthesized)
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
            # The stipulated candidate names the two hand-written checks explicitly, not
            # `list(CATALOG)`: a scripted fixture is supposed to be a *fixed* candidate, and
            # one defined as "everything in the catalog" silently becomes a different
            # candidate — and at >2 checks a budget-infeasible one — as soon as synthesis
            # adds vocabulary.
            *[{"policy_json": json.dumps({"checks": list(HAND_WRITTEN),
                                          "max_revisions": 1})}
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
    print(f"Check catalog: {len(CATALOG)} checks "
          f"({len(synthesized or {})} synthesized, {len(CATALOG) - len(synthesized or {})} "
          f"hand-written): {sorted(CATALOG)}", flush=True)
    print("Full assemble() stack; memory learning not invoked on evaluation data.", flush=True)
    print("Budget per task: <=2 RLM attempts, <=4 check calls; each RLM <=6 turns, <=4 subqueries.", flush=True)
    print("These are call ceilings, not a dollar cap. Final test is never sent to the proposer.", flush=True)
    report = search.run(dev, validation, test)
    report.update(mode=mode, caveat="Shared synthetic template; no claim of legal or domain transfer.",
                  solver_runs=solver.runs, proposal_runs=proposal_runs,
                  model=args.model if args.live else "scripted",
                  evaluator="fixed exact values/total v1", check_catalog=CATALOG,
                  synthesized_checks=sorted(synthesized or {}))
    (root / "report.json").write_text(json.dumps(report, indent=2, default=str))
    for i, row in enumerate(report["rounds"], 1):
        # `rule` is printed because the two promotion rules are different claims: one says
        # the champion got better, the other says it got cheaper while holding. A run log
        # reading `promoted=True` alone does not say which was demonstrated.
        saved = (row.get("cost") or {}).get("saved")
        print(f"round {i}: promoted={row['promoted']} rule={row.get('rule')} "
              f"gain={row.get('gain')} "
              f"cost_saved={f'{saved:+.0%}' if saved is not None else None} "
              f"error={row.get('error')}")
    for name, result in report["test"].items():
        print(f"test {name}: score={result['score']} solves={result['solves']} checks={result['checks']}")
    print(f"Artifacts: {root / 'report.json'}")
    return 1 if any(r["score"] is None for r in report["test"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
