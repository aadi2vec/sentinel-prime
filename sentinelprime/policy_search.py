"""Bounded search over *execution policies*, separate from supplemental-memory edits.

A candidate selects registered checks and a revision count. It cannot supply Python,
change the evaluator, or raise its execution ceiling. The runner executes the policy;
the proposer does not merely append advice to the solver prompt.

Fresh validation batches drive promotion; a final test is scored only after selection.
This is a small experimental selection rule, not statistical evidence of generality.
Callables are trusted application code: separating their inputs is not OS isolation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import dspy


@dataclass(frozen=True)
class Policy:
    checks: tuple[str, ...] = ()
    max_revisions: int = 0

    def __post_init__(self):
        if (not isinstance(self.checks, tuple)
                or any(not isinstance(c, str) or not c for c in self.checks)
                or len(set(self.checks)) != len(self.checks)
                or type(self.max_revisions) is not int or self.max_revisions < 0):
            raise ValueError("policy needs unique check names and a nonnegative integer revision count")

    @classmethod
    def parse(cls, raw: str) -> "Policy":
        data = json.loads(raw)
        if (not isinstance(data, dict) or set(data) != {"checks", "max_revisions"}
                or not isinstance(data["checks"], list)):
            raise ValueError("policy fields must be exactly checks (list), max_revisions (int)")
        return cls(tuple(data["checks"]), data["max_revisions"])

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Budget:
    """Hard outer-call ceilings, not a token or monetary budget."""
    max_solves: int = 2
    max_checks: int = 4

    def __post_init__(self):
        if (type(self.max_solves) is not int or self.max_solves < 1
                or type(self.max_checks) is not int or self.max_checks < 0):
            raise ValueError("invalid execution budget")


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    feedback: str


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    task: str
    gold: object  # Only the evaluator receives this object, never solve/check/propose.


@dataclass
class RunResult:
    answer: str = ""
    solves: int = 0
    checks: int = 0
    error: str | None = None
    trace: list[dict] = field(default_factory=list)


class PolicyRunner:
    def __init__(self, solve: Callable, checks: dict[str, Callable], budget: Budget):
        self.solve, self.checks, self.budget = solve, dict(checks), budget

    def validate(self, policy: Policy) -> None:
        if not isinstance(policy, Policy):
            raise ValueError("proposer must return a Policy")
        unknown = set(policy.checks) - self.checks.keys()
        if unknown:
            raise ValueError(f"unknown checks: {sorted(unknown)}")
        attempts = 1 + policy.max_revisions
        if attempts > self.budget.max_solves or attempts * len(policy.checks) > self.budget.max_checks:
            raise ValueError("policy exceeds fixed execution budget")

    def run(self, policy: Policy, task: str) -> RunResult:
        self.validate(policy)  # Refuse before executing anything.
        result, feedback = RunResult(), ""
        try:
            for attempt in range(policy.max_revisions + 1):
                result.solves += 1
                result.answer = self.solve(task, feedback)
                if not isinstance(result.answer, str):
                    raise ValueError("solver must return a string")
                result.trace.append({"kind": "solve", "attempt": attempt,
                                     "answer": result.answer})
                failures = []
                for name in policy.checks:
                    result.checks += 1
                    verdict = self.checks[name](task, result.answer)
                    if (not isinstance(verdict, CheckResult) or type(verdict.passed) is not bool
                            or not isinstance(verdict.feedback, str)):
                        raise ValueError(f"invalid check result from {name}")
                    result.trace.append({"kind": "check", "name": name, **asdict(verdict)})
                    if not verdict.passed:
                        failures.append(f"{name}: {verdict.feedback}")
                if not failures:
                    break
                # Revisions get only actual check observations and the preceding answer.
                feedback = json.dumps({"previous_answer": result.answer, "failures": failures})
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        return result


class PolicyArchive:
    """Single-writer atomic snapshots with retained promotion/rollback history.

    This is not a distributed consensus protocol or tamper-proof audit store. The
    archive's reserved families prevent accidental test reuse after restarting a run.
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"current": 0, "policies": [asdict(Policy())],
                          "events": [], "used_families": []}
        self.current  # Validate the active serialized policy immediately.

    @property
    def current(self) -> Policy:
        return Policy.parse(json.dumps(self.state["policies"][self.state["current"]]))

    @property
    def events(self) -> list[dict]:
        return list(self.state["events"])

    def _save(self, state: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(state, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self.state = state

    def reserve(self, families: set[str]) -> None:
        if families & set(self.state["used_families"]):
            raise ValueError("family already used in this archive; supply fresh evaluation data")
        state = json.loads(json.dumps(self.state))
        state["used_families"] += sorted(families)
        self._save(state)  # Reserve before trials, including when a later process crashes.

    def record(self, policy: Policy, promoted: bool, evidence: dict) -> None:
        state = json.loads(json.dumps(self.state))
        previous = state["current"]
        if promoted:
            state["policies"].append(asdict(policy))
            state["current"] = len(state["policies"]) - 1
        state["events"].append({"kind": "trial", "from": previous,
                                "to": state["current"], "candidate_policy": asdict(policy),
                                "fingerprint": policy.fingerprint, **evidence})
        self._save(state)

    def rollback(self, version: int) -> None:
        if type(version) is not int or not 0 <= version < len(self.state["policies"]):
            raise ValueError("unknown policy version")
        state = json.loads(json.dumps(self.state))
        state["events"].append({"kind": "rollback", "from": state["current"], "to": version})
        state["current"] = version
        self._save(state)


class ProposeExecutionPolicy(dspy.Signature):
    """Propose one bounded execution policy from development failures.

    Select registered checks and a revision count. Return ONLY a JSON object with
    checks (unique ordered names) and max_revisions (nonnegative integer). Respect
    the fixed worst-case solve/check ceilings. You cannot edit the evaluator.
    """
    current_policy: str = dspy.InputField()
    development_feedback: str = dspy.InputField()
    check_catalog: str = dspy.InputField()
    budget: str = dspy.InputField()
    policy_json: str = dspy.OutputField()


class PolicyProposer(dspy.Module):
    def __init__(self, catalog: dict[str, str], budget: Budget):
        super().__init__()
        self.catalog, self.budget = dict(catalog), budget
        self.propose = dspy.Predict(ProposeExecutionPolicy)

    def forward(self, policy: Policy, feedback: list[dict]) -> Policy:
        pred = self.propose(current_policy=json.dumps(asdict(policy)),
                            development_feedback=json.dumps(feedback),
                            check_catalog=json.dumps(self.catalog),
                            budget=json.dumps(asdict(self.budget)))
        return Policy.parse(pred.policy_json)


class PolicySearch:
    def __init__(self, runner: PolicyRunner, evaluate: Callable, propose: Callable,
                 archive: PolicyArchive, min_gain: float = 0.01):
        if not math.isfinite(min_gain) or min_gain <= 0:
            raise ValueError("min_gain must be finite and positive")
        self.runner, self.evaluate, self.propose = runner, evaluate, propose
        self.archive, self.min_gain = archive, min_gain

    def _score(self, policy: Policy, cases: list[Case]) -> dict:
        rows = []
        for case in cases:
            result = self.runner.run(policy, case.task)
            score, error = None, result.error
            if error is None:
                try:
                    score = float(self.evaluate(case, result.answer))
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("evaluator score must be finite and within [0,1]")
                except Exception as exc:
                    score, error = None, f"evaluation_error: {type(exc).__name__}: {exc}"
            rows.append({"id": case.id, "score": score, "error": error,
                         "solves": result.solves, "checks": result.checks,
                         "answer": result.answer, "trace": result.trace})
        complete = all(r["error"] is None for r in rows)
        return {"score": sum(r["score"] for r in rows) / len(rows) if complete else None,
                "solves": sum(r["solves"] for r in rows),
                "checks": sum(r["checks"] for r in rows), "rows": rows}

    def run(self, development: list[Case], validation: list[list[Case]],
            test: list[Case]) -> dict:
        groups = [development, *validation, test]
        if not validation or any(not g for g in groups):
            raise ValueError("nonempty development, validation batches, and test required")
        families, ids = set(), set()
        for group in groups:
            group_families = {c.family for c in group}
            if any(not c.id or not c.family for c in group):
                raise ValueError("case id and family are required")
            if families & group_families:
                raise ValueError("family overlaps development, validation batches, or test")
            families |= group_families
            for case in group:
                if case.id in ids:
                    raise ValueError("duplicate case id")
                ids.add(case.id)
        self.archive.reserve(families)
        initial, rounds = self.archive.current, []
        self.runner.validate(initial)
        for batch in validation:
            champion = self.archive.current
            dev = self._score(champion, development)
            # No gold, evaluator exceptions (which may contain gold), or holdout rows.
            feedback = [{"task": c.task, "answer": r["answer"], "score": r["score"],
                         "execution_trace": r["trace"], "incomplete": r["error"] is not None}
                        for c, r in zip(development, dev["rows"])]
            try:
                candidate = self.propose(champion, feedback)
                self.runner.validate(candidate)
            except Exception as exc:
                evidence = {"promoted": False, "error": f"{type(exc).__name__}: {exc}",
                            "development": dev}
                self.archive.record(champion, False, evidence)
                rounds.append(evidence)
                continue
            # Each candidate consumes a new batch; selection never sees the final test.
            baseline = self._score(champion, batch)
            trial = self._score(candidate, batch)
            complete = baseline["score"] is not None and trial["score"] is not None
            gain = trial["score"] - baseline["score"] if complete else None
            # Conservative small-sample rule: mean improvement AND no observed case regression.
            promoted = bool(complete and gain >= self.min_gain and all(
                b["score"] <= t["score"] for b, t in zip(baseline["rows"], trial["rows"])))
            evidence = {"promoted": promoted, "gain": gain, "development": dev,
                        "baseline": baseline, "candidate": trial}
            self.archive.record(candidate, promoted, evidence)
            rounds.append(evidence)
        return {"budget": asdict(self.runner.budget), "rounds": rounds,
                "initial_policy": asdict(initial), "champion_policy": asdict(self.archive.current),
                "test": {"initial": self._score(initial, test),
                         "champion": self._score(self.archive.current, test)}}
