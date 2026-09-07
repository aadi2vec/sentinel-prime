"""Assembled RLM adapter and a deliberately small verification-policy laboratory.

The fixtures exercise arithmetic and revised source values, not legal expertise. In
scripted mode both the solver's mistakes and proposer choices are stipulated. The real
RLM/interpreter, assembler, policy runner, and selection protocol still execute.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Callable

import dspy

from sentinelprime.assembly import assemble
from sentinelprime.policy_search import Case, CheckResult


class AssembledSolver:
    """Every attempt uses the shared assembly point and a fresh workspace/ledger.

    Defaults remain fully wired: memory admission, audit, credit, monitoring, and
    proposal recording. We do not call learn() on evaluation outcomes: otherwise two
    learning mechanisms and holdout leakage would confound this experiment. Children
    and semantic caching retain assemble()'s explicit opt-in defaults.

    lm_factory is a model seam for the network-free demo, not a second agent builder.
    RLM limits are identical across policies; outer policy budgets count whole solves.
    This adapter retains the current trusted-code interpreter, NOT an OS sandbox.
    """
    def __init__(self, root: str | Path, *, lm=None, lm_factory: Callable | None = None,
                 max_iters: int = 6, max_llm_calls: int = 4):
        if (lm is None) == (lm_factory is None):
            raise ValueError("provide exactly one of lm or lm_factory")
        if any(type(n) is not int or n < 1 for n in (max_iters, max_llm_calls)):
            raise ValueError("RLM limits must be positive integers")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lm, self.lm_factory = lm, lm_factory
        self.max_iters, self.max_llm_calls = max_iters, max_llm_calls
        self.runs: list[dict] = []

    def __call__(self, task: str, feedback: str) -> str:
        workdir = Path(tempfile.mkdtemp(prefix="attempt-", dir=self.root)).resolve()
        documents = workdir / "documents"
        documents.mkdir()
        (documents / "task.txt").write_text(task)
        lm = self.lm_factory(task, feedback) if self.lm_factory else self.lm
        # Never reconstruct PrimeAgent/ContinualHarness here or silently thin the stack.
        system = assemble(lm=lm, sub_lm=lm, root=workdir / "state")
        system.agent.rlm.max_iters = self.max_iters
        system.agent.rlm.max_llm_calls = self.max_llm_calls
        prompt = (
            "Solve the task in documents/task.txt. Read it through the interpreter. "
            "Return the requested answer directly in SUBMIT(deliverable=...), not a file path. "
            "Do not alter input documents.\n"
        )
        if feedback:
            prompt += "A previous attempt failed these checks. Recompute and correct it:\n" + feedback
        record = {"workdir": str(workdir), "system": system.describe(),
                  "max_iters": self.max_iters, "max_llm_calls": self.max_llm_calls,
                  "revision": bool(feedback), "model": str(getattr(lm, "model", "scripted")),
                  "memory_learning": "not invoked during evaluation"}
        try:
            with dspy.track_usage() as usage:
                pred = system.agent.run_task(prompt, workdir=str(workdir),
                                             task_id=workdir.name, context={"trial": workdir.name})
            record["usage"] = usage.get_total_tokens()
            record["trajectory"] = system.agent.last_trajectory
            record["cache"] = system.agent.last_cache_stats
            return str(pred.deliverable)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.runs.append(record)
            (workdir / "execution.json").write_text(json.dumps(record, indent=2, default=str))


CATALOG = {
    "arithmetic": "Check that total equals the sum of the answer's line values.",
    "source_revision": "Check that line values include every source item with its latest amended value.",
}


def _answer(raw: str) -> dict | None:
    try:
        answer = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if (not isinstance(answer, dict) or set(answer) != {"values", "total"}
            or not isinstance(answer["values"], dict)
            or any(not isinstance(k, str) or type(v) is not int for k, v in answer["values"].items())
            or type(answer["total"]) is not int):
        return None
    return answer


def arithmetic(task: str, raw: str) -> CheckResult:
    answer = _answer(raw)
    ok = answer is not None and answer["total"] == sum(answer["values"].values())
    return CheckResult(ok, "Return valid JSON and recompute total as the sum of values.")


def source_revision(task: str, raw: str) -> CheckResult:
    source = json.loads(task)
    values = dict(source["base"])
    for amendment in source["amendments"]:
        values.update(amendment)
    answer = _answer(raw)
    return CheckResult(answer is not None and answer["values"] == values,
                       "Use every base item and apply amendments in order; last value wins.")


CHECKS = {"arithmetic": arithmetic, "source_revision": source_revision}


def make_cases(prefix: str, start: int, count: int = 3) -> list[Case]:
    """Different generated instances, not different domains or unseen task templates.

    Family IDs denote independent source bundles. The same task template is shared
    deliberately; success here cannot establish cross-template or cross-domain transfer.
    Gold is constructed separately and is never included in the public task string.
    """
    cases = []
    for n in range(start, start + count):
        base = {"alpha": n + 2, "beta": 2 * n + 3}
        amendments = [{"alpha": n + 9}]
        public = {"instruction": "Return JSON with values (final item amounts) and total. "
                  "Apply amendments in order to the base values; latest amendment wins.",
                  "base": base, "amendments": amendments}
        gold = {"values": {"alpha": n + 9, "beta": 2 * n + 3}, "total": 3 * n + 12}
        cases.append(Case(f"{prefix}-{n}", f"source-bundle-{n}", json.dumps(public), gold))
    return cases


def evaluate(case: Case, raw: str) -> float:
    # Fixed exact scoring, outside all candidate-controlled checks and revision logic.
    answer = _answer(raw)
    if answer is None:
        return 0.0
    return (float(answer["values"] == case.gold["values"])
            + float(answer["total"] == case.gold["total"])) / 2


def scripted_lm(task: str, feedback: str):
    """Stipulated mistakes and repairs through the REAL assembled DSPy RLM loop."""
    from dspy.utils.dummies import DummyLM

    # A revision repairs only the checks actually requested by the execution policy.
    failures = json.loads(feedback)["failures"] if feedback else []
    fix_source = any(f.startswith("source_revision:") for f in failures)
    fix_arithmetic = any(f.startswith("arithmetic:") for f in failures)
    code = (
        "import json\n"
        "source = json.load(open('documents/task.txt'))\n"
        "values = dict(source['base'])\n"
    )
    if fix_source:
        code += "for amendment in source['amendments']: values.update(amendment)\n"
    code += f"total = sum(values.values()) + {0 if fix_arithmetic else 1}\n"
    code += "SUBMIT(deliverable=json.dumps({'values': values, 'total': total}))"
    return DummyLM([{"reasoning": "scripted mechanism fixture; not learned reasoning",
                     "code": code}])
