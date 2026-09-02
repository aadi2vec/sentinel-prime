"""PrimeAgent: dspy.RLM per-turn reasoning wired to the ContinualHarness ledger.

The outer graph is runtime-decided (the RLM emits an action per turn and may
spawn children), but the inner nodes — generate_action, extract, and the
harness's propose — are ordinary DSPy predictors, so the whole agent stays
GEPA-optimizable. The ledger enters only as a frozen `guidance` input read once
at task start; the base task instructions are never mutated.
"""
from __future__ import annotations

import dspy

from sentinelprime.harness import ContinualHarness
from sentinelprime.interpreter import InterpreterFactory


class PrimeTask(dspy.Signature):
    """Complete the legal task using the documents in the working directory.
    Produce the requested deliverable."""

    task: str = dspy.InputField()
    guidance: str = dspy.InputField(
        desc="Supplemental learned guidance — apply it, but do not override the task."
    )
    deliverable: str = dspy.OutputField()


class PrimeAgent(dspy.Module):
    def __init__(self, harness: ContinualHarness, root_lm, sub_lm=None,
                 spawn_manager=None, rlm=None) -> None:
        super().__init__()
        self.harness = harness
        self.root_lm = root_lm
        self.sub_lm = sub_lm
        self.spawn_manager = spawn_manager
        self._current_workdir = "."
        if rlm is None:
            tools = [spawn_manager.spawn_child] if spawn_manager is not None else []
            rlm = dspy.RLM(
                PrimeTask,
                tools=tools,
                sub_lm=sub_lm,
                interpreter_factory=InterpreterFactory(lambda: self._current_workdir),
            )
        self.rlm = rlm

    def run_task(self, task: str, workdir: str) -> dspy.Prediction:
        # Reproducibility invariant: freeze the ledger snapshot at task start.
        self._current_workdir = workdir
        guidance = self.harness.read() or "(no learned guidance yet)"
        with dspy.context(lm=self.root_lm):
            return self.rlm(task=task, guidance=guidance)

    def learn(self, trajectory: list[dict], feedback):
        # Applied only BETWEEN tasks — never mid-task.
        return self.harness.refine(trajectory, feedback)
