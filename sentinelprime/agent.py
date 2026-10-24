"""PrimeAgent: dspy.RLM per-turn reasoning wired to the ContinualHarness ledger.

The outer graph is runtime-decided (the RLM emits an action per turn and may
spawn children), but the inner nodes — generate_action, extract, and the
harness's propose — are ordinary DSPy predictors, so the whole agent stays
GEPA-optimizable. The ledger enters only as a frozen `guidance` input read once
at task start; the base task instructions are never mutated.

`run_task` is where the online loop's optional machinery attaches, so the live
path exercises the same seams the scripted harness does:

  - `context` is forwarded to `harness.read(context=…)`, so ReuseController's
    scope/currency gates see real task state and a lesson whose source has moved
    is withheld from the prompt.
  - `monitor` receives the real RLM trajectory (`pred.trajectory`) and the run's
    SubQueryCache stats, so live thrash emits a `replan` audit event.
  - `last_trajectory` exposes that trajectory so the caller can hand it to
    `learn()` rather than refining on an empty list.
  - `last_coverage` reports which workspace documents the run never named, an
    LM-free read on the failure mode that has no reasoning in it at all.

All four are opt-in: with no context and no monitor the behavior is unchanged,
and a workspace with no documents/ directory reports no coverage.
"""
from __future__ import annotations

import os
from typing import Any

import dspy
import pydantic
from dspy.primitives.repl_types import REPLHistory

from sentinelprime.children import ChildSessionManager
from sentinelprime.context import EditableContext, context_tools
from sentinelprime.contextfns import context_fn_tools
from sentinelprime.credit import grounding_from_trajectory
from sentinelprime.grounding import document_coverage, document_names
from sentinelprime.errorbudget import EDIT_ADMISSION_RATE, TURN_LIMIT_RATE
from sentinelprime.harness import ContinualHarness, RefineResult
from sentinelprime.interpreter import InterpreterFactory
from sentinelprime.subcache import SubQueryCache


class CachingRLM(dspy.RLM):
    """dspy.RLM whose sub-LLM tools are wrapped with a per-run SubQueryCache.

    A fresh cache is created for each forward pass (intra-run dedup only —
    cross-task reuse is the gated ReuseController's job), and its hit/miss
    stats are exposed on ``last_cache_stats`` after the run for measurement.
    Sequential use only, matching the rest of PrimeAgent.
    """

    def __init__(self, *args, subquery_embedder=None,
                 subquery_similarity_threshold: float = 0.9, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_cache_stats: dict | None = None
        self._active_cache: SubQueryCache | None = None
        # Optional intra-run semantic dedup: when an embedder is supplied, near-duplicate
        # sub-queries reuse a prior answer. None -> exact content-hash dedup only.
        self._subquery_embedder = subquery_embedder
        self._subquery_similarity_threshold = subquery_similarity_threshold

    def _prepare_execution_tools(self) -> dict:
        cache = SubQueryCache(
            embedder=self._subquery_embedder,
            similarity_threshold=self._subquery_similarity_threshold,
        )
        self._active_cache = cache
        return cache.wrap(super()._prepare_execution_tools())

    def forward(self, interpreter=None, /, **input_args):
        result = super().forward(interpreter, **input_args)
        if self._active_cache is not None:
            self.last_cache_stats = self._active_cache.stats()
        return result

    async def aforward(self, interpreter=None, /, **input_args):
        # dspy's aforward calls _prepare_execution_tools too, so the sub-query cache is
        # already wrapped; this only mirrors forward's stats capture.
        result = await super().aforward(interpreter, **input_args)
        if self._active_cache is not None:
            self.last_cache_stats = self._active_cache.stats()
        return result


class _ContextHistory(REPLHistory):
    """A `REPLHistory` that renders from an `EditableContext` instead of its own entries.

    The entries list is kept intact and correct — `dspy.RLM` builds `pred.trajectory` from
    it, and nothing may remove from that — while `format()`, which is what the adapter
    actually serializes into the prompt, comes from the context. That split *is* the two
    views: the prompt shows the edited context, the trajectory keeps every turn.
    """

    _ctx: Any = pydantic.PrivateAttr(default=None)

    def format(self) -> str:
        return self._ctx.render() if self._ctx is not None else super().format()


class ContextEditingRLM(CachingRLM):
    """`CachingRLM` whose REPL history is editable in place by the model.

    `dspy.RLM.forward` threads an immutable `REPLHistory` through `_execute_iteration` and
    assigns the return value back, so one override is the whole seam: let the base class run
    the turn, record what executed into the context, and hand back a history that renders
    from the context rather than from its own entries.

    Ordering is load-bearing and happens to fall out correctly. The model's edit tools run
    *inside* `_execute_code`, so by the time the base class has built the next history the
    edits have already been applied — to turns that existed *before* this one. The turn that
    did the editing is appended after. A model therefore cannot evict the turn it is
    currently in, which is not a rule anybody enforces, just the only order the loop admits.
    """

    def __init__(self, *args, context=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context = context

    def _as_view(self, history: REPLHistory) -> REPLHistory:
        if self.context is None:
            return history
        view = _ContextHistory(entries=list(history.entries),
                               max_output_chars=history.max_output_chars)
        # Rewrapped every turn because REPLHistory.append() is frozen and returns the base
        # class, so the view cannot survive a turn on its own.
        view._ctx = self.context
        return view

    def _execute_iteration(self, repl, variables, history, iteration, input_args,
                           output_field_names):
        result = super()._execute_iteration(repl, variables, self._as_view(history),
                                            iteration, input_args, output_field_names)
        if self.context is None:
            return result
        if isinstance(result, dspy.Prediction):
            # The submitting turn is appended to the *Prediction's* trajectory and never
            # returned as a history, so this branch is not a shortcut — skipping it drops
            # the one turn that produced the deliverable from the grounding view, and a
            # deliverable whose own turn is invisible reads as ungrounded by construction.
            self._record(getattr(result, "trajectory", None) or [])
            return result
        self._record(entry.model_dump() for entry in result.entries)
        return self._as_view(result)

    def _record(self, rows) -> None:
        """Append whatever executed since the last turn. Indexed by count, not identity.

        `_execute_iteration` is handed a history it did not necessarily produce — `forward`
        reassigns it each turn — so "which turns are new" is answered by how many the
        context already holds, not by diffing against the history that came in.
        """
        rows = list(rows)
        for row in rows[len(self.context.executed):]:
            self.context.append(row["reasoning"], row["code"], row["output"])


class PrimeTask(dspy.Signature):
    """Complete the legal task using the documents in the working directory.
    Produce the requested deliverable."""

    # FIELD ORDER IS LOAD-BEARING. Adapters render input fields in declaration order, and
    # dspy.RLM appends the growing repl_history last, so this order puts the prompt's
    # stable content first: [instructions][ledger guidance][per-task input][repl history].
    # That is what makes the ledger block part of a provider-cacheable prefix shared across
    # tasks; with `task` first, the prompt diverges before the ledger is ever reached.
    guidance: str = dspy.InputField(
        desc="Supplemental learned guidance — apply it, but do not override the task."
    )
    task: str = dspy.InputField()
    deliverable: str = dspy.OutputField()


def cacheable_prefix(guidance: str, tasks: list[str],
                     signature: type[dspy.Signature] = PrimeTask) -> str:
    """The byte-identical prompt prefix shared by `tasks` under one ledger state.

    Provider prompt caching bills the shared leading segment of a request, so the only
    thing that matters is where the prompt first *diverges* between two tasks. Rendering
    the real adapter messages and taking their longest common prefix measures that
    boundary directly — no provider, no tokens, no billing data needed. `read()`'s
    byte-stable item ordering is what keeps this prefix from moving on its own.

    Returns "" for fewer than two tasks (nothing to share).
    """
    adapter = dspy.settings.adapter or dspy.ChatAdapter()
    rendered = [
        "\n".join(m["content"] for m in
                  adapter.format(signature, [], {"guidance": guidance, "task": t}))
        for t in tasks
    ]
    if len(rendered) < 2:
        return ""
    prefix = rendered[0]
    for other in rendered[1:]:
        limit = min(len(prefix), len(other))
        i = 0
        while i < limit and prefix[i] == other[i]:
            i += 1
        prefix = prefix[:i]
    return prefix


class PrimeAgent(dspy.Module):
    def __init__(self, harness: ContinualHarness, root_lm, sub_lm=None,
                 spawn_manager=None, rlm=None, subquery_embedder=None,
                 monitor=None, error_budget=None, editable_context: bool = True,
                 context_gate=None, enable_children: bool = False, child_runner=None,
                 max_child_depth: int = 1) -> None:
        super().__init__()
        self.harness = harness
        self.root_lm = root_lm
        self.sub_lm = sub_lm
        self.spawn_manager = spawn_manager
        # Children are opt-in: they cost a thread pool and a sub-agent's tokens. When on,
        # the manager is built lazily on first spawn, rooted at the current task's workdir,
        # so an agent that never delegates never allocates one.
        self.enable_children = enable_children or spawn_manager is not None
        self._child_runner = child_runner
        self.max_child_depth = max_child_depth
        # Optional thrash detector. When set, every run_task feeds the real RLM
        # trajectory + sub-query cache stats to it, so a live loop (not just the
        # scripted run_lab harness) emits `replan` audit events. None -> no monitoring.
        self.monitor = monitor
        # Optional error budget. `monitor` detects and annotates; this is the first
        # collaborator permitted to *stop* the loop, and `learn()` is where it is consulted.
        # None -> every round refines, exactly as before it existed (invariant 7).
        self.error_budget = error_budget
        # Last run's RLM trajectory and monitor verdict, so the caller can hand the
        # *real* trajectory to learn() instead of an empty list.
        self.last_trajectory: list[dict] = []
        self.last_monitor_decision = None
        # The guidance ids actually surfaced for the last task — the exposure set credit
        # assignment scores. Not the whole ledger: a lesson gating withheld cannot be
        # blamed for that task's outcome.
        self.last_exposed_ids: list[str] = []
        # Which of the workspace's documents the last run ever named. In due diligence the
        # dominant failure is not faulty reasoning, it is never opening a document — and
        # that costs no LM call to detect. None when the workdir has no documents/ dir.
        self.last_coverage = None
        self.last_child_errors: list = []
        # What the prefix scheduler did with the last task's fan-out. None when no child was
        # ever spawned, so "no children" reads differently from "children, none warm" — the
        # same off/empty distinction `last_context_stats` makes below.
        self.last_child_scheduling: dict | None = None
        # What the last run's context editing cost and removed. None when the flag is off,
        # so a caller can tell "no edits" from "no write path" — the same distinction the
        # budget's off/on/frozen split makes one level up.
        self.last_context_stats: dict | None = None
        self._current_workdir = "."
        self._subquery_embedder = subquery_embedder
        # The write path. On by default, like every other mechanism here — the cost is that
        # it breaks invariant 5 by construction, so `context.reprefill_chars` prices the
        # break on every run and `describe()` records `ctx=on`. A run with it on is not
        # comparable to one with it off; that is a reading, not a defect.
        # One instance for the agent's lifetime, reset per task: the edit tools are bound
        # to it at RLM construction, and rebinding per task is how a tool ends up editing
        # the previous task's context.
        context = EditableContext(gate=context_gate) if editable_context else None
        if rlm is None:
            # Bind both halves. Spawning without collecting is what made subagents dead
            # weight before: the model could start work it had no way to read back.
            tools = [self.spawn_child, self.collect_child] if self.enable_children else []
            if context is not None:
                # Primitives and templates together. The templates are not a second
                # mechanism — they expand into the primitives and hit the same gate — but
                # they are the vocabulary the model edits in, and a surface offering only
                # one-turn edits is one on which a six-turn dead end costs six calls.
                tools = tools + context_tools(context) + context_fn_tools(context)
            factory = ContextEditingRLM if context is not None else CachingRLM
            extra = {"context": context} if context is not None else {}
            rlm = factory(
                PrimeTask,
                tools=tools,
                sub_lm=sub_lm,
                interpreter_factory=InterpreterFactory(lambda: self._current_workdir),
                subquery_embedder=subquery_embedder,
                **extra,
            )
        self.rlm = rlm
        # Derived from the RLM, never assumed from the flag. The context is only real if
        # something is writing turns into it, and a caller-supplied `rlm=` (every stub in
        # the test suite, any custom loop) is not. Setting it from `editable_context` alone
        # would hand `_after_task` an empty grounding view to read `last_trajectory` out
        # of — a mechanism reporting as on while silently discarding the run, which is the
        # one failure mode this repo refuses everywhere else.
        self.context = getattr(rlm, "context", None)

    def spawn_child(self, task: str, name: str = "child") -> str:
        """Start a sub-agent on `task` in its own session directory. Returns its child id.

        Call with keyword arguments: spawn_child(task="...").
        Returns immediately — the child runs in the background. Pass the id to
        collect_child(child_id=...) to read its result.
        """
        return self._children().spawn_child(task, name=name).child_id

    def collect_child(self, child_id: str) -> str:
        """Wait for a spawned sub-agent and return its deliverable.

        Call with keyword arguments: collect_child(child_id=...).
        """
        return self._children().result(child_id)

    def _children(self):
        if self.spawn_manager is None:
            self.spawn_manager = ChildSessionManager(
                self._child_runner or self._default_child_runner,
                root_dir=self._current_workdir,
                max_depth=self.max_child_depth,
            )
        return self.spawn_manager

    def _default_child_runner(self, task: str, name: str, session_dir: str) -> str:
        """A child is another PrimeAgent sharing the parent's ledger — read-only.

        It reads the same learned guidance but never calls learn(), so the ledger keeps a
        single writer (the parent, between tasks) even though children run on threads.
        Children cannot spawn their own children: depth is capped at the parent's limit.

        The child gets its own context — contexts are per-run by construction — but the
        *same* gate object, which is the opposite of the ledger's single-writer rule and
        deliberately so. An ungated child would be a write path running with no admission
        control while `describe()` reports the gate as on, and the ladder's adaptive
        statistics are counts of refusals: a refusal is evidence about the bar wherever it
        was proposed, so pooling parent and child observations sharpens the ordering rather
        than contaminating it.
        """
        child = PrimeAgent(self.harness, root_lm=self.root_lm, sub_lm=self.sub_lm,
                           subquery_embedder=self._subquery_embedder,
                           context_gate=(self.context.gate if self.context else None))
        pred = child.run_task(task, workdir=session_dir)
        return getattr(pred, "deliverable", str(pred))

    @property
    def last_cache_stats(self) -> dict | None:
        # Per-run sub-query cache stats, when the RLM tracks them (CachingRLM).
        return getattr(self.rlm, "last_cache_stats", None)

    def _before_task(self, task: str, workdir: str, context: dict | None) -> str:
        # Reproducibility invariant: freeze the ledger snapshot at task start.
        self._current_workdir = workdir
        if self.context is not None:
            # Invariant 6's analogue for the write path: nothing survives the task
            # boundary. Reset in place rather than rebinding, because the edit tools hold
            # a reference to this instance from RLM construction — a fresh object would
            # leave them editing the previous task's context.
            self.context.reset()
            # The one thing a mid-task edit can be grounded in. `refine()`'s gate grounds a
            # lesson in observed rubric failures, which do not exist yet here — so the gate
            # is handed the task, and handing it the empty string instead would leave the
            # judge rung scoring an edit against nothing while reporting as a grounding
            # check. Set after reset(), which clears it.
            self.context.task = task
        # `context` carries the live task state (document/playbook versions, matter ids)
        # that ReuseController gates on, so a lesson whose source has moved is withheld
        # from the prompt. None -> ungated read, the base behavior.
        exposed = self.harness.admissible_items(context=context)
        self.last_exposed_ids = [item.id for item in exposed]
        return self.harness.read(context=context) or "(no learned guidance yet)"

    def _after_task(self, pred, task_id: str):
        # Bound every child's lifetime to the task that spawned it: the parent edits the
        # ledger between tasks, and a child still reading it then would race that write.
        if self.spawn_manager is not None:
            self.last_child_errors = self.spawn_manager.drain()
            # Read after draining, not during: the counts are only complete once every
            # child has been admitted and the last pilot has been waited on.
            self.last_child_scheduling = self.spawn_manager.scheduling()
        # dspy.RLM returns the REPL history as `trajectory` ([{reasoning, code, output}]),
        # which is exactly the shape ProgressMonitor and refine() consume.
        self.last_trajectory = list(getattr(pred, "trajectory", None) or [])
        if self.context is not None:
            # The grounding view, not the prompt view. Everything downstream of here —
            # coverage, the monitor, refine(), credit assignment — is asking what the run
            # *did*, and an evicted turn still happened: dropping it would make a genuine
            # pass read as ungrounded and a document that was opened read as never opened.
            # Only `render()` is allowed to forget.
            self.last_trajectory = self.context.full_trajectory()
            self.last_context_stats = self.context.stats()
        docs = document_names(os.path.join(self._current_workdir, "documents"))
        self.last_coverage = (document_coverage(self.last_trajectory, docs)
                              if docs else None)
        if self.monitor is not None:
            self.last_monitor_decision = self.monitor.check_and_record(
                self.last_trajectory,
                self.last_cache_stats,
                self.harness.audit_log,
                self.harness.backend.current_version().number,
                task_id,
            )
        return pred

    async def arun_task(self, task: str, workdir: str, context: dict | None = None,
                        task_id: str = "") -> dspy.Prediction:
        """Async twin of run_task.

        A single RLM loop cannot be parallelised — turn N+1 reads turn N's REPL output —
        so this does not make one task faster. What it enables is overlapping *whole
        tasks*, which is where a sweep's wall time actually goes.
        """
        guidance = self._before_task(task, workdir, context)
        with dspy.context(lm=self.root_lm):
            pred = await self.rlm.acall(task=task, guidance=guidance)
        return self._after_task(pred, task_id)

    def run_task(self, task: str, workdir: str, context: dict | None = None,
                 task_id: str = "") -> dspy.Prediction:
        guidance = self._before_task(task, workdir, context)
        with dspy.context(lm=self.root_lm):
            pred = self.rlm(task=task, guidance=guidance)
        # dspy.RLM returns the REPL history as `trajectory` ([{reasoning, code, output}]),
        # which is exactly the shape ProgressMonitor and refine() consume.
        return self._after_task(pred, task_id)

    def learn(self, trajectory: list[dict], feedback, criterion_texts=None):
        # Applied only BETWEEN tasks — never mid-task.
        # With the rubric's criterion texts, the same trajectory answers a second question
        # before it is scored: did the run actually read the facts each criterion names?
        # A pass it could not have derived is dropped rather than credited — otherwise a
        # lucky guess keeps a useless lesson's success rate up and it never retires.
        # Omitted -> no filter, and credit behaves exactly as it did before.
        grounded = (grounding_from_trajectory(trajectory, criterion_texts)
                    if criterion_texts else None)
        # Credit first: this task's outcome is evidence about the guidance that was in the
        # prompt for it, and refine() consumes that evidence when it retires. Before the
        # budget gate, deliberately: a freeze stops self-modification, not measurement, and
        # halting observation would blind the thing that decides when to thaw.
        self.harness.credit(self.last_exposed_ids, feedback, grounded=grounded)
        budget = self.error_budget
        if budget is None:
            return self.harness.refine(trajectory, feedback)

        self._observe_turn_limit(budget, trajectory)
        # The freeze record has to name the task the breach was *observed* on, not whichever
        # task happened to be current when the budget was constructed.
        budget.task_id = feedback.task_id
        version = self.harness.backend.current_version().number
        budget.version = version
        if not budget.allows_refine():
            # A RefineResult, never None: a None would make every caller branch on the
            # freeze. An empty reversibility window is the honest record — the round
            # happened and changed nothing, so rollback(from_version) is a no-op rather
            # than undoing somebody else's edits.
            return RefineResult([], [], [], version, version)

        result = self.harness.refine(trajectory, feedback)
        self._observe_admission(budget, result)
        return result

    def _observe_turn_limit(self, budget, trajectory: list[dict]) -> None:
        """The share of tasks that ran to the RLM's iteration limit.

        `monitor.py` names the failure this aggregates: the live context "appends to a
        monotone history with no progress signal and no backtrack, so a wrong strategy gets
        *elaborated* until max_iters". One task doing that is noise; half of them doing it
        is a degraded system. Skipped when the limit is unknown — an unmeasurable round
        must not be scored as a healthy one.
        """
        if TURN_LIMIT_RATE not in budget.objectives:
            return
        max_iters = getattr(self.rlm, "max_iters", None)
        if not max_iters:
            return
        budget.observe(TURN_LIMIT_RATE, 1.0 if len(trajectory) >= max_iters else 0.0)

    @staticmethod
    def _observe_admission(budget, result) -> None:
        """The share of proposed edits the ladder admitted.

        Rejected edits produce no audit record, so this rate is a signal nothing else in
        the repo can see, and a collapse in it means the proposer is paying the ladder's
        cost every round to produce almost nothing. A round where the proposer offered
        nothing is not an admission failure, so it is not observed at all.
        """
        if EDIT_ADMISSION_RATE not in budget.objectives:
            return
        admitted = len(result.created) + len(result.updated) + len(result.deleted)
        proposed = admitted + len(result.rejected)
        if proposed:
            budget.observe(EDIT_ADMISSION_RATE, admitted / proposed)
