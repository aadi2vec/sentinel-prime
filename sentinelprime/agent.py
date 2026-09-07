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

All three are opt-in: with no context and no monitor the behavior is unchanged.
"""
from __future__ import annotations

import dspy

from sentinelprime.children import ChildSessionManager
from sentinelprime.harness import ContinualHarness
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
                 monitor=None, enable_children: bool = False, child_runner=None,
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
        # Last run's RLM trajectory and monitor verdict, so the caller can hand the
        # *real* trajectory to learn() instead of an empty list.
        self.last_trajectory: list[dict] = []
        self.last_monitor_decision = None
        # The guidance ids actually surfaced for the last task — the exposure set credit
        # assignment scores. Not the whole ledger: a lesson gating withheld cannot be
        # blamed for that task's outcome.
        self.last_exposed_ids: list[str] = []
        self.last_child_errors: list = []
        self._current_workdir = "."
        self._subquery_embedder = subquery_embedder
        if rlm is None:
            # Bind both halves. Spawning without collecting is what made subagents dead
            # weight before: the model could start work it had no way to read back.
            tools = [self.spawn_child, self.collect_child] if self.enable_children else []
            rlm = CachingRLM(
                PrimeTask,
                tools=tools,
                sub_lm=sub_lm,
                interpreter_factory=InterpreterFactory(lambda: self._current_workdir),
                subquery_embedder=subquery_embedder,
            )
        self.rlm = rlm

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
        """
        child = PrimeAgent(self.harness, root_lm=self.root_lm, sub_lm=self.sub_lm,
                           subquery_embedder=self._subquery_embedder)
        pred = child.run_task(task, workdir=session_dir)
        return getattr(pred, "deliverable", str(pred))

    @property
    def last_cache_stats(self) -> dict | None:
        # Per-run sub-query cache stats, when the RLM tracks them (CachingRLM).
        return getattr(self.rlm, "last_cache_stats", None)

    def run_task(self, task: str, workdir: str, context: dict | None = None,
                 task_id: str = "") -> dspy.Prediction:
        # Reproducibility invariant: freeze the ledger snapshot at task start.
        self._current_workdir = workdir
        # `context` carries the live task state (document/playbook versions, matter ids)
        # that ReuseController gates on, so a lesson whose source has moved is withheld
        # from the prompt. None -> ungated read, the base behavior.
        exposed = self.harness.admissible_items(context=context)
        self.last_exposed_ids = [item.id for item in exposed]
        guidance = self.harness.read(context=context) or "(no learned guidance yet)"
        with dspy.context(lm=self.root_lm):
            pred = self.rlm(task=task, guidance=guidance)
        # dspy.RLM returns the REPL history as `trajectory` ([{reasoning, code, output}]),
        # which is exactly the shape ProgressMonitor and refine() consume.
        # Bound every child's lifetime to the task that spawned it: the parent edits the
        # ledger between tasks, and a child still reading it then would race that write.
        if self.spawn_manager is not None:
            self.last_child_errors = self.spawn_manager.drain()
        self.last_trajectory = list(getattr(pred, "trajectory", None) or [])
        if self.monitor is not None:
            self.last_monitor_decision = self.monitor.check_and_record(
                self.last_trajectory,
                self.last_cache_stats,
                self.harness.audit_log,
                self.harness.backend.current_version().number,
                task_id,
            )
        return pred

    def learn(self, trajectory: list[dict], feedback):
        # Applied only BETWEEN tasks — never mid-task.
        # Credit first: this task's outcome is evidence about the guidance that was in the
        # prompt for it, and refine() consumes that evidence when it retires.
        self.harness.credit(self.last_exposed_ids, feedback)
        return self.harness.refine(trajectory, feedback)
