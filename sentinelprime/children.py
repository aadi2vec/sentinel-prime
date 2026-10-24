"""Non-blocking agentic children for the PrimeAgent RLM loop.

spawn_child admits a child immediately (returns a handle) and runs its work on
a thread pool; the parent collects the result later. This makes runtime fan-out
cheap and keeps the RLM turn loop from blocking on child completion. The child
runner is injected so the manager has no LM dependency of its own.

**Fan-out and prefix caching pull in opposite directions, which is not obvious and is why
this module has a scheduler at all.** Every child here shares the parent's ledger, so every
child's prompt opens with the same `guidance` block — invariant 5 exists to keep that prefix
byte-stable precisely so the provider can cache it. But a cache is only warm once something
has populated it, and four children launched simultaneously against a cold prefix all miss:
they are in flight before any of them has returned. Maximal parallelism and maximal cache
reuse are the same decision made two different ways.

So `prefix_pilot` runs the first child of each distinct prefix alone and releases the rest
once it returns. That is a real trade and not a free win — N children go from one wall-clock
unit to two — and the counts are recorded (`scheduling()`) rather than asserted, because a
mechanism that claims to buy warm starts should be able to say how many it bought.

Three honest limits:

  - **The signal is pilot *completion*, which is later than pilot *prefix warm*.** The cache
    is populated by the pilot's first request, not its last, so waiting for the whole child
    over-waits by however long the child runs after its first turn. Nothing in a `dspy.LM`
    call exposes "the prefix is now cached", and inventing a shorter wait would mean guessing
    a duration — a magic number dressed as an optimisation.
  - **A warm start is assumed, never observed.** Providers do not report prefix-cache hits
    through this path, so `warm_starts` counts children that *were scheduled to* benefit. It
    is a count of a scheduling decision, not of a cache hit.
  - **One hung pilot must not strand its whole group**, so the wait is bounded by
    `pilot_timeout` and a group whose pilot exceeds it launches cold rather than not at all.
    Degrading to the unscheduled behaviour is the right failure: slower and correct beats a
    fan-out that never starts.
"""
from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SpawnHandle:
    child_id: str
    name: str
    session_dir: str
    # Which prefix group scheduled this child, and whether it waited for that group's pilot.
    # On the handle rather than in a side table so a collected result can be attributed: a
    # child that ran cold because its pilot timed out is a different observation from one
    # that ran cold because it *was* the pilot.
    prefix_key: str = ""
    warm: bool = False


class ChildSessionManager:
    def __init__(
        self,
        run_child: Callable[[str, str, str], Any],
        root_dir: str,
        max_depth: int = 2,
        depth: int = 0,
        max_workers: int = 4,
        prefix_pilot: bool = True,
        pilot_timeout: float = 300.0,
    ) -> None:
        self._run_child = run_child
        self._root_dir = root_dir
        self._max_depth = max_depth
        self._depth = depth
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._registry: dict[str, tuple[SpawnHandle, Future]] = {}
        self._lock = threading.Lock()
        self._prefix_pilot = prefix_pilot
        self._pilot_timeout = pilot_timeout
        # One pilot per prefix group. The pilot's own future is what later children of the
        # group wait on, so the group needs no separate synchronisation primitive — the
        # thing being waited for is exactly the thing that warms the cache.
        self._pilots: dict[str, Future] = {}
        self._pilot_timeouts = 0

    def spawn_child(self, task: str, name: str = "child",
                    prefix_key: str = "") -> SpawnHandle:
        """Admit a child immediately; dispatch it behind its prefix group's pilot.

        `prefix_key` defaults to `""` — one group for every child — because that is the
        truth when children share the parent's ledger: identical guidance, identical prefix.
        A caller that gives its children genuinely different prompt prefixes should key them
        apart, or they will queue behind a pilot whose cache they cannot use.
        """
        if self._depth >= self._max_depth:
            raise RecursionError(f"max child depth {self._max_depth} reached")
        child_id = uuid.uuid4().hex[:8]
        session_dir = os.path.join(self._root_dir, f"{name}-{child_id}")
        os.makedirs(session_dir, exist_ok=True)
        with self._lock:
            pilot = self._pilots.get(prefix_key) if self._prefix_pilot else None
            # Claiming pilot status under the lock is what makes two simultaneous spawns of
            # a new group produce one pilot and one follower rather than two pilots.
            is_pilot = self._prefix_pilot and pilot is None
            handle = SpawnHandle(child_id, name, session_dir, prefix_key,
                                 warm=not is_pilot and pilot is not None)
            if is_pilot or pilot is None:
                future = self._pool.submit(self._run_child, task, name, session_dir)
            else:
                future = self._pool.submit(self._run_behind, pilot, task, name, session_dir)
            if is_pilot:
                self._pilots[prefix_key] = future
            self._registry[child_id] = (handle, future)
        return handle  # non-blocking admission: returned before child finishes

    def _run_behind(self, pilot: Future, task: str, name: str, session_dir: str) -> Any:
        """Wait for the group's pilot to warm the prefix, then run.

        The pilot's *outcome* is deliberately ignored — `exception()` is not re-raised. A
        pilot that failed still sent its request and still warmed the prefix, and a sibling
        that would have succeeded must not be cancelled because an unrelated child did not.
        The pilot's own error is still reported through `drain()`, where it belongs.
        """
        try:
            pilot.exception(timeout=self._pilot_timeout)
        except TimeoutError:
            with self._lock:
                self._pilot_timeouts += 1
        return self._run_child(task, name, session_dir)

    def scheduling(self) -> dict:
        """What the prefix scheduler did, for the run log. Counts decisions, not cache hits."""
        with self._lock:
            handles = [handle for handle, _ in self._registry.values()]
            return {
                "prefix_pilot": self._prefix_pilot,
                "children": len(handles),
                "prefix_groups": len({h.prefix_key for h in handles}),
                # Children scheduled behind a warm prefix. Not a measured hit rate — see the
                # module docstring's second limit.
                "warm_starts": sum(1 for h in handles if h.warm),
                "cold_starts": sum(1 for h in handles if not h.warm),
                "pilot_timeouts": self._pilot_timeouts,
            }

    def result(self, child_id: str, timeout: float | None = None) -> Any:
        return self._registry[child_id][1].result(timeout=timeout)

    def done(self, child_id: str) -> bool:
        return self._registry[child_id][1].done()

    def children(self) -> list[SpawnHandle]:
        with self._lock:
            return [handle for handle, _ in self._registry.values()]

    def drain(self) -> list[tuple[str, BaseException]]:
        """Wait for every admitted child, returning (child_id, error) for any that raised.

        Non-blocking admission is a *turn-loop* optimization, not a licence to leak
        workers past the task that spawned them: the parent edits the ledger between
        tasks, and a child still reading it then would race that write. Draining bounds
        every child's lifetime to its parent's task while keeping the pool reusable.
        """
        errors: list[tuple[str, BaseException]] = []
        with self._lock:
            pending = list(self._registry.items())
        for child_id, (_handle, future) in pending:
            exc = future.exception()
            if exc is not None:
                errors.append((child_id, exc))
        return errors

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
