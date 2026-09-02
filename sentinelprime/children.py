"""Non-blocking agentic children for the PrimeAgent RLM loop.

spawn_child admits a child immediately (returns a handle) and runs its work on
a thread pool; the parent collects the result later. This makes runtime fan-out
cheap and keeps the RLM turn loop from blocking on child completion. The child
runner is injected so the manager has no LM dependency of its own.
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


class ChildSessionManager:
    def __init__(
        self,
        run_child: Callable[[str, str, str], Any],
        root_dir: str,
        max_depth: int = 2,
        depth: int = 0,
        max_workers: int = 4,
    ) -> None:
        self._run_child = run_child
        self._root_dir = root_dir
        self._max_depth = max_depth
        self._depth = depth
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._registry: dict[str, tuple[SpawnHandle, Future]] = {}
        self._lock = threading.Lock()

    def spawn_child(self, task: str, name: str = "child") -> SpawnHandle:
        if self._depth >= self._max_depth:
            raise RecursionError(f"max child depth {self._max_depth} reached")
        child_id = uuid.uuid4().hex[:8]
        session_dir = os.path.join(self._root_dir, f"{name}-{child_id}")
        os.makedirs(session_dir, exist_ok=True)
        handle = SpawnHandle(child_id, name, session_dir)
        future = self._pool.submit(self._run_child, task, name, session_dir)
        with self._lock:
            self._registry[child_id] = (handle, future)
        return handle  # non-blocking admission: returned before child finishes

    def result(self, child_id: str, timeout: float | None = None) -> Any:
        return self._registry[child_id][1].result(timeout=timeout)

    def done(self, child_id: str) -> bool:
        return self._registry[child_id][1].done()

    def children(self) -> list[SpawnHandle]:
        with self._lock:
            return [handle for handle, _ in self._registry.values()]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
