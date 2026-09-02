# PrimeAgent RLM Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `ContinualHarness` actually run against a live agent by wiring `dspy.RLM` to a real filesystem/bash interpreter, non-blocking `spawn_child` sub-agents, and a `PrimeAgent` module that injects the harness ledger as frozen supplemental guidance.

**Architecture:** `PrimeAgent` (a `dspy.Module`) composes three new pieces around the existing `ContinualHarness`: (1) `LocalInterpreter`, an in-process `CodeInterpreter` that swaps DSPy's Deno/Pyodide WASM sandbox for real host-process fs/bash so the RLM can read the task's document bundle; (2) `ChildSessionManager.spawn_child`, non-blocking admission of concurrent child RLM loops; (3) the agent that freezes a `harness.read()` snapshot at task start (reproducibility invariant) and applies `refine()` only between tasks. The RLM's `generate_action`/`extract` and the harness's `propose` remain classic DSPy predictors, so the whole thing stays GEPA-optimizable in a later plan.

**Tech Stack:** Python 3.10+, `dspy>=2.6` (installed: 3.3.1), stdlib only otherwise (`concurrent.futures`, `subprocess`, `contextlib`). No external services, no benchmark.

## Global Constraints

- Python 3.10+. New code lives under `sentinelprime/` (the spec's `primeagent/` layout predates the rename; do **not** create a `primeagent/` package). Tests under `tests/`.
- All models are swappable `dspy.LM` instances passed in by the caller — never hardcode a provider or model id.
- **Base prompt immutable / ledger supplemental:** the harness ledger is injected as an *additional* `guidance` input field, never by mutating the RLM's task instructions.
- **Reproducibility invariant:** `PrimeAgent.run_task()` reads the ledger exactly once, at the start of the task; `refine()` edits are applied only *between* tasks via `PrimeAgent.learn()`. No mid-task ledger mutation.
- **`LocalInterpreter` is NOT a security sandbox.** It runs model-generated code with the host process's OS permissions. Every module docstring and the README note must state it is not to be run against untrusted input. Do not add sandboxing theatre that implies otherwise.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit. Mock LMs — no test may make a real network LM call.
- Run tests with `.venv/bin/pytest`.

---

## File Structure

- `sentinelprime/interpreter.py` — `LocalInterpreter` (in-process `CodeInterpreter`), `_confine_path`, `InterpreterFactory`.
- `sentinelprime/children.py` — `SpawnHandle`, `ChildSessionManager`.
- `sentinelprime/agent.py` — `PrimeTask` signature, `PrimeAgent(dspy.Module)`.
- `sentinelprime/__init__.py` — add `PrimeAgent` to exports.
- `tests/test_interpreter.py`, `tests/test_children.py`, `tests/test_agent.py`.

---

### Task 1: `LocalInterpreter` REPL core

Swap DSPy's WASM sandbox for an in-process Python REPL that keeps state across `execute()` calls, captures stdout, exposes host-side tools, and returns `FinalOutput` when the model calls `SUBMIT()`. This is the piece the RLM drives every turn.

**Files:**
- Create: `sentinelprime/interpreter.py`
- Test: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: `dspy.primitives.code_interpreter.{FinalOutput, CodeExecutionError}`.
- Produces: `LocalInterpreter` implementing the `CodeInterpreter` protocol — `tools` (mutable dict property), `start() -> None`, `execute(code: str, variables: dict | None = None) -> FinalOutput | str | None`, `shutdown() -> None`, plus attributes `output_fields: list[dict]` and `_tools_registered: bool` that `dspy.RLM` sets/reads. `SUBMIT(**kwargs)` inside executed code raises an internal `_SubmitSignal` that `execute` converts to `FinalOutput(kwargs)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interpreter.py
import pytest
from dspy.primitives.code_interpreter import FinalOutput, CodeExecutionError, CodeInterpreter
from sentinelprime.interpreter import LocalInterpreter


def test_state_persists_and_stdout_captured():
    interp = LocalInterpreter()
    interp.start()
    assert interp.execute("x = 21") is None            # no print -> None
    assert interp.execute("print(x * 2)") == "42\n"    # state carried across calls


def test_submit_returns_final_output():
    interp = LocalInterpreter()
    interp.output_fields = [{"name": "deliverable"}]
    interp.start()
    result = interp.execute("SUBMIT(deliverable='done')")
    assert isinstance(result, FinalOutput)
    assert result.output == {"deliverable": "done"}


def test_runtime_error_becomes_code_execution_error():
    interp = LocalInterpreter()
    interp.start()
    with pytest.raises(CodeExecutionError):
        interp.execute("1 / 0")


def test_syntax_error_propagates():
    interp = LocalInterpreter()
    interp.start()
    with pytest.raises(SyntaxError):
        interp.execute("def (:")


def test_injected_tool_is_callable_from_code():
    interp = LocalInterpreter()
    interp.tools["echo"] = lambda **kw: kw["msg"]
    interp.start()
    assert interp.execute("print(echo(msg='hi'))") == "hi\n"


def test_satisfies_protocol():
    assert isinstance(LocalInterpreter(), CodeInterpreter)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interpreter.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sentinelprime.interpreter'`

- [ ] **Step 3: Write minimal implementation**

```python
# sentinelprime/interpreter.py
"""In-process CodeInterpreter for dspy.RLM.

This replaces DSPy's default Deno/Pyodide WASM interpreter (which has no
filesystem or network) with a plain in-process Python REPL, so the RLM can
read the task's document bundle and shell out with `subprocess` directly.

SECURITY: this is NOT a sandbox. Executed code runs with THIS process's OS
permissions — full filesystem and subprocess access. It is a durable control
environment for a trusted agent, not an isolation boundary. Never run it
against untrusted input.
"""
from __future__ import annotations

import contextlib
import io
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeExecutionError, FinalOutput


class _SubmitSignal(Exception):
    """Raised by the injected SUBMIT() to unwind the exec and carry outputs."""

    def __init__(self, output: dict):
        self.output = output


class LocalInterpreter:
    """A CodeInterpreter that execs model code in a persistent namespace."""

    def __init__(self) -> None:
        self._ns: dict[str, Any] | None = None
        self._tools: dict[str, Callable[..., Any]] = {}
        # dspy.RLM sets output_fields (SUBMIT field metadata) and toggles
        # _tools_registered before each forward pass; expose both.
        self.output_fields: list[dict] = []
        self._tools_registered = False

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        return self._tools

    def start(self) -> None:
        if self._ns is None:
            self._ns = {"__name__": "__rlm__"}
        self._register()

    def _register(self) -> None:
        assert self._ns is not None

        def SUBMIT(**kwargs: Any):
            raise _SubmitSignal(kwargs)

        self._ns["SUBMIT"] = SUBMIT
        self._ns.update(self._tools)
        self._tools_registered = True

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        if self._ns is None:
            self.start()
        if not self._tools_registered:
            self._register()
        if variables:
            self._ns.update(variables)

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<rlm>", "exec"), self._ns)
        except _SubmitSignal as sig:
            return FinalOutput(sig.output)
        except SyntaxError:
            raise
        except Exception as exc:  # runtime error in model code -> recoverable
            raise CodeExecutionError(str(exc)) from exc

        out = buf.getvalue()
        return out if out else None

    def shutdown(self) -> None:
        self._ns = None
        self._tools_registered = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interpreter.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinelprime/interpreter.py tests/test_interpreter.py
git commit -m "feat: LocalInterpreter — in-process CodeInterpreter for dspy.RLM"
```

---

### Task 2: Workdir scoping, path confinement, and `InterpreterFactory`

`dspy.RLM` takes a **zero-argument** `interpreter_factory`. Provide one that binds the task's document directory, exposes an `execution_instructions` string (RLM injects it into the action prompt), and ship a best-effort path-confinement helper for document scoping.

**Files:**
- Modify: `sentinelprime/interpreter.py`
- Test: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: `LocalInterpreter` from Task 1.
- Produces:
  - `_confine_path(workdir: str, path: str) -> str` — resolves `path` against `workdir` and returns the absolute path, raising `ValueError` if it escapes `workdir`.
  - `InterpreterFactory` — callable object. `InterpreterFactory(workdir_source)` where `workdir_source` is a `str` or a zero-arg `Callable[[], str]`; `__call__()` returns a fresh `LocalInterpreter` whose `workdir` attribute is the resolved dir. Has class attribute `execution_instructions: str`. A stable `InterpreterFactory` instance can therefore back a persistent RLM while the workdir varies per task (Task 4 relies on this).
  - `LocalInterpreter` gains an optional `workdir: str | None` constructor arg and attribute.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_interpreter.py
from dspy.primitives.code_interpreter import _validate_interpreter_factory
from sentinelprime.interpreter import _confine_path, InterpreterFactory


def test_confine_path_allows_inside(tmp_path):
    p = _confine_path(str(tmp_path), "docs/a.txt")
    assert p.startswith(str(tmp_path))


def test_confine_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        _confine_path(str(tmp_path), "../../etc/passwd")


def test_factory_is_zero_arg_and_makes_fresh_interpreters(tmp_path):
    factory = InterpreterFactory(str(tmp_path))
    _validate_interpreter_factory(factory)          # dspy accepts it
    a, b = factory(), factory()
    assert a is not b
    assert a.workdir == str(tmp_path)
    assert isinstance(factory.execution_instructions, str)
    assert factory.execution_instructions            # non-empty


def test_factory_accepts_callable_workdir_source():
    current = {"dir": "/tmp/one"}
    factory = InterpreterFactory(lambda: current["dir"])
    assert factory().workdir == "/tmp/one"
    current["dir"] = "/tmp/two"
    assert factory().workdir == "/tmp/two"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interpreter.py -q`
Expected: FAIL with `ImportError: cannot import name '_confine_path'`

- [ ] **Step 3: Write minimal implementation**

Change the `LocalInterpreter.__init__` signature to accept a workdir:

```python
    def __init__(self, workdir: str | None = None) -> None:
        self.workdir = workdir
        self._ns: dict[str, Any] | None = None
        self._tools: dict[str, Callable[..., Any]] = {}
        self.output_fields: list[dict] = []
        self._tools_registered = False
```

Then append to `sentinelprime/interpreter.py`:

```python
import os


def _confine_path(workdir: str, path: str) -> str:
    """Resolve `path` under `workdir`; raise if it escapes.

    Best-effort document scoping — a convenience for the agent, NOT a security
    boundary (executed code can still call open() on any absolute path).
    """
    root = os.path.realpath(workdir)
    resolved = os.path.realpath(os.path.join(root, path))
    if resolved != root and not resolved.startswith(root + os.sep):
        raise ValueError(f"path {path!r} escapes workdir {workdir!r}")
    return resolved


class InterpreterFactory:
    """Zero-arg factory binding a workdir for dspy.RLM's interpreter_factory."""

    execution_instructions = (
        "You are in a real Python process with full filesystem and subprocess "
        "access. The task's documents live in the working directory. This is a "
        "control environment, NOT a security sandbox."
    )

    def __init__(self, workdir_source: "str | Callable[[], str]") -> None:
        self._workdir_source = workdir_source

    def _resolve(self) -> str:
        src = self._workdir_source
        return src() if callable(src) else src

    def __call__(self) -> LocalInterpreter:
        return LocalInterpreter(workdir=self._resolve())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interpreter.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinelprime/interpreter.py tests/test_interpreter.py
git commit -m "feat: workdir-scoped InterpreterFactory + path confinement"
```

---

### Task 3: `ChildSessionManager` — non-blocking `spawn_child`

`spawn_child(task, name)` registers a child, kicks off its work concurrently, and returns a thin handle *immediately* — results come back later via the handle, not up the call stack. A depth guard caps recursion. The child runner is injected (a `Callable`), so this task is fully testable without any LM.

**Files:**
- Create: `sentinelprime/children.py`
- Test: `tests/test_children.py`

**Interfaces:**
- Consumes: nothing from this repo (stdlib only).
- Produces:
  - `SpawnHandle` dataclass: `child_id: str`, `name: str`, `session_dir: str`.
  - `ChildSessionManager(run_child, root_dir, max_depth=2, depth=0, max_workers=4)` where `run_child: Callable[[str, str, str], Any]` is invoked as `run_child(task, name, session_dir)` and returns the child's deliverable. Methods: `spawn_child(task, name="child") -> SpawnHandle` (non-blocking; raises `RecursionError` at depth cap), `result(child_id, timeout=None) -> Any`, `done(child_id) -> bool`, `children() -> list[SpawnHandle]`, `shutdown() -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_children.py
import threading
import pytest
from sentinelprime.children import ChildSessionManager, SpawnHandle


def test_spawn_is_non_blocking_then_result_available(tmp_path):
    release = threading.Event()

    def run_child(task, name, session_dir):
        release.wait(timeout=2)
        return f"done:{task}"

    mgr = ChildSessionManager(run_child, str(tmp_path))
    handle = mgr.spawn_child("read the NDA", name="reader")
    assert isinstance(handle, SpawnHandle)
    assert mgr.done(handle.child_id) is False   # returned before child finished
    release.set()
    assert mgr.result(handle.child_id, timeout=2) == "done:read the NDA"
    mgr.shutdown()


def test_depth_guard_rejects_beyond_max(tmp_path):
    mgr = ChildSessionManager(lambda *a: None, str(tmp_path), max_depth=2, depth=2)
    with pytest.raises(RecursionError):
        mgr.spawn_child("too deep")
    mgr.shutdown()


def test_registry_tracks_children(tmp_path):
    mgr = ChildSessionManager(lambda *a: "ok", str(tmp_path))
    h1 = mgr.spawn_child("a")
    h2 = mgr.spawn_child("b")
    ids = {h.child_id for h in mgr.children()}
    assert {h1.child_id, h2.child_id} == ids
    mgr.shutdown()


def test_session_dir_created(tmp_path):
    import os
    mgr = ChildSessionManager(lambda *a: "ok", str(tmp_path))
    handle = mgr.spawn_child("a", name="worker")
    assert os.path.isdir(handle.session_dir)
    assert "worker" in handle.session_dir
    mgr.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_children.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sentinelprime.children'`

- [ ] **Step 3: Write minimal implementation**

```python
# sentinelprime/children.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_children.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinelprime/children.py tests/test_children.py
git commit -m "feat: ChildSessionManager — non-blocking spawn_child admission"
```

---

### Task 4: `PrimeAgent` — compose RLM + interpreter + spawn_child + harness

The convergence. `PrimeAgent.run_task()` freezes the ledger into a `guidance` input at task start and runs the RLM; `PrimeAgent.learn()` applies `refine()` between tasks. The RLM is a stable attribute (a persistent `dspy.RLM` backed by an `InterpreterFactory` that reads the current workdir), so it stays GEPA-visible via `named_predictors()`. Composition logic is tested with a stub RLM (no LM); a scripted-LM smoke test exercises the real RLM + `LocalInterpreter` end to end.

**Files:**
- Create: `sentinelprime/agent.py`
- Modify: `sentinelprime/__init__.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `ContinualHarness` (has `.read()`, `.refine(trajectory, feedback) -> RefineResult`), `InterpreterFactory` (Task 2), `ChildSessionManager` (Task 3), `dspy.RLM`.
- Produces:
  - `PrimeTask(dspy.Signature)` with inputs `task: str`, `guidance: str` and output `deliverable: str`.
  - `PrimeAgent(dspy.Module)`:
    - `__init__(self, harness, root_lm, sub_lm=None, spawn_manager=None, rlm=None)`. When `rlm is None`, builds one `dspy.RLM(PrimeTask, tools=[spawn_manager.spawn_child] if spawn_manager else [], sub_lm=sub_lm, interpreter_factory=InterpreterFactory(lambda: self._current_workdir))`. An injected `rlm` (any callable returning a `dspy.Prediction`-like object) overrides this for testing.
    - `run_task(self, task: str, workdir: str) -> Prediction` — sets `self._current_workdir = workdir`, reads the frozen `guidance = self.harness.read() or "(no learned guidance yet)"`, runs the RLM under `dspy.context(lm=self.root_lm)`, returns the prediction.
    - `learn(self, trajectory: list[dict], feedback) -> RefineResult` — delegates to `self.harness.refine(trajectory, feedback)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent.py
import dspy
from sentinelprime.agent import PrimeAgent, PrimeTask
from sentinelprime.harness import ContinualHarness
from sentinelprime.memory import JsonMemoryBackend
from sentinelprime.feedback import parse_lab_result


class StubRLM:
    """Records inputs; returns a canned Prediction. No LM involved."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return dspy.Prediction(deliverable="STUB", trajectory=[])


def _harness(tmp_path):
    return ContinualHarness(JsonMemoryBackend(str(tmp_path / "state.json")))


def test_run_task_injects_frozen_guidance(tmp_path):
    harness = _harness(tmp_path)
    rlm = StubRLM()
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=rlm)
    pred = agent.run_task("summarize the SPA", workdir=str(tmp_path))
    assert pred.deliverable == "STUB"
    assert rlm.calls[0]["task"] == "summarize the SPA"
    # empty ledger -> placeholder guidance, base task untouched
    assert rlm.calls[0]["guidance"] == "(no learned guidance yet)"


def test_guidance_reflects_ledger_but_only_between_tasks(tmp_path):
    harness = _harness(tmp_path)
    # Seed the ledger directly via the backend so read() is non-empty.
    from sentinelprime.memory import MemoryItem
    harness.backend.write([MemoryItem(id="n1", scope="global", kind="note",
                                      text="check change-of-control", created_at="t")])
    rlm = StubRLM()
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=rlm)
    agent.run_task("t1", workdir=str(tmp_path))
    assert "change-of-control" in rlm.calls[0]["guidance"]


def test_learn_delegates_to_refine(tmp_path, monkeypatch):
    harness = _harness(tmp_path)
    seen = {}

    def fake_refine(trajectory, feedback):
        seen["called"] = (trajectory, feedback)
        return "REFINED"

    monkeypatch.setattr(harness, "refine", fake_refine)
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=StubRLM())
    fb = parse_lab_result({"task_id": "t", "criteria": [{"id": "c1", "passed": False, "reason": "x"}]})
    out = agent.learn(trajectory=[{"a": 1}], feedback=fb)
    assert out == "REFINED"
    assert seen["called"][0] == [{"a": 1}]


def test_default_rlm_is_named_predictor(tmp_path):
    harness = _harness(tmp_path)
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"))
    names = dict(agent.named_predictors())
    # RLM's inner predictors and the harness proposer are all GEPA-visible.
    assert any("generate_action" in n for n in names)
    assert any("propose" in n for n in names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_agent.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'sentinelprime.agent'`

- [ ] **Step 3: Write minimal implementation**

```python
# sentinelprime/agent.py
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
```

Update exports:

```python
# sentinelprime/__init__.py
from sentinelprime.harness import ContinualHarness
from sentinelprime.agent import PrimeAgent

__all__ = ["ContinualHarness", "PrimeAgent"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_agent.py -q`
Expected: PASS (4 passed). If `named_predictors()` does not descend into the injected `dspy.RLM`, confirm `self.rlm` is set as a direct attribute (it is) — dspy collects nested `Predict` modules from sub-`Module` attributes.

- [ ] **Step 5: Write the end-to-end smoke test (scripted LM, no network)**

```python
# append to tests/test_agent.py
def test_end_to_end_with_scripted_lm(tmp_path, monkeypatch):
    """Real dspy.RLM + LocalInterpreter driven by a scripted LM.

    The scripted LM returns fixed reasoning+code so the RLM writes a file,
    reads it back, and SUBMITs — exercising the interpreter's fs access and
    SUBMIT->FinalOutput path without any network call.
    """
    doc = tmp_path / "note.txt"
    doc.write_text("change-of-control clause present")

    from sentinelprime.harness import ContinualHarness
    from sentinelprime.memory import JsonMemoryBackend

    # A scripted LM: first turn writes code, second turn SUBMITs.
    from dspy.utils.dummies import DummyLM
    lm = DummyLM([
        {"reasoning": "read the doc",
         "code": "```python\nwith open('note.txt') as f: text = f.read()\nprint(text)\n```"},
        {"reasoning": "submit",
         "code": "```python\nSUBMIT(deliverable=text)\n```"},
    ])

    harness = ContinualHarness(JsonMemoryBackend(str(tmp_path / "state.json")))
    agent = PrimeAgent(harness=harness, root_lm=lm, sub_lm=lm)
    pred = agent.run_task("summarize note", workdir=str(tmp_path))
    assert "change-of-control" in pred.deliverable
```

- [ ] **Step 6: Run the smoke test**

Run: `.venv/bin/pytest tests/test_agent.py::test_end_to_end_with_scripted_lm -q`
Expected: PASS. If `DummyLM`'s response keys do not match the RLM action signature fields (`reasoning`, `code`), inspect `dspy.utils.dummies.DummyLM` in the installed dspy and adjust the scripted dicts to the exact field names the RLM's `generate_action` predictor expects. If `DummyLM` cannot drive the RLM's adapter reliably, mark this single test `@pytest.mark.integration` and skip by default — the composition is already covered by Steps 1–4; do not weaken the unit tests to compensate.

- [ ] **Step 7: Run the full suite and commit**

Run: `.venv/bin/pytest -q`
Expected: PASS (all prior tests plus the new ones).

```bash
git add sentinelprime/agent.py sentinelprime/__init__.py tests/test_agent.py
git commit -m "feat: PrimeAgent — RLM + interpreter + spawn_child + harness ledger"
```

---

## Self-Review

**Spec coverage (§6, §7, §7.1):**
- §6 custom `interpreter_factory` with real fs/bash → Tasks 1–2. `tools=[spawn_child]` → Task 4. Sandbox caveat documented → Global Constraints + module docstrings.
- §7 `spawn_child` non-blocking admission, `SpawnHandle` (child_id/name/session_dir), recursion depth 2, in-memory registry → Task 3.
- §7.1 reproducibility invariant (freeze ledger at task start, refine between tasks) → Task 4 `run_task`/`learn`. Static nodes GEPA-visible via `named_predictors()` → Task 4 Step 1 last test.
- §11 testing: interpreter path confinement (Task 2), spawn admission + registry (Task 3), harness edit/rollback (already covered by Plan A), one small e2e (Task 4 Step 5).

**Deliberately out of scope (their own later plans):** credit assignment / retrieval / decay (§5.1), GEPA runner (§10 `optimize/`), Harvey LAB adapter + curve (§8, `eval/`), TraceMind backend (§5.2). This plan produces a runnable, tested `PrimeAgent` without any of them.

**Type consistency:** `InterpreterFactory` accepts str-or-callable in Task 2 and is used with the callable form in Task 4 — consistent. `ChildSessionManager.spawn_child(task, name)` signature in Task 3 matches the `tools=[spawn_manager.spawn_child]` wiring in Task 4. `harness.read()` / `harness.refine()` match the Plan A `ContinualHarness` API.

**Risk flagged for the executor:** the Task 4 Step 5 smoke test depends on `DummyLM`'s exact response contract in dspy 3.3.1, which Step 6 says to verify against the installed source and downgrade to a skipped integration test rather than fake a pass. This is the one place the plan cannot fully pin the interface from static reading.
