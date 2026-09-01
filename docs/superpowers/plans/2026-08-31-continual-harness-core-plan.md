# ContinualHarness Core (Plan A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the standalone, dependency-light headline module — `ContinualHarness` (online, label-free, reversible memory ledger) plus its config, session store, feedback parser, and pluggable JSON memory backend — fully unit-tested with no external services.

**Architecture:** Five focused modules under a new `primeagent/` package. A `MemoryBackend` protocol abstracts storage; `JsonMemoryBackend` is the zero-dependency default (a versioned `harness_state.json` with snapshot/rollback). `ContinualHarness` is a `dspy.Module` whose `refine()` proposes create/update/delete edits to the ledger from a trajectory + rubric-failure feedback, and whose `read()` serializes the ledger into a supplemental prompt block. The LM-calling part (`propose`) is separated from the deterministic edit-application core so the core is testable without any LM.

**Tech Stack:** Python 3.10+, `dspy` (core dep — only lightweight `Module`/`Predict`/`Signature` used here), `pyyaml`, `pytest`.

## Global Constraints

- Python **3.10+** (uses `X | None` unions and `match`). Copy verbatim into `pyproject.toml` `requires-python = ">=3.10"`.
- All models are declared as swappable `dspy.LM(...)` instances — **nothing hardcoded**, no baked-in provider or API key.
- **No external services** in Plan A: no network, no TraceMind MCP, no Harvey repo. Tests must run offline.
- `ContinualHarness` base prompt is **immutable**; the ledger is **supplemental only** (prepended, never overwrites).
- Every `refine()` is **reversible**: before/after snapshots, `rollback(version)` restores.
- TDD: write the failing test first; commit after each green task.
- New code lives under `primeagent/` and `tests/`. Do **not** modify the legacy `rlm/` package or `main.py`.

---

### Task 1: Package scaffold + provider-agnostic config

**Files:**
- Create: `pyproject.toml`
- Create: `primeagent/__init__.py`
- Create: `primeagent/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `config.example.yaml`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `primeagent.config.Models` — dataclass with fields `root_lm, sub_lm, reflection_lm, judge_lm`, each a `dspy.LM`.
  - `primeagent.config.load_config(path: str | None = None) -> Models`.
  - `primeagent.config._make_lm(spec: dict) -> dspy.LM` where `spec = {"model": str, "params": dict}`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "primeagent"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["dspy>=2.6", "pyyaml>=6.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
include = ["primeagent*"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Install into the project venv**

Run: `.venv/bin/pip install -e ".[dev]"`
Expected: dspy, pyyaml, pytest install; `.venv/bin/python -c "import dspy"` succeeds.

- [ ] **Step 3: Create empty `primeagent/__init__.py` and `tests/__init__.py`**

Both files are empty.

- [ ] **Step 4: Create `config.example.yaml`**

```yaml
root_lm:
  model: "openai/gpt-4o"
  params: {temperature: 0.2}
sub_lm:
  model: "openai/gpt-4o-mini"
  params: {temperature: 0.0}
reflection_lm:
  model: "openai/gpt-4o"
  params: {temperature: 0.7}
judge_lm:
  model: "openai/gpt-4o"
  params: {temperature: 0.0}
```

- [ ] **Step 5: Write the failing test**

```python
# tests/test_config.py
import textwrap
import dspy
from primeagent.config import load_config, _make_lm, Models


def test_make_lm_builds_dspy_lm_with_params():
    lm = _make_lm({"model": "openai/gpt-4o-mini", "params": {"temperature": 0.0}})
    assert isinstance(lm, dspy.LM)
    assert lm.model == "openai/gpt-4o-mini"


def test_load_config_maps_all_four_roles(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(textwrap.dedent("""
        root_lm: {model: "openai/root"}
        sub_lm: {model: "openai/sub"}
        reflection_lm: {model: "openai/reflect"}
        judge_lm: {model: "openai/judge"}
    """))
    models = load_config(str(cfg))
    assert isinstance(models, Models)
    assert models.root_lm.model == "openai/root"
    assert models.sub_lm.model == "openai/sub"
    assert models.reflection_lm.model == "openai/reflect"
    assert models.judge_lm.model == "openai/judge"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primeagent.config'`.

- [ ] **Step 7: Write minimal implementation**

```python
# primeagent/config.py
from __future__ import annotations
from dataclasses import dataclass
import yaml
import dspy


@dataclass
class Models:
    root_lm: dspy.LM
    sub_lm: dspy.LM
    reflection_lm: dspy.LM
    judge_lm: dspy.LM


def _make_lm(spec: dict) -> dspy.LM:
    return dspy.LM(model=spec["model"], **spec.get("params", {}))


def load_config(path: str | None = None) -> Models:
    if path is None:
        raise ValueError("config path is required (no default provider is baked in)")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Models(
        root_lm=_make_lm(raw["root_lm"]),
        sub_lm=_make_lm(raw["sub_lm"]),
        reflection_lm=_make_lm(raw["reflection_lm"]),
        judge_lm=_make_lm(raw["judge_lm"]),
    )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml config.example.yaml primeagent/__init__.py primeagent/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: package scaffold + provider-agnostic dspy.LM config"
```

---

### Task 2: SessionStore (JSONL trajectory)

**Files:**
- Create: `primeagent/session.py`
- Create: `tests/test_session.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `primeagent.session.SessionStore(path: str)` with:
    - `.append(event: dict) -> None` — append one JSON object as a line.
    - `.read() -> list[dict]` — read all events in order.
    - `.__iter__()` — iterate events lazily.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session.py
from primeagent.session import SessionStore


def test_append_and_read_roundtrip(tmp_path):
    store = SessionStore(str(tmp_path / "t.jsonl"))
    store.append({"role": "action", "code": "print(1)"})
    store.append({"role": "observation", "text": "1"})
    events = store.read()
    assert events == [
        {"role": "action", "code": "print(1)"},
        {"role": "observation", "text": "1"},
    ]


def test_read_missing_file_returns_empty(tmp_path):
    store = SessionStore(str(tmp_path / "none.jsonl"))
    assert store.read() == []


def test_iter_yields_in_order(tmp_path):
    store = SessionStore(str(tmp_path / "t.jsonl"))
    for i in range(3):
        store.append({"i": i})
    assert [e["i"] for e in store] == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primeagent.session'`.

- [ ] **Step 3: Write minimal implementation**

```python
# primeagent/session.py
from __future__ import annotations
import json
import os
from typing import Iterator


class SessionStore:
    def __init__(self, path: str):
        self.path = path

    def append(self, event: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def read(self) -> list[dict]:
        return list(self)

    def __iter__(self) -> Iterator[dict]:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_session.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add primeagent/session.py tests/test_session.py
git commit -m "feat: JSONL SessionStore for trajectory events"
```

---

### Task 3: Feedback model + LAB rubric parser

**Files:**
- Create: `primeagent/feedback.py`
- Create: `tests/test_feedback.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `primeagent.feedback.CriterionResult` — dataclass `{id: str, passed: bool, reason: str}`.
  - `primeagent.feedback.Feedback` — dataclass `{task_id: str, score: float, criteria: list[CriterionResult]}` with property `.failures -> list[CriterionResult]` (only `passed is False`) and `.as_text() -> str`.
  - `primeagent.feedback.parse_lab_result(raw: dict) -> Feedback` — parses a LAB-judge result dict of shape `{"task_id": str, "criteria": [{"id","passed","reason"}, ...]}`; `score` = fraction passed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feedback.py
from primeagent.feedback import parse_lab_result, Feedback, CriterionResult


SAMPLE = {
    "task_id": "ma-001",
    "criteria": [
        {"id": "c1", "passed": True, "reason": "cited clause 4.2"},
        {"id": "c2", "passed": False, "reason": "missed change-of-control provision"},
        {"id": "c3", "passed": False, "reason": "no source for indemnity cap"},
    ],
}


def test_parse_produces_feedback_with_fractional_score():
    fb = parse_lab_result(SAMPLE)
    assert isinstance(fb, Feedback)
    assert fb.task_id == "ma-001"
    assert abs(fb.score - (1 / 3)) < 1e-9
    assert len(fb.criteria) == 3


def test_failures_returns_only_failed_criteria():
    fb = parse_lab_result(SAMPLE)
    failed_ids = [c.id for c in fb.failures]
    assert failed_ids == ["c2", "c3"]
    assert all(isinstance(c, CriterionResult) for c in fb.failures)


def test_as_text_lists_failure_reasons():
    fb = parse_lab_result(SAMPLE)
    text = fb.as_text()
    assert "change-of-control" in text
    assert "indemnity cap" in text
    assert "cited clause 4.2" not in text  # passing criteria excluded


def test_empty_criteria_scores_zero_without_error():
    fb = parse_lab_result({"task_id": "x", "criteria": []})
    assert fb.score == 0.0
    assert fb.failures == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_feedback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primeagent.feedback'`.

- [ ] **Step 3: Write minimal implementation**

```python
# primeagent/feedback.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CriterionResult:
    id: str
    passed: bool
    reason: str


@dataclass
class Feedback:
    task_id: str
    score: float
    criteria: list[CriterionResult]

    @property
    def failures(self) -> list[CriterionResult]:
        return [c for c in self.criteria if c.passed is False]

    def as_text(self) -> str:
        if not self.failures:
            return "All rubric criteria passed."
        lines = [f"- [{c.id}] {c.reason}" for c in self.failures]
        return "Failed rubric criteria:\n" + "\n".join(lines)


def parse_lab_result(raw: dict) -> Feedback:
    criteria = [
        CriterionResult(id=c["id"], passed=bool(c["passed"]), reason=c.get("reason", ""))
        for c in raw.get("criteria", [])
    ]
    score = (sum(1 for c in criteria if c.passed) / len(criteria)) if criteria else 0.0
    return Feedback(task_id=raw["task_id"], score=score, criteria=criteria)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_feedback.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add primeagent/feedback.py tests/test_feedback.py
git commit -m "feat: Feedback model + LAB rubric-failure parser"
```

---

### Task 4: MemoryBackend protocol + JsonMemoryBackend (snapshot/rollback)

**Files:**
- Create: `primeagent/memory.py`
- Create: `tests/test_memory.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `primeagent.memory.MemoryItem` — dataclass `{id: str, scope: str, kind: str, text: str, created_at: str, meta: dict}` (`scope` ∈ `"session"|"global"`; `kind` ∈ `"note"|"memory"|"sub_agent_spec"`).
  - `primeagent.memory.Version` — dataclass `{number: int, parent: int | None, created_at: str}`.
  - `primeagent.memory.MemoryBackend` — `typing.Protocol` with: `read(scope: str | None = None, query: str | None = None) -> list[MemoryItem]`, `write(items: list[MemoryItem]) -> None` (upsert by `id`), `delete(ids: list[str]) -> None`, `current_version() -> Version`, `snapshot() -> Version`, `rollback(version: int) -> None`.
  - `primeagent.memory.JsonMemoryBackend(path: str)` — implements `MemoryBackend`, persisting to `harness_state.json`. `query` is ignored (no vectors in the JSON backend). `snapshot()` saves a deep copy of current items under a new incrementing version and returns that `Version`. `rollback(n)` restores items from the snapshot with `number == n`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory.py
import pytest
from primeagent.memory import MemoryItem, JsonMemoryBackend


def _item(id, scope="global", kind="note", text="t"):
    return MemoryItem(id=id, scope=scope, kind=kind, text=text, created_at="2026-01-01", meta={})


def test_write_then_read_returns_items(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    be.write([_item("a"), _item("b")])
    ids = sorted(i.id for i in be.read())
    assert ids == ["a", "b"]


def test_write_upserts_by_id(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    be.write([_item("a", text="old")])
    be.write([_item("a", text="new")])
    items = be.read()
    assert len(items) == 1
    assert items[0].text == "new"


def test_read_filters_by_scope(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    be.write([_item("a", scope="global"), _item("b", scope="session")])
    assert [i.id for i in be.read(scope="session")] == ["b"]


def test_delete_removes_items(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    be.write([_item("a"), _item("b")])
    be.delete(["a"])
    assert [i.id for i in be.read()] == ["b"]


def test_snapshot_then_mutate_then_rollback_restores(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    be.write([_item("a", text="orig")])
    v = be.snapshot()
    be.write([_item("a", text="changed"), _item("b")])
    assert {i.id for i in be.read()} == {"a", "b"}
    be.rollback(v.number)
    items = be.read()
    assert [i.id for i in items] == ["a"]
    assert items[0].text == "orig"


def test_state_persists_across_reinstantiation(tmp_path):
    path = str(tmp_path / "s.json")
    JsonMemoryBackend(path).write([_item("a")])
    assert [i.id for i in JsonMemoryBackend(path).read()] == ["a"]


def test_rollback_unknown_version_raises(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    with pytest.raises(KeyError):
        be.rollback(999)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primeagent.memory'`.

- [ ] **Step 3: Write minimal implementation**

```python
# primeagent/memory.py
from __future__ import annotations
import copy
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass
class MemoryItem:
    id: str
    scope: str
    kind: str
    text: str
    created_at: str
    meta: dict = field(default_factory=dict)


@dataclass
class Version:
    number: int
    parent: int | None
    created_at: str


@runtime_checkable
class MemoryBackend(Protocol):
    def read(self, scope: str | None = None, query: str | None = None) -> list[MemoryItem]: ...
    def write(self, items: list[MemoryItem]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def current_version(self) -> Version: ...
    def snapshot(self) -> Version: ...
    def rollback(self, version: int) -> None: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonMemoryBackend:
    def __init__(self, path: str):
        self.path = path
        self._items: dict[str, MemoryItem] = {}
        self._version = 0
        self._parent: int | None = None
        self._history: dict[int, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            raw = json.load(f)
        self._items = {i["id"]: MemoryItem(**i) for i in raw.get("items", [])}
        self._version = raw.get("version", 0)
        self._parent = raw.get("parent_version")
        self._history = {int(k): v for k, v in raw.get("history", {}).items()}

    def _persist(self) -> None:
        raw = {
            "version": self._version,
            "parent_version": self._parent,
            "items": [asdict(i) for i in self._items.values()],
            "history": self._history,
        }
        with open(self.path, "w") as f:
            json.dump(raw, f, indent=2)

    def read(self, scope: str | None = None, query: str | None = None) -> list[MemoryItem]:
        items = list(self._items.values())
        if scope is not None:
            items = [i for i in items if i.scope == scope]
        return items

    def write(self, items: list[MemoryItem]) -> None:
        for it in items:
            self._items[it.id] = it
        self._persist()

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._items.pop(i, None)
        self._persist()

    def current_version(self) -> Version:
        return Version(number=self._version, parent=self._parent, created_at=_now())

    def snapshot(self) -> Version:
        self._parent = self._version
        self._version += 1
        self._history[self._version] = [asdict(i) for i in self._items.values()]
        self._persist()
        return Version(number=self._version, parent=self._parent, created_at=_now())

    def rollback(self, version: int) -> None:
        snap = self._history[version]  # raises KeyError if unknown
        self._items = {i["id"]: MemoryItem(**i) for i in copy.deepcopy(snap)}
        self._persist()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_memory.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add primeagent/memory.py tests/test_memory.py
git commit -m "feat: MemoryBackend protocol + versioned JsonMemoryBackend"
```

---

### Task 5: ContinualHarness (read + reversible refine)

**Files:**
- Create: `primeagent/harness.py`
- Modify: `primeagent/__init__.py` (export `ContinualHarness`)
- Create: `tests/test_harness.py`

**Interfaces:**
- Consumes:
  - `primeagent.memory.MemoryBackend`, `MemoryItem`, `JsonMemoryBackend`.
  - `primeagent.feedback.Feedback` (uses `.as_text()`).
- Produces:
  - `primeagent.harness.RefineResult` — dataclass `{created: list[str], updated: list[str], deleted: list[str], from_version: int, to_version: int}`.
  - `primeagent.harness.ProposeLedgerEdits` — `dspy.Signature` (inputs `trajectory_summary`, `rubric_failures`, `current_ledger`; output `edits` as a JSON string).
  - `primeagent.harness.ContinualHarness(dspy.Module)` with:
    - `__init__(self, backend: MemoryBackend)`.
    - `read(self, scope: str | None = None) -> str` — supplemental prompt block; `""` when the ledger is empty.
    - `refine(self, trajectory: list[dict], feedback: Feedback) -> RefineResult` — calls `self.propose(...)`, applies edits, snapshots before and after.
    - `_apply_edits(self, edits: list[dict]) -> tuple[list[str], list[str], list[str]]` — deterministic, no LM.
    - `rollback(self, version: int) -> None` — delegates to backend.
    - `self.propose` is a `dspy.Predict(ProposeLedgerEdits)`, exposed to GEPA via `named_predictors()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness.py
import json
from primeagent.memory import JsonMemoryBackend, MemoryItem
from primeagent.feedback import parse_lab_result
from primeagent.harness import ContinualHarness, RefineResult


def _backend(tmp_path):
    return JsonMemoryBackend(str(tmp_path / "h.json"))


def _seed(be, id="n1", kind="note", text="always check change-of-control", scope="global"):
    be.write([MemoryItem(id=id, scope=scope, kind=kind, text=text, created_at="2026-01-01", meta={})])


def test_read_empty_ledger_returns_empty_string(tmp_path):
    h = ContinualHarness(_backend(tmp_path))
    assert h.read() == ""


def test_read_serializes_items_grouped_by_kind(tmp_path):
    be = _backend(tmp_path)
    _seed(be, id="n1", kind="note", text="note-body")
    be.write([MemoryItem(id="s1", scope="global", kind="sub_agent_spec",
                         text="extract CoC clauses", created_at="2026-01-01",
                         meta={"name": "coc_extractor", "when_to_use": "M&A docs"})])
    block = ContinualHarness(be).read()
    assert "note-body" in block
    assert "coc_extractor" in block
    assert "M&A docs" in block


def test_apply_edits_create_update_delete(tmp_path):
    be = _backend(tmp_path)
    _seed(be, id="keep", text="keep me")
    _seed(be, id="old", text="delete me")
    h = ContinualHarness(be)
    created, updated, deleted = h._apply_edits([
        {"op": "create", "id": "new1", "kind": "note", "text": "fresh insight", "scope": "global"},
        {"op": "update", "id": "keep", "kind": "note", "text": "updated body", "scope": "global"},
        {"op": "delete", "id": "old"},
    ])
    assert created == ["new1"]
    assert updated == ["keep"]
    assert deleted == ["old"]
    ids = {i.id: i.text for i in be.read()}
    assert ids == {"keep": "updated body", "new1": "fresh insight"}


def test_refine_is_reversible(tmp_path, monkeypatch):
    be = _backend(tmp_path)
    _seed(be, id="n1", text="original")
    h = ContinualHarness(be)

    # Stub the LM-calling predictor to return deterministic edits.
    edits = [{"op": "create", "id": "n2", "kind": "note", "text": "learned", "scope": "global"}]
    monkeypatch.setattr(h, "propose",
                        lambda **kw: type("P", (), {"edits": json.dumps(edits)})())

    fb = parse_lab_result({"task_id": "t", "criteria": [
        {"id": "c1", "passed": False, "reason": "missed CoC clause"}]})
    result = h.refine(trajectory=[{"role": "action", "code": "..."}], feedback=fb)

    assert isinstance(result, RefineResult)
    assert result.created == ["n2"]
    assert result.from_version != result.to_version
    assert {i.id for i in be.read()} == {"n1", "n2"}

    h.rollback(result.from_version)
    assert {i.id for i in be.read()} == {"n1"}


def test_refine_exposes_predictor_to_gepa(tmp_path):
    h = ContinualHarness(_backend(tmp_path))
    names = dict(h.named_predictors())
    assert any("propose" in n for n in names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'primeagent.harness'`.

- [ ] **Step 3: Write minimal implementation**

```python
# primeagent/harness.py
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import dspy

from primeagent.memory import MemoryBackend, MemoryItem
from primeagent.feedback import Feedback


@dataclass
class RefineResult:
    created: list[str]
    updated: list[str]
    deleted: list[str]
    from_version: int
    to_version: int


class ProposeLedgerEdits(dspy.Signature):
    """Propose small, additive edits to a supplemental memory ledger so a future run
    avoids the rubric failures just observed. Never rewrite the base task; only add,
    refine, or remove supplemental notes / memories / reusable sub-agent specs.
    Return a JSON list of edit ops. Each op is one of:
      {"op":"create","id":<str>,"kind":"note|memory|sub_agent_spec","text":<str>,"scope":"session|global","meta":{...}}
      {"op":"update","id":<existing id>,"kind":...,"text":<str>,"scope":...}
      {"op":"delete","id":<existing id>}
    """
    trajectory_summary: str = dspy.InputField()
    rubric_failures: str = dspy.InputField()
    current_ledger: str = dspy.InputField()
    edits: str = dspy.OutputField(desc="JSON list of edit ops")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContinualHarness(dspy.Module):
    def __init__(self, backend: MemoryBackend):
        super().__init__()
        self.backend = backend
        self.propose = dspy.Predict(ProposeLedgerEdits)

    def read(self, scope: str | None = None) -> str:
        items = self.backend.read(scope=scope)
        if not items:
            return ""
        notes = [i for i in items if i.kind == "note"]
        memories = [i for i in items if i.kind == "memory"]
        specs = [i for i in items if i.kind == "sub_agent_spec"]
        out: list[str] = ["## Learned guidance (supplemental — do not override the task)"]
        if notes:
            out.append("### Notes")
            out += [f"- {n.text}" for n in notes]
        if memories:
            out.append("### Memories")
            out += [f"- {m.text}" for m in memories]
        if specs:
            out.append("### Reusable sub-agents")
            for s in specs:
                name = s.meta.get("name", s.id)
                when = s.meta.get("when_to_use", "")
                out.append(f"- {name}: {when} — {s.text}")
        return "\n".join(out)

    def _apply_edits(self, edits: list[dict]) -> tuple[list[str], list[str], list[str]]:
        created: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []
        existing = {i.id for i in self.backend.read()}
        for op in edits:
            kind = op.get("op")
            if kind == "delete":
                self.backend.delete([op["id"]])
                deleted.append(op["id"])
                continue
            item_id = op.get("id") or str(uuid.uuid4())
            item = MemoryItem(
                id=item_id,
                scope=op.get("scope", "global"),
                kind=op["kind"],
                text=op["text"],
                created_at=_now(),
                meta=op.get("meta", {}),
            )
            self.backend.write([item])
            (updated if item_id in existing else created).append(item_id)
        return created, updated, deleted

    def refine(self, trajectory: list[dict], feedback: Feedback) -> RefineResult:
        before = self.backend.snapshot()
        pred = self.propose(
            trajectory_summary=json.dumps(trajectory)[:4000],
            rubric_failures=feedback.as_text(),
            current_ledger=self.read() or "(empty)",
        )
        edits = json.loads(pred.edits)
        created, updated, deleted = self._apply_edits(edits)
        after = self.backend.snapshot()
        return RefineResult(created, updated, deleted, before.number, after.number)

    def rollback(self, version: int) -> None:
        self.backend.rollback(version)
```

- [ ] **Step 4: Export from the package**

```python
# primeagent/__init__.py
from primeagent.harness import ContinualHarness

__all__ = ["ContinualHarness"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_harness.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -v`
Expected: PASS (all tasks green — config, session, feedback, memory, harness).

- [ ] **Step 7: Commit**

```bash
git add primeagent/harness.py primeagent/__init__.py tests/test_harness.py
git commit -m "feat: ContinualHarness with reversible label-free refine + read"
```

---

## Notes for the implementer

- **Why `propose` is stubbed in tests, not run against an LM:** the reversibility and edit-application logic is the property we must guarantee; it must be tested deterministically. The LM call (`propose`) is a thin seam. An end-to-end refine against a real/`DummyLM` model belongs in Plan B, where `dspy.settings.lm` is configured.
- **GEPA-readiness:** `self.propose` being a `dspy.Predict` means `named_predictors()` exposes it automatically — no extra wiring. Plan B's GEPA runner can optimize `ProposeLedgerEdits.instructions` like any DSPy module.
- **Backend swap:** everything talks to `MemoryBackend`, so Plan B's optional `TraceMindBackend` drops in without touching `ContinualHarness`.
