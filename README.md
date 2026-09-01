# sentinel-prime

**A DSPy-native, self-improving agent harness.** The headline contribution is
`ContinualHarness` — an **online, label-free, reversible** memory ledger that lets a DSPy
program improve itself across live sessions. This is the piece DSPy lacks today: it ships
offline, labeled optimizers (GEPA/MIPRO) but nothing for online, unlabeled, runtime
self-improvement.

Scaffolding around the harness: `dspy.RLM` for per-turn reasoning and `dspy.GEPA` for offline
optimization. The target domain and evaluation is Harvey **LAB** (Legal Agent Benchmark), M&A
due-diligence slice.

> Status: **Plan A shipped** — the standalone `ContinualHarness` core (config, session store,
> feedback parser, versioned memory backend, harness) is implemented and tested (21 tests).
> The agent runtime + Harvey LAB eval (Plan B) is next. See `docs/superpowers/`.

## Why this exists

- **Two complementary learning loops.** `ContinualHarness.refine()` runs *online* after each task
  with **no labels** — it reads the trajectory and rubric-failure text and proposes small
  create/update/delete edits to a supplemental memory ledger. `dspy.GEPA` runs *offline* on a
  labeled subset. The base prompt is never mutated; the ledger is supplemental only.
- **Reversible by construction.** Every `refine()` writes before/after snapshots; `rollback(version)`
  restores prior state — the safety property GEPA structurally cannot provide.
- **Pluggable memory backend.** A narrow `MemoryBackend` protocol; the default `JsonMemoryBackend`
  is zero-dependency. An optional TraceMind backend (a local memory OS over MCP) can drop in.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quickstart

```python
from sentinelprime import ContinualHarness
from sentinelprime.memory import JsonMemoryBackend
from sentinelprime.feedback import parse_lab_result

harness = ContinualHarness(JsonMemoryBackend("harness_state.json"))

# after a run, feed the trajectory + rubric failures back — no labels needed
feedback = parse_lab_result({
    "task_id": "ma-001",
    "criteria": [{"id": "c1", "passed": False, "reason": "missed change-of-control clause"}],
})
result = harness.refine(trajectory=[...], feedback=feedback)

# the ledger is now injectable into the next run's prompt
guidance = harness.read()

# and every refine is reversible
harness.rollback(result.from_version)
```

## Configuration

Models are declared as swappable `dspy.LM` instances — nothing hardcoded, no baked-in provider.
Copy `config.example.yaml` and point each role (`root_lm`, `sub_lm`, `reflection_lm`, `judge_lm`)
at any provider DSPy supports. Provider API keys are read from the environment
(see `.env.example`).

## Testing

```bash
.venv/bin/pytest -q
```

## Roadmap

- **Plan A (done):** `ContinualHarness` core — config, `SessionStore`, `Feedback` parser,
  `MemoryBackend` + `JsonMemoryBackend`, `ContinualHarness`.
- **Plan B (next):** `dspy.RLM` integration — real fs/bash interpreter, non-blocking `spawn_child`
  sub-agents, `PrimeAgent`, offline GEPA runner, Harvey LAB adapter + self-improvement curve,
  optional TraceMind backend.
- **Phase 2 (future):** a Rust durable-execution runtime (daemon/supervisor/leases,
  crash-recovery, reconnect/replay) over a hybrid gRPC control plane + ZeroMQ message bus.

Design docs live in `docs/superpowers/specs/` and the implementation plan in
`docs/superpowers/plans/`.
