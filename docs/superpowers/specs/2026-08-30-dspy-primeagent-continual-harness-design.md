# dspy.PrimeAgent — A DSPy-Native, Self-Improving RLM Harness

**Status:** Design — awaiting implementation approval
**Date:** 2026-08-30
**Headline deliverable:** `dspy.ContinualHarness` — online, label-free, reversible self-improvement for DSPy programs
**Scaffolding:** `dspy.RLM` (per-turn reasoning) + `dspy.GEPA` (offline optimization)
**Domain & eval:** Harvey **LAB** (Legal Agent Benchmark), M&A due-diligence slice

---

## 1. Goal & Positioning

Build a DSPy-native, single-process, Prime-Agent-shaped harness whose **novel contribution is
`dspy.ContinualHarness`**: an online, label-free, reversible memory ledger that lets a DSPy
program improve itself across live sessions. This is precisely the piece the Prime Agent
breakdown identifies as having **no DSPy equivalent** — DSPy ships offline, labeled,
compile-time optimizers (GEPA/MIPRO) but nothing for online, unlabeled, runtime
self-improvement (the paper's "Continual Harness" / `/refine`).

`dspy.RLM` supplies the per-turn reasoning loop; `dspy.GEPA` supplies offline optimization.
The system is demonstrated on Harvey **LAB**, whose criterion-level rubric-failure output is the
**shared feedback signal** for both learning loops — so the online loop needs no hand-labeling.

**Why this positioning (project is a resume + open-source portfolio piece):** a faithful clone of
`PrimeIntellect-ai/prime-agent` already exists and adds nothing new. The differentiated,
community-relevant artifact is a clean, adoptable DSPy module that fills a real gap in a widely
used framework, plus a demoable self-improvement curve on a **named, credible, MIT-licensed
legal benchmark**.

### 1.1 The hero result
Harvey LAB reports frontier agents complete **<10%** of tasks end-to-end under its strict
all-pass rubric. The headline demo: *"`dspy.PrimeAgent` with a continual harness climbs from
X% to Y% on the M&A due-diligence slice across sessions, with zero labeled training data."*
This is the README hero plot and the resume line.

---

## 2. Scope

**In scope (Phase 1, this spec):** A-scope — `dspy.RLM` as the per-turn primitive, real agentic
`spawn_child` sub-agents (non-blocking admission), the `ContinualHarness` ledger, cross-*call*
session state, GEPA offline optimization, and a Harvey LAB eval harness. Single process.

**Out of scope (Phase 1) / deferred to Phase 2 (Rust):** the full C-scope replica —
daemon/worker/supervisor topology, atomic launch leases, crash recovery, reconnect/replay
(protocol v4), cross-*restart* kernel persistence. Phase 1 interfaces are designed so Phase 2
can grow into them over the same IPC bridge (see §12).

---

## 3. Architecture (single process, Phase 1)

```
                        ┌─────────────────────────────────────────┐
   LAB task ───────────▶│            PrimeAgent (dspy.Module)      │
 (instructions,         │                                         │
  documents, rubric)    │  ┌───────────────────────────────────┐  │
                        │  │ ContinualHarness  (HEADLINE)      │  │
                        │  │  harness_state.json (versioned)   │  │
                        │  │  • prompt notes                   │  │
                        │  │  • reusable sub-agent specs       │  │
                        │  │  • skill descriptions / memories  │  │
                        │  │  read()   → inject into RLM instr.│  │
                        │  │  refine(trajectory, feedback)     │  │
                        │  │  snapshot()/rollback() reversible │  │
                        │  └───────────────┬───────────────────┘  │
                        │                  │ injects guidance      │
                        │                  ▼                       │
                        │  ┌───────────────────────────────────┐  │
                        │  │ dspy.RLM  (per-turn reasoning)    │  │
                        │  │  custom interpreter_factory →     │  │
                        │  │   real fs/bash over the doc set   │  │
                        │  │  tools=[spawn_child, ...]         │  │
                        │  └───────────────┬───────────────────┘  │
                        │                  │ spawn_child(task)     │
                        │                  ▼                       │
                        │  ┌───────────────────────────────────┐  │
                        │  │ ChildSessionManager               │  │
                        │  │  non-blocking admission + registry│  │
                        │  │  child = its own RLM sub-agent    │  │
                        │  │  results via async message/file   │  │
                        │  └───────────────────────────────────┘  │
                        │                                         │
                        │  SessionStore (JSONL trajectory)        │
                        └──────────────────┬──────────────────────┘
                                           ▼
        deliverable ─────▶ Harvey LAB evaluation/ (LLM judge, all-pass rubric)
                                           │ per-criterion pass/fail + reasons
                    ┌──────────────────────┴───────────────────────┐
                    ▼ (online, per-session)          ▼ (offline, periodic)
            ContinualHarness.refine()          dspy.GEPA.compile()
            edits harness_state.json           rewrites RLM instructions
            (no labels, reversible)            (labeled slice, versioned artifact)
```

---

## 4. The Two Learning Loops (the differentiator)

The loops are **complementary, not redundant** — mirroring the breakdown's `/refine` vs. GEPA
distinction. The demo reports **both curves** (harness-only, and harness+GEPA stacked) so the
online loop's isolated contribution is visible.

| | `ContinualHarness.refine()` (online) | `dspy.GEPA.compile()` (offline) |
|---|---|---|
| **Trigger** | After each LAB task, live | Periodic, on a labeled dev subset |
| **Data needed** | None — reads the just-run trajectory + rubric-failure text | trainset + valset + metric |
| **What it edits** | Additive `harness_state.json` ledger (notes, sub-agent specs, memories) | The RLM's `generate_action`/`extract` instruction text |
| **Base prompt** | Never touched (immutable) — supplemental only | Is itself the thing being rewritten |
| **Reversible** | Yes — before/after snapshots, `rollback(version)` | New versioned compiled artifact |
| **Output** | Incrementally growing memory, re-injected next run | A compiled program version deployed in place |

---

## 5. `ContinualHarness` — Mechanism (headline module)

A `dspy.Module` with a persisted, versioned ledger.

**State (`harness_state.json`):**
```python
{
  "version": int,
  "parent_version": int | None,
  "notes": [ { "id", "text", "created_at", "source_task" } ],
  "sub_agent_specs": [ { "id", "name", "when_to_use", "instructions" } ],
  "memories": [ { "id", "text", "created_at" } ]
}
```

**API:**
- `read() -> str` — serialize the ledger into a compact block **prepended to the RLM signature's
  instructions** at each turn. The base system prompt stays immutable; the harness is purely
  supplemental (this is the paper's key invariant).
- `refine(trajectory, feedback) -> RefineResult` — a `dspy.Module` (itself GEPA-optimizable via
  `named_predictors()`) that reads the trajectory + LAB rubric-failure text and proposes small
  **create / update / delete** edits to the ledger. **No labels required** — the rubric failures are
  the signal.
- `snapshot()` / `rollback(version)` — every `refine` writes a before/after snapshot; rollback
  restores. This reversibility is the property GEPA structurally cannot provide and is central to
  the module's novelty and safety.

**Scope tiers:** a session-local ledger (per LAB task run) and a global ledger
(`~/.primeagent/harness/`) for cross-task lessons — the M&A-specific strategy notes and reusable
sub-agent specs that accumulate across the slice and drive the self-improvement curve.

### 5.2 Pluggable `MemoryBackend` (default JSON, optional TraceMind)

`ContinualHarness` reads/writes its ledger through a narrow `MemoryBackend` interface so the
storage engine is swappable. This keeps the OSS repo runnable standalone while letting the
user's own **TraceMind** memory OS plug in as a first-class backend.

```python
class MemoryBackend(Protocol):
    def read(self, scope: str, query: str | None) -> list[MemoryItem]: ...
    def write(self, scope: str, items: list[MemoryItem]) -> None: ...
    def snapshot(self) -> Version: ...
    def rollback(self, version: Version) -> None: ...
```

- **`JsonMemoryBackend` (default, zero-dep):** the `harness_state.json` ledger described above.
  The repo runs with no external services.
- **`TraceMindBackend` (optional):** talks to TraceMind's **`tm-mcp`** MCP server (JSON-RPC 2.0
  over stdio) — the same Python↔Rust seam as the Phase-2 bridge. Mapping:
  - `ContinualHarness.read()` → `memory_query` (UCB1-bandit retrieval into the hot tier;
    vector → optional ColBERT rerank → k-hop graph → episodic scan).
  - `ContinualHarness.refine()` writes → `memory_store` (notes/memories) and TraceMind's
    **Procedural** layer (versioned how-to sequences with lifecycle FSM) for **reusable
    sub-agent specs** — versioning gives reversibility for free (`version` + `parent_id`).
  - Reversibility → TraceMind's provenance/superseded chains; `rollback` restores a prior
    procedure/version.
  - Confidence **decay** (forgetting curve) is the natural expiry policy for stale ledger notes,
    replacing manual pruning.
  - **Design invariant preserved:** TraceMind's KG is never mutated by gradient descent — only
    by heuristics + feedback — which is exactly the label-free, reversible property the harness
    requires.

The backend is selected by config; `TraceMindBackend` is optional and gated behind its own
extra so it never becomes a hard dependency of the published module.

---

## 6. `dspy.RLM` Integration

- Used directly as the per-turn primitive. `generate_action` / `extract` are exposed to GEPA via
  `named_predictors()`, so offline optimization rewrites their instructions like any DSPy module.
- **Custom `interpreter_factory`** providing *real* filesystem/bash access scoped to the task's
  document bundle (LAB provides a doc set per task). The default DSPy interpreter is
  Deno+Pyodide WASM with no fs/network; we swap in a real subprocess-backed interpreter.
- **`tools=[spawn_child, ...]`** — `spawn_child` is the injection point for real agentic children
  (not `llm_query`, which is a bare completion with no tool loop).
- **Sandbox caveat (documented):** the custom interpreter runs model-generated code with the
  process's OS permissions — a durable control environment, **not a security sandbox**. Not to be
  run against untrusted inputs. Same trust posture the Prime Agent paper states explicitly.

---

## 7. `ChildSessionManager` — Real Agentic Children (A-scope, no daemon)

- `spawn_child(task, name) -> handle` — **non-blocking admission**: registers the child, returns a
  thin handle immediately (mirrors `RLMSpawnHandle`: `child_id`, `name`, `session_dir`), and runs
  the child's own RLM loop concurrently via `asyncio`.
- Results return **later** as async agent-messages / files the parent reads — *not* up the call
  stack. This makes fan-out cheap and non-blocking by construction.
- Recursion depth default **2** (root → child → grandchild), configurable.
- In-memory child registry. (A durable, restart-surviving registry is the Phase-2 upgrade.)

### 7.1 Workflow Model — static graph vs. runtime-decided graph

The design deliberately combines two graph regimes. Stating the split explicitly is part of the
DSPy-native contribution, because DSPy today only has the first.

- **Static graph (compile-time):** classic DSPy — modules wired in `forward()`, shape fixed before
  any token, prompts optimized at fixed nodes by GEPA/MIPRO.
- **Runtime-decided graph (agentic):** the model emits an action per turn; the graph *is* the
  trajectory and only exists after the run. `dspy.RLM` is exactly this.

`dspy.PrimeAgent` is a **runtime-decided graph whose nodes are static DSPy predictors** — dynamic
outer shape, GEPA-optimizable inner nodes. Three patterns compose:

1. **Loop** — the RLM turn loop (`generate_action → execute → REPLHistory → …`), bounded by
   `max_iters` / `max_llm_calls`. Termination is model- *or* budget-decided.
2. **Dynamic fan-out** — `spawn_child` branches the graph at runtime via **non-blocking
   admission**: children run their own RLM loops concurrently; results return as async
   messages/files, *not* up the call stack. The runtime graph is a growing forest bounded by
   recursion depth.
3. **Static nodes inside** — `generate_action` / `extract` remain classic predictors exposed via
   `named_predictors()`, so GEPA can rewrite them even though the outer shape is dynamic. This is
   why both learning loops coexist.

**Managed agents = the learning loop crystallizing the dynamic graph.** A "managed agent" is a
registered, named, reusable sub-agent selected by name (rather than an ad-hoc free-text spawn).
The harness ledger's `sub_agent_specs` (`{name, when_to_use, instructions}`) **is** that registry,
and `ContinualHarness.refine()` is what **authors** it: when it detects a recurring child task, it
writes a new spec; next session `read()` injects it and `spawn_child` invokes it **by name**
instead of re-deriving it. With the TraceMind backend (§5.2) these specs live in the **Procedural**
layer — versioned (`version`+`parent_id`), so a managed agent can improve across runs and roll back
if a new version regresses the rubric.

The net novel claim (README-worthy): *the continual harness turns an unstructured dynamic agent
graph into a growing library of versioned, managed sub-agents — with no labels.* DSPy has neither
runtime-dynamic graphs nor a managed-agent registry today.

**Reproducibility invariant (required for a credible curve):** freeze the ledger version at the
**start** of each LAB task — the whole task runs against one immutable `read()` snapshot — and apply
`refine()` edits only **between** tasks. Mid-task ledger mutation would make the self-improvement
curve non-reproducible and blur learning vs. async-fan-out noise.

---

## 8. Eval Harness (Harvey LAB)

- Thin adapter over the vendored `harveyai/harvey-labs` `harness/` + `evaluation/` (MIT license;
  vendored as a git submodule under `third_party/`).
- Runs `PrimeAgent` on the M&A due-diligence slice, collects deliverables, invokes LAB's
  LLM-judge, and parses **per-criterion pass/fail + reasons** into a `Feedback` object consumed
  by both learning loops.
- **Dev loop (subset):** a hand-picked ~30–50 task subset, cheap enough to re-run frequently
  during development.
- **Headline run (full M&A slice):** run occasionally to produce the reported self-improvement
  curve (pass-rate vs. session index).
- The LAB LLM-judge model is held **fixed** across runs so scores are comparable.

---

## 9. Provider-Agnostic Config

`config.py` (optionally backed by `config.yaml`) declares every model as a swappable
`dspy.LM(...)` — nothing hardcoded, no baked-in provider or budget:
- `root_lm` — per-turn RLM reasoner (strong)
- `sub_lm` — cheap/fast model for short document reads inside the RLM
- `reflection_lm` — strong model for GEPA reflection, run rarely/offline
- `judge_lm` — LAB's LLM judge, fixed for comparability

Any provider (Anthropic / OpenAI / Google / mix) works via DSPy's `LM` abstraction.

---

## 10. Proposed File Layout

```
primeagent/
  __init__.py             # exposes PrimeAgent, ContinualHarness
  agent.py                # PrimeAgent(dspy.Module) — composes the pieces
  harness.py              # ContinualHarness (HEADLINE)
  children.py             # ChildSessionManager, spawn handle
  interpreter.py          # custom interpreter_factory (real fs/bash)
  session.py              # SessionStore (JSONL trajectory)
  config.py               # provider-agnostic dspy.LM config
  optimize/
    gepa_runner.py        # offline GEPA compile driver
eval/
  lab_adapter.py          # Harvey LAB harness/judge adapter
  feedback.py             # parse rubric failures → Feedback
  run_curve.py            # produces the self-improvement curve + plot
third_party/harvey-labs/  # vendored MIT benchmark (git submodule)
tests/                    # unit tests per module
docs/                     # README hero plot, this design, sandbox caveat
```

---

## 11. Testing

- Per-module unit tests with mocked LMs:
  - `ContinualHarness` edit application + snapshot/rollback correctness
  - `feedback.py` rubric-failure parsing
  - `ChildSessionManager` non-blocking admission + registry
  - `interpreter.py` document-scoping / path confinement
- One small end-to-end integration test on a single LAB task with a cheap model.

---

## 12. Phase 2 (Future) — A Durable-Execution Managed Runtime for Runtime-Decided Agent Graphs

Phase 2 replaces the single-process Python host with a **Rust daemon** that is, in modern terms, a
**durable-execution managed runtime** for agents — the same category as Temporal, LangGraph
Platform, Cloudflare Agents (Durable Objects), and AWS Bedrock AgentCore — but specialized for
**runtime-decided agent graphs** rather than static workflow definitions. It delivers the full
Prime Agent runtime that sits entirely outside DSPy, mirroring Prime Agent's polyglot design
(TypeScript host + Python kernel over a typed stdio bridge) with the host language as Rust.

**Why this is the interesting version of the problem.** A static DAG has known checkpoint
boundaries and is easy to make durable. Our graph is **grown at runtime** by the model, with async
`spawn_child` fan-out (§7.1) — there is no predefined workflow to checkpoint. So the runtime's
source of truth is the **event-sourced trajectory itself**: every action, spawn, host-request, and
agent-message is appended to a durable log, and any agent's state is reconstructed by **replaying**
that log. This is the distributed-systems contribution — *durable execution for agent forests whose
shape is unknown until they run* — and it is strictly harder than durably running a fixed graph.

**Managed-runtime responsibilities the daemon owns** (control plane), leaving the Python worker to
own only reasoning (execution plane):

| Managed-runtime capability | Realized as |
|---|---|
| Durable execution / crash-resume | event-sourced trajectory log + replay (protocol v4) |
| Exactly-resume-where-stopped | `dill`-snapshotted cross-restart kernel state |
| One-agent-one-identity (actor/durable object) | durable child registry surviving restarts |
| Process ownership / failover | atomic launch lease; crashed supervisor transparently replaced |
| Backpressure / retries | 250ms→1s→5s backoff, 3-failure threshold |
| Idle → cheap (hibernation) | object-store rehydrate under inference latency (§A.3) |
| Reconnect / catch-up | generation + sequence-cursor protocol; clients replay only what they missed |

**Build vs. buy (deliberate choice).** Phase 1 builds *nothing* (single process). Phase 2 builds
the runtime *by hand in Rust* — because for a resume/OSS piece the hand-built durable runtime **is**
the distributed-systems credential; adopting Temporal/LangGraph Platform/Durable Objects would hide
exactly the hard part worth showing. The IPC bridge (below) is kept clean enough that an adopter
who wants durability without the ops could swap our daemon for a managed platform — but the
reference implementation is ours.

**The seam is the IPC bridge, defined in Phase 1.** Every host-crossing call (`rlm.run`,
`agent_message.*`, `compact`, ledger read/write) is modeled as a typed `host_request` /
`host_reply` pair. In Phase 1 both sides are Python; in Phase 2 the host side becomes Rust and
the Python worker is a subprocess the Rust daemon spawns and supervises. **Only bounded
metadata crosses the boundary** — credentials, provider execution, transcript writes, and
scheduling never enter Python.

### 12.1 Transport — hybrid gRPC control plane + ZeroMQ message bus

Two traffic classes with different shapes, so two transports:

- **Control plane → gRPC (protobuf; `tonic` on Rust, `grpcio` on Python).** The typed,
  request/reply, bounded-metadata calls — `rlm.run`, `compact`, ledger read/write, lease/registry
  ops. gRPC matches the "typed `host_request`/`host_reply`" design directly (protobuf *is* the
  contract), gives cross-language codegen, deadlines/cancellation, mTLS, and mature observability.
  This is most of the surface and the enterprise-trust story.
- **Message bus → ZeroMQ (`ROUTER/DEALER` + `PUB/SUB`, msgpack payloads).** The high-fan-out,
  async, actor-mailbox traffic — `spawn_child` dispatch and out-of-band `agent_message.*` delivery
  (§7.1). Spawning many short-lived children and delivering results *out of the call stack* is
  ZMQ's native pattern: `ROUTER/DEALER` multiplexes N workers over one socket without per-worker
  connection bookkeeping; `PUB/SUB` broadcasts agent-messages. Expressing this on gRPC bidi
  streams would mean far more connection management for exactly the dynamic part of the system.

**Why not one transport:** gRPC alone makes dynamic mailbox fan-out awkward; ZMQ alone throws away
the typed contract and enterprise tooling that the control plane wants. The split puts each where
it is strongest.

**Durability caveat (applies to both):** neither gRPC nor ZMQ persists anything — they move bytes.
Crash-safety comes solely from the event-sourced trajectory log; the transport reconnects, the log
replays.

```
┌────────────────────────────────────────┐
│  Rust host  (Phase 2, C-scope)          │
│   daemon · supervisor · atomic leases   │
│   worker lifecycle · crash recovery     │
│   protocol v4 (reconnect/replay)        │
│   session store · routing · durability  │
└──────┬───────────────────────────┬──────┘
       │ gRPC (typed control plane)│ ZeroMQ ROUTER/DEALER + PUB/SUB
       │ bounded metadata          │ spawn dispatch · agent-messages
┌──────▼───────────────────────────▼──────┐
│  Python worker(s)  (Phase 1, unchanged)  │
│   dspy.RLM · ContinualHarness · GEPA     │
└──────────────────────────────────────────┘
```

**Phase 2 scope:** detached process groups, leader-election-style atomic launch lease (crashed
supervisor transparently replaced), 250ms→1s→5s backoff with 3-failure threshold, durable child
registry surviving restarts, `dill`-snapshotted cross-restart kernel state, and a
generation-and-sequence-cursor protocol so reconnecting clients replay only what they missed.
Each phase is independently shippable and resume-worthy: Phase 1 is the DSPy/ML contribution,
Phase 2 is the distributed-systems contribution.

---

## 13. Explicitly Out of Scope (both phases, for now)

- Multi-tenant hosting / auth.
- Fine-tuning any model (closed APIs only, provider-agnostic).
- A GUI/TUI client (headless + scripts only for Phase 1).
- Replicating Prime Agent's compaction summarizer as a separate optimized module (could be a
  later GEPA target; not required for the headline result).

---

## 14. Success Criteria (Phase 1)

1. `dspy.ContinualHarness` is a clean, importable, tested `dspy.Module` with working
   `read` / `refine` / `snapshot` / `rollback` and no labeled-data dependency.
2. `dspy.PrimeAgent` runs end-to-end on a Harvey LAB M&A task and produces a graded
   deliverable.
3. A reproducible **self-improvement curve** on the M&A slice showing harness-only pass-rate
   rising across sessions from a cold ledger, with harness+GEPA plotted alongside.
4. Provider-agnostic config demonstrated by running the same pipeline on ≥2 providers (or
   documented as trivially swappable).
5. README with the hero plot, quickstart, and the sandbox/trust caveat.

---

## 15. Appendix A — Research & Commercial Vision (beyond Phase 1)

This appendix captures the longer-horizon thesis. **None of it is required for the Phase-1
headline result**; it exists to show the research/commercial arc the clean Phase-1 module opens
up, and to keep Phase-1 interfaces honest about where they lead.

### A.1 Memory as three tiers (Complementary Learning Systems framing)

The `ContinualHarness` "memory" generalizes to a three-tier hierarchy, mirroring CLS theory
(fast hippocampal store + slow neocortical consolidation):

| Tier | Substrate | Write speed | Reversible | Phase-1 status |
|---|---|---|---|---|
| **T1 Context** | In-prompt ledger block (`read()`) | Instant | Trivially | ✅ built |
| **T2 External store** | TraceMind KG / JSON ledger | Fast | Yes (versioned) | ✅ built (§5.2) |
| **T3 Weights** | LoRA adapter "engram" | Slow | Yes (swap adapter) | Phase 2+ |

The **Engram thesis** — "memory *is* weight updates" — enters at **T3**: recurring, high-value
lessons that have proven stable in T2 get **consolidated** into a LoRA adapter, exactly as
hippocampal traces consolidate into cortex. Consolidation is gated (only promote T2 items with
sustained positive rubric impact), so it stays label-free and reversible (a bad adapter is
swapped out, not "unlearned").

### A.2 Hybrid reward — verifiable + non-verifiable

Legal deliverables give a naturally **decomposed reward**, so we don't have to choose RLVR vs.
preference learning:

- **Verifiable (RLVR-style):** citation-grounding (does the cited clause exist / say that?),
  schema/format compliance, numeric/date consistency, required-section presence. Cheap,
  deterministic, high-trust.
- **Non-verifiable:** LAB's rubric-criterion LLM-judge + human/preference signal on
  analysis quality, completeness, risk framing. Expensive, noisy, but where legal value lives.

The **same rubric-failure text** already drives the Phase-1 harness loop; the extension is to
also use it as a reward for T3 consolidation (GRPO/DPO-style on an OSS model) once we move
off closed APIs.

### A.3 Object-store-backed low-latency sandboxes

Agent turns are **I/O-bound with multi-second LLM gaps**. That gap is free time. Design:

- Session filesystem/kernel state lives in an **object store** (S3/MinIO), not pinned RAM.
- On resume, **lazy page-in / rehydrate hides behind inference latency** — by the time the
  model returns the next action, the working set is warm.
- Idle sessions cost object-store bytes, not compute — the unit economics that make
  **long-running** agents affordable at scale.

This yields two archetypes on one substrate: **low-latency** (kept warm, interactive) vs.
**long-running** (checkpoint to object store, rehydrate on event) — a control-plane /
execution-plane split where **isolation is bought per-tenant** (Firecracker/gVisor/Kata microVMs)
rather than shared.

### A.4 OSS models + multi-LoRA serving for scale

Once T3 exists, per-tenant / per-practice-area adapters are the moat and the scaling story:

- **S-LoRA / Punica-style multi-LoRA serving:** one base model, thousands of cheap swappable
  adapters — serve 1M customers' personalized "engrams" without 1M full models.
- **Catastrophic-forgetting mitigations:** per-domain adapters, replay buffers from T2, EWC.
- The accumulating, reversible, per-tenant T2→T3 memory is the **data flywheel** — the
  defensible asset a faithful clone cannot replicate.

### A.5 Phased sequencing (research → product)

1. **Phase 1 (this spec):** T1+T2 text memory, GEPA offline, closed APIs, LAB curve. *Ships now.*
2. **Phase 2:** Rust runtime (C-scope §12) **+** swap to an OSS base model, add **T3 LoRA
   consolidation** with the hybrid reward. Distributed-systems + RL contribution.
3. **Phase 3:** object-store sandboxes + multi-LoRA serving; the commercial platform.

Each phase is independently shippable and resume-worthy.

---

## 16. References

- Prime Agent breakdown (source PDF), grounded in `PrimeIntellect-ai/prime-agent`.
- Recursive Language Models (Zhang, Kraska, Khattab, 2025).
- `dspy/predict/rlm.py`, `dspy/teleprompt/gepa/gepa.py` — stanfordnlp/dspy.
- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (arXiv:2507.19457).
- Harvey LAB (Legal Agent Benchmark) — `harveyai/harvey-labs` (MIT), harvey.ai blog.
- Harvey BigLaw Bench — harvey.ai blog / `harveyai/biglaw-bench`.
```
