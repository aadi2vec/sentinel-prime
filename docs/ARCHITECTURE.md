# SentinelPrime — Architecture & Benchmark Walkthrough

SentinelPrime is a DSPy-native, **online / label-free / reversible** self-improvement
harness for a legal-reasoning agent. The thesis in one line: *DSPy optimizes offline and
irreversibly; regulated legal reasoning needs online learning that is reversible and
auditable.* Every self-improvement edit becomes a provenance-scoped, verifier-gated,
reversible entry in an append-only audit ledger a compliance officer can read.

This document is a top-to-bottom walkthrough of the system, how the pieces fit, how to
set it up, and how to run the LAB benchmark loop.

---

## 1. The two loops and the two caches

Two things run at different timescales, and it is worth separating them up front because
almost every component belongs to exactly one:

```
                     ┌───────────────────────── ONLINE (per task, no labels) ─────────────────────────┐
  task ─► PrimeAgent.run_task ─► dspy.RLM loop ─► trajectory ─► Feedback(rubric) ─► harness.refine()
             ▲            │                             │                                   │
             │        guidance                     SubQueryCache                       (propose→verify→
        harness.read() (gated,                  (intra-run dedup)                        apply→audit)
         deterministic prefix)                                                               │
             └───────────────────────────────── ledger + audit log ◄─────────────────────────┘

                     ┌───────────── OFFLINE (labeled subset) ─────────────┐
                     dspy.GEPA / MIPRO tunes the named predictors
                     (propose, verify, generate_action, extract)
```

- **Online learning loop** — `refine()` runs after any live task using only the trajectory
  and the rubric-failure *text*. No gold labels. This is the gap vs. `dspy.GEPA`, which is
  offline and needs a labeled trainset.
- **Offline optimization loop** — every LM-calling node in the system is an ordinary
  `dspy.Predict` (proposer, verifier, RLM's generate_action/extract), so `named_predictors()`
  exposes them and GEPA/MIPRO can tune the online machinery itself.

And two *distinct* cache layers that people conflate at their peril:

- **`SubQueryCache`** — intra-run dedup of the RLM's `llm_query` sub-calls. Bounded to one
  trajectory; blast radius is one run. Exact by default, semantic on request (§4).
- **`ReuseController`** — *cross-task* admission of a prior conclusion. Carries compliance
  risk (a near-match can be wrong; the world moves), so it is a gated controller, not a
  cache (§5).

---

## 2. Component map

| Module | Role | Loop / layer |
|---|---|---|
| `memory.py` | `MemoryBackend` protocol + `JsonMemoryBackend` — versioned WAL + copy-on-write ledger (`snapshot`/`rollback`). | ledger substrate |
| `audit.py` | `AuditRecord` (the product atom) + append-only, content-addressed `AuditLog`; `explain()` renders the compliance chain. | audit |
| `harness.py` | `ContinualHarness` — `read()` (supplemental prompt block), `refine()` (propose→verify→apply→audit), `rollback()`, `explain()`. | online |
| `verifier.py` | `PredictVerifier` — GEPA-optimizable admission gate; edits must be *justified* before entering the ledger. | online |
| `reuse.py` | `ReuseController` — three-gate admission (scope-valid ∧ causally-current ∧ verifier-admitted) for cross-task reuse. | cross-task |
| `planner.py` | `CostLadderPlanner` — orders verification checks cheapest-detection-first, short-circuits; ordering driven by ledger statistics. | verification |
| `monitor.py` | `ProgressMonitor` — detects thrash (reasoning stall + cache hit-rate spike), records a `replan` audit event. | online |
| `subcache.py` | `SubQueryCache` — intra-run sub-query dedup, exact + optional semantic. | intra-run |
| `agent.py` | `PrimeAgent` + `CachingRLM` — wires the RLM per-turn reasoning to the ledger; `run_task`/`learn`. | online |
| `interpreter.py` | `InterpreterFactory` / `LocalInterpreter` — workdir-confined code execution for the RLM REPL. | runtime |
| `children.py` | `ChildSessionManager` — non-blocking `spawn_child` sub-agent admission. | runtime |
| `session.py` | `SessionStore` — per-session persistence. | runtime |
| `feedback.py` | `Feedback` + `parse_lab_result` — turns a LAB rubric result into the label-free signal `refine()` consumes. | eval adapter |
| `config.py` | Swappable `dspy.LM` roles (`root_lm`, `sub_lm`, `reflection_lm`, `judge_lm`). | config |

---

## 3. The refine() pipeline (the heart of the system)

`ContinualHarness.refine(trajectory, feedback)` is where a run's failures become a durable,
reversible, audited lesson. The protocol, in order:

1. **Snapshot BEFORE.** `backend.snapshot()` — `before.number` is the exact restore point
   that undoes this whole round.
2. **Propose (LM, label-free).** `ProposeLedgerEdits` reads `trajectory_summary`,
   `rubric_failures` (the failure *text* only), and the current ledger, and returns a JSON
   list of create/update/delete ops. No gold answers enter here.
3. **Verify / admit (optional gate).** When a `verifier` is configured, each proposed edit
   must clear `PredictVerifier` (admit == yes ∧ score ≥ min_score). Rejected edits never
   touch the ledger and never produce an audit record. The verdict's justification is
   threaded into the audit trail.
4. **Apply (deterministic, no LM).** `_apply_edits` upserts/deletes against the backend and
   classifies each op as created/updated/deleted against the pre-batch id set.
5. **Snapshot AFTER + audit.** `backend.snapshot()` closes the reversibility window
   `before..after`; `_emit_audit` writes one provenance-scoped `AuditRecord` per admitted
   edit (with `content_hash` dedup, `trajectory_digest`, and the verifier justification).

Reversibility: `rollback(before.number)` restores the ledger exactly. Explainability:
`explain(version)` renders `failure → verification → ledger_edit → reversibility` for the
edits at that version.

### read() — the supplemental prompt block

`read()` serializes the ledger into an *additional* guidance block that is prepended to the
task prompt; the base task prompt is immutable. Two guarantees matter:

- **Reuse gating** — with a `context`, each item is passed through `ReuseController`; provably
  stale external conclusions are dropped, intrinsic (document-invariant) ones are kept.
- **Prefix-cache determinism** — items are emitted in a pinned `(kind, id)` order so the
  block is a byte-stable prompt prefix across reads. The backend does not promise an order,
  so `read()` pins it — otherwise provider prompt caching silently invalidates.

---

## 4. Sub-query cache: exact → semantic

Inside one RLM run, `llm_query`/`llm_query_batched` re-hit the sub-LM for overlapping
document chunks with zero memory. `SubQueryCache` wraps those tools with:

- **Exact tier (always on):** content-hash dedup, dependency-free.
- **Semantic tier (opt-in):** pass an `embedder`; prompts within `similarity_threshold`
  cosine reuse a prior answer, counted separately as `semantic_hits`.

```python
from sentinelprime.agent import PrimeAgent
# exact-only (default) — hermetic, no embedding dependency
agent = PrimeAgent(harness, root_lm=lm)
# semantic dedup — near-duplicate chunk questions collapse
agent = PrimeAgent(harness, root_lm=lm, subquery_embedder=my_embed_fn)
```

Why intra-run and not cross-task: near-duplicate chunk questions are the dominant waste
*within* a trajectory and the blast radius is one run, so a cosine threshold is a reasonable
heuristic there. Cross-task reuse is a compliance decision — that is the `ReuseController`'s
job, with scope/currency/verifier gates that a bare cosine cannot provide.

---

## 5. Cost-ladder planner & progress monitor (the newest layers)

**`CostLadderPlanner`** — verifying an edit is asymmetric: a deterministic probe ("does the
cited span exist? does the citation resolve?") is orders of magnitude cheaper than the
generative verifier, and catches most rejections. The planner runs checks as a *conjunctive
short-circuit ladder*, ordered by `cost / P(fail)` ascending (best detection-per-dollar
first). The failure probabilities come from the audit log — `from_audit_log(...)` derives
each check's empirical failure rate from how often each rubric criterion appears in past
failures. Result: a strictly cheaper expected cost than an arbitrary (e.g. verifier-first)
order.

**`ProgressMonitor`** — the first slice of stop-and-rethink: *measurement before machinery*.
The shipped RLM loop is greedy single-path, so a wrong strategy gets elaborated until
`max_iters`. The monitor detects thrash from signals the run already emits — a reasoning
stall (consecutive `reasoning` steps near-identical by Jaccard overlap) and a `SubQueryCache`
hit-rate spike (re-asking answered questions) — and records a `replan` event onto the same
append-only audit log. It detects and records only; abandoning a subtree needs trajectory
checkpoints (see the plan's design notes), which this meter will later trigger.

---

## 6. Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"     # Python 3.10+
cp .env.example .env                   # add a provider API key
```

Models are swappable `dspy.LM` roles — nothing hardcoded. Copy `config.example.yaml` and
point `root_lm` / `sub_lm` / `reflection_lm` / `judge_lm` at any provider DSPy supports;
provider keys are read from the environment.

Run the test suite (fully hermetic — mocked LMs, no network):

```bash
.venv/bin/pytest -q
```

Live smoke test against a real LM (auto-detects provider from the API key present):

```bash
.venv/bin/python scripts/smoke_live.py
```

---

## 7. Running the LAB benchmark loop

The target evaluation is Harvey **LAB** (Legal Agent Benchmark), M&A due-diligence slice.
The benchmark measures a **self-improvement curve**: does the pass-rate rise across a
sequence of tasks *because* the online ledger accumulates and gates lessons?

**Status of the pieces:**

- ✅ *Eval adapter* — `parse_lab_result(raw)` converts a LAB rubric result
  (`{task_id, criteria:[{id, passed, reason}]}`) into the label-free `Feedback` the harness
  consumes; `score` is the pass-rate.
- ✅ *Agent runtime* — `PrimeAgent.run_task` (RLM + workdir-confined interpreter) and
  `PrimeAgent.learn` (→ `refine`).
- ✅ *Self-improvement loop harness* — `scripts/run_lab.py` drives the run→feedback→refine
  cycle over a task set and reports the pass-rate curve.
- ⏳ *Dataset* — the LAB M&A slice itself is external. `scripts/run_lab.py` ships with a tiny
  built-in fixture set so the loop is runnable today; point `--tasks <dir>` at the real slice
  when available.

The loop each task iteration performs:

```
for task in tasks:
    guidance = harness.read(context=task.context)   # gated + deterministic prefix
    pred     = agent.run_task(task.prompt, workdir=task.workdir)
    result   = grade(pred, task.rubric)             # -> LAB rubric result dict
    fb       = parse_lab_result(result)
    monitor.check_and_record(trajectory, agent.last_cache_stats, audit_log, version, task.id)
    agent.learn(trajectory, fb)                     # refine(): propose→verify→apply→audit
```

Run it on the built-in fixtures:

```bash
.venv/bin/python scripts/run_lab.py                 # scripted LM, no network
.venv/bin/python scripts/run_lab.py --live          # real LM (needs an API key)
```

The script prints, per task, the pass-rate and the ledger deltas (created/updated/deleted),
then the final self-improvement curve and the audit-log path so any edit can be inspected
with `harness.explain(version)`.

---

## 8. What is deliberately parked

Vector clocks (single writer → length-1 today), Byzantine quorums, trajectory-level
checkpoints/backtracking, and a Pareto-under-uncertainty strategy frontier are designed but
not built — see `docs/superpowers/plans/2026-09-05-sentinelprime-audit-ledger-plan.md`
("Design notes" and "Parked"). The Rust durable-execution runtime (WAL + supervision +
deterministic replay — the audit-guarantee-under-failure layer) is Phase 2.
```
