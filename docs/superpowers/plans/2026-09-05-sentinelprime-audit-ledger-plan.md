# SentinelPrime — Auditable Reasoning Ledger Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. TDD throughout; mock LMs, no network calls in tests. Run tests with `.venv/bin/pytest`.

**Positioning (the product):** Not "a self-improving agent." An **auditable reasoning ledger for regulated decisions**. The already-shipped `ContinualHarness` (reversible, online, label-free) is a write-ahead log with copy-on-write restore — the one property GEPA/MIPRO structurally cannot offer. Verification, compliance, and long-term reasoning reduce to one question: *can you prove why the system concluded what it concluded, show the guidance it used was valid at the time, and reconstruct that state later?* Reversibility is what makes all three answerable.

**One-line thesis:** the audit ledger is read two ways — as **provenance** (`explain()` establishes trust) and as **statistics** (a planner minimizes cost). Cost optimization and verification are the same ledger, read twice.

## Scope discipline (what this plan does and does NOT build)

The source design doc (`SentinelPrime → auditable reasoning ledger`) is adopted for *positioning*. Its architecture is over-scoped and over-analogized for implementation. This plan translates only the load-bearing, presently-buildable core and sequences the rest honestly.

**Build now (Tasks 1–3):**
- `AuditRecord` emitted on every `refine()` — the product's atom.
- `harness.explain(version)` — the causal-chain compliance UI.
- `SubQueryCache` — intra-run `llm_query` dedup (the "subquery stuff"), with hit/miss stats.

**Roadmap, designed but not built here (Section: Roadmap):** `ReuseController` (auditable semantic cache), verifier admission-control on `refine()`, cost-ladder planner over verifier levels, Rust durable runtime.

**Explicitly parked (do NOT implement):**
- **Vector clocks / "version vector".** The harness has one sequential writer; a version vector is length-1 (an integer) until multiple mutable sources exist. Revisit only when playbook/jurisdiction/counterparty become independent versioned inputs.
- **Byzantine 3f+1 consensus, LLM-debate quorum.** Pitch-deck material; not a demo component.
- **"Nash bargaining" for compute allocation.** It's knapsack-by-value-density; do not name it Nash, do not build a bargaining solver.
- **No class named after Ctesibius/Selinger/Talmud etc.** Citations live in docs, never in identifiers.

## Global constraints

- Python 3.10+. New code under `sentinelprime/`. Tests under `tests/`.
- All models are swappable `dspy.LM` instances passed by the caller.
- **Immutability invariant preserved:** the audit layer is *additive* — it observes `refine()`, it never mutates the base prompt or changes ledger semantics.
- **Reversibility preserved:** every `AuditRecord` carries the `(from_version, to_version)` window so `rollback(from_version)` still undoes exactly one round.
- The audit log is append-only. Records are never edited or deleted (a rolled-back edit is a *fact that happened*).

---

## Task 1: `AuditRecord` + emission on `refine()`

Every self-improvement edit becomes a reconstructable, provenance-scoped fact. `refine()` already computes everything needed — the version window (`before.number`/`after.number`), the cause (`feedback` + trajectory), and the touched ids (`_apply_edits`). This task captures that into an append-only log.

**Files:** create `sentinelprime/audit.py`, `tests/test_audit.py`; modify `sentinelprime/harness.py`.

**Interfaces:**
- `AuditRecord` dataclass:
  - `edit_id: str` — the ledger item id touched.
  - `op: str` — `"create" | "update" | "delete"`.
  - `scope: str` — provenance scope: `"intrinsic"` (depends only on document invariants → reusable across sessions) vs `"external"` (depends on mutable state → must be re-derived). Default: map from the item's `scope` meta (`global`→intrinsic candidate, `session`→external) but allow the edit op to set it explicitly via `meta["provenance"]`. Store what was given; do not infer cleverly.
  - `from_version: int`, `to_version: int` — the reversibility window (rollback point).
  - `cause_task_id: str`, `cause_failures: str` — verbatim `feedback.task_id` and `feedback.as_text()`.
  - `trajectory_digest: str` — `sha256` of the JSON trajectory (bounded), for provenance without storing the whole thing.
  - `score: float` — `feedback.score` (the calibrated pass-rate scalar; later, the verifier logit-expectation).
  - `created_at: str` (ISO-8601 UTC).
  - `content_hash: str` — `sha256` over `(op, edit_id, text, cause_task_id)` → content-addressed id. Identical edits collapse to one `content_hash` (CSE/memoization: dedup, do not re-append duplicates).
- `AuditLog` (append-only): `append(record) -> None`, `records() -> list[AuditRecord]`, `by_version(version) -> list[AuditRecord]` (records whose window ends at `version`), `seen(content_hash) -> bool`. Persists to a JSON path alongside the memory backend; loads on init.

- [ ] **Step 1: Write the failing test** (`tests/test_audit.py`)

```python
import json
from sentinelprime.audit import AuditRecord, AuditLog


def test_append_and_roundtrip(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    rec = AuditRecord(
        edit_id="checklist.change_of_control", op="create", scope="intrinsic",
        from_version=4, to_version=5, cause_task_id="ma-001",
        cause_failures="- [c1] missed change-of-control clause",
        trajectory_digest="abc", score=0.88, created_at="t",
        content_hash="h1",
    )
    log.append(rec)
    # survives reload
    reloaded = AuditLog(str(tmp_path / "audit.json"))
    assert reloaded.records()[0].edit_id == "checklist.change_of_control"
    assert reloaded.by_version(5)[0].op == "create"


def test_content_hash_dedup(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    assert log.seen("h1") is False
    r = AuditRecord(edit_id="k", op="create", scope="intrinsic", from_version=1,
                    to_version=2, cause_task_id="t", cause_failures="", trajectory_digest="d",
                    score=1.0, created_at="t", content_hash="h1")
    log.append(r)
    assert log.seen("h1") is True
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/pytest tests/test_audit.py -q` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `sentinelprime/audit.py`** — the `AuditRecord` dataclass + `AuditLog` (JSON load/persist, `append` is a no-op if `content_hash` already `seen`).

- [ ] **Step 4: Wire into `harness.refine()`** — the harness gains an optional `audit_log: AuditLog | None`. Inside `refine()`, after `_apply_edits`, for each edit build an `AuditRecord` (compute `content_hash`, `trajectory_digest`) and `append` it. Guard on `audit_log is not None` so existing Plan-A/B behavior is unchanged when no log is supplied. `refine()` still returns the same `RefineResult`.
  - Failing test: pass an `AuditLog` to a `ContinualHarness`, run `refine()` with a mock `propose` returning one create-op, assert one `AuditRecord` with the correct `from_version`/`to_version`/`cause_failures` landed in the log.

- [ ] **Step 5: Run full suite + commit** — `.venv/bin/pytest -q`.
```
git add sentinelprime/audit.py sentinelprime/harness.py tests/test_audit.py
git commit -m "feat: AuditRecord + append-only audit log emitted on refine()"
```

---

## Task 2: `harness.explain(version)` — the compliance UI

Reconstruct a ledger edit's derivation as a chain a compliance officer reads: `failure → (verifier justification) → ledger edit → reversibility point → later reuse`. This is the product's core UI and its cheapest demo.

**Files:** modify `sentinelprime/audit.py` (add `explain`), `sentinelprime/harness.py` (delegate); test in `tests/test_audit.py`.

**Interface:** `explain(version: int) -> str` — renders the record(s) at that version as an aligned, labeled block. Verbatim worked shape:

```
[failure       ] task ma-001: missed change-of-control clause
[ledger_edit   ] create 'checklist.change_of_control' (v4->v5, scope=intrinsic, id=9da2...)
[reversibility ] rollback(from_version=4) restores prior ledger state exactly
```

The `[verification]` and `[reuse]` lines are appended later (Tasks in Roadmap) — `explain` renders whatever fields are present, so it degrades gracefully now.

- [ ] **Step 1: Failing test** — build a log with one record at `to_version=5`, assert `explain(5)` contains `ma-001`, `create`, `v4->v5`, `scope=intrinsic`, and a `rollback(from_version=4)` line.
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement `AuditLog.explain(version)`; add `ContinualHarness.explain(version)` delegating to the log.**
- [ ] **Step 4: Run + commit** — `feat: harness.explain(version) — causal-chain compliance view`.

---

## Task 3: `SubQueryCache` — intra-run `llm_query` dedup (the subquery stuff)

Distinct layer from the ledger. Inside a single RLM run, `llm_query`/`llm_query_batched` (dspy `rlm.py:298-323`) re-hit the sub-LM for overlapping document chunks with zero memory. This wraps those tools with a content-hash cache and hit/miss counters, so the demo can show a token/call reduction. This is the *measurement* piece that later justifies the planner.

**Files:** create `sentinelprime/subcache.py`, `tests/test_subcache.py`.

**Interfaces:**
- `SubQueryCache`:
  - `wrap(tools: dict[str, Callable]) -> dict[str, Callable]` — takes the `{"llm_query", "llm_query_batched"}` dict RLM injects and returns wrapped versions that consult the cache first (exact `sha256(prompt)` key). `llm_query_batched` dedups within the batch and across prior calls.
  - `stats() -> dict` — `{"hits": int, "misses": int, "calls": int}`.
  - Optional `near_dup` hook (embedding cosine) left as a no-op stub with a clear TODO — do not pull in an embedding dep now.
- Integration seam: the wrapped tools are injected via the same mechanism `LocalInterpreter`/`InterpreterFactory` already use. A follow-up wires `PrimeAgent` to install the cache per run; **this task only builds and unit-tests the cache in isolation** (no RLM/LM needed — pass a fake `llm_query`).

- [ ] **Step 1: Failing test** — fake `llm_query = lambda p: f"ans:{p}"`; wrap; call twice with the same prompt; assert one underlying call, `stats()["hits"] == 1`; batched call with a repeated prompt dedups.
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement `SubQueryCache`.**
- [ ] **Step 4: Run + commit** — `feat: SubQueryCache — content-hash dedup for RLM sub-LLM calls`.

---

## Roadmap (designed, not built in this plan)

Each item has an acceptance criterion so a later plan can pick it up directly. Ordered by leverage.

1. **`ReuseController.should_reuse(record, current_context)` — auditable semantic cache.** ✅ *Built.* `sentinelprime/reuse.py` runs the three gates in order (scope-valid ∧ causally-current[stub] ∧ verifier-admitted) and returns a reasoned `ReuseDecision`; wired into `harness.read(context=…)` via `_admissible`. Tests in `test_reuse.py` + `test_harness.py` read-gating suite.
2. **Verifier admission-control on `refine()`.** ✅ *Built.* `sentinelprime/verifier.py` — `PredictVerifier` (GEPA-optimizable `dspy.Predict` over `VerifyLedgerEdit`) returns a `VerifierVerdict(admitted, score, justification)`. `ContinualHarness(verifier=…)` gates each proposed edit; rejected edits never touch the ledger and never produce an audit record. The verdict justification is threaded into `AuditRecord.verification` and rendered by `explain()` as the `[verification]` line. Tests: `test_verifier.py` (4) + refine-gating tests in `test_audit.py`.
3. **Cost-ladder planner over verifier levels.** ✅ *Built.* `sentinelprime/planner.py` — `CostLadderPlanner` orders checks by `cost / P(fail)` ascending (best detection-per-dollar first), short-circuits on the first failure. `from_audit_log` derives each check's empirical failure rate from `criterion_failure_counts` over the ledger. `expected_cost` computes the short-circuited-conjunction cost; the planner's order is strictly cheaper than an arbitrary static order. Tests: `test_planner.py` (5).
4. **Prefix-cache discipline (off-the-shelf lever, one guardrail).** ✅ *Built (guardrail).* `harness.read()` now emits items in a pinned `(kind, id)` order, so the block is a byte-stable prompt prefix independent of backend insertion order. Tests: `test_read_is_deterministic_regardless_of_insertion_order`, `test_read_of_unchanged_ledger_is_byte_identical`. (Structuring the full immutable-base+ledger prefix for a specific provider's cache remains a wiring task.)
5. **Rust durable runtime (Phase 2) as the audit-guarantee-under-failure layer.** WAL + supervision + deterministic replay — sold as "the record survives a crash and replays exactly," the compliance moat, not engineering hygiene.

## Integration gaps (built in isolation, not yet wired into the live loop)

These components are shipped and unit-tested, but the *live* `PrimeAgent` path does not yet
exercise them end to end. Recorded here (post PR #2 merge, 2026-09-06) so a later plan can close
each seam. None require new algorithms — they are wiring + threading.

**Status (2026-09-06, Track A): all five closed.** `scripts/run_lab.py` carries
`--no-verifier` / `--no-monitor` / `--no-context` / `--no-semantic-cache`, and the fixtures are
built so each switch moves a measured number (curve, audit records, replans, rejected,
withheld, prefix boundary). What is *not* closed is endpoint-side measurement: a real
cached-token/cost reduction and a real `semantic_hits > 0` both need a live provider.

- [x] **Thread `context` through `PrimeAgent.run_task` → `harness.read(context=…)`.** `read()` already
  accepts a `context` and gates reuse through `ReuseController._admissible`, but `PrimeAgent` calls
  `read()` with no context, so the currency/scope gates never see real task state. Acceptance: a
  live run passes task/document identifiers into `read`, and an out-of-scope cached lesson is
  demonstrably withheld.
  *Closed:* `PrimeAgent.run_task(..., context=…)` forwards to `harness.read(context=…)`;
  `ContinualHarness.admissible_items()` exposes the gated set so the withheld count is
  measurable; `run_lab.py` amends a document mid-run so a revision-scoped lesson goes stale
  (5 withheld with gating on, 0 with `--no-context`).
- [x] **Wire the `CostLadderPlanner` into `refine()`'s verifier step.** The planner exists and derives
  failure rates `from_audit_log`, but `refine()` calls the verifier directly per edit. Seam: order the
  verifier's checks (or multiple verifier levels) via the ladder so the cheapest high-detection check
  short-circuits first. Acceptance: a refine round with two verifier levels runs them in ladder order
  and stops on first rejection, with the order recorded.
  *Closed:* `LadderVerifier` presents `refine()`'s single-verifier interface and fans out over
  priced `VerifierLevel` rungs ordered by `CostLadderPlanner`. The live ladder is
  `GroundingProbe` (deterministic, cost 1) then `PredictVerifier` (cost 100); the executed
  order, the rejecting rung and the cost spent land in the `AuditRecord.verification` and so in
  `explain()`. `RefineResult.rejected` makes the gate's work countable. `from_audit_log`
  fits ladders whose rungs map to rubric criteria; since these rungs do not, the live ladder
  runs `adaptive=True` and re-derives P(fail) from its **own** observed rejections — rejected
  edits leave no audit record, so the ladder is the only component that sees them.
- [x] **Wire `ProgressMonitor` into the live loop (not just `run_lab.py`).** `PrimeAgent.run_task` should
  feed the real RLM trajectory + `SubQueryCache` stats to `monitor.check_and_record` each run so a
  `replan` audit event is emitted on live thrash. Currently only the scripted/live `run_lab` harness
  calls it. Acceptance: a looping live trajectory produces a `replan` `AuditRecord`.
  *Closed:* `PrimeAgent(monitor=…)`; `run_task` feeds `pred.trajectory` + `last_cache_stats`
  to `monitor.check_and_record` and exposes `last_trajectory` / `last_monitor_decision`, so
  the live loop learns from the real trajectory instead of `[]`.
- [x] **Prefix-cache: full immutable-base + ledger prompt structuring for a provider cache.** The
  `read()` byte-stability guardrail is in; the remaining work is structuring the actual prompt so a
  provider (e.g. OpenAI/Anthropic prompt caching) can cache the stable prefix. Acceptance: repeated
  runs over an unchanged ledger show a measured cached-prefix token/cost reduction.
  *Closed structurally:* `PrimeTask` now declares `guidance` before `task`, so the prompt is
  `[instructions][ledger][task][repl_history]` and the ledger falls inside the shared prefix —
  with the original order the prompt diverged before the ledger was reached, so it could never
  be cached. `agent.cacheable_prefix()` measures the divergence boundary from the real adapter
  messages (798 chars shared across the two fixtures); `run_lab.py` prints it each run.
  *Still open:* the token/cost half of the acceptance needs a live provider, as do
  provider-specific cache markers (Anthropic `cache_control`), which sit in litellm/DSPy.
- [x] **Semantic sub-cache embedder in the live path.** `SubQueryCache` accepts an injectable embedder
  (default None = exact-only). Live `PrimeAgent` currently passes no embedder, so only exact dedup
  runs. Acceptance: a live run with a real embedder shows `semantic_hits > 0` on paraphrased subqueries.
  *Closed:* `subcache.make_embedder` adapts a batch embedder (`dspy.Embedder`) to the cache's
  one-prompt contract; live runs build it from `EMBED_MODEL` by default (`--no-semantic-cache`
  ablates). `SubQueryCache` disables its semantic tier with a warning if the embedder raises,
  so a dead embeddings endpoint costs the run its dedup *rate*, never the run.
  *Still open:* `semantic_hits > 0` against a real endpoint — proven hermetically only.

The two larger design items below (trajectory checkpoints/backtracking, Pareto-under-uncertainty
strategy frontier) remain **parked** — they need new machinery, not just wiring.

## Design notes: stop-and-rethink & strategy frontier (roadmap)

The shipped `dspy.RLM` loop is greedy single-path: it appends to a monotone
`REPLHistory` (rlm.py:727-733) with no progress signal and no backtrack, so a wrong
strategy gets *elaborated* until `max_iters` forces an extract. Addressing this is the
exploration half of the token-bleed problem. Build in this order — each prerequisite is
strict:

1. **Progress/value signal (prerequisite for any stopping rule).** No stop is possible
   without a "getting warmer?" measure. Ranked options: (a) self-rated confidence — cheap
   but a stuck model is confidently wrong, weak alone; (b) a cheap verifier probe on partial
   state (the asymmetric cost ladder reused as a progress meter) — the honest one; (c)
   loop/novelty detection — a spiking `SubQueryCache` hit-rate already signals the agent is
   re-asking answered questions (spinning), plus embedding-similarity of consecutive
   `reasoning` steps.
2. **Trajectory-level checkpoints.** "Rethink" without checkpoints = restart from zero. Lift
   the ledger's WAL + copy-on-write (snapshot/rollback) from the ledger level to the *trajectory*
   level: snapshot REPL namespace + history at decision points so a subtree can be abandoned
   and a different strategy tried from a branch point. Greedy-no-backtrack becomes DFS-with-backtracking.
3. **Trigger policy (optimal stopping).** A watchdog raising a `replan` interrupt on: value
   flat over a window, loop detected, or budget-fraction spent with no value gain. **Default
   greedy; escalate to search only on a trigger** — the greedy loop is cheap *because* it
   commits, so cheap monitors run every turn and expensive branch/replan fires only on threshold.

**Pareto frontier of strategies (the GEPA question) — coherent, with one crux.** A frontier
over *un-executed* strategies is only as good as the cheap estimator of each strategy's (value,
cost) *without running it*. GEPA can measure objectives because it is offline+labeled and evolves
a population against a trainset/metric (irreversible, in-place rewrite). At inference there are no
labels, so frontier coordinates are surrogates: expected cost from the audit-ledger statistics
(the Selinger catalog — which checks fail, how often, at what cost) and expected value from a cheap
verifier; the frontier is therefore *provisional*. Correct structure: planner enumerates N
candidate strategies → surrogate-score each on (expected value, expected cost) → keep the
non-dominated set **plus high-variance candidates** (Pareto-under-uncertainty = bandit/MCTS
territory; TraceMind's UCB1 bandit is reusable machinery) → execute in cost order under a probe
budget, updating estimates online and pruning as truth arrives. This is **not "GEPA at inference"**;
it is the online, label-free, *reversible* cousin — every prune/branch-switch sits on the reversible
ledger, so it is undoable and recorded.

**Product tie-in (why this is more than robustness):** every "abandon tree, replan" event is
premium audit content — "at iteration 7, plateau detected, rolled back to checkpoint at iteration 3,
switched strategy X→Y; here is the frontier snapshot and why X was dominated." That answers the
due-diligence question a client sues over ("why didn't you pursue Z?" → "we did; it was dominated;
here's the record"), and folds straight into `explain()`.

**First slice (measurement before machinery):** ✅ *Built.* `sentinelprime/monitor.py` — `ProgressMonitor`
detects thrash from signals already emitted (`SubQueryCache` hit-rate spike + consecutive-`reasoning`
Jaccard similarity over a window) and `check_and_record` appends a `replan` `AuditRecord` on trigger.
It detects/records only; abandoning a subtree needs trajectory checkpoints (still parked). Tests:
`test_monitor.py` (5) — fires on a looping trajectory, stays quiet on a healthy one.

## The pitch, one line

DSPy optimizes offline and irreversibly; regulated legal reasoning needs online learning that is reversible and auditable. SentinelPrime is the write-ahead reasoning ledger — provenance-scoped, verifier-gated, with a reuse controller that reads the same ledger for both cost and trust — that makes a self-improving legal agent something a compliance officer can sign off on.
