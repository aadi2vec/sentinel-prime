# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # setup (Python 3.10+)
.venv/bin/pytest -q                                          # full suite (hermetic, no network)
.venv/bin/pytest tests/test_policy_search.py -v              # one file
.venv/bin/pytest tests/test_harness.py::test_read_empty_ledger_returns_empty_string  # one test
.venv/bin/python scripts/policy_lab.py --out lab_runs/<new>  # policy search, scripted, no network
.venv/bin/python scripts/policy_lab.py --live --model PROVIDER/MODEL \
    --rounds 2 --cases 3 --out lab_runs/<new>                # same loop, real proposer
```

`--out` must name a directory that does not exist: a run that resumed into a used archive
could silently re-read a final test it had already spent. `policy_lab.py` does **not** load
`.env`; put the provider credentials in the process environment for `--live`.

Always use `.venv/bin/...` — there is no runner script or task file. No linter/formatter is
configured. `graveyard/` holds retired code and is excluded from collection via
`testpaths` in `pyproject.toml`; read `graveyard/README.md` before restoring anything.

## Architecture

SentinelPrime is an **agent-systems research harness**: can an RLM improve the machinery it
uses to solve problems — decomposition, context allocation, verification, recovery,
delegation — by proposing and testing bounded changes to its own execution policy?

Harvey LAB (M&A due-diligence slice) is a **demanding evaluation environment, not a product
direction**. Legal is the stress test. Do not reintroduce product framing; that direction was
retired on 2026-09-07 and its plan is in `graveyard/docs/`.

The current mutable surface is deliberately one thing: **which verification checks run, and
whether a failed check triggers another solver attempt**. Everything else in this repo —
the ledger, the audit trail, the admission ladder, credit assignment — is *supporting
machinery and instrumentation* for that search, not the thesis.

Read `docs/plans/2026-09-07-rlm-policy-evolution-plan.md` (plan of attack) and
`docs/superpowers/specs/2026-09-07-execution-policy-search-design.md` (design) first. They
are the only two live documents.

### The policy search (`policy_search.py`, `policy_experiment.py`, `scripts/policy_lab.py`)

A candidate `Policy` selects registered checks and a bounded revision count. It **cannot**
supply code, alter a check's implementation, change the evaluator, or raise the `Budget`.
`Policy.parse` rejects unknown fields outright — that is the evaluation boundary as a type
rather than a convention, and it is the invariant the whole design rests on: a candidate that
can redefine success improves its score without improving its problem-solving.

`Case.gold` reaches the evaluator only, never solve/check/propose. Development, each
validation batch, and the final test are **family-disjoint**, reserved before work begins so
a restart cannot re-spend the test set. Incomplete evaluation cannot promote. `PolicyArchive`
versions every promotion with its evidence and supports `rollback(version)`.

**Every report carries a matched-compute control**, and it is the headline number.
`champion - initial` moves whenever the champion simply runs the solver more often;
`matched_compute_control(policy)` spends the same solver attempts with the diagnosis removed,
so `compute_adjusted_gain` is what checking actually bought. The control lives on
`runner.control_runner()` — a sibling the candidate runner has never heard of, so `validate`
refuses any policy naming `CONTROL_CHECK`. Do not move it into the candidate catalog: a
candidate that can select an always-failing check wins by burning solver calls, which is the
confound being measured. An incomplete control reports `None`, never a fallback to the
unadjusted gain.

Two honest limits remain, kept in the docstrings: the checkers are hand-written, not
synthesized; and the budget counts solver attempts and check calls, not tokens, so nothing
here establishes cost efficiency.

### Two loops, two caches — every module belongs to exactly one

- **Online loop (per task, no labels):** `PrimeAgent.run_task` → `dspy.RLM` turns → trajectory
  → `parse_lab_result` → `ContinualHarness.refine()` → ledger + audit log → next task's
  `harness.read()`.
- **Offline loop:** every LM-calling node is a plain `dspy.Predict` (`ProposeLedgerEdits`,
  `VerifyLedgerEdit`, RLM's generate_action/extract), so `named_predictors()` exposes them and
  GEPA can tune the *online machinery itself*. Preserve this when adding LM calls — do not
  hand-roll prompts outside a Signature.
- **`SubQueryCache`** (`subcache.py`) is *intra-run* dedup of RLM `llm_query` calls; blast
  radius is one trajectory, so a cosine threshold is acceptable there. **`ReuseController`**
  (`reuse.py`) is *cross-task* admission of a prior conclusion — a gated compliance decision,
  not a cache. Keep these two separate; conflating them is the main design hazard.

### Two logs, two purposes — the same split one level up

`AuditLog` records what *entered the ledger*: compliance, read by a human. `ProposalLog`
(`proposals.py`) records what the *verifier was asked about*, admitted or not: training data,
read by an optimizer. The invariant that rejected edits produce no audit record is exactly
what makes the audit log an all-positive dataset, so a gate fit to it learns to admit
everything — that is why the second log exists, and why the first one must not absorb it.
Same hazard as the two caches: do not conflate them.

`ProposalLog` is JSONL (appended live during a multi-hour sweep; a truncated trailing line is
dropped) while `AuditLog` is one JSON document. Outcome rows are appended, never edited —
`outcomes()` reports last-wins, so the file stays append-only.

### One assembly point (`assembly.py`)

`assemble(lm=..., root=...)` is the only place the system is wired, and every entry point
calls it — including `policy_experiment.py`, so a candidate policy is expressible only as
`assemble()` arguments and cannot reach around the assembler. Before it existed, six entry
points each hand-built an agent and had begun to drift apart; those runners are now in
`graveyard/scripts/`, but the rule they motivated is the reason this module exists.

Two rules keep it that way:

- **The default is the configuration the project claims** — gate, audit, credit, monitor,
  and proposal log all on. A thinner default is how five of six call sites ended up
  under-wired; nobody chose it, it was just the shortest constructor.
- **Ablation stays a flag, not a branch.** `gate=None` / `credit=False` / `record=False`
  switch one mechanism without taking a different assembly path that might differ in some
  second way nobody controlled for.

`gate` is `None | "probe" | "ladder"` — `"probe"` is the hermetic deterministic-only rung
set, which is what a no-network run needs. Two combinations are **refused rather than
degraded**: `credit=True, audit=False` (CreditAssigner reads targets off the AuditRecord, so
it would attribute nothing while reporting itself on) and `program=` with a non-ladder gate
(there is no judge to load the compiled prompt into, and ignoring it would report a tuned
run that used the untuned bar). A mechanism that reports as on while doing nothing produces
a null result that looks like evidence — worse than a crash.

`Assembly.describe()` renders the whole configuration on one line, machinery fingerprint
included, and every script logs it at start. Two runs of "the same" arm with different
compiled judges are not the same experiment, and the arm name alone does not say which.

### The refine() protocol (`harness.py` — the heart)

`snapshot BEFORE` → `propose` (LM, sees only trajectory + rubric-failure *text*) → optional
`verifier` admission gate → `_apply_edits` (deterministic, no LM) → `snapshot AFTER` →
`_emit_audit`. `RefineResult.from_version` is the exact restore point: `rollback(from_version)`
undoes the whole round; `RefineResult.rejected` is the only record of what the gate stopped
(rejected edits deliberately produce no audit record).

The `verifier` is normally a `LadderVerifier` (`verifier.py`): it presents the single
`verify(op, feedback, trajectory)` interface `refine()` expects and fans out internally over
priced `VerifierLevel` rungs that `CostLadderPlanner` orders by `cost / P(fail)`, stopping at
the first rejection. That is how the planner reaches the online loop without `refine()`
learning about ladders. Live rungs: `GroundingProbe` (deterministic, cost 1) then
`PredictVerifier` (generative, cost 100), with `adaptive=True` so the ladder re-derives
P(fail) from its own observed rejections — rejected edits leave no audit record, so the
ladder is the only thing that sees them. `CostLadderPlanner.from_audit_log` is the
alternative statistics source, and fits only ladders whose rungs map to rubric criteria.

### Credit assignment (`credit.py`)

The second defence: admission control asks whether an edit is *grounded*, credit assignment
asks whether it *helped*. A lesson is scored against the criterion ids the proposer declared in
`meta.targets` (recorded on `AuditRecord.targets`) over the tasks where it was actually
exposed, and retired inside `refine()`'s snapshot window when it stops clearing them.

Three things here are load-bearing and easy to break:
1. **Declared targets, not the round's failure text.** The fallback blames a lesson answering
   `c1` whenever `c2` fails — enough to retire good guidance. Keep `meta.targets` flowing.
2. **A bounded window, not a lifetime tally.** The environment is non-stationary; a lifetime
   average convicts a lesson for a period already fixed and can never recover.
3. **`forget()` on retire.** Without it a rewritten lesson inherits old blame and is retired on
   sight — the ledger oscillates instead of converging.

### Where the live loop attaches (`agent.py`)

`PrimeAgent.run_task(task, workdir, context=None, task_id="")` owns three seams:
`context` → `harness.read(context=…)` (reuse gating sees real task state); `pred.trajectory`
+ `last_cache_stats` → `monitor.check_and_record` (live thrash emits a `replan` audit event);
and `last_trajectory` → what the caller passes to `learn()`. Refine on the real trajectory,
never on `[]`. `last_exposed_ids` is the credit-assignment exposure set — the guidance actually
surfaced, not the whole ledger.

Sub-agents are opt-in (`enable_children=True`), lazily built on first spawn, and expose **both**
`spawn_child` and `collect_child`: spawning without collecting is what made them dead weight
before. dspy's tool bridge is `invoke(**kwargs)`, so tools must be called with **keyword
arguments** in the REPL (`spawn_child(task=...)`); a positional call raises. Children share the
parent's harness read-only and never call `learn()`, and `run_task` drains them so no worker
outlives the task that spawned it — the parent writes the ledger between tasks and a child
still reading it would race that write.

### Invariants that the whole design rests on — do not break

1. **Supplemental-only.** `read()` emits an *additional* prompt block; the base task prompt is
   never mutated. An empty ledger must return `""` so nothing is prepended.
2. **Reversible.** Every `refine()` brackets its edits between two backend snapshots.
3. **Label-free.** Only the trajectory and `feedback.as_text()` reach the proposer. No gold
   answers in the online path.
4. **Append-only audit.** `AuditLog` records are never edited or deleted — a rolled-back edit
   is still a fact that happened. Duplicate edits collapse by `content_hash`.
5. **Prefix-cache determinism.** `read()` sorts items before emitting; the backend promises no
   order, and an unstable order silently invalidates provider prompt caching. Field order in
   `PrimeTask` is part of the same guarantee: `guidance` is declared *before* `task` so the
   ledger sits inside the prompt prefix shared across tasks. Reordering those fields silently
   destroys cacheability — `agent.cacheable_prefix()` is the regression check.
6. **Learning happens only between tasks.** `PrimeAgent.learn()` is never called mid-task;
   `run_task` freezes the ledger snapshot at task start.
7. **Optional collaborators degrade to base behavior.** `audit_log`, `verifier`,
   `reuse_controller`, `monitor`, `proposal_log`, and `read(context=...)` are all `None`-able;
   when unset the code must behave exactly as before they existed (gating only ever *removes*
   provably-stale reuse). This is what makes ablation possible — do not make any of them
   mandatory.
8. **The optimizer is auditable too.** Every `AuditRecord` carries
   `machinery = machinery_fingerprint(harness)` — a digest of the proposer's and verifier's
   instructions and demos. GEPA rewrites exactly those, so without the stamp "prove the
   guidance was valid at the time" holds for the ledger and quietly fails one level up, at
   the prompt that admitted it. An object with no predictors fingerprints to `""` (an
   unstamped record), never to the digest of nothing — that would print a plausible hash for
   a run that had no prompts.

### Known gaps, deliberately deferred

Documented in module docstrings, not bugs to "fix" incidentally: no credit assignment (edits
are applied whether or not they later help), no relevance ranking in `read()` (it serializes
the whole ledger), no decay/dedup of lessons, `ReuseController.causally_current` is a stub
(single writer). Vector clocks, Byzantine quorums, and trajectory checkpoints are explicitly
**parked** — see the "Parked" section of the 2026-09-05 plan before building any of them.

### Security note

`interpreter.py` replaces DSPy's Deno/Pyodide WASM interpreter with an **in-process** Python
REPL so the RLM gets real filesystem and subprocess access. It is *not* a sandbox;
`_confine_path` is a convenience, not a boundary. Never point it at untrusted input.

## Conventions

- Tests are hermetic: no network, no real LM. Stub the LM seam by assigning over it —
  `harness.propose = _StubPropose(...)` / `_StubVerifier` (see `tests/test_audit.py`) — rather
  than mocking DSPy internals. TDD is the working style used throughout the plan docs.
- Module docstrings carry the *why* (the design argument and the honest limitations); inline
  comments mark the invariants. Match that density — it is the point of the repo, which is a
  portfolio/OSS piece where the reasoning is the deliverable.
- Citations and analogies belong in `docs/`, never in identifiers (no classes named after
  people or historical mechanisms).
- Models are always swappable `dspy.LM` instances passed in by the caller; nothing is
  hardcoded and `load_config` refuses to assume a default provider.
- `rlm/` contains only stale `__pycache__` from a vendored prototype — no source. Ignore it.
