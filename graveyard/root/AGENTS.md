# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # setup (Python 3.10+)
.venv/bin/pytest -q                                          # full suite (hermetic, no network)
.venv/bin/pytest tests/test_harness.py -v                    # one file
.venv/bin/pytest tests/test_harness.py::test_read_empty_ledger_returns_empty_string  # one test
.venv/bin/python scripts/run_lab.py                          # scripted self-improvement demo (no network)
.venv/bin/python scripts/run_lab.py --live --epochs 3        # same loop against a real LM
.venv/bin/python scripts/run_lab.py --no-verifier            # ablate one component (see below)
.venv/bin/python scripts/smoke_live.py                       # one-task live smoke test
.venv/bin/python scripts/gepa_runner.py --inspect            # is the trainset viable? no LM
.venv/bin/python scripts/gepa_runner.py --target verifier --out compiled/verifier.json
.venv/bin/python scripts/paired_ab.py --arm gated            # learning + untuned ladder
.venv/bin/python scripts/paired_ab.py --arm tuned            # + the compiled judge
```

`run_lab.py` takes `--no-verifier` / `--no-monitor` / `--no-context` / `--no-credit` /
`--no-children` / `--no-semantic-cache` (the last two are live-only). Each switch is an ablation: the
fixtures are built so the proposer emits an ungrounded lesson, the simulated agent thrashes
while unguided and *follows* bad guidance, and a document is amended mid-run — so turning a
component off changes the curve and the printed counts. Keep it that way; a fixture set where
ablation is a no-op cannot attribute the curve to a mechanism. Note that `_SimulatedAgent`
being misled is a *stipulation*, not evidence about real LMs — say so whenever you quote
those deltas.

Always use `.venv/bin/...` — there is no runner script or task file. No linter/formatter is configured.

Live runs need `.env` (copy `.env.example`); both scripts load it with their own tiny dotenv
reader and auto-detect the provider from whichever API key is present, preferring
`OPENAI_MODEL` (default `openai/gpt-5.6-luna`) with an optional `OPENAI_BASE_URL` proxy.
`config.yaml` is gitignored; `config.example.yaml` is the template.

## Architecture

SentinelPrime is a DSPy-native **online / label-free / reversible** self-improvement harness
for a legal-reasoning agent (target eval: Harvey LAB, M&A due-diligence slice). The thesis:
DSPy's optimizers (GEPA/MIPRO) are offline, labeled, and irreversible; this adds an online,
unlabeled, *reversible and auditable* loop on top. `docs/ARCHITECTURE.md` is the full
walkthrough; the design/plan docs live in `docs/superpowers/{specs,plans}/`.

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

### The offline loop (`optimize.py` + `scripts/gepa_runner.py`)

The claim is *not* "our predictors are GEPA-visible" (any DSPy module can say that). It is
that **the online loop manufactures the offline optimizer's training set, without labels**:
`refine()` writes a `ProposalRecord`, `credit()` writes an `OutcomeRecord`, and
`label_proposals` turns those into supervised examples nobody annotated.

Three things here are load-bearing:

1. **Labels are not probe/judge agreement.** Training the generative judge to agree with
   `GroundingProbe` collapses it into a 100x-more-expensive copy of the probe, and the
   ladder's second rung becomes dead weight. Labels come from the two places a verdict is
   *earned*: probe **rejections** (ungrounded by construction) and downstream credit
   outcomes. Probe *admissions* are deliberately left unlabelled.
2. **Precedence is probe over outcome.** A lesson that scored well downstream while
   ungrounded is the lucky-guess case; the deterministic signal outranks the outcome.
3. **A single-class trainset is refused, not scored.** `trainset_report().viable` is the
   guard — an all-positive log yields an excellent-looking score for a gate that admits
   everything.

Probe-rejection rows are drawn from proposals the ladder short-circuits, so by default the
judge is trained on a region it never faces. `LadderVerifier(explore=eps)` is the fix: on
that fraction of *rejected* rounds it runs the rungs behind the rejection anyway, recording
**shadow** verdicts that never touch the decision, never enter `last_verdicts`, and never
appear in the compliance trail (their cost is reported separately as
`last_exploration_cost`). Those rows join the judge's real input distribution at rate eps.
It also thaws the frozen P(fail) statistics the class docstring warns about — one mechanism,
both payoffs. Use `--explore 0.2` on a run whose purpose is to produce a trainset.

The residual gap is *measured*, not asserted: `TrainsetReport.on_distribution_rate` and
`.caveat` print next to every compile, `balance()` stops the cheap stratum swamping the
expensive one, and `stratified_scores()` splits a gain by where it came from — the pooled
mean is the number that can lie. `rollout_metric` remains the real downstream measure at one
agent run per candidate, injected so the expensive path stays opt-in.

### The experiment arms (`scripts/paired_ab.py`)

`control → learning → gated → tuned`, each adding exactly one mechanism to the previous:
the ledger, then the admission ladder, then the GEPA-compiled judge. `BASELINE` pins what
each is scored against — `tuned` vs `gated`, never `tuned` vs `control`, which would bundle
three mechanisms into one number and credit whichever the reader had in mind.

### One assembly point (`assembly.py`)

`assemble(lm=..., root=...)` is the only place the system is wired, and every script calls
it. Before it existed, six entry points each hand-built an agent: `paired_ab.py` had the
full stack, `run_lab.py` hand-rolled a private ladder that could neither explore nor load a
compiled program, and `lab_eval.py` / `smoke_live.py` silently ran with no gate and no
training record. Nothing was broken; "what is a wired SentinelPrime agent?" simply had six
answers that had started to drift.

Two rules keep it that way:

- **The default is the configuration the project claims** — gate, audit, credit, monitor,
  and proposal log all on. A thinner default is how five of six call sites ended up
  under-wired; nobody chose it, it was just the shortest constructor.
- **Ablation stays a flag, not a branch.** `gate=None` / `credit=False` / `record=False`
  switch one mechanism without taking a different assembly path that might differ in some
  second way nobody controlled for.

`gate` is `None | "probe" | "ladder"` — `"probe"` is the hermetic deterministic-only rung
set that `run_lab`'s scripted mode needs. Two combinations are **refused rather than
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
