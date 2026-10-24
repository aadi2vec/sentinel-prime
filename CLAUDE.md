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
.venv/bin/python scripts/synthesize_checks.py                # replay stage; reports only
.venv/bin/python scripts/synthesize_checks.py --live --model PROVIDER/MODEL \
    --out lab_runs/<new>/checks.json                         # synthesize + admit checks
.venv/bin/python scripts/policy_lab.py --checks lab_runs/<new>/checks.json \
    --out lab_runs/<new2>                                    # search over the merged catalog
.venv/bin/python scripts/parity_gate.py --live --model PROVIDER/MODEL \
    --per-kind 4 --out lab_runs/<new>                        # §3a Phase 0 kill gate
```

`synthesize_checks.py` and `parity_gate.py` both **withhold their result without `--live`**:
the first proposes nothing (a deterministic stand-in proposer would be a hand-written check
wearing synthesis's name), the second runs an echo baseline through the real RLM loop as the
instrument's self-test and declines to return a verdict.

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

There are now **two** mutable surfaces, and they are mutable by different actors over
different timescales — do not conflate them:

1. **The offline surface: which verification checks run, and whether a failed check triggers
   another solver attempt.** Mutated by `PolicySearch` between validation batches, against
   labelled cases, with an archive and a rollback.
2. **The online surface: the model's own live context.** Mutated by the model mid-task,
   label-free, through a gated tool surface. This is the 2026-10-19 addition.

Everything else in this repo — the ledger, the audit trail, the admission ladders, credit
assignment — is *supporting machinery and instrumentation* for those two, not the thesis.

Read `docs/PLAN.md` first. It is the only plan, and §3 names the only work in flight; a task
that cannot be traced to it is drift. Earlier planning documents were deleted deliberately.

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

**Two promotion rules, and the evidence records which fired** (`PolicySearch._promotion`).
The original rule was quality-only — mean gain ≥ `min_gain`, no per-case regression — and it
had a structural blind spot: a candidate reaching the same answers with fewer outer calls
scores `gain == 0`, below any positive `min_gain`, so the search *could not find efficiency
even when handed one*. That is a rule that cannot express half the objective, not a tuning
choice. So:

- **quality** — mean up by ≥ `min_gain`, no per-case regression. Cost is not consulted.
- **efficiency** — no per-case regression, mean *not lower*, and ≥ `min_cost_gain` fewer
  outer calls. This is the success gate: a cost saving is only credited where quality held.

The asymmetry is the point: **quality may be bought with compute; compute may not be bought
with quality.** `rounds[i]["rule"]` is `"quality" | "efficiency" | None`, because a champion
promoted for being cheaper and one promoted for being better are different claims and
`promoted: true` alone does not say which.

The standing objection, mitigated and not solved: dropping a check always lowers cost, and
on a batch where that check never fires it looks free — so the efficiency rule is biased
toward deleting checks the current batch does not exercise. Per-case non-regression,
`min_cost_gain`, and a fresh family-disjoint batch per round hold it in check; none is a
proof. A check that matters *nowhere in the validation set* will be deleted and should be.

The catalog a candidate selects from comes from `policy_experiment.build_catalogs()`, which
returns the descriptions and the callables **as one pair** — the proposer can only name what
the description catalog lists and the runner can only run what the callable catalog holds, so
building them apart is how a check ends up unreachable or unrunnable. With no argument it is
the two hand-written checks, byte-identical to the pre-synthesis baseline; with admitted
`CheckSpec`s from `scripts/synthesize_checks.py` (via `policy_lab.py --checks`) it is those
plus the synthesized ones. A synthesized name colliding with a hand-written check is refused,
not resolved: shadowing `arithmetic` would change what the baseline arm measured while still
reporting itself as the same catalog.

A scripted fixture candidate must name its checks explicitly, never `list(CATALOG)` — a
"candidate" defined as the whole catalog silently becomes a different one, and past three
checks a budget-infeasible one, as soon as synthesis adds vocabulary.

Two honest limits, kept in the docstrings: synthesis has never been run with a model, so every
reading so far is from the hand-written pair; and the budget counts solver attempts and check
calls, not tokens — so a candidate may spend more of the common ceiling than the baseline, and
a gain does not by itself establish compute efficiency. The second limit is now load-bearing
rather than incidental: "cost" in the efficiency rule means **outer calls**, so an efficiency
promotion is evidence about call count and nothing else. A candidate with fewer calls may
still spend more tokens per call.

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
  proposal log, error budget, write path, and context gate all on. A thinner default is how
  five of six call sites ended up under-wired; nobody chose it, it was just the shortest
  constructor. There are **no exceptions left**: `editable_context` defaulted off until
  2026-10-19 and now defaults on, which means invariant 5 is broken by default. That is a
  deliberate reading, not a regression — see the write-path section — but it does mean runs
  recorded before that date are a different experiment from runs recorded after it.
- **Ablation stays a flag, not a branch.** `gate=None` / `credit=False` / `record=False` /
  `editable_context=False` / `context_gate=False` switch one mechanism without taking a
  different assembly path that might differ in some second way nobody controlled for.

`gate` is `None | "probe" | "ladder"` — `"probe"` is the hermetic deterministic-only rung
set, which is what a no-network run needs, and it applies to **both** ladders (ledger and
context) because a no-network run is a no-network run on both paths. Four combinations are
**refused rather than degraded**: `credit=True, audit=False` (CreditAssigner reads targets
off the AuditRecord, so it would attribute nothing while reporting itself on),
`error_budget=True, audit=False` (the freeze is an audit event, and an unrecorded freeze is a
silent outage — the run reads as having had nothing to learn rather than as one that was
halted), `program=` with a non-ladder gate (there is no judge to load the compiled prompt
into, and ignoring it would report a tuned run that used the untuned bar), and
`context_gate=True, editable_context=False` (no edits exist to admit, so the gate would be
built, reported on, and never consulted). A mechanism that reports as on while doing nothing
produces a null result that looks like evidence — worse than a crash.

`context_gate` is tri-state (`None | True | False`) and the only one that is. `None` means
"follow `editable_context`", so switching the write path off stays *one* flag rather than
two; an explicit `False` is the real ablation — **the write path running ungated, which is
the CLM baseline this repo's thesis argues against and therefore has to be expressible.**

`Assembly.describe()` renders the whole configuration on one line, machinery fingerprint
included, and every script logs it at start. Two runs of "the same" arm with different
compiled judges are not the same experiment, and the arm name alone does not say which.
`ctx` is three-valued for the same reason `budget` is: `off | open | gated`, because an
ungated write path and a gated one are the two arms of the central comparison.

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

**Retirement closes a validity interval; it no longer deletes the row** (2026-10-19, see
below). `_retire` must test *in force*, not *present in the ledger* — those stopped being the
same thing, and the weaker test re-retires a lesson that already left force, emitting a second
audit record under a new `cause_task_id` (hence a new `content_hash`, hence no dedup) for one
departure.

### Valid time (`memory.py`, `harness.in_force_at`) — the two axes live in two objects

A `MemoryItem` carries `valid_at` / `invalid_at`: the period the guidance was **in force**.
Transaction time — when the system came to believe it — is already in `AuditLog`, append-only,
one record per edit. Keeping them apart is what makes this bitemporal *as a system* rather than
four columns on a row: two copies of one fact drift, and invariant 4 makes the audit log's copy
the unfalsifiable one.

**Why it was needed.** Invariant 8 claims the system can *prove the guidance was valid at the
time*. For the guidance itself that was false the moment a lesson was retired: `_retire` called
`backend.delete`, so the row — and the text the run was actually told — was gone by the time
anyone asked. The only reconstruction was `backend.rollback()`, which mutates the live ledger to
answer a read-only question and discards everything written since.

Four things are load-bearing:
1. **The filter sits outside reuse gating.** `admissible_items` drops out-of-force items
   *before* the `context is None` early return, because an expired interval is not a decision
   about this context. Behind the gate, a retired lesson would reappear on every ungated read —
   which is every read that passes no context.
2. **The interval is half-open, `[valid_at, invalid_at)`.** A lesson and the rewrite that
   replaced it are then never both in force for one instant, so a historical read cannot
   surface two versions of one lesson with no way to tell which the run saw.
3. **A write reopens a closed interval.** The mirror of `forget()` on retire, and load-bearing
   for the same reason: retirement now leaves a row behind and `forget()` clears its blame so a
   rewrite gets a fresh trial — but a rewrite inheriting the closed interval would sit in the
   ledger, unblamed, and never be read again. `_apply_edits` constructs a *fresh* `MemoryItem`,
   which is what does it; do not turn that into an in-place mutation.
4. **`in_force_at` is the reader that stops these being fields written and never read** —
   constraint 4's own failure mode, and the reason Phase 4 was parked until the ledger had a
   question only valid time could answer. `ContinualHarness.explain()` consumes it, so the
   compliance chain now carries a `[guidance]` line quoting what was in force entering the
   round, alongside the `[machinery]` line that already said "the admission bar in force at the
   time". A lesson *this round* retired is unioned back onto that line: it was in force going
   into the round, and the half-open interval correctly excludes it at the record's instant.

Comparison is lexicographic on the raw ISO-8601 strings — exact for `_now()`'s fixed-width
`+00:00` output, and what lets a fixture use `"2026-01-01"`. A different producer (local offset,
other precision) would compare wrong; `_now()` is the only one on the live path.

Two honest limits, both in the docstrings. **One interval per item, not a history of them**: the
ledger holds one row per id, so a rewrite overwrites the text and the period before it is
unanswerable here — but it fails as a *gap* rather than as the current text mislabelled, because
the rewrite moves `valid_at` forward too. Superseded text is recovered from the transaction
axis, not this one. And `explain()`'s instant is when the *edit* was recorded, not when the task
ran, so a lesson added and retired between the two would not show; no `AuditRecord` carries the
task's start time.

A proposed `delete` op still deletes. Retirement and deletion are different claims — "this
stopped working over an interval" versus "this is wrong", which has no interval to close — and
collapsing them loses the distinction.

### The error budget (`errorbudget.py`)

The third defence, and the **only component permitted to act**. Admission asks whether *this*
edit is grounded; credit asks whether *this* lesson helped; neither can see the case where both
are working correctly and the system is still getting worse. `ProgressMonitor` deliberately
stops at detection, so until this existed every signal was an annotation.
`PrimeAgent.learn()` consults `allows_refine()` and a breach halts learning rather than
reporting it. On by default via `assemble()`.

Four things are load-bearing:
1. **Not task success.** §5's 29pp band across identical configs would freeze the loop on
   sampling noise. The online defaults are properties of the *machinery*: `turn_limit_rate`
   (share of tasks that ran to `max_iters` — the thrash `monitor.py` describes, in aggregate)
   and `edit_admission_rate` (share of proposals the ladder admitted — invisible elsewhere,
   because rejected edits leave no audit record). §5's own primary metric needs a graded
   corpus, so it lives in `check_set_objectives()` for `synthesize_checks.py`, not here.
2. **Register nothing you cannot feed.** An unobserved objective never reaches
   `min_observations`, so it never breaches and the budget reports itself as guarding a metric
   it is blind to — the `credit=True, audit=False` failure one level up. Objective names are
   module constants shared by producer and consumer so a typo breaks both sides identically.
3. **A freeze stops self-modification, not measurement.** `learn()` calls `harness.credit()`
   *before* the gate and returns a `RefineResult` with an empty reversibility window
   (`from_version == to_version`) rather than `None` — a `None` would make every caller branch
   on the freeze, and halting observation would blind whatever decides when to thaw.
4. **`thaw()` is manual and clears the window.** An auto-thaw resumes without anyone having
   looked at why it stopped; resuming with the breaching observations still in scope would
   re-trip immediately and make the manual thaw useless.

### Where the live loop attaches (`agent.py`)

`PrimeAgent.run_task(task, workdir, context=None, task_id="")` owns three seams:
`context` → `harness.read(context=…)` (reuse gating sees real task state); `pred.trajectory`
+ `last_cache_stats` → `monitor.check_and_record` (live thrash emits a `replan` audit event);
and `last_trajectory` → what the caller passes to `learn()`. Refine on the real trajectory,
never on `[]`. `last_exposed_ids` is the credit-assignment exposure set — the guidance actually
surfaced, not the whole ledger.

Sub-agents are opt-in (`enable_children=True`), lazily built on first spawn, and expose **both**
`spawn_child` and `collect_child`: spawning without collecting is what made them dead weight
before. dspy wraps each tool in a `def invoke(**kwargs)` that it renames to the original
function and re-signs with the original signature, so in the REPL a tool is called **by its own
name with keyword arguments** (`spawn_child(task=...)`) — there is no `.invoke` attribute to
reach for, and a positional call raises. Children share the
parent's harness read-only and never call `learn()`, and `run_task` drains them so no worker
outlives the task that spawned it — the parent writes the ledger between tasks and a child
still reading it would race that write.

**Fan-out and prefix caching pull in opposite directions** (`children.py`), which is why
there is a scheduler there at all. Every child shares the parent's ledger, hence the same
`guidance` prefix that invariant 5 keeps byte-stable so the provider can cache it — but a
cache is only warm once something has populated it, and four children launched at once
against a cold prefix all miss. Maximal parallelism and maximal reuse are the same decision
made two ways. `prefix_pilot=True` (default) runs the first child of each `prefix_key` alone
and releases the rest when it returns. It is a real trade, not a free win — N children go
from one wall-clock unit to two — which is why `scheduling()` reports the counts rather than
asserting a benefit, and why children remain opt-in. Three honest limits in the docstring:
the signal is pilot *completion*, not pilot *prefix warm* (nothing in a `dspy.LM` call
exposes the latter, and a shorter wait would be a guessed duration); `warm_starts` counts a
scheduling decision, never an observed cache hit; and a hung pilot is bounded by
`pilot_timeout` so its group launches cold rather than never. A failed pilot's exception is
deliberately *not* re-raised into its siblings — it still sent its request and still warmed
the prefix, and it is still reported through `drain()`.

### The write path (`context.py`, `contextgate.py`, `contextfns.py`) — on by default

`assemble(editable_context=True)` (the default) lets the model edit its own REPL history in
place. **This is the repo's novel contribution, and the contribution is not the write path —
it is the write path *with an admission gate on it*.** CLM hands the model unrestricted
write access to its context; the thesis here is that a label-free write path needs admission
control for the same reason the ledger does, and `LadderVerifier` already is one.

Five primitives and three templates, all on the live tool surface by default:
`evict_step`, `compact_step`, `write_sketchpad`, `mark_context`, `rewind_context`;
`drop_failures`, `drop_duplicates`, `fold_range`.

It changes what the prompt *is* mid-run, which is exactly the prefix determinism invariant 5
protects — so **invariant 5 is broken by default as of 2026-10-19**. That is why every edit
is priced and why `describe()` renders `ctx=off|open|gated`. Runs before and after that date
are not comparable; the run log says which, rather than leaving it to be inferred.

The seam is one override. `dspy.RLM.forward` threads an immutable `REPLHistory` through
`_execute_iteration` and reassigns the return value, so `ContextEditingRLM` lets the base class
run the turn, records what executed into the `EditableContext`, and returns a `_ContextHistory`
whose `format()` renders from the context instead of its own entries. Four things there are
load-bearing:

1. **Two views, and only one may forget.** `render()` feeds the prompt; `full_trajectory()`
   feeds everything downstream — coverage, the monitor, `refine()`, credit assignment. Those
   are asking what the run *did*, and `grounding_from_trajectory` reads only REPL `output`, so
   an evicted evidence turn would make a genuine pass read as ungrounded and an opened document
   read as never opened. Same split as `AuditLog`/`ProposalLog`, one level down.
2. **The submitting turn arrives on the other branch.** `_process_execution_result` appends it
   to the `Prediction`'s trajectory and never returns it as a history, so the `isinstance(result,
   Prediction)` branch must still record — skipping it drops the one turn that produced the
   deliverable. `_record` indexes by `len(context.executed)`, not by diffing the incoming
   history, because `forward` reassigns that history every turn.
3. **`render()` is ordered most-stable-first**: blocks, eviction notice, sketchpad. The
   tempting layout leads with the sketchpad where the model will read it, but the sketchpad is
   the most frequently rewritten element and anything above the blocks re-charges all of them.
   Measured: a 9-char note cost 771 re-prefill chars (99% of the prefix) leading, 0 trailing.
4. **Every edit is priced.** `ReprefillEvent.chars` is the rendered text after the divergence
   point, because an editable-context win that is really a cache miss is the failure mode. The
   honest limit, kept in the docstring: it counts **characters, not tokens or dollars**.

The non-evictable floor is structural, not checked: `task` and `guidance` are `PrimeTask`
signature fields and the tool surface can only name a history step, so the illegal edit is
inexpressible rather than rejected. `context.reset()` runs in `_before_task` — invariant 6's
analogue — and resets *in place*, because the tools hold a reference bound at RLM construction
and a fresh object would leave them editing the previous task's context.

Two more things are derived, not assumed, and both were bugs waiting to happen:
`agent.context` is read off the RLM (`getattr(rlm, "context", None)`), never set from the
flag — a caller-supplied `rlm=` (every stub in the test suite) writes no turns, so setting it
from the flag would hand `_after_task` an empty grounding view and silently discard the run.
And a child agent gets its own context but the **same gate object**, because an ungated child
is a write path with no admission control while `describe()` reports the gate as on.

#### The context ladder (`contextgate.py`)

Same `LadderVerifier`, same `CostLadderPlanner`, same `verify(op, feedback, trajectory)`
interface as the ledger's — reusing it verbatim is what makes this the *same* mechanism
rather than a lookalike that drifts. A separate instance, though: the two gates see
different ops at different times, so sharing one would pool two P(fail) distributions into
one ordering.

**What changes is what an edit can be grounded in, and this is the load-bearing difference.**
`refine()` runs *after* grading, so `GroundingProbe` can ask whether a lesson is about an
observed rubric failure. A context edit happens *mid-task*: nothing is graded, and
`feedback.as_text()` would be `""` — lexical overlap against nothing is 0.0 for every
candidate, so the ledger's own probe would reject every context edit while reporting itself
as a grounding check. The only thing available is the task, which is what `TaskStatement`
carries and what `_before_task` sets.

Three rungs, priced the ledger's hundred-to-one: `RetentionProbe` (1) refuses an edit that
drops the **last visible copy** of a figure — §5's distribution is 29 omissions and ~0
fabrications, so an edit that removes the only copy of a number manufactures the dominant
error; `ReprefillProbe` (2) refuses an edit that will not pay back the prefix it invalidates
within `horizon_turns`; `PredictContextVerifier` (100) is a `dspy.Predict` over a Signature,
so it reaches `named_predictors()` and GEPA tunes this bar too.

Three things in `_edit` are load-bearing:
1. **The edit is judged on the context it would actually produce** — applied, rendered, then
   undone if refused. A rung that *simulated* the edit would give the gate a second
   rendering path and have it judging a layout the prompt never sees.
2. **A refusal is not priced but is recorded.** Nothing was sent, so no prefix was re-billed;
   but `refusals` is the only record a refused edit leaves (there is no `ReprefillEvent`),
   which is exactly why `ProposalLog` exists one level up.
3. **Invalidation is paid once; a saving recurs.** An earlier version compared the two as if
   both were one-off, which refused every edit near the head — precisely backwards, since a
   head edit has the most remaining turns to amortise over. Found by *running* a template,
   not by reading: evicting a 10-char failed attempt invalidated 264 chars and was refused at
   "26.4x". The honest limit is now stated rather than hidden — `horizon_turns=8` is an
   assumption, because nothing in the loop knows how many turns are left.

A context rollback restores `blocks`/`evictions`/`sketchpad`/`_next_step` and **never**
`executed` or `reprefills` — invariant 4's reasoning one level down: a turn that ran is a
fact, and characters that were re-sent were really billed. `_next_step` restores only
*upward*, so a rewind cannot reuse a step number the run already issued. The gate's own
internal snapshots use `_snapshot()`, which does **not** enter the model's checkpoint
namespace — a rewind target it never marked would make the tool's behaviour depend on how
many edits the gate happened to refuse.

#### Context functions are templates, never code (`contextfns.py`)

The obvious reading of CLM's "model-defined context functions" is code synthesis. This repo
refuses it: `interpreter.py` is not a sandbox, and §8 excludes code synthesis for the same
reason `Policy.parse` rejects unknown fields — a candidate that supplies the implementation
can redefine what the operation means. So a context function is a **parameterised template**
over the primitives, exactly `Policy`'s shape one level up.

Two properties make them more than sugar: every expansion still passes the gate individually
(a template is not a bypass), and a composite is **atomic** where the primitives are not —
one tool call gets one outcome, which is what `ContextCheckpoint` was built for and why the
reversibility layer had to land first. One tool per function rather than a dispatcher,
because dspy builds the tool surface from each function's own signature, and a dispatcher
would describe its real parameters only in prose.

The honest limit: `drop_failures` matches a substring, so it will catch a turn that printed
"Error" while succeeding. The ladder cannot catch that — such an edit loses no figure.

### Invariants that the whole design rests on — do not break

1. **Supplemental-only.** `read()` emits an *additional* prompt block; the base task prompt is
   never mutated. An empty ledger must return `""` so nothing is prepended. A ledger whose rows
   are all out of force is an empty ledger for this purpose.
2. **Reversible.** Every `refine()` brackets its edits between two backend snapshots. The
   write path's analogue is `ContextCheckpoint`, which restores the **prompt view only** —
   never `executed` or `reprefills`, because a rewind must not launder a turn out of
   existence or make backtracking look free. Closing a validity interval is a `write` inside
   the same window, so `rollback(from_version)` reopens it exactly as it used to restore a
   deleted row.
3. **Label-free.** Only the trajectory and `feedback.as_text()` reach the proposer. No gold
   answers in the online path.
4. **Append-only audit.** `AuditLog` records are never edited or deleted — a rolled-back edit
   is still a fact that happened. Duplicate edits collapse by `content_hash`. As of
   2026-10-19 the *ledger* is append-only for retirement too: a lesson that stops earning its
   place has its validity interval closed rather than its row deleted, so "prove the guidance
   was valid at the time" holds for the guidance and not only for the edit that admitted it.
5. **Prefix-cache determinism.** `read()` sorts items before emitting; the backend promises no
   order, and an unstable order silently invalidates provider prompt caching. Field order in
   `PrimeTask` is part of the same guarantee: `guidance` is declared *before* `task` so the
   ledger sits inside the prompt prefix shared across tasks. Reordering those fields silently
   destroys cacheability — `agent.cacheable_prefix()` is the regression check.
   **`editable_context=True` breaks this invariant by construction and is now the default**
   (2026-10-19), which is why every edit is priced (`context.reprefill_chars`), why
   `ReprefillProbe` can refuse one, and why `describe()` prints `ctx`. What survives of the
   invariant: the *supplemental block* is still sorted and still prefix-stable, `render()` is
   ordered most-stable-first so the volatile elements sit at the tail, and `cacheable_prefix`
   is a module-level function unaffected by the flag. With `editable_context=False` this
   holds unchanged.
6. **Learning happens only between tasks.** `PrimeAgent.learn()` is never called mid-task;
   `run_task` freezes the ledger snapshot at task start.
7. **Optional collaborators degrade to base behavior.** `audit_log`, `verifier`,
   `reuse_controller`, `monitor`, `proposal_log`, `read(context=...)`, `EditableContext.gate`,
   and `prefix_pilot` are all `None`-able/off-able; when unset the code must behave exactly
   as before they existed (gating only ever *removes* provably-stale reuse; `gate=None`
   applies every well-formed edit; `prefix_pilot=False` submits every child immediately).
   This is what makes ablation possible — do not make any of them mandatory.
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
(single writer). Vector clocks and Byzantine quorums are explicitly **parked** — see the
"Parked" section of the 2026-09-05 plan before building any of them. Trajectory checkpoints
*unparked* on 2026-10-19 and shipped as `ContextCheckpoint`; the parked note is stale.

**Owed, and known to be owed: tests for the write path and Phases 2–3.** `context.py`,
`contextgate.py`, and `contextfns.py` have no test file, and `children.py`'s prefix scheduler
and `PolicySearch._promotion`'s efficiency rule are untested. This was a deliberate
instruction (feature parity first, tests later), not an oversight, and it is the one place
this repo departs from its own TDD convention. The 452 passing tests cover everything else.
All three have been exercised end to end by hand — which is how `render()`'s ordering bug and
`ReprefillProbe`'s one-off-vs-recurring bug were found — but exercised is not tested.

Phase 4 (valid time) is **not** part of that debt: it touches the ledger core that
`test_harness`/`test_memory`/`test_credit` already cover, so it was built test-first and has
`tests/test_bitemporal.py`. Two of those tests were rewritten after a mutation check showed
them passing for the wrong reason — a re-retirement guard that was really being satisfied by
`targets()` resolving differently, and an `explain()` assertion on an id that the
`[ledger_edit]` line already carried. If you add to that file, mutate the line you think you
are protecting and watch the test fail.

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
