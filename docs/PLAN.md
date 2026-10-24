# The plan

**One document. If a task cannot be traced to §3, it is not the work.**

---

## 1. The goal

> Can an RLM improve the machinery it uses to solve problems — how it decomposes work,
> manages context, checks results, and learns from failure?

Not "can it remember better advice." Can it get better at *how it computes*.

Harvey LAB (M&A due diligence) is the hard environment this is tested in. It is not a
product direction. No legal application is being built.

## 2. The one experiment

Improve a single decision: **when and how to verify.**

```
fixed policy → collect real failures → synthesize a candidate checking strategy
            → does it catch mistakes on tasks it has not seen, inside the budget?
            → promote or discard
```

Success is an independently measured improvement on unseen tasks at a fixed ceiling. It is
**not** more instructions, agreement with the agent's own judge, or a benchmark rewritten to
reward the system.

## 3. What is missing, and nothing else is the work

> Amended 2026-10-18 — see §3a. All four items below are built and on the live path; items
> 3 and 4's readings have not been taken. A second program (the write path) is admitted in
> phases behind a kill gate.
>
> Amended again 2026-10-19 — the write path's Phases A, 0, 1, 2 and 3 are all built, wired
> and **on by default**, which means the kill gate is no longer in front of the work it was
> meant to gate. §3a is kept as written because the reasoning still holds and the reading is
> still worth taking; what changed is what a `cancel` verdict would now cost. See the amended
> kill condition below.

The search substrate is built and merged: bounded candidate policies, an evaluator the
candidate cannot touch, family-disjoint dev/validation/test with reservation, promotion,
rollback, and a matched-compute control. What does not exist is the experiment.

| Verb in §2 | Today |
|---|---|
| collect real failures | tasks are synthetic; the solver's mistakes are **scripted** |
| synthesize a checking strategy | synthesis is wired (`--checks`) but **has never been run with a model** |
| on tasks it has not seen | one generated template, shared across all splits |

Four tasks close it, in order:

1. **A LAB-backed family.** `Case.task` is a real LAB task, `Case.gold` its rubric criteria,
   `evaluate` the `RubricJudge`. Real failures in.
2. **A failure-mode taxonomy.** Which ways a deliverable actually fails, counted on the
   corpus, and for each mode whether a deterministic check could detect it at all. Without
   it, synthesis invents checks against imagined faults: the observed distribution is 29
   omissions, 17 partials and ~0 fabrications, so a check that verifies claims *made*
   addresses almost nothing. This is the prerequisite for item 3, not a parallel track.

3. **Check synthesis.** *(built, unread)* A `dspy.Signature` from criterion text to a
   **parameterised** check — span existence, amount match, structural presence. Not arbitrary
   code: `interpreter.py` is explicitly not a sandbox, so the synthesizer selects and fills a
   template rather than emitting Python. A synthesized check joins the catalog only after
   agreeing with the judge on held-out already-graded criteria.

   `scripts/synthesize_checks.py` is the replay stage and `policy_lab.py --checks` is where its
   output re-enters: `build_catalogs()` merges admitted specs into the catalog the proposer
   selects from and the runner executes, as one pair, because a check in only one half is
   either unreachable or unrunnable. Three refusals are deliberate — a name colliding with a
   hand-written check (shadowing `arithmetic` would change what the baseline measured while
   reporting the same catalog), a malformed entry anywhere in a catalog file (loading the good
   rows would run a different check set under the admitted one's name), and **scripted mode**,
   which proposes nothing at all. A deterministic stand-in proposer would be a hand-written
   check wearing synthesis's name, which is exactly the limit this item exists to remove, so
   without `--live` the script reports what is discriminative and stops. The current corpus
   offers **30** discriminative criteria over 6 graded runs.
4. **Replay scoring.** *(built)* A candidate is scored against **stored** runs, not fresh agent runs —
   in the existing substrate, a `PolicyRunner` whose `solve` is a lookup. Available today:
   368 graded criterion rows, 10 deliverables, 2 trajectories.

### 3a. Amendment 2026-10-18 — the write path

§3 is now built end to end (`lab.py`, `taxonomy.py`, `synthesis.py`, `replay.py`), and both
halves of Phase A are on the live path rather than merely present — which is a distinction
worth stating, because a module nothing calls is indistinguishable from an unwritten one:

| Mechanism | Where it is consulted | Default |
|---|---|---|
| Error budget + freeze | `assemble()` → `PrimeAgent.learn()` consults `allows_refine()` before `refine()` | **on**; `error_budget=False` ablates |
| Check synthesis | `scripts/synthesize_checks.py` → `policy_lab.py --checks` → `build_catalogs()` | on **when a catalog is supplied**; the run log prints the split |

Two things follow from the same rule the assembler already enforces:

- `error_budget=True, audit=False` is **refused**, alongside `credit=True, audit=False`. The
  freeze is an audit event, and an unrecorded freeze is a silent outage — the run reads as
  having had nothing to learn rather than as one that was halted.
- The online budget registers only objectives the live loop actually feeds
  (`turn_limit_rate`, `edit_admission_rate`). §5's primary metric is a property of a check
  set over a graded corpus and `assemble()` has none, so registering it there would produce
  an objective nobody observes: never at `min_observations`, therefore never in breach, and
  reported as guarding a metric it is structurally blind to. `check_set_objectives()` is
  registered by `synthesize_checks.py`, which does have the corpus.

Neither default is task success. §5 records a pooled criterion rate spanning 0.400–0.691
across five runs of one *identical* configuration, so a budget defined on it would freeze
the loop on sampling noise and the freeze would carry no information.

A second program is admitted here: an RLM manages *when and what to read* into context, but
appends what it retrieves, so the live context only grows. The write side — an editable live
context the model edits in place, and context-management functions the model defines rather
than the harness predefining — is a real gap, and it is the gap this harness's own
`monitor.py` already names ("appends to a monotone history with no progress signal and no
backtrack, so a wrong strategy gets *elaborated* until max_iters").

It is admitted **in phases**. The claim that the write path is this system's bottleneck is
still not established — §5's observed distribution is 29 omissions, 17 partials and ~0
fabrications, plus 3 of 8 documents declined on one task, and those are read-path failures.
Phase 0 was built to settle that first.

**Amended 2026-10-19: Phase 0 is no longer a precondition for Phase 1.** The gate's reading
requires a live model and no session has had credentials, so "blocked on the Phase 0 reading"
was in practice an indefinite stall on a measurement nobody could take — and the gate was
this plan's own invention, not an external requirement. Phase 1 was built instead, behind a
flag that defaults off. The gate survives as a *reading*, not a precondition: `parity_gate.py`
still answers whether the baseline needed the write path, and a `cancel` verdict is still the
reason to retire Phase 1 rather than extend it. What changed is the order, not the question.

**Amended again, same day: the flag now defaults on, and Phases 2–3 are built.** The write
path is the configuration this project claims, so it is not an opt-in extra sitting beside
the thing that gets run. The consequence is recorded rather than argued: **invariant 5 is
broken by default from 2026-10-19**, every run from here is a different experiment from the
ones archived in `lab_runs/`, and `describe()` prints `ctx=off|open|gated` so a run log says
which. The write path still prices every edit, and now a gate can refuse one.

| Phase | Work | Status |
|---|---|---|
| A | Check synthesis (item 3) · error budget + freeze over the check-set metric | **built, wired, on by default** — `synthesis.py`, `errorbudget.py`; synthesis itself unread |
| 0 | Parity task family: needle retention, in-place sketchpad edit, KV offload/evict, log triage | **built, unread** — `parity.py`, `scripts/parity_gate.py` |
| 1 | Editable live context · re-prefill accounting · non-evictable floor · eviction record | **built, wired, on by default** — `context.py`, `agent.ContextEditingRLM`, `assemble(editable_context=True)`; tests owed |
| 2 | Admission ladder on context edits · context checkpoint/rollback · context-fn templates | **built, wired, on by default** — `contextgate.py`, `ContextCheckpoint`, `contextfns.py`, `assemble(context_gate=…)`; tests owed |
| 3 | Success-gated promotion · prefix-aware child scheduling | **built, wired, on by default** — `PolicySearch._promotion`, `ChildSessionManager(prefix_pilot=True)`; tests owed |
| 4 | Bitemporal lesson provenance (`valid_at`/`invalid_at`) | **built, wired, on by default** — `MemoryItem.in_force_at`, `harness.in_force_at`, retirement closes an interval; `tests/test_bitemporal.py` |

Four constraints carry over from the existing invariants and are not negotiable per-phase:

1. **Phase 1 breaks invariant 5 by construction** — an in-place edit is exactly what
   invalidates the prompt prefix `read()`'s sort order and `PrimeTask`'s field order exist to
   protect. So re-prefill cost is counted in the same change. An editable-context win that is
   actually a cache miss is the failure mode. *Amended with the default flip: it is still a
   flag (`editable_context=False` restores the invariant exactly) but it is no longer the
   non-default branch, so the accounting is the only thing standing between this phase and
   that failure mode. Phase 2's `ReprefillProbe` turns the accounting into a refusal, which
   is what "not negotiable" has to mean once the default is on.*
2. **Eviction must record what it dropped.** `grounding_from_trajectory` reads only REPL
   `output`; evict the turn holding the evidence and a criterion that genuinely passed reads
   as ungrounded, so `credit.observe` silently drops the credit. Mutable context without an
   eviction record is a credit-assignment bug, not a feature.
3. **The task statement and guidance block are structurally non-evictable** — the
   live-context analogue of invariant 1.
4. **Phase 4 stays parked until Phase 1 lands.** Valid-time separation only carries
   information once sources are mutable; before that `invalid_at` is a field written and
   never read, which is the `credit=True, audit=False` failure — a mechanism that reads as
   rigor while doing nothing. Same park as `ReuseController.causally_current`. *Phase 1 has
   landed, so this is now unparked.* **Built 2026-10-19, and the constraint is what shaped
   it:** the fields were only admitted once there were two readers that could not be
   satisfied without them — `admissible_items` (every online read) and
   `ContinualHarness.explain` (the compliance chain). Had it landed as a pair of columns and
   a `rollback()`, it would have been exactly the failure this constraint names.

Phase 1 as built has one seam and two views. The seam is `dspy.RLM._execute_iteration`, which
`forward` reassigns the history from each turn — so a single override lets the base class run
the turn, records what executed into an `EditableContext`, and returns a history whose
`format()` renders from the context instead of its own entries. The two views are the point:
`render()` feeds the prompt and may forget, `full_trajectory()` feeds grounding and may not
(constraint 2). The same split as `AuditLog`/`ProposalLog`, one level down.

Constraint 1 already paid for itself. The first layout put the sketchpad at the top of the
rendered context, where the model would read it first; measured on a three-turn run, a
9-character sketchpad note charged **771 re-prefill characters — 99% of the prefix**, because
anything above the turn blocks invalidates all of them. Reordering to most-stable-first
(blocks, eviction notice, sketchpad) took the same note to **0**. Without the accounting that
would have shipped as an editable-context win.

### Phase 2 as built: the gate is the contribution

CLM gives the model unrestricted write access to its context. The claim this repo is making
is narrower and testable: **a label-free write path needs admission control for the same
reason the ledger does**, and `LadderVerifier` already is one. So the context edits go
through the *same* ladder — same `verify(op, feedback, trajectory)` interface, same
`CostLadderPlanner` ordering by `cost / P(fail)`, same adaptive statistics, same `explore`
— rather than a parallel mechanism that would drift from it. `assemble(context_gate=False)`
expresses the ungated arm, which is the CLM baseline and therefore has to be expressible.

**What changes is what an edit can be grounded in, and this was the design problem.**
`refine()` runs after grading, so `GroundingProbe` asks whether a lesson is about an observed
rubric failure. A context edit happens mid-task: nothing is graded, and `feedback.as_text()`
is `""` — lexical overlap against nothing is 0.0 for every candidate, so the ledger's own
probe would reject *every* context edit while still reporting itself as a grounding check.
The only thing available is the task, which is what `TaskStatement` carries.

Phase 2 ordering was forced, and worth recording: **2b had to precede 2a.** The gate judges
an edit on the context it would actually produce — applied, rendered, then undone if refused
— which needs checkpoint/restore. The alternative, a rung that simulates the edit, would give
the gate a second rendering path and have it judging a layout the prompt never sees.

Constraint 1 paid for itself a second time, and this one is the better finding.
`ReprefillProbe` originally compared invalidation to saving as though both were one-off
costs. They are not: **invalidation is paid once, the saving recurs on every remaining turn.**
Comparing them directly refuses every edit near the head of the context — exactly backwards,
since a head edit has the most turns left to amortise over. Found by *running* a context
template over a three-turn context, not by reading the code: evicting a 10-character failed
attempt invalidated 264 characters and was refused at "26.4x". The rule is now `invalidated`
vs `saved × horizon_turns`, and `horizon_turns = 8` is stated as an assumption because
nothing in the loop knows how many turns are left — `max_iters` is a ceiling, not a forecast.

Context functions (2c) are **parameterised templates, never code** — §8's exclusion of code
synthesis is not negotiable and `interpreter.py` is not a sandbox. They are `Policy`'s shape
one level down: select from a registry, bind a bounded parameter, never reach the
implementation. Two properties earn them their place: every expansion still passes the gate
individually, so a template is not a bypass; and a composite is **atomic** where the
primitives deliberately are not, because one tool call should have one outcome and half a
fold is worse than none of it.

### Phase 3 as built: two findings, both about rules that could not express the objective

**3a — the promotion rule was quality-only, and therefore blind to half the goal.** A
candidate reaching the same answers with fewer outer calls scores `gain == 0`, below any
positive `min_gain`, so `PolicySearch` could not promote an efficiency win *even when handed
one*. There are now two rules and the evidence records which fired: **quality** (mean up by
`min_gain`, no per-case regression) and **efficiency** (no per-case regression, mean not
lower, ≥ `min_cost_gain` fewer outer calls). The asymmetry is the whole content: quality may
be bought with compute; compute may not be bought with quality. The standing objection is
recorded rather than solved — dropping a check always lowers cost, and on a batch where it
never fires that looks free.

**3b — fan-out and prefix caching pull in opposite directions.** Every child shares the
parent's ledger, hence the same `guidance` prefix that invariant 5 keeps byte-stable so the
provider can cache it. But four children launched at once against a cold prefix all miss:
they are in flight before any has returned. Maximal parallelism and maximal cache reuse are
the same decision made two ways, which is not obvious and is why `children.py` has a
scheduler now. `prefix_pilot` runs the first child of a prefix group alone and releases the
rest when it returns — a real trade (N children go from one wall-clock unit to two), so
`scheduling()` reports the counts rather than asserting a benefit. Three limits stated: the
observable signal is pilot *completion* rather than pilot *prefix warm*; `warm_starts` counts
a scheduling decision, never a measured cache hit; and a hung pilot is bounded by
`pilot_timeout` so its group launches cold rather than never.

**Owed across all three phases: tests.** This was an explicit instruction — feature parity
first — and it is the one place the repo departs from its own TDD convention. It is recorded
here so it reads as a debt rather than as an oversight.

### Phase 4 as built: the parked constraint was the specification

Phase 4's entry in the table was one line, and the constraint that parked it turned out to be
the only design input that mattered: *before sources are mutable, `invalid_at` is a field
written and never read.* So the work was not "add two timestamps" but "find the question the
ledger could not answer", and there was one, stated in the repo's own invariant 8: **prove the
guidance was valid at the time.** That held for the *edit* — `AuditLog` is append-only, and
`machinery_fingerprint` even records which admission bar was in force. It was false for the
*guidance*, because `_retire` deleted the row, so by the time anyone asked what a run had been
told, the text was gone. The only reconstruction was `backend.rollback()` — which mutates the
live ledger to answer a read-only question and discards everything written since.

So: valid time (`valid_at`/`invalid_at`) goes on the item, transaction time stays in the audit
log, and retirement **closes an interval instead of deleting a row**. The two axes deliberately
live in two objects; duplicating transaction time onto the row is how the two copies drift, and
invariant 4 is what makes the log's copy the unfalsifiable one. Two readers justify the fields —
`admissible_items` on every online read (so the loop's behaviour is unchanged from deletion) and
`explain()`, which now renders a `[guidance]` line quoting what was in force entering the round.

Three findings worth keeping, all from writing the tests first:

- **"Present in the ledger" and "in force" stopped being the same predicate**, and `_retire`
  tested the wrong one. The weaker test re-retires a lesson that already left force, emitting a
  second audit record under a new `cause_task_id` — a new `content_hash`, so no dedup — for one
  departure.
- **A write must reopen a closed interval**, which is the exact mirror of credit's
  `forget()`-on-retire rule. Retirement now leaves a row behind and `forget()` clears its blame
  so a rewrite gets a fresh trial; a rewrite inheriting the closed interval would sit in the
  ledger, unblamed, and never be read again.
- **Two tests passed for the wrong reason and were caught by mutating the line they claimed to
  protect**, not by reading them. A re-retirement guard was really being satisfied by
  `targets()` resolving differently once the retire record became the latest for that id, and an
  `explain()` assertion named an id the `[ledger_edit]` line already carried. Both now assert on
  something only the new code can produce.

The honest limit, in the docstring: **one interval per item, not a history of them.** A rewrite
overwrites the text and opens a new interval, so the period before it is unanswerable from the
ledger. It fails in the safe direction — a *gap*, not the current text mislabelled as having
been in force then, because the rewrite moves `valid_at` forward too — and superseded text is
recovered from the transaction axis, which keeps every version. Separately, `explain()`'s instant
is when the edit was recorded rather than when the task ran; no `AuditRecord` carries the latter.

Still excluded, unchanged from §8: weight training and harness→weights distillation (it also
breaks `machinery_fingerprint` as the auditability mechanism), arbitrary code synthesis
(`interpreter.py` is not a sandbox), and a learned recursion-budget controller (it collides
with §6 and `max_revisions` is unmeasurable under replay).

**Kill condition, stated in advance:** if Phase 0 shows the baseline RLM already handles
surgical in-place edits competently, the write path is not this system's bottleneck — Phase 1
is retired rather than extended, and Phases 2–3 are cancelled. Reading it after Phase 1 was
built costs the Phase 1 work if the verdict is `cancel`; reading it never costs the ability
to tell a win from a coincidence, which is worse.

**Amended 2026-10-19: the gate was overtaken by the build, and that is recorded rather than
rewritten.** Phases 1, 2 and 3 were all built on an explicit instruction before the reading
was taken, so a `cancel` verdict now costs three phases rather than one. The arithmetic above
still favours reading it — three phases of sunk work is still cheaper than a permanent
inability to distinguish a write-path win from a baseline the write path never improved on —
but the gate has stopped being a gate and is now a post-hoc validity check on work already in
the request path. Two consequences worth naming instead of discovering later:

- A `cancel` verdict does not mean "delete the write path". It means the *efficiency* claim is
  unsupported, because a baseline that already edits competently makes the measured saving
  attributable to the baseline. The admission gate on the write path (`contextgate.py`) is a
  separate claim and survives a `cancel`: it is an argument about what an ungated label-free
  write path admits, which does not depend on the baseline being bad at editing.
- Because the flag now defaults on, the baseline arm is `assemble(editable_context=False)`
  rather than "the code as of last week". That arm is still reachable, which is the only
  reason the reading is still takeable at all — see constraint 1.

The condition is now code, not prose — `parity.gate_verdict` holds the thresholds
(`correct_rate >= 0.8`, `mean_footprint <= 0.2`, read off the `sketchpad` kind) and returns
one of `cancel` / `build` / `inconclusive`. Three outcomes rather than two because footprint
on a wrong answer measures nothing about editing: a baseline that simply cannot do the task
is not evidence either way, and collapsing that case into `build` is how a gate becomes a
formality. The reading requires a live model:

```bash
.venv/bin/python scripts/parity_gate.py --live --model PROVIDER/MODEL \
    --per-kind 4 --out lab_runs/<new>
```

Without `--live` the script runs an echo baseline through the real RLM loop and withholds
the verdict. That run is the instrument's self-test, not the gate: an echo scores
`correct_rate` 0 at `footprint` 0.0 on every kind — it keeps everything and edits nothing —
which is what confirms the two measurements are actually separable before a paid run.

## 4. Why replay, and what it cannot tell us

A live task run costs ~218s and ~270k tokens and carries the variance in §5. One agent run per
candidate makes the search unaffordable and unreadable at once. Replay makes the agent run a
fixed cost amortised over every candidate ever tested.

It buys detection only. A stored run cannot be re-solved, so `max_revisions` is unmeasurable
under replay. Evaluation is therefore two-stage, answering different questions:

| Stage | Cost | Question |
|---|---|---|
| Replay | ~0 | does this check set find real mistakes without crying wolf? |
| Live | one agent run per surviving candidate | does acting on the flags actually fix them? |

Replay ranks and filters; live confirms the survivors. A candidate that wins on replay and
loses live is itself a finding.

## 5. Measurements that constrain everything above

Recorded here because the documents holding them were deleted.

| Finding | Value |
|---|---|
| Pooled criterion rate, five runs of one *identical* no-ledger config, one LAB task | **0.400 – 0.691** (29pp) |
| A hand-written oracle lesson vs. an empty ledger | **−5.4pp**, at 40–84% more tokens |
| Judge calls that hit rate limits and were scored as criterion failures | **44 of 220 (20%)** — fixed |
| Failure modes among 50 failed criteria | 29 omissions, 17 partials, **~0 hallucinations** |
| Documents the agent declined to analyse, on one task | 3 of 8, scoring ~0 against 50–85% for those it read |

Two consequences. Pooled criterion rate cannot settle a comparison at affordable n, so the
primary metric is a property of the *check set* over a fixed corpus — precision and recall at
a cost ceiling — which agent sampling variance does not enter. And the dominant real failure
is omission, not fabrication, so a check that only verifies claims *made* addresses almost
nothing; coverage and completeness checks matter more than citation checks.

## 6. The boundary the candidate may not cross

It may change execution. It may not change what counts as success, or what it may spend.

`Policy.parse` rejects unknown fields, `Case.gold` reaches the evaluator only, splits are
family-disjoint and reserved before work begins, and the matched-compute control lives on a
sibling runner no candidate has heard of.

One known leak, to be closed when the policy surface expands beyond check selection:
`assemble()` exposes `credit_min_exposures` / `credit_min_success_rate`, and `build_ladder`
exposes `min_score`. Those define success. When a policy can reach `assemble()` arguments,
split the signature into a searchable `PolicyConfig` and a frozen `EvaluationConfig`.

## 7. How this fails

Written in advance, so a null result is a result.

- **Synthesized checks do not beat the hand-written ladder on replay.** Synthesis fails.
- **Replay ranking does not survive the live run.** Detection does not imply correction, and
  the cheap loop is not a substitute for the expensive one.
- **A policy wins on LAB and dies on a second task family.** We learned a template, not a
  mechanism. A second family is how this gets detected — *after* §3, not before, because until
  checks are synthesized there is no vocabulary for one to test.
- **Every candidate lands inside the noise.** The objective is unreadable and variance
  reduction is the only remaining job. `graveyard/scripts/noise_floor.py` is the instrument.

## 8. Not doing

A legal product, design partners, packaging, a UI, weight training, a second task family
before §3 is done, and any further hardening of the scripted demo. The audit, provenance and
reversibility machinery stays as instrumentation, never as the claim.
