# sentinel-prime

> **The predictive-recursion direction moved to [recursio](https://github.com/aadi2vec/recursio)
> on 2026-09-12.** Its PRD and execution plan live there, along with a copy of seven
> domain-neutral modules from this repo. This repository keeps the RLM research frontier:
> failure-driven self-improvement, the admission ladder, credit assignment, replay scoring,
> the gated editable context, and the judged corpus in `lab_tasks/` and `lab_runs/`.

**Can an RLM improve the machinery it uses to solve problems?**

Not "can it remember more advice" — can it change *how it computes*: how work is decomposed,
how context is allocated, when results get checked, when to try again, when to delegate. This
repo is a harness for asking that experimentally. Offline, the system proposes bounded changes
to its own execution policy, trials them in isolation, evaluates them on unseen tasks under a
fixed ceiling, and promotes or discards. Online, the model edits its own live context mid-task
— through an admission gate, which is the part that is novel here.

Harvey LAB (M&A due diligence) is the first demanding environment. **It is the stress test,
not the product.** No legal application is being built here.

```
round 1: promoted=False gain=0.0
round 2: promoted=True  gain=1.0
test initial:  score=0.0 solves=3 checks=0
test champion: score=1.0 solves=6 checks=12
```

That is `scripts/policy_lab.py`, verbatim, hermetic and network-free. Read [Status](#status)
before it means anything: the solver's failures and the proposal sequence are **scripted**.
What executes for real is the assembled RLM, the interpreter, the checks, the selection rule,
and the persistence.

## Two mutable surfaces

There was deliberately only one until 2026-10-19. There are now two, and they are mutated by
**different actors over different timescales** — conflating them is the main design hazard in
the repo.

**1. The offline surface: which verification checks run, and whether a failed check triggers
another solver attempt.** Mutated by `PolicySearch` between validation batches, against
labelled cases, with an archive and a rollback.

A candidate `Policy` selects registered checks and a bounded revision count. It **cannot**
supply code, alter a check's implementation, change the evaluator, or raise its budget.
`Policy.parse` rejects unknown fields outright — the evaluation boundary is a type, not a
convention. `Case.gold` reaches the evaluator only, never the solver, checker, or proposer.
Development, each validation batch, and the final test are family-disjoint and reserved before
work begins, so a restart cannot re-spend the test set.

That constraint is the whole game. A candidate that can redefine success improves its score
without improving its problem-solving, and model agreement does not become truth by being
called a quorum.

**2. The online surface: the model's own live context.** Mutated by the model *mid-task*,
label-free, through a gated tool surface.

This is the newer half, and it is the repo's novel claim. The CLM line of work (see
[`docs/PLAN.md`](docs/PLAN.md) §3a for the framing and the citations) hands the model
unrestricted write access to its own context. The thesis here is that a **label-free write path
needs admission control for the same reason the ledger does** — and `LadderVerifier` already is
one. So the contribution is not the write path; it is the write path *with the admission ladder
on it*. `assemble(context_gate=False)` is the ungated CLM-shaped baseline, and it is
expressible precisely because it is the arm this repo argues against.

### Two promotion rules on the offline surface, because the original could not state half the objective

`PolicySearch._promotion` was quality-only — mean gain ≥ `min_gain`, no per-case regression. A
candidate reaching the same answers with *fewer outer calls* scores `gain == 0`, below any
positive threshold, so the search **could not find efficiency even when handed one.** That is a
rule that cannot express half its own objective, not a tuning choice.

- **quality** — mean up by ≥ `min_gain`, no per-case regression. Cost is not consulted.
- **efficiency** — no per-case regression, mean *not lower*, and ≥ `min_cost_gain` fewer outer
  calls.

**Quality may be bought with compute; compute may not be bought with quality.** The evidence
records which rule fired (`rounds[i]["rule"]`), because a champion promoted for being cheaper
and one promoted for being better are different claims and `promoted: true` does not say which.

The standing objection, mitigated and not solved: dropping a check always lowers cost, so on a
batch where that check never fires it looks free. Per-case non-regression, `min_cost_gain`, and
a fresh family-disjoint batch per round hold it in check; none is a proof.

## Status

**Proven:** 452 tests, hermetic, TDD'd. The policy-search substrate runs end to end — bounded
candidates, schema and budget validation, family reservation, staged promotion, rollback, and
persisted per-attempt evidence. Every agent comes from `assemble()`, so a policy is
expressible only as assembler arguments and cannot reach around the assembler.

**Not proven, and this is the important part:** that any model improves its own computation.
`policy_lab.py` is a *mechanism* demo with a scripted solver and a scripted proposal sequence.
Synthesis has never been run with a model, so every reading so far comes from the two
hand-written checks. The budget counts solver attempts and check calls, not tokens — so a
candidate can spend more of the shared ceiling than the baseline, and a gain does not by itself
establish compute efficiency. There are **no real-model results.**

**Owed, and known to be owed:** tests for the write path. `context.py`, `contextgate.py`, and
`contextfns.py` have no test file, and `children.py`'s prefix scheduler and the efficiency
promotion rule are untested. This was a deliberate instruction — feature parity first, tests
later — not an oversight, and it is the one place this repo departs from its own TDD
convention. All three have been exercised end to end by hand, which is how `render()`'s
ordering bug and `ReprefillProbe`'s one-off-vs-recurring bug were found. Exercised is not
tested. Valid time (`memory.py` / `harness.in_force_at`) is **not** part of that debt: it
touches the well-covered ledger core, so it was built test-first.

**Runs before and after 2026-10-19 are not comparable.** `editable_context` defaults on, which
breaks prefix-cache determinism by construction. That is why every context edit is priced and
why `describe()` prints `ctx=off|open|gated` — the run log says which experiment it was rather
than leaving it to be inferred.

**What was measured, and what it cost the previous direction:**

| Finding | Value |
|---|---|
| Pooled criterion rate across five runs of one *identical* no-ledger config | **0.400 – 0.691** (29pp spread) |
| A hand-written oracle lesson vs. an empty ledger (arm F) | **−5.4pp**, at 40–84% more tokens |
| Judge calls that hit rate limits and were scored as criterion *failures* | **44 of 220 (20%)** |

The third was a real defect. The exception text flowed into `Feedback.as_text()` — the
proposer's only view of why a run fell short — so a learning arm wrote ledger lessons about
the provider's rate limiter. Fixed: `RubricJudge` retries, then emits a third `error` verdict
that `to_feedback` excludes from both the pooled rate and the proposer's input.

The first two are why the object of study moved from *what the agent remembers* to *how the
agent computes*. Content-learning effects on this benchmark are smaller than the measurement
noise, and no amount of mechanism fixes an unreadable objective. Both numbers come from
`graveyard/scripts/arm_f.py`, retired because it answered its question.

## The mechanisms

Everything here is *supporting machinery and instrumentation* for the two mutable surfaces, not
the thesis.

| | What it does | Why it is there |
|---|---|---|
| `policy_search.py` | bounded candidates, fixed evaluator, family-disjoint splits, two promotion rules, rollback | the search, and the boundary that keeps it honest |
| `policy_experiment.py` | full-`assemble()` RLM adapter, check catalog, generated tasks, independent scorer | a trial that runs the real stack |
| `assembly.py` | `assemble(lm=..., root=...)` — the only wiring path; `describe()` fingerprints the config | a candidate cannot reach around the assembler |
| `ContinualHarness` | `read()` / `refine()` / `rollback()` / `explain()` / `in_force_at()` — versioned WAL + copy-on-write ledger | supporting machinery: reversible, auditable state |
| `LadderVerifier` | short-circuited cost ladder: cheap deterministic probe → generative judge | verification is asymmetric — and it is the fixed policy the search must beat |
| `AuditLog` | append-only, content-addressed record per edit | a rolled-back edit is still a fact that happened |
| `ProposalLog` | JSONL record of what the verifier was *asked* about, admitted or not | the audit log is all-positive by construction, so a gate fit to it learns to admit everything |
| `CreditAssigner` | retires a lesson whose targeted criteria stop passing | a polluted ledger is worse than an empty one |
| `MemoryItem.in_force_at` | valid time on ledger items; retirement **closes an interval** instead of deleting a row | "prove the guidance was valid at the time" was true for the edit and false for the guidance |
| `PrimeAgent` | `dspy.RLM` + workdir interpreter + sub-agents | the thing being improved |

### New in the write path (2026-10-19)

| | What it does | Why it is there |
|---|---|---|
| `context.py` | `ContextEditingRLM` + `EditableContext`: five primitives on the live tool surface (`evict_step`, `compact_step`, `write_sketchpad`, `mark_context`, `rewind_context`), every edit priced in re-prefill chars | the online mutable surface. Two views — `render()` feeds the prompt and may forget; `full_trajectory()` feeds coverage/monitor/refine/credit and may not |
| `contextgate.py` | the same `LadderVerifier`, separate instance: `RetentionProbe` (1) → `ReprefillProbe` (2) → `PredictContextVerifier` (100) | a mid-task edit has nothing graded to be grounded in, so the ledger's own probe would reject every edit while reporting itself as a grounding check |
| `contextfns.py` | three composite templates (`drop_failures`, `drop_duplicates`, `fold_range`) over the primitives — **templates, never synthesized code** | a candidate that supplies the implementation can redefine what the operation means. Each expansion still passes the gate individually; the composite is atomic where the primitives are not |
| `ContextCheckpoint` | rewind restores the prompt view only — never `executed` or `reprefills` | a rewind must not launder a turn out of existence or make backtracking look free |
| `errorbudget.py` | SLO objectives on *machinery* metrics (`turn_limit_rate`, `edit_admission_rate`); a breach **halts** learning | the only component permitted to act. Admission asks if an edit is grounded, credit asks if it helped; neither sees both working while the system still degrades |
| `synthesis.py` | proposes checks from rubric criteria and admits them on replay precision/recall | the hand-written pair is §3's named honest limit |
| `parity.py` | the deterministic context-management task family and the Phase 0 verdict thresholds | LAB's 29pp band is an unreadable objective; a cheap deterministic task is a readable one |
| `children.py` | `prefix_pilot=True` runs the first child of each prefix group alone, then releases the rest | fan-out and prefix caching pull in opposite directions — maximal parallelism and maximal reuse are the same decision made two ways |
| `lab.py` | Harvey LAB loader and rubric judge, using **LAB's own prompt** | grading our own loop against a rubric we wrote would not be an evaluation |

## The active request path, and the flags that change it

`assemble()`'s defaults **are** the configuration the project claims. A thinner default is how
five of six call sites once ended up under-wired — nobody chose it, it was just the shortest
constructor. There are no exceptions left: everything below is on in the request path unless
you switch it off.

```
gate=ladder audit=on credit=on monitor=on record=on budget=on ctx=gated explore=0 machinery=9f337b069ddf
```

That is `Assembly.describe()`, logged at the start of every script. Two runs of "the same" arm
with different compiled judges are not the same experiment, and the arm name alone does not say
which — hence the machinery fingerprint.

| Flag | Default | What switching it off ablates |
|---|---|---|
| `gate` | `"ladder"` | `None` applies every well-formed edit (no admission control); `"probe"` is the hermetic deterministic-only rung set. Applies to **both** ladders — a no-network run is a no-network run on both paths |
| `audit` | `True` | the append-only compliance record |
| `credit` | `True` | lesson retirement — edits are applied whether or not they later help |
| `monitor` | `True` | live thrash detection (`replan` audit events) |
| `record` | `True` | the `ProposalLog` training-data stream |
| `error_budget` | `True` | the freeze. Detection continues; nothing acts on it |
| `editable_context` | `True` | **the write path.** `False` is the read-path baseline arm and restores prefix-cache determinism |
| `context_gate` | `None` | tri-state, and the only one. `None` follows `editable_context`; **explicit `False` is the real ablation — the write path running ungated, which is the CLM-shaped baseline the thesis argues against** |
| `enable_children` | `False` | sub-agents are opt-in because `prefix_pilot` is a real trade, not a free win |
| `explore` / `seed` | `0.0` / `0` | on this fraction of *rejected* rounds, the ladder runs the rungs it short-circuited past anyway, so a rung the greedy ordering starved still accumulates statistics. Defaults to 0 because it spends the expensive rung on edits already rejected |
| `program` | `None` | a GEPA-compiled judge prompt; requires `gate="ladder"` |

**Ablation stays a flag, not a branch** — so switching one mechanism never takes a different
assembly path that might differ in some second way nobody controlled for.

**Four combinations are refused rather than degraded**, because a mechanism that reports as on
while doing nothing produces a null result that looks like evidence — worse than a crash:

| Refused | Why |
|---|---|
| `credit=True, audit=False` | `CreditAssigner` reads targets off the `AuditRecord`, so it would attribute nothing while reporting itself on |
| `error_budget=True, audit=False` | the freeze is an audit event; an unrecorded freeze reads as a run with nothing to learn rather than one that was halted |
| `program=…` with a non-ladder gate | there is no judge to load the prompt into, so a tuned run would report having used the untuned bar |
| `context_gate=True, editable_context=False` | no edits exist to admit, so the gate would be built, reported on, and never consulted |

## Relationship to Prime Agent, and to CLM

The continual-harness *idea* is not ours. [Prime Intellect's Prime
Agent](https://github.com/PrimeIntellect-ai/prime-agent) introduced it: durable supplemental
state refined through small evidence-backed updates that never rewrite the immutable base
prompt, with snapshots supporting rollback. `ContinualHarness` is a DSPy-native
reimplementation of that design, and the taxonomy (`note | memory | sub_agent_spec`) is theirs.

The editable-live-context *idea* is not ours either. It comes from the CLM line of work, which
frames the model's own context as a mutable workspace and reports accuracy and FLOP wins from
letting the model manage it. The one-line framing: **RLM is the read path — when and what to
read into context. CLM is the write path.**

What is ours sits one level out of each. Prime Agent refines the agent's *context*; this
searches over the agent's *execution policy* — with an evaluator the candidate cannot touch, a
budget it cannot raise, and held-out families it cannot see. CLM gives the write path to the
model unrestricted; this puts an **admission ladder** on it, on the argument that a label-free
write path needs one for the same reason the ledger does. Neither is a straight port.

Two things are deliberately *not* borrowed: code synthesis for context functions (templates
instead — `interpreter.py` is not a sandbox, and a candidate that supplies the implementation
can redefine the operation), and harness-to-weights distillation, which `docs/PLAN.md` §8
excludes.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Python 3.10+
.venv/bin/pytest -q
```

## Run it

```bash
# the policy search
.venv/bin/python scripts/policy_lab.py --out lab_runs/first     # scripted, no network
.venv/bin/python scripts/policy_lab.py --live --model PROVIDER/MODEL \
    --rounds 2 --cases 3 --out lab_runs/live-first              # real proposer

# synthesize checks, then search over the merged catalog
.venv/bin/python scripts/synthesize_checks.py                   # replay stage; reports only
.venv/bin/python scripts/synthesize_checks.py --live --model PROVIDER/MODEL \
    --out lab_runs/syn/checks.json
.venv/bin/python scripts/policy_lab.py --checks lab_runs/syn/checks.json --out lab_runs/second

# does the baseline RLM already edit its context competently?
.venv/bin/python scripts/parity_gate.py --live --model PROVIDER/MODEL \
    --per-kind 4 --out lab_runs/gate
```

`synthesize_checks.py` and `parity_gate.py` both **withhold their result without `--live`**:
the first proposes nothing, because a deterministic stand-in proposer would be a hand-written
check wearing synthesis's name; the second runs an echo baseline through the real loop as the
instrument's self-test and declines to return a verdict, because its question is about a
model's behavior.

`--out` must name a directory that does not exist — a run resuming into a used archive could
silently re-read a final test it had already spent. Live mode does not load `.env`; put
credentials in the process environment. No default provider is assumed.

Artifacts: `archive.json`, `report.json`, and per-attempt `execution.json` carrying assembly
descriptions, model identities, RLM trajectories, cache statistics, and reported usage.

## The plan

[`docs/PLAN.md`](docs/PLAN.md) — the goal, the one experiment, what is missing, and four ways
it fails. It is the only plan; earlier planning documents were deleted rather than archived,
because having a menu of extensions to hand is what kept pulling work away from the
experiment. Retired *code* is in [`graveyard/`](graveyard/README.md).
