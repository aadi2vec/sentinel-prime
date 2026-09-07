# sentinel-prime

**Can an RLM improve the machinery it uses to solve problems?**

Not "can it remember more advice" — can it change *how it computes*: how work is decomposed,
how context is allocated, when results get checked, when to try again, when to delegate. This
repo is a harness for asking that experimentally. The system proposes bounded changes to its
own execution policy, trials them in isolation, evaluates them on unseen tasks under a fixed
ceiling, and promotes or discards.

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

## The one mutable surface

Deliberately one thing, so attribution stays possible: **which verification checks run, and
whether a failed check triggers another solver attempt.**

A candidate `Policy` selects registered checks and a bounded revision count. It **cannot**
supply code, alter a check's implementation, change the evaluator, or raise its budget.
`Policy.parse` rejects unknown fields outright — the evaluation boundary is a type, not a
convention. `Case.gold` reaches the evaluator only, never the solver, checker, or proposer.
Development, each validation batch, and the final test are family-disjoint and reserved before
work begins, so a restart cannot re-spend the test set.

That constraint is the whole game. A candidate that can redefine success improves its score
without improving its problem-solving, and model agreement does not become truth by being
called a quorum.

## Status

**Proven:** 314 tests, hermetic, TDD'd. The policy-search substrate runs end to end — bounded
candidates, schema and budget validation, family reservation, staged promotion, rollback, and
persisted per-attempt evidence. Every agent comes from `assemble()`, so a policy is
expressible only as assembler arguments and cannot reach around the assembler.

**Not proven, and this is the important part:** that any model improves its own computation.
`policy_lab.py` is a *mechanism* demo with a scripted solver and a scripted proposal sequence.
The checkers are hand-written, not synthesized. The budget counts solver attempts and check
calls, not tokens — so a candidate can spend more of the shared ceiling than the baseline, and
a gain does not by itself establish compute efficiency. There are **no real-model results.**

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

| | What it does | Why it is there |
|---|---|---|
| `policy_search.py` | bounded candidates, fixed evaluator, family-disjoint splits, staged promotion, rollback | the search, and the boundary that keeps it honest |
| `policy_experiment.py` | full-`assemble()` RLM adapter, check catalog, generated tasks, independent scorer | a trial that runs the real stack |
| `assembly.py` | `assemble(lm=..., root=...)` — the only wiring path | a candidate cannot reach around the assembler |
| `ContinualHarness` | `read()` / `refine()` / `rollback()` / `explain()` — versioned WAL + copy-on-write ledger | supporting machinery: reversible, auditable state |
| `LadderVerifier` | short-circuited cost ladder: cheap deterministic probe → generative judge | verification is asymmetric — and it is the fixed policy the search must beat |
| `AuditLog` | append-only, content-addressed record per edit | a rolled-back edit is still a fact that happened |
| `CreditAssigner` | retires a lesson whose targeted criteria stop passing | a polluted ledger is worse than an empty one |
| `PrimeAgent` | `dspy.RLM` + workdir interpreter + sub-agents | the thing being improved |
| `lab.py` | Harvey LAB loader and rubric judge, using **LAB's own prompt** | grading our own loop against a rubric we wrote would not be an evaluation |

## Relationship to Prime Agent

The continual-harness *idea* is not ours. [Prime Intellect's Prime
Agent](https://github.com/PrimeIntellect-ai/prime-agent) introduced it: durable supplemental
state refined through small evidence-backed updates that never rewrite the immutable base
prompt, with snapshots supporting rollback. `ContinualHarness` is a DSPy-native
reimplementation of that design, and the taxonomy (`note | memory | sub_agent_spec`) is theirs.

What is ours sits one level out. Prime Agent refines the agent's *context*. This searches over
the agent's *execution policy* — with an evaluator the candidate cannot touch, a budget it
cannot raise, and held-out families it cannot see.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Python 3.10+
.venv/bin/pytest -q
```

## Run it

```bash
.venv/bin/python scripts/policy_lab.py --out lab_runs/first     # scripted, no network
.venv/bin/python scripts/policy_lab.py --live --model PROVIDER/MODEL \
    --rounds 2 --cases 3 --out lab_runs/live-first              # real proposer
```

`--out` must name a directory that does not exist — a run resuming into a used archive could
silently re-read a final test it had already spent. Live mode does not load `.env`; put
credentials in the process environment. No default provider is assumed.

Artifacts: `archive.json`, `report.json`, and per-attempt `execution.json` carrying assembly
descriptions, model identities, RLM trajectories, cache statistics, and reported usage.

## Documents

Two, and only two, are live:

- [Plan of attack](docs/plans/2026-09-07-rlm-policy-evolution-plan.md) — phases, invariants, go/no-go
- [Design](docs/superpowers/specs/2026-09-07-execution-policy-search-design.md) — policy
  representation, the evaluation boundary, replay evaluation, and four ways this fails

Everything else was retired on 2026-09-07 into [`graveyard/`](graveyard/README.md), kept rather
than deleted because the surviving documents cite its measurements.
