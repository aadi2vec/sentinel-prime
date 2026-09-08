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

The search substrate is built and merged: bounded candidate policies, an evaluator the
candidate cannot touch, family-disjoint dev/validation/test with reservation, promotion,
rollback, and a matched-compute control. What does not exist is the experiment.

| Verb in §2 | Today |
|---|---|
| collect real failures | tasks are synthetic; the solver's mistakes are **scripted** |
| synthesize a checking strategy | the proposer **picks from two hand-written checks** |
| on tasks it has not seen | one generated template, shared across all splits |

Three tasks close it, in order:

1. **A LAB-backed family.** `Case.task` is a real LAB task, `Case.gold` its rubric criteria,
   `evaluate` the `RubricJudge`. Real failures in.
2. **Check synthesis.** A `dspy.Signature` from criterion text to a **parameterised** check —
   span existence, amount match, structural presence. Not arbitrary code: `interpreter.py` is
   explicitly not a sandbox, so the synthesizer selects and fills a template rather than
   emitting Python. A synthesized check joins the catalog only after agreeing with the judge
   on held-out already-graded criteria.
3. **Replay scoring.** A candidate is scored against **stored** runs, not fresh agent runs —
   in the existing substrate, a `PolicyRunner` whose `solve` is a lookup. Available today:
   368 graded criterion rows, 10 deliverables, 2 trajectories.

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
