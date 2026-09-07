# Execution-policy search — design

**Date:** 2026-09-07
**Status:** Design. No results. Nothing here is a claim about what the system can do.
**Supersedes as project direction:** `docs/plans/2026-09-07-evidence-runtime-product-strategy.md`
(§10 packaging, §11 supervised pilot, design partners). See §8.

---

## 1. The question

> Can an RLM improve the machinery it uses to solve problems — how it decomposes work,
> coordinates computation, manages context, checks results, and learns from failure?

This is an agent-systems research project. Harvey LAB is the first demanding environment and
nothing more: a stress test, not a product surface. The result we are after is *which parts of
its own computation a harness can learn to improve, and where that process stops working.*

The shift from what this repo has been doing is one level of abstraction. `refine()` proposes
changes to the agent's **context** — lessons in a ledger. This proposes changes to the
**execution policy** — the bounded set of decisions about how work gets done and checked.
Content-learning accumulates advice; policy-learning searches over how the system computes.

### 1.1 Why the change of object, stated as a falsifiable reason

We measured it. Across five runs of a no-ledger configuration on one LAB task, the pooled
criterion rate spanned **0.400 – 0.691** (29pp). A hand-written oracle lesson (arm F,
`scripts/arm_f.py`) moved it **−5.4pp** while costing 40–84% more tokens. That is not a
refutation of content-learning — the run was contaminated (§1.2) and the targeted failure did
not reproduce — but it establishes that *content-learning effects on this benchmark are
smaller than the measurement noise*, and no amount of mechanism fixes an unreadable objective.

Policy-learning has a better-conditioned objective, and §4 is the reason.

### 1.2 What that experiment also found, now fixed

44 of 220 judge calls (20%) hit rate limits and were scored as criterion **failures**. Beyond
deflating the rate, the exception text flowed through `to_feedback` into `Feedback.as_text()`,
which is the proposer's only view of why a run fell short — so a learning arm wrote ledger
lessons about the provider's rate limiter. Fixed on 2026-09-07: `RubricJudge` retries with
backoff and then emits a third verdict, `error`, which `to_feedback` excludes from both the
pooled rate and the proposer's input; `judge_errors()` reports the count; `all_pass(fb,
results)` refuses the claim when a criterion was never graded.

Any result computed before that fix is suspect, including arm F's.

---

## 2. What already exists

| Piece | Where | What it gives this design |
|---|---|---|
| `Policy(checks, max_revisions)`, `Budget`, `Case`, `PolicyRunner`, `PolicyArchive`, `PolicySearch` | `policy_search.py` | the search substrate: bounded candidates, family-disjoint dev/validation/test, staged promotion, rollback |
| `assemble(**kwargs) -> Assembly` | `assembly.py` | the only wiring path. A candidate configuration *is* an `assemble()` kwargs dict, so a policy cannot be expressed outside SentinelPrime |
| `Check(name, cost, run)`, `CostLadderPlanner.order/expected_cost` | `planner.py` | the check type and cost/P(fail) ordering. Nothing generates checks |
| `LadderVerifier`, `VerifierLevel`, `GroundingProbe` | `verifier.py` | one hand-written checking policy, currently fixed |
| 368 graded criterion rows + trajectories | `lab_runs/` | the replay corpus (§4) |

`policy_search.py` already enforces the constraint this design depends on: `Case.gold` is
documented *"Only the evaluator receives this object, never solve/check/propose"*, and
`Policy.parse` rejects an `evaluator` field outright.

---

## 3. The evaluation boundary

**The candidate may change execution. It may not change what counts as success, or how much
it is allowed to spend.** Otherwise it improves its score without improving its problem-solving,
and model agreement does not become truth by being called a quorum.

The boundary is currently leaky in one specific place. `assemble()` exposes
`credit_min_exposures` and `credit_min_success_rate`, and `build_ladder` exposes `min_score`.
Those are not execution parameters — they define what counts as a lesson earning its place and
what counts as an acceptable edit. A search that can reach them lowers the bar instead of
clearing it.

**Required change:** split `assemble()`'s signature into a searchable group and a frozen group,
so the separation is a type rather than a convention.

```
assemble(lm, root, *, policy: PolicyConfig, evaluation: EvaluationConfig, ...)
```

- `PolicyConfig` — searchable: which checks, their order, revision count, explore rate,
  child spawning, sub-query caching.
- `EvaluationConfig` — frozen, never written by a proposer: judge, rubric, credit thresholds,
  admission score floors, `Budget`.

`PolicySearch` accepts only a `PolicyConfig`. A proposer that emits an `EvaluationConfig` field
raises, exactly as `Policy.parse` already does for `evaluator`.

---

## 4. Replay evaluation — the economics of the whole program

A live task run costs ~218s and ~300k tokens, and carries 29pp of variance. Scoring one
candidate policy per agent run makes policy search unaffordable and unreadable at the same time.

**It doesn't need one.** A checking policy is a function from `(trajectory, deliverable) →
flagged claims`. Its score is measured against verdicts that already exist. There are 368
graded criterion rows on disk with trajectories, and `arm_f.py`/`paired_ab.py` now persist
every new one.

In the existing substrate this is not new machinery: **replay is a `PolicyRunner` whose `solve`
is a lookup into stored runs.** The candidate's checks execute for real; only the solver is
served from cache. Deterministic checks cost nothing, and the agent run is amortised across
every candidate ever tested.

**The honest limitation, stated up front.** Replay can score *detection* — did the check catch
a mistake the judge later marked wrong. It cannot score *correction*, because a stored run
cannot be re-solved: `max_revisions` is unmeasurable under replay. So evaluation is two-stage,
and the stages answer different questions:

| Stage | Cost | Answers | Scores |
|---|---|---|---|
| **Replay** | ~0 | does this check set find real mistakes without crying wolf? | precision/recall at a cost ceiling |
| **Live** | 1 agent run per candidate | does acting on the flags actually fix them? | pooled rate, at fixed budget |

Replay ranks and filters. Live confirms the survivors. A candidate that wins on replay and
loses live is itself a finding — it says detection transfers and correction does not.

---

## 5. First experiment: learn when and how to verify

Smallest interesting slice of the loop. The system improves one decision: its checking strategy.

1. **Freeze the baseline.** Current ladder, current budget, via `assemble()`. Score it on
   replay.
2. **Collect failures.** From the graded rows: criterion text, judge reasoning, trajectory.
3. **Synthesize candidate checks.** A `dspy.Signature` from criterion text to a deterministic
   `Check` — span existence, arithmetic reconciliation, structural presence, re-execution.
   Each emitted check is admitted to the catalog only after agreeing with the judge on
   *held-out already-graded* criteria, so a check earns its place before it can be chosen.
4. **Propose a policy.** `PolicyProposer` selects from the catalog under `Budget`.
5. **Trial by replay** on family-disjoint validation batches. Score: **caught-at-budget** —
   failures flagged minus false alarms, under a ceiling from `CostLadderPlanner.expected_cost`.
6. **Promote or discard** through `PolicyArchive`. Promotion is staged; `rollback(version)`
   is the undo.
7. **One live run** per promoted policy, on the held-out test families, to test whether the
   replay estimate transfers.

**Primary metric:** precision/recall of the check set at a cost ceiling. This is a property of
the checking strategy over a fixed corpus, so agent sampling variance does not enter it. That
is the point.

---

## 6. The generality guard

If LAB is the only environment while the policy language is being designed, the language will
be legal-shaped by the time a second family arrives, and "we built a general mechanism" will be
untestable after the fact.

**Constraint on the check vocabulary: a check may not name documents, clauses, or rubric kinds.**
It may only express spans, arithmetic, structure, and re-execution. A second task family's
adapter is stubbed now — empty is fine — so the representation is forced to stay portable while
changing it is still free.

Transfer remains an empirical question. The *representation* must not prejudge it.

---

## 7. What would falsify this

Stated in advance, so a null result is a result:

- **Synthesized checks do not beat the hand-written ladder on replay.** Then check synthesis
  fails and the ladder's rungs are as good as this method gets.
- **Replay ranking does not survive the live run.** Then detection does not imply correction,
  and the cheap evaluation loop is not a substitute for the expensive one.
- **A policy wins on LAB and dies on family two.** Then we learned legal procedure, not a
  mechanism — which is exactly what §6 exists to detect rather than hide.
- **Every candidate lands inside the noise.** Then the objective is still unreadable and
  variance reduction is the only remaining job.

---

## 8. Non-goals

Retired deliberately, and named because a document in this repo currently proposes them:
a legal review product, design-partner recruitment, tenant access controls, packaging or
pricing, a compliance platform, a UI. The audit, provenance and reversibility machinery stays
— as *instrumentation* that makes the search auditable, not as a product claim.

`docs/plans/2026-09-07-evidence-runtime-product-strategy.md` §§10–11 should be marked
superseded rather than deleted; the execution-architecture material in its §4 remains useful.

---

## 9. Open questions

1. **Does the check catalog need to be closed?** An LM-synthesized check is code. Admitting it
   to a catalog that then runs in-process is a real hazard given `interpreter.py` is explicitly
   not a sandbox. Options: restrict synthesis to a parameterised template set (safe, less
   expressive) or isolate execution (expressive, more work). Unresolved.
2. **What is the right budget unit?** `Budget` counts solves and checks. Tokens are what
   actually cost money and what the 40–84% arm F regression showed up in. A check that is cheap
   in calls and expensive in context would be mispriced.
3. **Does `explore` belong in the policy?** It changes what the ladder observes, which changes
   the statistics that order it — a policy parameter that alters its own evaluation inputs. It
   may need to sit on the frozen side.
