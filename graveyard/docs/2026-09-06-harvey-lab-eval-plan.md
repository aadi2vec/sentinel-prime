# Evaluation plan — SentinelPrime on Harvey LAB (M&A due diligence)

**Date:** 2026-09-06
**Status:** Plan — no results yet. Everything below is design; nothing here is a claim.
**Benchmark:** [harveyai/harvey-labs](https://github.com/harveyai/harvey-labs) (MIT), corporate-M&A slice.

---

## 0. Why this document exists

Every mechanism in this repo is wired, ablatable, and tested against synthetic fixtures whose
agent is *stipulated* to behave a certain way. That proves the machinery runs. It proves
nothing about legal reasoning. The gap between "the harness works" and "the harness helps" is
the whole remaining risk in the project, and this is the plan to close it.

The failure mode to avoid is producing a rising curve and calling it self-improvement. A curve
can rise from task ordering, judge drift, or a lucky seed. **The result is the gap between two
arms, never the shape of one line.** §4 is the part of this plan that matters.

---

## 1. What LAB actually is (grounding)

From the benchmark's own docs and Harvey's M&A announcement:

- **Task layout:** `tasks/<practice-area>/<task>/[<scenario>]/task.json` plus a `documents/`
  folder. `task.json` carries `title`, `instructions`, `work_type`
  (`analyze` | `draft` | `review` | `research`), `deliverables` (expected output filenames),
  `criteria` (inline pass/fail rubric), and `tags`.
- **Workspace:** the agent reads `/workspace/documents` (read-only) and writes
  `/workspace/output`. Documents include `.docx`, `.xlsx`, `.pptx`, `.pdf` and plaintext.
- **Scoring:** *all-pass* — `score = 1.0 if every criterion passed else 0.0`. Criteria are
  atomic and binary, covering facts, conclusions, citations, severity ratings,
  recommendations, deadlines, dollar amounts and formatting. Judged by LLM judges, with
  `scores_sonnet.json` / `scores_gpt.json` / `scores_dual.json` outputs.
- **Reported aggregates:** all-pass rate, **pooled criterion pass rate**, criteria-level
  heatmaps, document coverage, token usage, latency, estimated cost.
- **Scale of the M&A slice:** a VDR in the tens of millions of tokens; one exemplar is a
  $458M acquisition with eight material contracts plus a 10-K and a deferred compensation
  plan, nine planted legal issues, and **57 rubric criteria for a single task**.
- **Difficulty:** frontier agents complete **<10%** of tasks end-to-end under all-pass.

Three consequences fall straight out of these facts and shape everything below.

### 1.1 All-pass is the wrong primary metric for a learning curve

At <10% completion with 57 binary criteria ANDed together, all-pass is a near-zero, near-zero-
variance signal. A harness could fix a third of the remaining criteria and move all-pass by
nothing. **Primary metric: pooled criterion pass rate. Secondary: all-pass rate.** Both are
already LAB-native aggregates, so this is not a metric we invented to look good — but we must
say plainly which one leads, because leading with pooled and quoting Harvey's <10% headline in
the same breath would be a sleight of hand.

Conveniently, pooled criterion pass rate is *also* what `parse_lab_result` already computes as
`Feedback.score`, and criterion-level pass/fail is exactly the label-free signal `refine()`
consumes. The benchmark's native granularity and the harness's learning signal are the same
object. That is the strongest argument that LAB is the right benchmark for this system.

### 1.2 The criterion taxonomy maps onto the verifier ladder

LAB's criteria split into kinds that are *asymmetrically* cheap to check:

| Criterion kind | Cheap deterministic probe | Needs the generative judge |
|---|---|---|
| citations | does the cited span exist in the VDR? | — |
| dollar amounts, deadlines | regex + arithmetic against source | — |
| formatting | deliverable structure check | — |
| facts | string/entity presence | ambiguous phrasing |
| conclusions, severity, recommendations | — | yes |

This is the criterion-specific mapping that `CostLadderPlanner.from_audit_log` was built for
and that the current synthetic rungs could not honestly supply. **Task 3 below builds real
per-kind probes**, at which point the planner's ledger-derived statistics become the live
ordering source and the `adaptive=True` fallback becomes a supplement rather than the whole
story.

### 1.3 Tens of millions of tokens makes cost a co-primary metric

At VDR scale, a self-improvement curve bought with 10× the tokens is not a result. LAB already
reports token usage and estimated cost per run. **Report cost per task alongside every curve,
and treat a quality gain with a cost regression as a negative result.** This is also where
`SubQueryCache` (intra-run sub-query dedup) and the prefix-cache work have to pay for
themselves or be cut.

---

## 2. How SentinelPrime plugs in

LAB's `ModelAdapter` interface (`chat`, `make_tool_result_messages`, `make_system_message`,
`make_user_message`) assumes the harness drives the agent loop. PrimeAgent has its own loop —
`dspy.RLM` turns, a REPL, sub-agents. Forcing it through `ModelAdapter` would mean throwing
away the thing under test.

**So: use LAB as dataset + rubric + judge, and drive the agent ourselves.**

```
task.json ─┐
           ├─► LabTask ─► PrimeAgent.run_task(instructions,
documents/ ─┘                                 workdir=<copy of documents>,
                                              context={matter, doc_shas, ...},
                                              task_id=<task id>)
                            │
                            └─► deliverables written to output/
                                        │
                          LAB judge over `criteria` ──► scores_*.json
                                        │
                             parse_lab_result ──► Feedback ──► agent.learn(...)
```

Two integration decisions worth stating up front:

- **We must use LAB's own judge, not our `judge_lm`.** `config.yaml` has a `judge_lm` role and
  it is tempting to grade ourselves. Grading our own learning loop with a model we configure
  is not an evaluation. If LAB's scorer cannot be invoked standalone over an output directory,
  the fallback is to reimplement it *exactly* against the published `eval-strategies.md` and
  report both judges — never a bespoke rubric.
- **Documents are copied, not mounted.** `LocalInterpreter` is explicitly not a sandbox and
  runs with process permissions. Give each task a disposable workdir copy so an agent bug
  cannot mutate the benchmark corpus.

---

## 3. Build tasks

| # | Task | Acceptance |
|---|---|---|
| 1 | `sentinelprime/lab.py`: `load_task(dir) -> LabTask` reading `task.json` + `documents/`; `to_feedback(scores_json) -> Feedback` | round-trips a real task dir; `Feedback.score` equals LAB's pooled criterion rate for that task |
| 2 | `scripts/run_lab.py --tasks <dir>`: replace the built-in fixtures with real LAB tasks; keep every existing ablation switch working | a single real task runs end to end and produces a `scores_*.json` |
| 3 | Criterion-kind probes (`CitationProbe`, `AmountProbe`, `FormatProbe`) as `VerifierLevel` rungs; planner ordered by `from_audit_log` with the real criterion mapping | ladder order derives from observed per-criterion failure rates; cost per admitted edit drops vs. the LM-only ladder |
| 4 | Run-artifact capture: per-task tokens, latency, cost, ledger size, rejections, retirements, prefix-cache hit rate | one JSON per run, sufficient to reproduce every number in a results table |
| 5 | Experiment runner: arms × seeds × task orders (§4), resumable | produces the §4 table unattended |

Task 3 is optional for a first result and required for the cost claim.

---

## 4. Experimental design — the part that decides whether any of this is real

### 4.1 Arms

Every arm sees the **same tasks in the same order** under the same seed. The only difference is
what the harness is allowed to do.

| Arm | Description | Tests |
|---|---|---|
| **A. Frozen** | Ledger empty and `refine()` disabled | the baseline agent; the control |
| **B. Full** | Everything wired | does online learning help at all? |
| **C. No verifier** | `--no-verifier` | does admission control matter? |
| **D. No credit** | `--no-credit` | does retirement matter? |
| **E. No context** | `--no-context` | does reuse gating matter? |
| **F. Oracle-ish ceiling** | ledger hand-written by a human from the same failures | how much of the headroom does the automatic loop capture? |

**The headline result is B − A**, reported with a confidence interval. Arm A is not optional:
without it, a rising curve is uninterpretable. Arm F is what keeps the result honest in the
other direction — if B captures 5% of the gap that F captures, say so.

### 4.2 Controlling for task order

Online learning across a sequence is order-sensitive, and a benchmark slice is small. So:

- **≥5 random task orders per arm**, fixed across arms (order *i* is identical in every arm).
- Report the curve as **mean ± SD across orders**, never a single run.
- Additionally report the **permutation null**: shuffle the pairing between "position in
  sequence" and "score" and recompute the trend. If the observed trend is inside that null
  distribution, there is no learning effect, whatever the line looks like.

### 4.3 Held-out generalization

A ledger that helps on the tasks that produced it may only have memorized them. Split by
**scenario/task family**, not by individual task:

- **Learn split** — the agent runs, fails, refines. Ledger grows here.
- **Held-out split** — different task family, same practice area. Ledger is **frozen**
  (`refine()` off, `read()` on).

Two numbers matter and they are different claims:
1. *In-sequence gain* — does the curve rise on the learn split? (weakest claim; contaminated)
2. **Transfer gain** — does the frozen ledger from the learn split beat an empty ledger on the
   held-out split? (the claim worth making)

If transfer gain is ~0, the honest write-up is "the harness memorizes within a task family and
does not generalize," and that is still a publishable finding about label-free self-improvement.

### 4.4 Judge variance

The judge is an LLM. Before attributing any delta to the harness:

- Use `scores_dual.json`; report per-judge numbers too.
- Re-judge one arm twice and report **judge-rerun variance** as a noise floor. Any effect
  smaller than that floor is not an effect.

### 4.5 Ledger health, reported alongside every curve

A curve that rises while the ledger grows without bound is a different (worse) result than one
that rises while the ledger stays small. Every run reports: ledger size over time, edits
rejected, lessons retired, lessons withheld by gating, and mean guidance-block tokens. These
are already instrumented.

---

## 5. Cost model and budget

Rough sizing before committing spend — a full grid is arms(6) × orders(5) × tasks × epochs,
and VDR-scale tasks are not cheap.

- Pilot on **one** M&A task (`corporate-ma/analyze-change-of-control-provisions-…`), arms A/B,
  1 order, to shake out the adapter and measure real cost per task.
- Extrapolate, then decide the grid. Cut orders before cutting arm A.
- Sub-query dedup and prefix caching are the two levers; measure their effect in the pilot,
  because at tens of millions of tokens they decide whether the full grid is affordable.

---

## 6. Threats to validity — state these in any write-up

1. **Contamination.** Frontier models may have seen public LAB tasks. Mitigate by reporting
   arm A's absolute score against Harvey's published baselines; a suspiciously high baseline is
   the tell.
2. **Judge gaming.** Label-free refinement optimizes against rubric-failure *text*. It can
   learn to satisfy the judge rather than do the legal work. The held-out split and any
   human spot-check of high-scoring deliverables are the only real defenses. **Spot-check at
   least 10 deliverables by hand** and say so.
3. **Ledger overfitting to one matter.** Reuse gating exists precisely for this; §4.3 tests it.
4. **We built the fixtures the mechanisms succeed on.** Everything in `run_lab.py` today is
   synthetic and partly stipulated. No number from it belongs in a results table.
5. **Small N.** One practice-area slice is not "legal reasoning." Scope claims to the slice.

---

## 7. Milestones

- **M1 — Adapter (no results).** Tasks 1–2. One real LAB task runs end to end and is scored.
- **M2 — Pilot.** Arms A/B, one task, one order. Deliverable: real cost per task, and the first
  honest statement about whether anything moves.
- **M3 — Headline.** Arms A–E, ≥5 orders, learn + held-out splits. Deliverable: B−A with a CI,
  transfer gain, and the ablation table on real data.
- **M4 — Cost.** Task 3 probes + prefix-cache measurement against the live provider. Deliverable:
  quality-per-dollar, which is the claim that distinguishes this from "add more LLM calls."

**M2 is the decision point.** If arm B does not beat arm A on the pilot, the correct move is to
say so in the README and pivot the project's claim to the audit/compliance layer — which stands
on its own and does not depend on the learning curve working.
