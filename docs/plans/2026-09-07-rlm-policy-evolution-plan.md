# RLM Harness Evolution: Plan of Attack

**Direction:** agent-systems research. Harvey LAB is a demanding evaluation environment,
not a decision to build a legal product. This plan supersedes the product-first direction
in `2026-09-07-evidence-runtime-product-strategy.md` for current implementation work.

## Research question

Can an RLM improve how it computes—decomposition, context allocation, verification,
recovery, and delegation—by proposing and testing changes to its own harness?

Start with one mutable surface: **which verification checks run, and whether their failures
trigger another solver attempt**. Success means independently measured improvement on new
tasks within a fixed execution ceiling. It does not mean accumulating more instructions,
agreement with the agent's own judge, or changing the benchmark to reward itself.

## Architecture and invariants

```text
Current policy + development failures
                  |
DSPy policy proposer -> bounded candidate JSON
                  |
Schema, check-catalog, and worst-case budget validation
                  |
Fresh assembled RLM attempts under candidate and current policy
                  |
Fixed external evaluator on a fresh validation batch
                  |
Promote or reject -> persist policy version and trial evidence
                  |
Next development round; final test only after policy selection
```

- Every actual agent comes from `sentinelprime.assembly.assemble()`. Preserve its full
  default gate, audit, credit, monitor, and proposal-log wiring. Children and semantic
  caching retain the assembler's opt-in defaults; do not claim these are enabled.
- Each solver attempt has a fresh ledger and workspace. This first experiment does not
  invoke `learn()` on evaluation data. The continual machinery stays wired, but only the
  execution policy changes; enabling two learning mechanisms would confound attribution.
- Candidate policies select registered checks and a bounded revision count. They cannot
  provide arbitrary code, modify check implementations, change the evaluator, or raise
  the fixed budget. The runtime actually performs checks and revisions.
- Solver/checker inputs contain the public task, not the evaluator's gold object. The
  proposer receives development observations only. No validation/test labels enter the
  proposer's feedback or supplemental memory.
- A new validation batch is consumed each round. Family boundaries separate development,
  validation batches, and final test. Reservation persists before work begins so restarting
  against the same archive cannot accidentally reuse a final test.
- Exceptions and invalid scores produce incomplete evaluation, not task failures. Incomplete
  validation cannot promote a candidate. The final test never participates in promotion.
- Versioned policy state is atomically replaced and history retained; rollback selects an
  earlier version. This is single-writer durability, not distributed consensus.

## Phase 1 — implemented: a runnable bounded policy laboratory

Files:

- `sentinelprime/policy_search.py`: policy contracts, execution controller, DSPy proposal
  signature, fixed-evaluator search, family reservation, promotion history, and rollback.
- `sentinelprime/policy_experiment.py`: full-assembly RLM adapter, fixed check catalog,
  generated source/arithmetic tasks, independent exact scorer, and scripted model seam.
- `scripts/policy_lab.py`: network-free default and explicit live-model mode.
- `tests/test_policy_search.py`: behavior, budget, leakage, failure, persistence, and actual
  assembled-RLM integration tests.

Run the mechanism demo:

```bash
.venv/bin/python scripts/policy_lab.py --out lab_runs/policy-lab-first
.venv/bin/pytest tests/test_policy_search.py -q
```

The output directory must be new. Artifacts include `archive.json`, `report.json`, and
per-attempt `execution.json` files containing assembly descriptions, model identities,
RLM trajectories, cache statistics, and reported token usage. Reported provider usage may
be unavailable for scripted models; call counts are reported separately.

The scripted RLM initially ignores source amendments and miscalculates a total. The first
scripted policy requests arithmetic checking and revision; it does not repair the source
error and is rejected. The next requests source-revision and arithmetic checks and can be
promoted when its revised answer improves on new source bundles. Both the solver behavior
and the proposal sequence are stipulated; the actual RLM, interpreter, assembler, checks,
selection, and persistence execute. This is a mechanism test, **not evidence of model
self-improvement**. The checkers are hand-written, not synthesized.

Live mode uses a DSPy predictor to propose policies from observed development results:

```bash
# Set the chosen provider's credentials in the process environment first.
.venv/bin/python scripts/policy_lab.py --live --model PROVIDER/MODEL \
  --rounds 2 --cases 3 --out lab_runs/policy-lab-live-first
```

No default provider is assumed; this script does not load `.env` automatically. No paid
run is required for the hermetic tests. Even live success on these generated templates
would not demonstrate general or legal reasoning improvement.

The initial budget is at most two solver attempts and four check invocations per task;
each assembled RLM is capped at six turns and four subqueries. These are execution-call
ceilings, not a hard token/dollar or wall-clock cap. Proposer calls also cost tokens. A
candidate can use more of the common ceiling than the baseline, so gains do not establish
compute efficiency; use matched-compute baselines in Phase 2.

## Phase 2 — establish a real effect before expanding the search space

1. Add two task families with independently checkable answers, such as table reconciliation
   and small code transformations. Preserve source-bundle/template splits explicitly.
2. Run real-model trials with repeated paired cases and counterbalanced arm order. Pin
   model settings, solver/check implementations, and data versions. Record all spend,
   including proposals, checks, subqueries, and revisions.
3. Compare no checks, fixed check-all, a fixed revision strategy, and the evolved policy.
   Add a matched-compute second-solve baseline so extra inference is not credited to learning.
4. Predefine quality, regression, and cost gates using enough independent examples. The
   Phase 1 rule (positive mean gain and no observed per-case regression) is deliberately
   simple and is not a statistical significance test.
5. Use a genuinely untouched test set after selection. Repeatedly reading and reacting to
   the same final-test report turns it into development data, even across fresh directories.

**Go/no-go:** if policy search cannot beat fixed verification at comparable resources,
investigate the failure before adding more recursive machinery. Flat results are useful.

## Phase 3 — learn verification strategies

Expand the policy language from check names to bounded compositions: retrieve evidence,
recompute, search for a counterexample, compare sources, then accept/revise/abstain. Keep
the operations registered and inspectable. The model may propose a new composition, not
execute arbitrary self-written validator code against production state.

Record verifier mistakes against external outcomes. Trial candidates on examples of false
acceptance and false rejection as well as normal tasks. Optimize DSPy verifier prompts or
strategy selection only against independent labels. Preserve the final evaluator outside
this search space. A second model agreeing with the first is not a new source of truth.

**Go/no-go:** a new checking strategy catches an unseen error class without unacceptable
false rejections or cost growth. Distinguish selecting existing checks (Phase 1) from
discovering new compositions (this phase).

## Phase 4 — make distributed-systems mechanisms earn their place

Add one mechanism at a time, only for a measured failure:

| Mechanism | Triggering problem | Experiment |
|---|---|---|
| Checkpoint/recovery | Expensive work lost on failure | Inject failures; compare recovery correctness and recomputation |
| Dependency invalidation | Stale intermediate conclusions after input change | Change one source; measure missed invalidations and work saved |
| Idempotency/deduplication | Duplicate execution or updates | Retry identical work; verify one logical effect and reduced calls |
| Backpressure/scheduling | Children exhaust shared resource limits | Compare quality/latency under one enforced shared budget |
| Speculative branches | Single-path reasoning gets stuck | Compare checked branch selection with matched-compute retries |

Do not equate model majority voting with correctness, or promise deterministic replay of
arbitrary Python/provider calls. Start with explicit artifacts and task boundaries rather
than an immediate Rust rewrite or arbitrary interpreter-state snapshots.

## Phase 5 — reconnect continual memory and Harvey LAB

Once execution-policy gains are measurable, introduce the existing continual harness as a
separate experimental axis: frozen versus learned memory, crossed with fixed versus learned
execution policy. Freeze identical starting ledger snapshots for comparisons. Only development
outcomes may update memory; validation/test runs get read-only copies of the chosen state.

Harvey LAB then tests whether the mechanism survives a difficult document-rich environment.
The current LAB adapter now distinguishes judge errors; verify that repair throughout the
evaluation/reporting path and build reliable external review signals. Legal remains
an evaluation environment; no legal UI, sales workflow, or compliance certification is needed.

## Limits of this implementation

The experiment is opt-in and leaves existing LAB entry points and memory semantics unchanged.
Its Python callables and current local interpreter are trusted code, not an adversarial isolation
boundary. The RLM can access host resources with the process's permissions; input separation
alone cannot protect hidden tests against a malicious generated program. Use OS-isolated workers
with only public task artifacts mounted before testing adversarial self-modification.

There is no autonomous weight training, arbitrary harness rewrite, learned check implementation,
or demonstrated superiority over Prime Agent here. The delivered first step is a controlled
experiment in which an assembled RLM's execution policy can change, be tested, and be reversed.
