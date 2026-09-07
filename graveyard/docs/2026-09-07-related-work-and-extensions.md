# Related work, repositioning, and candidate extensions

**Date:** 2026-09-07
**Status:** Design + literature review. No results. Nothing here is a claim about this system.
**Companion to:** `docs/plans/2026-09-06-harvey-lab-eval-plan.md` (which still governs what counts as a result).

---

## 0. Why this document exists

SentinelPrime was designed without a literature sweep. This is the sweep. Two findings:

1. Several of this repo's ideas have been published independently since it was started. Context
   evolution, gated memory updates, and self-generated training data are no longer novel.
2. There is now a *theory* — the generator/verifier gap — that predicts the shape of the curve
   this project is trying to produce, and identifies which component sets its ceiling.

The second finding is more important than the first. It converts several "we should probably"
items into pre-registered, falsifiable predictions, and it reorders the build queue.

---

## 1. Neighbours, and what each one takes off the table

| Work | What it does | What it removes from our novelty claim |
|---|---|---|
| **ACE** — Agentic Context Engineering ([2510.04618](https://arxiv.org/abs/2510.04618)) | Generator/Reflector/Curator evolve a context; incremental *delta* updates rather than rewrites, to avoid **context collapse** (iterative rewriting erodes detail). +10.6% agent tasks, +8.6% finance; matches production agents with smaller open models. | "Evolving a supplemental context beats a static prompt." |
| **TAME** — Trustworthy Test-Time Evolution of Agent Memory ([2602.03224](https://arxiv.org/pdf/2602.03224)) | Gating + verification before accepting memory updates; targets **memory poisoning** and **memory drift**; ships a benchmark. | "Memory updates need an admission gate." Closest neighbour to our thesis. |
| **SEAL** — Self-Adapting Language Models ([2506.10943](https://arxiv.org/abs/2506.10943)) | Model emits "self-edits" (its own finetuning data + update directives); RL loop rewarded by *downstream performance of the updated model*. | "Reward proposed knowledge by whether it measurably helps." |
| **Language Models Need Sleep** ([2606.03979](https://arxiv.org/html/2606.03979v1)) | Knowledge Seeding (upward distillation, smaller/plastic self -> larger stable net) + Dreaming (RL-generated synthetic data; each dream gets an isolated LoRA trial with a **binary reward: did downstream performance improve**; ReST-EM). Beats SEAL (48.9 vs 46.7 SQuAD; 80% vs 72.5% few-shot). | "Sleep/dream consolidation of experience." Note: their consolidation is explicitly **irreversible** — plastic params are reset and pruned after consolidation. |
| **DGM** — Darwin Godel Machine ([2505.22954](https://arxiv.org/abs/2505.22954), ICLR 2026) | Archive of self-modifying coding agents; branch from *any* ancestor. SWE-bench 20.0→50.0, Polyglot 14.2→30.7. Beats no-self-improvement and greedy hill-climbing. | "Branch descendants and keep the archive." |
| **Evo-Memory** ([2511.20857](https://arxiv.org/pdf/2511.20857)) | Benchmark: restructures static datasets into sequential task streams; 10+ memory modules over 10 datasets; proposes ExpRAG and ReMem. | Nothing — this is a *tool for us*. See §4.6. |
| **ShinkaEvolve / AlphaEvolve / CodeEvolve** ([2509.19349](https://arxiv.org/pdf/2509.19349)) | LLM-as-mutation-operator program evolution; bandit LLM ensemble; **novelty-based rejection filtering** for sample efficiency. | Nothing directly — but supplies a mechanism we lack (§4.5). |
| **Introspection threshold** ([2607.04277](https://arxiv.org/html/2607.04277v1)) | Argues current LLMs have only "quasi-introspection" — fragmentary self-knowledge without completeness, reliability, or causal grounding — so sustained RSI is blocked. | Nothing — cite it *for* us (§3). |

Not extractable at time of writing (PDFs would not decode; do not cite until read):
`2607.13104` (self-improvement survey), `2606.04536` (parametric memory).

---

## 2. The finding that reorders the build queue: the generator/verifier gap

The **sharpening mechanism** ([2412.01951](https://arxiv.org/pdf/2412.01951)): self-improvement works
because verification sharpens the output distribution toward high-quality sequences — models
verify better than they generate.

The **solver-verifier gap model** ([2507.00075](https://arxiv.org/html/2507.00075v2)) formalises it.
Track solver capability `U_s` and verifier capability `U_v`; the gap is `G(t) = U_s - U_v`. The
predicted dynamics are **exponential convergence to a fixed point**, with `G(t)` narrowing
exponentially. The ceiling is set by the initial capabilities: **a larger initial verifier-solver
gap predicts better final performance.** Verification signals form a hierarchy from formal
verifiers (strongest) to intrinsic self-assessment (weakest); the failure modes — self-confirming
loops, model collapse, diversity collapse — follow from violating that hierarchy.

Honest caveat from the same paper: empirically on math, accuracy and uncertainty decreased
*together* — unreliable verification misleads the loop in a way the clean model does not capture.
That is TAME's drift failure mode, and it is what the ladder exists to resist.

### 2.1 Three consequences for this repo

1. **The curve should be exponential-to-plateau, not linear.** Fit an exponential; report the
   asymptote and its CI. This is a pre-registered prediction: a *linear* curve is evidence that
   something other than self-improvement is producing it.

2. **Headroom is measurable before the grid is run.** Solver = the agent's pooled criterion rate.
   Verifier = Best-of-N with the agent selecting among its own candidates. The gap predicts how
   much `B - A` is available. Cheaper than arm F, and complementary: **arm F measures context
   headroom, the gap measures verification headroom.**

3. **Task 3 of the eval plan is promoted from optional to critical.** The criterion-kind probes
   (`CitationProbe`, `AmountProbe`, `FormatProbe`) are currently listed as "optional for a first
   result, required for the cost claim." Under this theory, deterministic probes sit *higher in
   the verification hierarchy* than `PredictVerifier`, so shifting admission mass from opinion
   rungs to probe rungs **raises the ceiling**, not merely the price. Task 3 partly determines
   whether there is a result at all.

---

## 3. Repositioning

ACE has context evolution. TAME has gated memory updates. SEAL and Sleep have self-generated
training data rewarded by downstream performance. DGM has branching archives.

What none of them combine: **a cost-ordered, ablatable admission ladder; snapshot-bracketed
reversibility; an append-only audit trail; and credit-based retirement** — in a domain where
those are requirements rather than features. And §2 says the ladder's fidelity is the ceiling on
everything else.

**So the honest framing is that this is a verification project, not a learning project.** That
matches the code, matches the theory, and matches the M2 fallback already written into the eval
plan ("pivot the project's claim to the audit/compliance layer").

The introspection-threshold paper supports this directly: if models cannot reliably introspect,
externalised, verifiable, auditable memory is the right substrate — it does not require the model
to know itself, only the system to record what happened.

**Action:** add a Related Work section to the README citing ACE, TAME, SEAL, Sleep, and DGM, and
state the differentiation in one paragraph. Appearing not to know the neighbours is worse than
having them.

---

## 4. Candidate extensions

Ordered by (novelty x evidence produced) / effort. None of these change the M2 gate: if
`F - A` is ~0, no mechanism here matters.

### 4.1 Measure context collapse (analysis only, data already exists)

ACE *names* collapse; measuring it requires every intermediate version, which nobody keeps and we
do. `refine()` brackets every round between two snapshots and `AuditLog` is append-only, so we can
compute information retention across ledger versions: what fraction of facts present at `v_n`
survive to `v_n+k`, with and without the verifier.

**Acceptance:** a retention-vs-version curve from existing `lab_runs/` artifacts, with the
`--no-verifier` arm as contrast. Turns invariant #2 from a design principle into an instrument.

### 4.2 Adversarial ledger poisoning (the eval the domain hands us)

TAME identifies poisoning abstractly. Our setting makes it *realistic*: in M&A due diligence the
VDR contains documents drafted by the counterparty, and `interpreter.py` is explicitly not a
sandbox. Plant an injected instruction in a source document; measure whether it reaches the
ledger and persists across tasks; then ablate the defence (`GroundingProbe` only, full ladder,
+ credit retirement).

**Acceptance:** an attack-success-rate table across ladder configurations. This result stands
whether or not the learning curve moves, and it makes the compliance framing load-bearing.

### 4.3 The memory hierarchy, with one coherence protocol

Levels distinguished by **admission rule and blast radius**, not latency:

| Level | Scope | Admission | Blast radius | Reversal |
|---|---|---|---|---|
| L0 provider KV prefix | one prefix | byte-identical | none (cost only) | n/a |
| L1 `SubQueryCache` | one trajectory | cosine threshold | one trajectory | evaporates |
| L2 procedural store | cross-task | **gap** | many tasks | needs one |
| L3 ledger | cross-task | ladder + credit | many tasks | `rollback(version)` |
| L4 weights | permanent | — | everything | drop adapter |

**L2 is the real gap: there is no procedural memory.** The ledger holds *declarative* lessons
("check assignment clauses for change-of-control"). A successful trace is *procedural*. Different
retrieval key, different failure mode. This is the same slot as "promote successful REPL code into
named tools" — a tool is the compiled form of a successful path, and unlike a lesson it can be
**unit-tested**, which is a non-LM signal.

Three constraints on the design, each of which kills the naive version:

- **Do not store raw traces.** LAB all-pass is <10%, so "successful trace" is a near-empty set.
  Key on *criterion-level* success instead — which sub-path satisfied which criterion. Dense, and
  already computed by `parse_lab_result`.
- **Store abstracted plans with entities parameterised**, retrieved by *task shape* (work_type +
  criterion kinds + document types), not cosine similarity to the query. Replaying Matter A's
  trace on Matter B is exactly the hazard `ReuseController` exists to block.
- **L2 fights L0.** Invariant #5 exists so the provider can KV-cache a stable sorted prefix.
  Injecting per-task retrieved content at the front causes a full-prefix miss on every task —
  **a semantic cache can raise the bill.** Fix: two-block layout. Stable sorted ledger stays in
  the prefix; per-task retrieved material goes *after* the task text as a suffix.
  `agent.cacheable_prefix()` remains the regression check.

**Where the hardware analogy earns its keep: coherence, not levels.** The hard part of a real
hierarchy is invalidation, which here is document amendment: a source SHA changes and every
dependent entry at every level must be invalidated (at L4, "invalidate" = drop the adapter). That
is `ReuseController.causally_current`, currently a stub. Write invalidation **once**, keyed on
source-document provenance, apply at all levels. Per repo conventions the analogy stays in
`docs/` — do not name anything `L1Cache`.

### 4.4 Refutation memory

`RefineResult.rejected` is per-round and produces no audit record by design, and `current_ledger`
shows the proposer only what was *admitted*. Nothing prevents re-proposing an identical rejected
edit every round forever. Add a rejection store keyed by `content_hash`, fed back to the proposer
as "already refuted, with reason."

Payoffs: stops proposal cycling; gives `CostLadderPlanner.from_audit_log` a persistent statistics
source it currently lacks (rejections are visible only to `adaptive=True`); and makes "what the
system learned *not* to believe" an auditable artifact.

### 4.5 Novelty-based rejection filtering (imported from ShinkaEvolve)

"No decay/dedup of lessons" is on our known-gaps list. Novelty rejection is a published solution:
reject a proposed lesson too close to an existing one. Pairs with a token budget on `read()`, so
lessons compete for prefix space and convergence becomes measurable as *information density
rising while token count stays flat*.

### 4.6 Evo-Memory as the cheap benchmark

LAB is 57 ANDed criteria at VDR scale with <10% completion — we cannot iterate on it. Evo-Memory
is purpose-built for sequential test-time-learning evaluation, with 10 datasets and 10+ memory
modules already implemented, which also gives us baselines instead of only arm A.
**Iterate on Evo-Memory, claim on LAB.** Directly addresses threat #5 ("one practice-area slice
is not legal reasoning").

### 4.7 A branchable ledger archive (DGM's actual lesson)

DGM's load-bearing choice is the archive: branching from *any* ancestor beats greedy hill-climbing
because it escapes local optima. Our audit log is already an append-only archive; it is just not
branchable. `rollback(version)` + branch gives a DGM-style archive over *ledgers* rather than over
agent code — the branching-descendants idea, in the space where we already have reversibility, at
a fraction of the cost of weight branching.

### 4.8 If weights are ever touched: reversible consolidation

The Sleep paper's consolidation is explicitly irreversible. The available contribution is the
opposite: consolidate the ledger into a LoRA adapter but **retain the ledger as the rollback
path**. If the adapter regresses on held-out tasks, drop it — the guidance still exists as text.
Copy-on-write for weights, consistent with the WAL/COW framing this project already uses.

Note also that Sleep's dream-validation (isolated LoRA trial, binary reward on downstream
improvement) answers the "dreams need a simulator" objection: what is needed is a cheap trial-fit
plus a held-out measurement, not a world model. That is the same shape as the student
comprehension probe, at a higher price point.

### 4.9 Point the admission machinery at the deliverable

Every gate in this repo governs what enters *memory*; nothing checks the answer before it
ships. For an accuracy goal that is backwards — a bad lesson costs future tasks, a bad
answer costs this one. Require each material assertion in the deliverable to carry a source
span and run the same ladder over those claims pre-submission.

Also the sharpest available defence against threat #6.2 (judge gaming): a claim that
satisfies the rubric but resolves to no span becomes detectable.

**Acceptance:** claims-without-spans counted per run; deliverable-side rejection rate
reported next to the pooled criterion rate.

### 4.10 Probe synthesis — verifiers that write verifiers

`CostLadderPlanner` orders a *hand-written* rung set, which caps the system at whatever
probes we thought of and hard-codes it to legal. Instead, synthesize a deterministic probe
from the criterion text: "cites § 7.3" -> span existence; "within 30 days" -> date
arithmetic; "$12.5M escrow cap" -> amount match; "severity rating per issue" -> structure.
Each synthesized probe is a `Check(name, cost, run)` — the interface already exists — and is
admitted only after agreeing with the judge on held-out already-graded criteria.

Why this is the load-bearing idea for a *self-sustaining* loop: §2 says the ceiling is
verifier fidelity. Everything in §1 improves the generator and therefore chases an
asymptote someone else set. A system that improves its own verifiers moves its own limit.
It is also the safe direction — a bad probe rejects good edits (conservative, and visible
in `RefineResult.rejected`), where a bad generator emits confident nonsense.

And it is how this generalizes beyond legal without hand-porting: the *synthesizer* is
portable, the probes it emits are domain-specific. The synthesizer is a `dspy.Signature`,
so GEPA can tune it with probe-vs-judge agreement as the metric — no new infrastructure.
This makes eval-plan Task 3 self-writing rather than hand-written.

**Acceptance:** probes synthesized from a real LAB rubric; per-probe agreement with the
judge on held-out criteria; cost per admitted edit vs. the LM-only ladder.

### 4.11 Abstention and selective accuracy

There is no way for the agent to say "I do not know" or "escalate this." Under all-pass
scoring a confident error and an abstention both score zero, so LAB does not reward it —
but the accuracy goal does. Add per-claim confidence and report **risk-coverage**: accuracy
on the subset the agent claims, as a function of coverage. A system 95% accurate on the 40%
it is confident about is more useful than a uniform 65%, and only the second framing
measures critical thinking at all. Gives credit assignment a further signal: a lesson that
improves calibration is valuable even at flat pass rate.

### 4.12 An adversarial child

`spawn_child(task, name)` takes an arbitrary string and no role, and children are currently
near-dead weight. Give one a standing role: attack the deliverable — strongest
counterargument, missed document, contradicting clause — and require the parent to revise
or explicitly rebut before submitting. This exploits the generator/verifier gap at
*inference* time rather than learning time, and it is the most direct operationalization of
"critical thinking" available from machinery that already exists.

### 4.13 Built on 2026-09-07

Two deterministic, LM-free checks from this list are implemented (`sentinelprime/grounding.py`):

- **Document coverage.** `PrimeAgent.last_coverage` reports which workspace documents a run
  never named. Name-mention is a permissive proxy — a listing counts — so it bounds the
  failure from one side: a document whose name never appears was certainly not read.
- **The lucky-guess filter.** A `dspy.RLM` trajectory separates what the interpreter
  *returned* from what the model *wrote*. A criterion whose facts (amounts, section refs,
  durations, dates) appear only on the generated side was not grounded in the documents.
  `CreditAssigner.observe(..., grounded=...)` drops such passes instead of counting them,
  closing a live defect: a run that guessed right was crediting whichever lesson happened
  to be in the prompt, so a useless lesson kept a passing rate and never retired.

Dropped, not blamed — anchor matching is lexical, so "cannot prove the run saw it" has to
mean *no evidence*, never evidence against. Both are opt-in and only ever remove credit,
per invariant #7.

---

## 5. Revised gate

Before committing to the §4 queue or to any weight work, three cheap measurements, in order:

1. **Noise floor** — `scripts/noise_floor.py --agent` on one real task. If agent+judge spread
   exceeds any plausible effect, nothing downstream is measurable and variance reduction is the
   only job. *(Not yet run at time of writing.)*
2. **Generator/verifier gap** — solver = pooled criterion rate; verifier = Best-of-N with the
   agent selecting. Predicts available headroom (§2.1).
3. **Arm F - arm A** — hand-written oracle ledger vs. frozen baseline. Measures context headroom.

| Outcome | Read |
|---|---|
| gap large, `F-A` large | headroom is real; build §4.3 / §4.7 |
| gap small | verification is the bottleneck; build probes (eval-plan Task 3), not learning machinery |
| `F-A` ~ 0 | context space is exhausted; weights are the only route, or the claim moves to audit |
| noise floor > any effect | none of the above is measurable yet |

Regardless of outcome, §4.1 and §4.2 produce results, because they do not depend on the learning
curve moving.

---

## 6. Sources

- ACE — https://arxiv.org/abs/2510.04618
- TAME — https://arxiv.org/pdf/2602.03224
- SEAL — https://arxiv.org/abs/2506.10943
- Language Models Need Sleep — https://arxiv.org/html/2606.03979v1
- Darwin Godel Machine — https://arxiv.org/abs/2505.22954
- Evo-Memory — https://arxiv.org/pdf/2511.20857
- ShinkaEvolve — https://arxiv.org/pdf/2509.19349
- Sharpening mechanism — https://arxiv.org/pdf/2412.01951
- Solver-verifier gap — https://arxiv.org/html/2507.00075v2
- Introspection threshold — https://arxiv.org/html/2607.04277v1
- FLEX — https://arxiv.org/pdf/2511.06449
- Agentic test-time training — https://arxiv.org/pdf/2607.03441
- Awesome-Self-Evolving-Agents — https://github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents
