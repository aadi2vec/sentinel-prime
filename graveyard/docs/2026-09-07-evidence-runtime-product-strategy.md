# SentinelPrime: Evidence-Based RLM Infrastructure and Legal Review Product

**Date:** 2026-09-07  
**Status:** Proposed strategy, not an implementation specification or a claim of demonstrated performance.  
**Scope:** Preserve the ambition to build differentiated agent infrastructure, with legal document review as its first demanding application.

## 1. Thesis

SentinelPrime should become an RLM runtime that manages the evidence, coverage, verification, and cost obligations of document-intensive reasoning. Its continual harness should subsequently improve how that work is performed and checked.

The initial ambition remains general-purpose legal problem solving across large document collections with high recall and high precision. A narrow first workflow makes that ambition measurable; it does not limit the eventual engine to one checklist.

The infrastructure must deliver value on the first task, before it has learned anything. Its first benefit should be more complete, inspectable execution: explicit review obligations, source-linked claims, unresolved contradictions, controlled finalization, and recoverable work. Learning becomes an additional measurable benefit rather than the prerequisite for usefulness.

**Infrastructure promise:** “Run document reasoning with explicit evidence obligations, and improve the execution and verification policies through evaluated changes.”

**Application promise:** “For every finding, show the supporting evidence, the scope examined, the checks performed, and what remains uncertain.”

Neither promise means mathematical proof of arbitrary legal interpretation, guaranteed absence of hallucinations, or autonomous legal sign-off.

## 2. What exists today

The inspected working tree contains a DSPy RLM agent, versioned supplemental memory, memory-admission gates, audit records, lesson retirement, proposal/outcome logging, and an offline GEPA compilation path. The local hermetic suite passed 334 tests during this assessment. This establishes tested software behavior, not legal accuracy.

| Existing component | Implemented value | Important boundary |
|---|---|---|
| `PrimeAgent` / `CachingRLM` | Document investigation, optional children, intra-run query deduplication | No enforced evidence obligations for final findings |
| `ContinualHarness` | Supplemental guidance, snapshots, rollback, between-task adaptation | Adaptation is not demonstrated improvement |
| `LadderVerifier` | Ordered gates for proposed memory edits | Does not certify the final deliverable |
| `GroundingProbe` | Cheap vocabulary-overlap filter | Topical overlap is not factual grounding |
| Grounding and coverage utilities | Lexical evidence proxies and document-name coverage | Naming a document does not establish that its relevant contents were examined |
| `ProgressMonitor` | Detects and records suspected stalls | Does not currently enforce mid-run replanning |
| `CreditAssigner` | Tracks targeted outcomes while lessons are exposed | Correlational attribution; no matched counterfactual |
| `ProposalLog` and GEPA runner | Operational records can feed verifier prompt optimization | Weak labels and no demonstrated downstream benefit from tuning |
| Audit and reuse controls | Traceable memory changes and some dependency checks | Application-level JSON persistence; causal-currency check remains a stub |

Local artifacts contain early legal evaluation runs, but no completed comparison demonstrating learned-ledger or tuned-verifier gains was found in the inspected paired log. One repeated task ranged from approximately 49% to 62% criterion pass rate across three runs. A control task reported approximately 270,000 tokens and 218 seconds; these are diagnostic observations, not production unit economics.

Evaluation errors currently become failed criteria in `lab.py`; rate-limit failures are visible in local artifacts. Repair this before interpreting experiment deltas. Documentation also contains stale status statements and should eventually be reconciled with the implementation.

## 3. The infrastructure contribution

Using an open-source execution framework does not prevent a valuable infrastructure business. It does mean the framework composition alone is insufficient differentiation. The contribution must be an execution capability that a basic discovery agent does not provide, supported by controlled comparisons.

The proposed core abstraction is a **review obligation**: a scoped question the runtime must resolve with evidence, explicitly exclude, or leave unresolved. Obligations connect tasks, documents, candidate claims, verification checks, and completion decisions.

The model chooses investigative actions within this structure. The runtime tracks whether those actions have satisfied the requested work. More fluent prose or a longer trajectory is not progress by itself.

### Day-one capabilities

| Capability | Runtime behavior | Immediate value | How to test it |
|---|---|---|---|
| Coverage-aware execution | Maintain issue-by-document obligations and reconcile them before finalization | Fewer silent omissions; visible incomplete work | Material-issue recall and unresolved-obligation reporting |
| Evidence-preserving reads | Return stable source handles, spans, document versions, and extraction lineage | Findings can be traced and checked without reconstructing free-form trajectories | Citation fidelity and evidence reconstruction rate |
| Claim-aware finalization | Require each material finding to carry evidence and check states | Unsupported claims cannot silently appear as verified findings | Unsupported-claim rate at matched recall |
| Targeted verification | Route source checks, interpretation review, and contradiction searches according to claim type | Verification effort addresses identifiable failure modes | Error detection and total cost per reviewed finding |
| Progress-aware scheduling | Prefer unresolved obligations; intervene at action boundaries on repeated nonprogress | Less repeated exploration and fewer premature stops | Quality under equal token/time budgets |
| Dependency-aware recomputation | When a document changes, mark dependent findings stale and recheck affected work | Faster, safer amendment review | Missed invalidations and rerun cost versus full reruns |

These are proposed extensions. Existing caching, snapshots, and monitoring provide starting points, but do not already implement this runtime.

Start with coverage, source handles, and finalization checks. Defer adaptive scheduling and fine-grained recomputation until the required dependency and action records exist.

## 4. Execution architecture

```text
Document inventory, versions, extraction lineage
                         |
Task scope + explicit review obligations
                         |
RLM investigation through instrumented tools
                         |
Candidate claims + source evidence + dependency links
                         |
Source checks -> support checks -> contradiction investigation
                         |
Coverage reconciliation + finalization policy
                         |
Supported findings / unresolved findings / missing evidence
                         |
Report + evidence package + review record
```

### Proposed contracts

- `DocumentVersion`: source identity, hash, original artifact, extracted representation, parser version, and location mapping.
- `ReviewObligation`: issue, corpus scope, relevant documents, status, supporting action records, and exclusion or unresolved reason.
- `EvidenceSpan`: document version, exact text, source location, retrieval event, and surrounding context reference.
- `Claim`: proposition, scope, supporting and contradicting evidence, assumptions, dependencies, and unresolved questions.
- `CheckResult`: claim/check identity, bounded assertion checked, method/version, evidence, result, and cost.
- `ReviewDecision`: reviewer disposition, correction, rationale, and affected claims or lessons.

Check results distinguish `pass`, `fail`, `unknown`, `not_applicable`, and `execution_error`. A failed service call is not a substantive rejection. Confidence is not interchangeable with a check result.

Expose instrumented tools such as `read_source`, `search_sources`, `record_claim`, and `resolve_obligation`. Preserve original files and extraction mappings; text copied into a model-generated message is not trusted source evidence merely because it resembles a quote.

Use isolated workers and constrained tool capabilities for customer documents. The current in-process Python interpreter is not an isolation boundary. Any read path outside the instrumented tools must either be captured or explicitly lack evidence guarantees.

## 5. What verification can establish

Verification must operate on the delivered work as well as on proposed memory changes. Keep these two interfaces and records separate.

1. **Source integrity:** the cited span exists in the examined document version; preserve enough context to inspect qualifications.
2. **Factual support:** the evidence supports the stated facts. Validate arithmetic and structured values deterministically where possible.
3. **Coverage:** required review obligations have recorded dispositions. Process coverage is not proof that every relevant issue was recognized.
4. **Consistency:** investigate definitions, amendments, exceptions, schedules, and linked documents that could change a finding.
5. **Reasoning:** test the interpretation against explicit transaction facts and an applicable review checklist. If external legal authorities are required, acquire and version those sources too, or mark that portion outside the review scope.
6. **Uncertainty:** preserve missing evidence and unresolved interpretations rather than forcing a definitive answer.

LLM judges remain useful for semantic support and interpretation. They must inspect source evidence and identify reasons to reject or qualify claims. Agreement between models is not independent proof of correctness.

Check ordering must respect logical prerequisites; a cost optimizer must not move an interpretation check ahead of the evidence it requires. Cheap deterministic checks certify only their bounded assertions. An exact quotation can still be misleading or insufficient.

### Precision and recall

Favor broad candidate discovery during investigation. Apply stricter support requirements to affirmative findings during publication. Preserve filtered but plausible candidates in an unresolved review queue so precision improvements do not hide recall losses.

Distinguish “not found within the examined scope,” “does not exist,” and “does not apply.” Absence claims need explicit search scope and coverage evidence. Report precision, material-issue recall, abstention, and review burden together; a system that abstains on everything has not solved the task.

## 6. First legal application

Start with change-of-control and assignment consent review over contracts and amendments. The target user is a legal or transaction advisory reviewer; access to design partners is an assumption to validate.

The user uploads a document bundle and transaction description, selects a review checklist, and receives a consent matrix. Each finding links to source passages and records exceptions, contradictions, missing information, and the checks performed. The reviewer can approve, correct, or escalate it, then export the report and evidence package.

Example finding:

```text
Finding: Consent may be required before the proposed transaction.
Sources: Agreement X, amendment Y, transaction description Z.
Evidence: Stable source spans and document versions.
Exception: Affiliate-transfer exception; applicability unresolved.
Conflict: Amendment Y modifies the original consent provision.
Missing fact: Identity of the acquiring entity.
Disposition: Source checks passed; interpretation awaiting review.
```

The initial interface needs upload, findings, source navigation, unresolved work, correction controls, and export. It does not need a general chat product, autonomous legal approval, or a broad contract-management suite.

## 7. Continual improvement: three distinct loops

### A. Within-task execution adaptation

The runtime changes its next action based on unresolved obligations, missing evidence, and failed checks. For example, a contradiction check requests the amendment instead of asking the agent to rewrite the same report. This is execution control, not persistent learning.

### B. Between-task procedural learning

Verified failures or reviewer corrections produce proposed procedural lessons. The existing harness can admit, version, expose, retire, and roll back these lessons between tasks.

Separate general procedures (“check linked amendments before interpreting consent”) from matter-specific facts (“agreement X requires consent”). Give them different scopes and dependency rules. Human approval should initially govern promotion of customer-facing procedural changes.

A lesson is useful only if it improves later work. Use matched offline trials with and without the lesson, keeping the base model and contemporaneous ledger fixed. Repeat noisy trials, preserve interactions between lessons, and evaluate transfer to unseen matters. Existing credit rates are a screening signal, not causal evidence.

### C. Verifier improvement

Link mistaken admissions and rejections to independently established outcomes. Use expert corrections, exact source failures, executable checks, and controlled downstream trials to form training examples. GEPA can then optimize verifier instructions against those examples.

Current vocabulary-probe negatives are weak labels, not gold truth. Downstream pass rates inherit evaluator errors and confounded lesson attribution. Shadow execution of rejected proposals reveals the deeper judge's opinion; it does not reveal the downstream utility of the rejected lesson. Trial selected rejected lessons only in isolated evaluation environments to study false rejections.

Promote a candidate verifier only after held-out evaluation against the current version. Measure harmful admissions, useful proposals rejected, downstream quality, latency, and cost. Version and retain the old verifier for rollback. Evaluating agreement with the training labels alone cannot establish improvement.

No weight training or reinforcement-learning improvement is established by these loops. Claims of improving on Prime Agent or an RL-trained alternative require a precisely identified baseline, matched resources, and empirical results. Avoid claiming that the project surpasses a broader system from a comparison with a thin or underconfigured baseline.

## 8. The feedback asset

A corrected failure should produce both a possible lesson and a regression case.

Example: a missed amendment produces “inspect linked amendments” as candidate guidance and a test containing a clause whose interpretation changes under an amendment. Optimize on separate cases and test transfer to unseen document families.

The valuable dataset connects source evidence, attempted actions, proposed changes, reviewer corrections, and measured downstream effects. Raw trajectories or self-generated praise do not provide the same signal. Customer data must remain within authorized tenant and training boundaries; cross-customer learning requires an explicit permitted data policy.

## 9. Evidence required for the infrastructure claim

Use the same model, documents, tools, output requirements, and resource budgets across comparisons. Include a well-configured ordinary DSPy RLM baseline and a simpler retrieval-plus-structured-extraction workflow; the RLM's added complexity must earn its cost.

Run two experiment families rather than bundling all mechanisms into one score:

| Experiment | Comparison | Question |
|---|---|---|
| Day-one runtime | Baseline vs. coverage/evidence/finalization runtime; learning disabled | Does infrastructure improve the first task? |
| Memory adaptation | Fixed runtime with frozen vs. learned guidance | Does adaptation improve unseen tasks? |
| Memory admission | Ungated vs. fixed-gate learning | Does the gate prevent harmful learning? |
| Verifier adaptation | Fixed vs. optimized verifier with other mechanisms controlled | Does verifier tuning improve downstream results? |
| Document updates | Full rerun vs. dependency-directed rerun | Can cost fall without missing affected findings? |

Use independent expert-reviewed findings and evidence spans. Split by matter/document family and time to reduce leakage; reserve a final test set untouched by optimization. Count errors separately from unavailable evaluations, repeat noisy comparisons, and report uncertainty.

Primary metrics: material-issue recall, supported-finding precision, citation fidelity, missed contradictions, unresolved rate, reviewer correction time, total tokens/cost, and latency. Report quality at matched budgets and cost at matched quality. Do not infer correctness from coverage, lower cost from cache hits alone, or self-improvement from a rising synthetic curve.

The first release must earn its value with learning disabled. If learning adds no reliable benefit, retain the evidence runtime and keep learning experimental.

## 10. Packaging and commercial hypothesis

Build two surfaces over one engine:

- **SDK/runtime:** obligations, evidence tools, claim checks, completion policy, versioned runs, and evaluated guidance changes for agent engineering teams.
- **Legal review application:** the first complete workflow and source of expert feedback.

DSPy integration is the initial advantage for development and optimization, not a permanent adoption requirement. Keep domain checks and source adapters separate from execution contracts so other agents can eventually submit the same structured claims.

The potential differentiation is measurable execution quality and change control. Potential defensibility comes from validated failure cases, integration quality, reliable runtime behavior, and permissioned outcome data. Neither market demand nor exclusivity of these ideas has been established.

Avoid positioning solely as “an LLM judge,” “another memory store,” or a universal self-improving agent. Sell a concrete operational result and show its comparison evidence.

## 11. Delivery sequence

### Milestone 1: Trustworthy measurement and source records

Repair evaluation-error handling; define claim and obligation schemas; add versioned manifests and stable source spans. Build a small expert-reviewed evaluation set and establish baseline accuracy, coverage, cost, and review time.

### Milestone 2: First-task infrastructure value

Instrument source tools, enforce finalization dispositions, and add source/support/contradiction checks. Deliver one reviewable consent matrix and evidence export. Compare against baselines with learning disabled.

### Milestone 3: Supervised product pilot

Add isolated execution, persistent job state, tenant access controls, recovery, reviewer corrections, and usable source navigation. Recruit design partners and measure workflow value. Agree quality and time-saving targets with reviewers before evaluating the pilot.

### Milestone 4: Earned continual improvement

Turn corrections into regression cases, run controlled lesson trials, and evaluate a candidate verifier on held-out matters. Introduce promotion/rollback controls. Enable learned behavior only where evidence supports it.

A provisional 90-day allocation is 15 days for measurement and partner discovery, 30 for the evidence workflow, 30 for supervised pilots, and 15 for evaluating paid continuation. This is a planning hypothesis, not an engineering estimate or guaranteed schedule.

Defer Rust migration, weight training, autonomous probe synthesis, elaborate search trees, and additional agent roles until a measured bottleneck justifies them. Durable recovery should begin with explicit task and artifact checkpoints; do not promise deterministic replay of arbitrary Python or hosted model calls.

## 12. Decisions still requiring evidence

- Can coverage and claim checks improve first-task quality enough to justify their cost?
- Does RLM exploration outperform a simpler structured workflow on the chosen corpus?
- Can reviewers reliably establish useful ground truth at sustainable cost?
- Which procedures transfer across matters without leaking matter-specific facts?
- Does verifier optimization improve independent downstream outcomes?
- Will engineering teams buy the runtime directly, or will the application be the initial commercial entry point?

The next concrete deliverable is an evidence-backed legal review that works with an empty learned ledger. That establishes an infrastructure contribution immediately and creates the reliable feedback needed to make continual learning meaningful.

## References and context

Repository foundations: `docs/ARCHITECTURE.md`, `docs/plans/2026-09-06-harvey-lab-eval-plan.md`, `sentinelprime/agent.py`, `sentinelprime/harness.py`, `sentinelprime/verifier.py`, `sentinelprime/grounding.py`, and `sentinelprime/optimize.py`.

External context consulted during the discussion:

- [DSPy GEPA documentation](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/diving-deeper/gepa-in-depth.md): instruction optimization is an existing framework capability.
- [LangMem introduction](https://www.langchain.com/blog/langmem-sdk-launch): experience-based memory and behavior adaptation already have existing implementations.
- [Harvey Vault](https://www.harvey.ai/platform/vault): broad document analysis and due diligence are established product categories.
- [Ironclad AI overview](https://support.ironcladapp.com/hc/en-us/articles/12947738534935-Ironclad-AI-Overview): contract-review playbooks are an existing application category.
- [Vanta Questionnaire Automation](https://www.vanta.com/products/questionnaire-automation): evidence-oriented questionnaire workflows are an adjacent application hypothesis with existing competition.

These sources establish context, not a comprehensive competitor assessment or a claim that the proposed runtime is novel.
