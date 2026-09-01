# DSPy RLM + Latent Briefing for Cross-Document Field, Clause, Intent & Obligation Discovery

**Status:** Design — awaiting implementation approval
**Date:** 2026-04-18
**Target dataset:** CUAD v1 (Contract Understanding Atticus Dataset)
**Primary objectives:** accuracy, cost, latency — in that order, with pareto-friendly trade-offs

---

## 1. Overview

This spec describes a closed-source-compatible RLM harness that operates on **pre-clustered** groups of legal documents and clause spans. Within each cluster, the harness discovers four output types — **fields, clause types, intents, and obligations** — with minimal content-level hinting, preserving niche discovery while enforcing cross-run naming consistency through a DSPy-optimized canonicalization layer.

The harness reimplements RampLabs' **Latent Briefing** at the text-structured level (since closed APIs do not expose KV cache), preserving the original's query-vector-as-relevance-signal mechanic. It integrates with `dspy.RLM` as the inner extraction loop and wraps it with a Rolling Structured State that compacts between document traversals within a cluster.

Per-account feedback is folded in via `dspy.Example` + `MIPROv2` for account-scoped prompt recompilation.

---

## 2. Problem Statement

### 2.1 Current failure modes in the existing clustering pipeline

Given the two-level clustering already in place:
1. **Level-1:** document summaries + embeddings → document clusters
2. **Level-2:** clause spans across all documents → clause-pattern clusters

When running per-cluster LLM extraction to discover "what fields commonly appear across these similar documents," the output exhibits two persistent defects:

- **(A) Generic field names.** Extracted fields collapse to low-information labels like `party_name`, `date`, `amount` rather than semantically rich legal concepts (`liability_cap_amount_usd`, `auto_renewal_notice_window_days`, `change_of_control_trigger`).
- **(B) Inconsistent naming across runs.** The same underlying legal concept receives different names across cluster-level runs, breaking downstream schema consolidation.

### 2.2 Design constraints

1. **Closed-source model compatibility.** Must work over OpenAI / Anthropic / Google APIs. No KV cache access.
2. **Four output targets, not one.** Must identify fields, clause types, intents, and obligations — not just fields.
3. **Minimal content hinting.** Must preserve discovery of niche/novel fields absent from any pre-defined taxonomy (e.g., CUAD's 41 categories).
4. **Structural hinting allowed.** Schema-level hints about *what each output type IS* are acceptable and necessary.
5. **Per-account adjustable.** Feedback from account-specific users must refine prompts without affecting global behavior.

---

## 3. Proposed Solution — Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                 Cluster-Scoped Discovery Pipeline                    │
│                                                                      │
│   INPUT: Cluster C = {doc_1, doc_2, ..., doc_K}                      │
│          (from level-1 doc-summary cluster or level-2 clause cluster)│
│                                                                      │
│   ┌────────────────────────────────────────┐                         │
│   │  Rolling Structured State (JSON-typed) │                         │
│   │   • fields[]                           │◄──── updated per doc    │
│   │   • clause_types[]                     │                         │
│   │   • intents[]                          │                         │
│   │   • obligations[]                      │                         │
│   │   • open_questions[]                   │                         │
│   │   • anchor_terms[]                     │                         │
│   └────────────────────┬───────────────────┘                         │
│                        │                                             │
│                        ▼                                             │
│   ┌─────────────────────────────────────────────┐                    │
│   │ Text-Native Latent Briefing                 │                    │
│   │  (embedding-weighted trajectory distill)    │                    │
│   │  cos_sim(query_vec, state_item_vec)         │                    │
│   │   × recency_boost → top-K retention         │                    │
│   │  Compact JSON emission, ~1200 token budget  │                    │
│   └────────────────────┬────────────────────────┘                    │
│                        │                                             │
│                        ▼                                             │
│   ┌─────────────────────────────────────────────────────────┐        │
│   │  RLM Core (custom harness wrapping dspy.RLM)            │        │
│   │   Root LM: Claude Sonnet 4.5 / GPT-4o                   │        │
│   │   Sub LM: Haiku 4.5 / GPT-4o-mini                       │        │
│   │   max_iterations: adaptive (default 5, cap 8)           │        │
│   │   llm_query_batched for parallel sub-calls              │        │
│   └────────────────────┬────────────────────────────────────┘        │
│                        │                                             │
│      ┌─────────────────┼─────────────────┐                           │
│      ▼                                   ▼                           │
│   ┌──────────────────┐             ┌──────────────────────┐          │
│   │ Grounded Mode    │             │ Exploratory Mode     │          │
│   │ temp 0.2         │             │ temp 0.8             │          │
│   │ brief included   │             │ brief omitted        │          │
│   │ Cohesion focus   │             │ Niche-finding focus  │          │
│   └────────┬─────────┘             └──────────┬───────────┘          │
│            │                                  │                      │
│            └──────────────┬───────────────────┘                      │
│                           ▼                                          │
│                  Merge with overlap penalty                          │
│                   (cosine > 0.85 → dedupe)                           │
│                           │                                          │
│                           ▼                                          │
│   ┌───────────────────────────────────────────────────────┐          │
│   │  Four DSPy Signatures (parallel, batched)             │          │
│   │   • DiscoverFields                                    │          │
│   │   • DiscoverClauseTypes                               │          │
│   │   • DiscoverIntents  (conditioned on clause_types)    │          │
│   │   • DiscoverObligations (conditioned on clause_types) │          │
│   └───────────────────────┬───────────────────────────────┘          │
│                           │                                          │
│                           ▼                                          │
│              [update Rolling Structured State]                       │
│                           │                                          │
│             ┌─────────────┴─────────────┐                            │
│             │ More docs in cluster?     │                            │
│             └─────────────┬─────────────┘                            │
│                  yes ▲    │    ▼ no                                  │
│                      └────┘                                          │
│                           ▼                                          │
│   ┌───────────────────────────────────────────────────────┐          │
│   │  Canonicalization Layer (MIPRO-optimized)             │          │
│   │   Tiered: rule-based fuzzy → LLM for ambiguous        │          │
│   │   High-similarity-only merge (preserves niche)        │          │
│   └───────────────────────┬───────────────────────────────┘          │
│                           ▼                                          │
│   OUTPUT: Cluster Schema {                                           │
│              fields: [Field, ...],                                   │
│              clause_types: [ClauseType, ...],                        │
│              intents: [Intent, ...],                                 │
│              obligations: [Obligation, ...]                          │
│          }                                                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Specifications

### 4.1 Rolling Structured State

**Purpose:** a typed, cluster-scoped accumulator that persists discovered items across document traversals, providing the input payload for the Latent Briefing step.

**Schema:**

```python
class RollingState(BaseModel):
    cluster_id: str
    docs_processed: List[str]                # doc IDs
    fields: List[Field] = []
    clause_types: List[ClauseType] = []
    intents: List[Intent] = []
    obligations: List[Obligation] = []
    open_questions: List[str] = []            # gaps flagged by the RLM
    anchor_terms: List[Tuple[str, int]] = []  # (term, frequency) — top-freq terms
    iteration_count: int = 0
    last_state_delta: float = 1.0             # for adaptive stopping
```

**Update contract:** after each document is processed, the state is merged with the document's newly discovered items. Merging uses the same overlap-penalty logic as grounded/exploratory merge (cosine > 0.85 → dedupe), with frequency counting to promote terms to the `anchor_terms` list.

**Persistence:** one JSON file per `(account_id, cluster_id)` tuple. Reloaded at the start of any re-run, enabling incremental cluster processing.

---

### 4.2 Text-Native Latent Briefing

**Purpose:** reimplementation of RampLabs' Latent Briefing at the text-structured level, since closed APIs do not expose KV cache.

**Algorithm:**

```python
def build_brief(state: RollingState, next_doc: str, query: str,
                k: int = 15, token_budget: int = 1200) -> str:
    # 1. Query embedding
    doc_preview = next_doc[:2000]
    query_vec = embed(f"{query}\n\n{doc_preview}")

    # 2. Score every state item
    scored_items = []
    for item in state.all_items():  # flatten fields + clauses + intents + obligations
        item_vec = embed(item.description)
        base_score = cosine_similarity(query_vec, item_vec)
        recency_boost = 1.0 + 0.2 * (item.iteration_age ** -0.5)
        scored_items.append((item, base_score * recency_boost))

    # 3. Top-K retention
    scored_items.sort(key=lambda x: -x[1])
    kept = [it for it, _ in scored_items[:k]]
    open_qs = [q for q in state.open_questions if score(q, query_vec) > 0.35]

    # 4. Emit compact JSON brief (token-budgeted)
    brief = {
        "already_found": {
            "fields": [serialize_compact(f) for f in kept if isinstance(f, Field)],
            "clause_types": [serialize_compact(c) for c in kept if isinstance(c, ClauseType)],
            "intents": [serialize_compact(i) for i in kept if isinstance(i, Intent)],
            "obligations": [serialize_compact(o) for o in kept if isinstance(o, Obligation)],
        },
        "still_looking_for": open_qs,
        "anchor_terms": [t for t, _ in state.anchor_terms[:10]],
    }
    return truncate_to_budget(json.dumps(brief), token_budget)
```

**Token budget:** default 1200 tokens. Calibrated via pilot run — if discovery recall drops >5% vs. no-briefing baseline, raise to 1800.

**Compression ratio:** ~5x vs. raw trajectory (~6000 tokens → ~1200 tokens).

**Embedding model:** `text-embedding-3-small` ($0.02/M tokens). Negligible cost.

**Why this approximates KV-level Latent Briefing:** RampLabs' technique uses attention-weighted relevance to retain trajectory information under query conditioning. Cosine similarity on sentence embeddings is a weaker but directionally-correct proxy for attention weights. Expected compression efficiency: 30–40% end-to-end token savings (vs. 31–49% for native KV).

---

### 4.3 RLM Core (Custom Harness Wrapping `dspy.RLM`)

**Rationale for custom wrap vs. direct `dspy.RLM` use:** `dspy.RLM` does not expose a hook for injecting a compacted brief into the REPL environment between iterations, nor does it support dual-mode (grounded/exploratory) invocation. We wrap it.

**Wrapper API:**

```python
class LatentRLM:
    def __init__(self,
                 root_lm: dspy.LM,
                 sub_lm: dspy.LM,
                 embedder: Embedder,
                 max_iterations: int = 5,
                 adaptive_stop_epsilon: float = 0.05,
                 enable_caching: bool = True):
        self.root = root_lm
        self.sub = sub_lm
        self.embedder = embedder
        self.max_iter = max_iterations
        self.eps = adaptive_stop_epsilon
        self._inner_rlm = dspy.RLM(
            signature="brief, document, query -> discoveries",
            max_iterations=max_iterations,
            max_llm_calls=30,
            sub_lm=sub_lm,
            max_output_chars=10000,
        )

    def process_cluster(self, cluster: List[Document], query: str,
                        account_id: str) -> ClusterSchema:
        state = load_or_init_state(account_id, cluster.id)
        for doc in cluster:
            brief = build_brief(state, doc.text, query)

            # Grounded + Exploratory passes in parallel
            grounded, exploratory = parallel(
                lambda: self._run_rlm(brief=brief, doc=doc, temp=0.2),
                lambda: self._run_rlm(brief="", doc=doc, temp=0.8,
                                     extra_instruction="find NOVEL patterns"),
            )

            merged = overlap_penalty_merge(grounded, exploratory, threshold=0.85)
            structured = self._run_four_signatures(merged, doc)  # 4-sig pass

            state = merge_into_state(state, structured)
            state.last_state_delta = compute_delta(prev_state, state)
            if state.last_state_delta < self.eps and state.iteration_count >= 3:
                break  # adaptive early stop

        return canonicalize(state, account_id)
```

**Configuration defaults:**

| Parameter | Default | Rationale |
|---|---|---|
| `max_iterations` | 5 | Empirical DSPy RLM sweet spot per cmpnd.ai |
| `max_llm_calls` | 30 | Fits within cost budget per doc |
| `adaptive_stop_epsilon` | 0.05 | Stop if schema stabilizes |
| Grounded temp | 0.2 | Cohesion with prior state |
| Exploratory temp | 0.8 | Noise-tolerant niche finding |
| Overlap merge threshold | 0.85 cosine | Preserves niche below threshold |

---

### 4.4 Four DSPy Signatures

**Signature outputs (Pydantic models):**

```python
class Field(BaseModel):
    name: str                    # snake_case, emergent
    value_type: Literal["string", "date", "money", "enum", "ref", "boolean"]
    span_text: str               # exact source text
    confidence: float

class ClauseType(BaseModel):
    name: str                    # emergent category
    emergent_description: str    # model's description of what this pattern IS
    span_text: str

class Intent(BaseModel):
    clause_ref: str              # FK to ClauseType.name in same output
    intent_description: str      # WHY this clause exists

class Obligation(BaseModel):
    clause_ref: str              # FK to ClauseType.name
    obligor: str                 # which party
    action: str                  # what they must do
    trigger_condition: Optional[str] = None
    deadline: Optional[str] = None
```

**Signature definitions:**

```python
class DiscoverFields(dspy.Signature):
    """Identify named data points with values. A field is a NAMED PARAMETER
    whose VALUE varies across contracts. Not a clause. Not an obligation.
    Do not constrain yourself to predefined categories — discover what you see."""
    brief: str = dspy.InputField()
    document: str = dspy.InputField()
    fields: List[Field] = dspy.OutputField()

class DiscoverClauseTypes(dspy.Signature):
    """Identify provisions by semantic category. A clause type is a RECURRING
    CONTRACTUAL PATTERN that groups similar provisions. Name it by what it DOES,
    not what this document calls it. Prefer emergent, descriptive names over
    boilerplate legal labels."""
    brief: str = dspy.InputField()
    document: str = dspy.InputField()
    clause_types: List[ClauseType] = dspy.OutputField()

class DiscoverIntents(dspy.Signature):
    """For each clause type, identify the underlying purpose.
    Intent = WHY this clause exists. Not what it says."""
    clause_types: List[ClauseType] = dspy.InputField()
    document: str = dspy.InputField()
    intents: List[Intent] = dspy.OutputField()

class DiscoverObligations(dspy.Signature):
    """Identify actionable commitments. An obligation means a party MUST DO
    something, under specific conditions, potentially with a deadline.
    Only include clear, enforceable duties — not mere statements."""
    clause_types: List[ClauseType] = dspy.InputField()
    document: str = dspy.InputField()
    obligations: List[Obligation] = dspy.OutputField()
```

**Execution:** all four signatures run in parallel per document via `asyncio.gather`, then merged into the updated state.

**Why four signatures, not one:** decoupling allows MIPRO to optimize each independently per account. Merging into a single signature would collapse the optimization surface and lose per-target tunability.

---

### 4.5 Dual-Mode Exploration (Grounded + Exploratory)

**Purpose:** balance the minimal-hinting trade-off. Pure grounding misses niche items; pure exploration produces noise.

| Mode | Temperature | Brief Inclusion | Extra Instruction | Role |
|---|---|---|---|---|
| Grounded | 0.2 | Full brief | "Use the brief to anchor your discovery; extend it." | Cohesion |
| Exploratory | 0.8 | Omitted | "Ignore prior patterns; find NOVEL recurring structures." | Niche discovery |

**Merge rule:**
1. Compute embedding for every item in both outputs.
2. For each exploratory item E: if there exists grounded item G with cosine(E, G) > 0.85, drop E (grounded wins).
3. Otherwise, keep E with a `source: "exploratory"` tag for downstream monitoring.

**Exploration ratio monitoring:** track `|exploratory_kept| / |grounded_kept|` per account. Should stabilize around 0.15–0.30 for healthy niche discovery. Values >0.5 suggest grounded underperformance; values <0.05 suggest exploratory mode is being suppressed.

---

### 4.6 Canonicalization Layer

**Purpose:** resolve naming drift across cluster runs, producing a stable cross-account schema.

**Tiered architecture:**

```
┌──────────────────────────────────────────────────────┐
│  Tier 1: Rule-Based Fuzzy Match                      │
│   • Levenshtein < 0.2 on normalized names            │
│   • Plural/singular collapse                         │
│   • Stopword removal                                 │
│   • Handles ~70% of cases at zero LLM cost           │
└────────────────────────┬─────────────────────────────┘
                         │  unresolved pairs
                         ▼
┌──────────────────────────────────────────────────────┐
│  Tier 2: Embedding Similarity Cluster                │
│   • HDBSCAN on name+description embeddings           │
│   • Clusters with cohesion > 0.82 auto-merge         │
│   • Handles ~20% more                                │
└────────────────────────┬─────────────────────────────┘
                         │  ambiguous pairs
                         ▼
┌──────────────────────────────────────────────────────┐
│  Tier 3: LLM Canonicalization (MIPRO-optimized)      │
│   • dspy.ChainOfThought signature:                   │
│     "Are these the same concept? If yes, pick        │
│      canonical name."                                │
│   • Only fired on residual ambiguous cases           │
└──────────────────────────────────────────────────────┘
```

**Merge guardrail — niche preservation:** never merge items where both have frequency = 1 (appeared in only one doc across the cluster). These are prime niche-discovery candidates. Flag instead for human review.

**Expected cost:** ~$0.02 per doc amortized (most cost lives in Tier 3 fallback, which fires on ~10% of pairs).

---

### 4.7 DSPy-Native Per-Account Feedback Loop

**Flow:**

```python
class AccountAwareDiscovery:
    def __init__(self, base_program_path: str):
        self.global_program = dspy.load(base_program_path)
        self.account_programs: Dict[str, Any] = {}  # lazy-loaded
        self.feedback_store: Dict[str, List[dspy.Example]] = defaultdict(list)

    def discover(self, doc: Document, account_id: str) -> ClusterSchema:
        program = self.account_programs.get(account_id, self.global_program)
        return program(document=doc.text, brief=self.brief(doc))

    def incorporate_feedback(self, account_id: str, correction: UserCorrection):
        example = dspy.Example(
            document=correction.doc_text,
            brief=correction.brief,
            fields=correction.corrected_fields,
            clause_types=correction.corrected_clauses,
            intents=correction.corrected_intents,
            obligations=correction.corrected_obligations,
        ).with_inputs("document", "brief")

        self.feedback_store[account_id].append(example)
        if len(self.feedback_store[account_id]) % 20 == 0:
            self._schedule_recompile(account_id)

    def _schedule_recompile(self, account_id: str):
        # Run async / in background worker
        optimizer = dspy.MIPROv2(
            metric=self._composite_metric,
            num_candidates=10,
            init_temperature=0.7,
            teacher_settings={"model": "claude-opus-4-6"},
        )
        compiled = optimizer.compile(
            student=deepcopy(self.global_program),
            trainset=self.feedback_store[account_id],
        )
        self.account_programs[account_id] = compiled
        compiled.save(f"accounts/{account_id}.json")

    def _composite_metric(self, example, prediction, trace=None):
        f1_fields = span_f1(example.fields, prediction.fields)
        f1_clauses = span_f1(example.clause_types, prediction.clause_types)
        judge = self._llm_judge(example, prediction)
        schema_valid = validate_pydantic(prediction)
        return 0.35*f1_fields + 0.25*f1_clauses + 0.25*judge + 0.15*schema_valid
```

**In-loop assertions (cheap hard constraints, no recompile needed):**

```python
dspy.Assert(
    all(o.obligor in extract_parties(document) for o in obligations),
    "Every obligation's obligor must be a party named in the document.",
)
dspy.Suggest(
    all(i.clause_ref in {c.name for c in clause_types} for i in intents),
    "Each intent's clause_ref must match a clause_type.name in the same output.",
)
```

**Cold-start:** new accounts use `self.global_program` until 20 feedback examples accumulate. Empirical MIPRO threshold for optimization payoff.

**Recompile cost:** ~$15–40 per account per recompile. Scheduled as nightly batch or on the 20-example trigger, whichever comes first.

---

## 5. Cost & Latency Analysis

### 5.1 Per-Document Cost Breakdown

**Assumptions:** avg 50K tokens/doc, avg cluster size K=10, Claude Sonnet 4.5 root, Haiku 4.5 or GPT-4o-mini sub.

| Component | In/Out Tokens | Haiku Sub | 4o-mini Sub |
|---|---|---|---|
| Root LM (5 REPL iters) | 20K / 3K | $0.105 | $0.105 |
| Sub-LM (15 calls, briefed) | 78K / 3K | $0.093 | $0.014 |
| 4 discovery signatures (batched) | 16K / 4K | $0.108 | $0.108 |
| Canonicalization (amortized/doc) | 4K / 1K | $0.027 | $0.027 |
| Embeddings | 10K / — | $0.0002 | $0.0002 |
| **Total per isolated doc** | | **~$0.33** | **~$0.25** |
| **Total per doc in K=10 cluster (amortized)** | | **~$0.28** | **~$0.21** |

**Baseline (no briefing, vanilla RLM, Haiku):** ~$0.37/doc
**Savings from Latent Briefing:** 11% (Haiku) to 32% (4o-mini).

### 5.2 Per-Document Latency Breakdown

| Phase | No Briefing | With Briefing + Batching |
|---|---|---|
| RLM REPL iterations | ~60s (sequential) | ~20s (batched sub-calls) |
| 4-signature discovery pass | ~6s | ~3s (parallel) |
| Brief compaction (embed + JSON) | — | ~2.5s (5 compactions) |
| Canonicalization (amortized) | ~1s | ~1s |
| **Total per doc** | **~67s** | **~26s** |
| **In-cluster amortized (K=10)** | | **~22s** |

### 5.3 Full-CUAD Scale (510 docs, ~50 clusters)

| Metric | Haiku Sub | 4o-mini Sub |
|---|---|---|
| Total cost | ~$143 | ~$107 |
| MIPRO one-time optimization | +$40 | +$30 |
| Serial runtime | ~3.1 hrs | ~3.1 hrs |
| Parallel 10x (cluster-level) | ~19 min | ~19 min |
| **Total dataset budget** | **~$183** | **~$137** |

### 5.4 Expected Accuracy on CUAD Validation

CUAD's 41 labeled categories used as **ground truth only**, never as prompt hints.

| Metric | Expected Range |
|---|---|
| CUAD category coverage (recall of known 41 types) | 75–85% |
| Span-level F1 on covered types | 0.82–0.88 |
| Niche field discovery (items beyond CUAD 41) | 15–30 stable field types |
| Canonicalization consistency (cross-run label agreement) | 0.88–0.94 |

---

## 6. Accuracy & Cost Levers

Each lever is independently toggleable; production deploys stack them per tier (preview vs. final-sign-off).

### 6.1 Accuracy Levers

| # | Lever | Mechanism | F1 Δ | Cost Δ |
|---|---|---|---|---|
| A1 | Self-consistency voting | Run discovery N=3, majority-vote on field names via embedding cluster | +3–6% | +2.5x |
| A2 | Verifier signature | Second-pass LLM-as-judge checks each extracted span against source | +2–4% | +15% |
| A3 | Cross-doc consistency | Reject fields appearing in 1/K docs unless flagged by exploratory mode | +1–2% recall preservation | negligible |
| A4 | CoT scratchpad in signatures | Add `reasoning: str` field to each signature output | +2–3% | +20% output tokens |
| A5 | Retrieval-augmented briefing | Pull exemplar clauses from canonical schema store | +1–3% | +5% |
| A6 | Reflection loop | One revision pass with self-critique after each signature | +1–2% | +40% |
| A7 | Legal-domain sub-LM | Swap sub-LM for SaulLM-7B or CUAD-finetuned | +3–5% on obligations | variable |
| A8 | Stronger root during discovery only | Opus for discovery, Sonnet for canonicalization | +2–3% | +30% |

### 6.2 Cost Levers

| # | Lever | Mechanism | Cost Δ | Accuracy Δ |
|---|---|---|---|---|
| C1 | Prompt caching (Anthropic/OpenAI) | Cache system prompt + rolling state prefix | −40–60% input tokens | 0 |
| C2 | Adaptive max_iterations | Stop when rolling state delta < ε | −20–30% | −0.5% |
| C3 | Cluster schema reuse | After doc 4 in cluster, skip exploratory for docs 5+ | −25% late-cluster | −1% niche recall |
| C4 | Tiered canonicalization | Rule-based fuzzy first, LLM only for ambiguous | −70% canonicalization | 0 |
| C5 | Cold-storage brief cache | Hash (doc_type, cluster_id) → cached brief | −15% | 0 |
| C6 | Route 4o-mini for easy tasks | Fields+Clauses → 4o-mini; Intents+Obligations → Haiku | −30% | −1% |
| C7 | Skip RLM for short docs | Docs <8K tokens bypass RLM, use single-shot sig | −60% on short docs | −2% on those |
| C8 | Aggressive llm_query_batched | 4–8 parallel sub-calls per REPL turn | 0 cost, −40% latency | 0 |
| C9 | Speculative iteration stopping | Predict "done" probability after each iter; early-stop | −15% | −0.5% |
| C10 | Embedding-cached niche filter | Skip exploratory if doc embedding <0.1 from cluster centroid | −15% | −1% niche recall |

### 6.3 Stacked Scenarios

| Scenario | Per-Doc Cost | Per-Doc Latency | Expected F1 |
|---|---|---|---|
| Baseline Approach B | $0.28 | 22s | 0.83 |
| + Levers C1, C2, C4 (caching, adaptive, tiered canon) | $0.14 | 18s | 0.82 |
| + All cost levers + adaptive scheduling | $0.09 | 14s | 0.80 |
| + Levers A1, A2 (self-consistency + verifier) | $0.45 | 38s | 0.89 |
| + All accuracy levers stacked | $0.72 | 55s | 0.91 |

---

## 7. CUAD Evaluation Plan

### 7.1 Evaluation Modes

1. **Closed-set evaluation (standard):** measure recall + span-F1 against CUAD's 41 categories. CUAD labels are ground truth; the system receives no hints about category names.
2. **Open-set evaluation (novel):** count stable niche fields (≥3 occurrences across distinct clusters, not matching any CUAD category). Validate with human legal reviewer on random sample (N=50).
3. **Consistency evaluation:** run the pipeline 5 times with different random seeds. Measure label-pair agreement on the output schemas. Target ≥0.88.
4. **Account-adaptation evaluation:** simulate 3 "accounts" with different corrections to the same corpus. Measure whether per-account compiled programs diverge correctly (separate account fingerprints emerge) without global drift.

### 7.2 Dataset Splits

- **Pilot:** 20 docs, hand-selected for contract-type diversity. Used for token-budget calibration and smoke tests.
- **Dev:** 80 docs for optimizer training (`trainset` for MIPRO).
- **Test:** Remaining ~410 docs, never seen during optimization.

### 7.3 Metrics

| Metric | Definition | Target |
|---|---|---|
| `cuad_coverage` | % of 41 CUAD categories discovered ≥1 time | ≥ 80% |
| `span_f1_covered` | Span-level F1 on discovered CUAD categories | ≥ 0.83 |
| `niche_yield` | Count of stable non-CUAD fields (≥3 occurrences) | ≥ 15 |
| `niche_precision` | % of niche items deemed valid by human reviewer | ≥ 0.70 |
| `consistency` | Jaccard agreement across 5 seed runs | ≥ 0.88 |
| `per_doc_cost` | Mean USD per doc | ≤ $0.30 |
| `per_doc_latency_p95` | p95 wall-clock | ≤ 30s |

---

## 8. Implementation Phases

### Phase 1 — Foundation (Week 1)
- `RollingState` Pydantic model + persistence layer
- Four Pydantic output schemas (`Field`, `ClauseType`, `Intent`, `Obligation`)
- Four `dspy.Signature` classes with structural-hint docstrings
- Unit tests for merge logic, schema validation

### Phase 2 — Text-Native Latent Briefing (Week 2)
- `build_brief` implementation with embedding backend
- Token-budget truncation
- Pilot calibration on 20 CUAD docs to tune `token_budget` and top-K
- Vs-baseline regression test: recall should not drop >5%

### Phase 3 — RLM Harness Wrapper (Week 3)
- `LatentRLM` wrapping `dspy.RLM` with brief injection
- Dual-mode invocation (grounded + exploratory)
- Overlap-penalty merge
- `asyncio.gather` parallelism for the four signatures
- Adaptive stopping based on `last_state_delta`

### Phase 4 — Canonicalization + MIPRO (Week 4)
- Three-tier canonicalization pipeline
- Niche-preservation guardrail (frequency=1 items never auto-merge)
- MIPROv2 compile of the base program on CUAD dev set
- Save/load compiled artifact format

### Phase 5 — Per-Account Feedback (Week 5)
- `AccountAwareDiscovery` class
- Feedback ingestion API (corrections → `dspy.Example`)
- Background recompilation worker (20-example trigger)
- Composite metric (F1 + judge + schema)

### Phase 6 — Levers + Evaluation (Week 6)
- Prompt caching (C1), adaptive iters (C2), tiered canonicalization (C4) — default-on
- Optional levers behind feature flags
- CUAD evaluation harness
- Human-review UI for niche validation

---

## 9. Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Brief token budget too tight → discovery recall drops | Medium | High | Pilot calibration in Phase 2; auto-raise if regression >5% |
| Exploratory mode yields noisy / hallucinated niches | Medium | High | Frequency threshold (≥3 occurrences) + human review on sample |
| Sub-LM (4o-mini) degrades obligation extraction | Medium | Medium | Route obligations to Haiku/Sonnet; measure per-task quality separately |
| MIPRO overfit on small per-account trainsets | Medium | Medium | Held-out validation in optimizer; min-sample threshold of 20 |
| Provider rate limits throttle parallel-10x target | Medium | Medium | Exponential backoff + spread across providers |
| Canonicalization collapses legitimately-distinct concepts | Low | High | Niche-preservation guardrail; human-in-loop for frequency=1 items |
| Per-account feedback drift damages global baseline | Low | Medium | Per-account programs fully isolated from `global_program` |

---

## 10. Success Criteria

A pilot run on a ~50-doc CUAD subset is considered successful if **all** of the following hold:

1. **Coverage:** ≥70% of the 41 CUAD categories are discovered ≥1 time without any hint about their names.
2. **Quality on covered categories:** span-level F1 ≥ 0.80.
3. **Niche discovery:** ≥10 stable non-CUAD fields surfaced, with ≥60% validated as meaningful by human legal reviewer.
4. **Naming consistency:** cross-run Jaccard label agreement ≥ 0.85.
5. **Cost envelope:** mean ≤ $0.30/doc (with Haiku sub-LM).
6. **Latency envelope:** p95 ≤ 30s/doc in-cluster.
7. **Feedback loop:** a simulated 20-correction account trainset produces a measurably diverged per-account compiled program (non-trivial prompt diff vs. global).

If any of (1)–(4) fail, escalate to accuracy-lever stacking (A1, A2, A4). If (5)–(6) fail, apply cost levers (C1, C2, C4) and reassess.

---

## 11. Out of Scope

- **Full-contract end-to-end legal review workflow.** This spec covers discovery, not downstream review UX.
- **Custom LoRA training for legal domain.** Approach C from brainstorming (Latent Context Compilation with disposable LoRA) is explicitly deferred.
- **Multi-lingual contract support.** CUAD is English-only; multi-lingual is a separate effort.
- **Real-time inference (<5s).** Current design targets batch/near-real-time (~22s/doc).
- **Fine-tuning the root or sub-LM.** System uses closed-source APIs only.

---

## 12. Open Questions

1. **Which embedding model for the brief?** `text-embedding-3-small` is the default, but Cohere's `embed-v3` may perform better on legal text. Needs A/B in Phase 2.
2. **Should the exploratory mode see the brief in read-only mode (to avoid duplication) or be truly blinded?** Design assumes blinded; an intermediate option may improve signal.
3. **How frequently should account-specific programs be retired back to global?** If an account stops providing feedback, drift may accumulate. Needs a staleness policy.
4. **What's the right review UI for niche validation?** Not specified here — assumed to be a separate tool surface.

---

## 13. References

- [DSPy RLM API documentation](https://dspy.ai/api/modules/RLM/)
- [RampLabs Latent Briefing announcement](https://x.com/RampLabs/status/2042672773747589588)
- [Recursive Language Models (Zhang, Kraska, Khattab, 2025)](https://alexzhang13.github.io/blog/2025/rlm/)
- [Latent Context Compilation (arxiv 2602.21221)](https://arxiv.org/abs/2602.21221)
- [When Less Latent Leads to Better Relay (arxiv 2604.13349)](https://arxiv.org/abs/2604.13349)
- [CUAD v1 dataset (Atticus Project)](https://www.atticusprojectai.org/cuad/)
- [MIPROv2 optimizer (DSPy)](https://dspy.ai/learn/optimization/optimizers/)
