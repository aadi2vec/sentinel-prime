# sentinel-prime

**An auditable reasoning ledger for a self-improving DSPy agent.**

When an agent learns from its own mistakes at runtime, someone eventually has to answer three
questions about a conclusion it reached: *what guidance did it use, why was that guidance
there, and was it still valid at the time?* SentinelPrime is a DSPy-native harness where every
self-improvement edit is a provenance-scoped, verifier-gated, reversible entry in an
append-only ledger — so those questions have answers a compliance officer can read.

```
[failure       ] task ma-001: Failed rubric criteria:
- [c1] always extract the change of control clause
- [c2] always report the governing law
[verification  ] ladder[grounding_probe] admitted (cost 1): grounding_probe: grounding: 0.42 vocabulary overlap with the observed failures
[ledger_edit   ] create 'lesson.revision' (v1->v2, scope=external, id=285a62ed)
[reversibility ] rollback(from_version=1) restores prior ledger state exactly
```

That is `harness.explain(version)`, verbatim — reproduce it with
`.venv/bin/python scripts/run_lab.py --epochs 3`. On the live path the ladder has a second
rung, and the verification line reads
`ladder[grounding_probe -> llm_verifier] admitted (cost 101): …`.

## Relationship to Prime Agent

The continual-harness *idea* is not ours. [Prime Intellect's Prime
Agent](https://github.com/PrimeIntellect-ai/prime-agent) introduced it: durable supplemental
state (prompts, memories, skill descriptions, sub-agent specs) refined by `/refine` through
small evidence-backed updates that never rewrite the immutable base prompt, with snapshots
supporting rollback. Our `ContinualHarness` is a **DSPy-native reimplementation of that
design**, and the taxonomy (`note | memory | sub_agent_spec`) is theirs.

Two things here are ours, and they are the reason the project exists:

**1. The audit and admission layer.** Prime Agent has snapshots and rollback. This adds a
provenance-scoped, content-addressed, append-only `AuditRecord` per edit and an `explain()`
that renders the causal chain above; a **verifier admission ladder** that gates an edit
*before* it enters the ledger and records why; **reuse gating** that withholds a lesson whose
source has moved; and **credit assignment** that retires a lesson once the failures it claimed
to fix stop clearing. Rollback is a capability. A chain someone can audit is a product.

**2. It is DSPy-native, which makes the online loop itself optimizable.** The proposer and the
verifier are ordinary `dspy.Predict`s, so `named_predictors()` exposes them
(`verifier.verifiers[1].verify_predict`) and GEPA can tune *the machinery that does the online
learning*. Neither framework can do that alone: DSPy ships offline, labeled optimizers
(GEPA/MIPRO) and nothing for online unlabeled self-improvement; Prime Agent ships the online
loop but not as tunable modules.

## The mechanisms

| | What it does | Why it is there |
|---|---|---|
| `ContinualHarness` | `read()` / `refine()` / `rollback()` / `explain()` — versioned WAL + copy-on-write ledger | online, label-free, reversible self-improvement |
| `AuditLog` | append-only, content-addressed `AuditRecord` per edit | a rolled-back edit is still a fact that happened |
| `LadderVerifier` | short-circuited cost ladder: cheap deterministic probe → generative judge | verification is asymmetric; don't pay the LM to reject the obvious |
| `CreditAssigner` | retires a lesson whose targeted criteria stop passing | a polluted ledger is worse than an empty one |
| `ReuseController` | scope / currency / verifier gates on cross-task reuse | a near-match can be wrong, and the world moves |
| `ProgressMonitor` | detects thrash, records a `replan` audit event | the RLM loop is greedy with no progress signal |
| `SubQueryCache` | intra-run sub-query dedup, exact + optional semantic | VDR-scale context is where the tokens go |
| `PrimeAgent` | `dspy.RLM` + workdir interpreter + sub-agents, wired to the ledger | the thing being improved |

## Status — read this before believing anything

**Proven:** 147 tests, all TDD'd. Every mechanism is wired end to end and independently
ablatable, and the live path has been driven end to end with a scripted LM.

**Not proven:** that any of it improves legal reasoning. There are **no benchmark results
yet.** The Harvey LAB M&A slice is not in this repo; `scripts/run_lab.py` ships synthetic
fixtures whose simulated agent is *stipulated* to be misled by bad guidance. Those fixtures
demonstrate that the mechanisms engage and that ablating one changes behavior. They are not
evidence about real models. See [the evaluation
plan](docs/plans/2026-09-06-harvey-lab-eval-plan.md) for the design that would produce real
evidence — including why the headline number has to be a *gap between arms*, not the shape of
a single curve.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # Python 3.10+
cp .env.example .env                                         # add a provider API key
.venv/bin/pytest -q
```

## Quickstart

```python
from sentinelprime import ContinualHarness, PrimeAgent
from sentinelprime.audit import AuditLog
from sentinelprime.credit import CreditAssigner
from sentinelprime.memory import JsonMemoryBackend
from sentinelprime.feedback import parse_lab_result

audit = AuditLog("audit.json")
harness = ContinualHarness(
    JsonMemoryBackend("ledger.json"),
    audit_log=audit,
    credit_assigner=CreditAssigner(audit),   # retire lessons that stop earning their place
)
agent = PrimeAgent(harness, root_lm=lm, sub_lm=lm, enable_children=True)

pred = agent.run_task(task.instructions, workdir=task.dir,
                      context={"matter": "project-crestview", "doc_sha": sha},
                      task_id="ma-001")

feedback = parse_lab_result(grade(pred.deliverable))   # criterion pass/fail — no labels
result = agent.learn(agent.last_trajectory, feedback)  # credit → propose → verify → audit

print(harness.explain(result.to_version))              # the compliance chain
harness.rollback(result.from_version)                  # and it is all reversible
```

## Ablation

Every component is opt-in and independently switchable, so a benchmark can attribute a result
to a mechanism rather than to the stack:

```bash
.venv/bin/python scripts/run_lab.py --epochs 6                 # everything wired
.venv/bin/python scripts/run_lab.py --epochs 6 --no-verifier   # no admission control
.venv/bin/python scripts/run_lab.py --epochs 6 --no-credit     # lessons are never retired
```

On the synthetic fixtures, the verifier and credit assignment turn out to be **two independent
defenses against a polluted ledger**:

| Run | curve | rejected | retired |
|---|---|---|---|
| all wired | 50 → 100 → 100 → 100 → 100 → 100% | 1 | 0 |
| `--no-verifier` | 25 → 75 → 75 → **100** → 100 → 100% | 0 | 2 |
| `--no-verifier --no-credit` | 25 → 50 → 50 → 50 → 50 → 50% | 0 | 0 |

Admission control keeps the bad lesson out. With it off, credit assignment notices the lesson
is not earning its place and retires it — recovery takes four epochs instead of two. With both
off, the ledger stays poisoned. *(Synthetic fixtures. The agent is stipulated to follow bad
guidance; this shows what each gate protects against, not how a real model behaves.)*

## Roadmap

- **Now:** the LAB adapter and the M2 pilot in the [eval plan](docs/plans/2026-09-06-harvey-lab-eval-plan.md).
  If online learning does not beat a frozen-ledger control, this README says so and the project
  stands on the audit layer.
- **Next:** criterion-kind verifier probes (citations, amounts, formatting) so the cost ladder is
  ordered by real ledger statistics; provider prefix-cache measurement.
- **Phase 2:** a Rust durable-execution runtime — WAL, supervision, deterministic replay: the
  audit guarantee under failure.

Design docs in [`docs/`](docs/): [architecture](docs/ARCHITECTURE.md) ·
[eval plan](docs/plans/2026-09-06-harvey-lab-eval-plan.md) · specs and plans under
`docs/superpowers/`.
