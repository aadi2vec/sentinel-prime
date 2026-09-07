# Graveyard

Retired on 2026-09-07, when the project's object of study changed from **what the agent
remembers** to **how the agent computes**. Nothing here is imported by the live tree;
`pyproject.toml` sets `testpaths = ["tests"]` so none of it is collected.

Kept rather than deleted because several of these files contain measurements and arguments
that are still true and still cited by the surviving documents. Deleting them would strand
the citations.

The two live documents are:

- `docs/plans/2026-09-07-rlm-policy-evolution-plan.md` — the plan of attack
- `docs/superpowers/specs/2026-09-07-execution-policy-search-design.md` — the design

## What is here and why it was retired

| Path | Why |
|---|---|
| `sentinelprime/optimize.py` | Offline GEPA circuit over the proposer/verifier prompts. Tuning the *content* loop; the object of study is now the execution policy, and `policy_search.py` runs its own bounded proposal loop |
| `sentinelprime/config.py`, `session.py` | YAML config loader and a session store, used by nothing outside their own tests |
| `scripts/paired_ab.py` | The `control → learning → gated → tuned` content-learning experiment. Superseded by policy search; its measured control-arm results are quoted in the spec |
| `scripts/arm_f.py` | The oracle-ledger test of whether guidance helps at all. **Its result is the reason the project turned** — see spec §1.1. Retired because it answered its question |
| `scripts/run_lab.py` | Synthetic-fixture ablation demo. The fixtures stipulated the agent's response to guidance, so no number from it was ever evidence |
| `scripts/gepa_runner.py` | Front end for `optimize.py` |
| `scripts/noise_floor.py` | Agent/judge variance measurement. **The one most likely to be wanted back** — spec §7 names "every candidate lands inside the noise" as a way this project fails, and this is the instrument for detecting that |
| `scripts/lab_eval.py`, `smoke_live.py`, `watch_run.py`, `fetch_lab_tasks.py` | Runners and utilities for the retired experiments |
| `tests/test_optimize.py`, `test_config.py`, `test_session.py` | Tests of the above |
| `lab_eval/*.py` | Vendored Harvey LAB scoring code — 2,400 lines, imported by nothing, and needing `anthropic`, `mistralai`, `seaborn` and a `utils.stdio` that does not exist in this repo. Only `lab_tasks/_lab_eval/prompts/rubric_criterion.txt` was ever live, and it stays: `sentinelprime/lab.py` grades with LAB's own prompt rather than one of ours |
| `docs/ARCHITECTURE.md` | Describes the content-learning system as the thesis |
| `docs/2026-09-06-harvey-lab-eval-plan.md` | The B−A-on-pooled-rate experiment. Retired because the measured 29pp spread makes that comparison unreadable at affordable n |
| `docs/2026-09-07-evidence-runtime-product-strategy.md` | Legal-review product strategy — packaging, design partners, tenant controls. Retired by the scope decision: legal is the stress test, not the product. Its §4 execution-architecture material is still worth reading |
| `docs/2026-09-07-related-work-and-extensions.md` | Literature sweep (ACE, TAME, SEAL, DGM, solver-verifier gap). Retired as a *plan*; still the reference for why verification fidelity sets the ceiling |
| `docs/2026-08-*.md`, `docs/2026-09-0[15]-*.md` | Superseded specs and plans from the content-learning era |
| `root/AGENTS.md` | Codex-flavored duplicate of `CLAUDE.md` that documented the now-deleted scripts |
| `root/config.example.yaml` | Template for the retired `config.py` |
| `root/rlm/` | Stale `__pycache__` from a vendored prototype. No source, ever |

## Restoring something

```bash
git mv graveyard/<path> <original location>   # tracked files keep their history
```
