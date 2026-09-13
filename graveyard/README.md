# Graveyard

Retired on 2026-09-07, when the project's object of study changed from **what the agent
remembers** to **how the agent computes**. Nothing here is imported by the live tree;
`pyproject.toml` sets `testpaths = ["tests"]` so none of it is collected.

Kept rather than deleted because several of these files contain measurements and arguments
that are still true and still cited by the surviving documents. Deleting them would strand
the citations.

`docs/PLAN.md` is the only plan. The retired planning documents were deleted rather than
archived here: they were long menus of extensions, and having them to hand is what kept
pulling work away from the one experiment. Their load-bearing measurements were copied into
`docs/PLAN.md` before deletion, so nothing cites a file that no longer exists.

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
| `root/config.example.yaml` | Template for the retired `config.py` |
| `root/rlm/` | Stale `__pycache__` from a vendored prototype. No source, ever |

## Restoring something

```bash
git mv graveyard/<path> <original location>   # tracked files keep their history
```
