# Repo Split (sentinel-prime / recursio) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split one repository carrying two incompatible research bets into `aadi2vec/sentinel-prime` (RLM research frontier, keeps all history and evidence) and a new `aadi2vec/recursio` (predictive recursion, clean initial commit, seven-module spine).

**Architecture:** No code is written. Seven import-closed modules are *copied* (not moved) into a new repository, their `sentinelprime.` imports rewritten to `recursio.`, two test files trimmed of the 12 tests that reach outside the spine, and `agent.py`/`assembly.py` parked in a read-only `reference/` with imports deliberately left unrewritten. The predictive PRD and PLAN move; the execution-policy plan is restored in place.

**Tech Stack:** Python 3.12 (floor 3.10), `dspy>=2.6`, `pytest>=8.0`, setuptools, git, `gh` CLI via `GH_TOKEN`.

**Spec:** `docs/superpowers/specs/2026-09-12-repo-split-design.md`

## Global Constraints

- **Source repo path:** `/Users/aadityasrivathsan/Desktop/rlm_project` (remote `aadi2vec/sentinel-prime`).
- **New repo path:** `/Users/aadityasrivathsan/Desktop/recursio` (verified free). Remote `aadi2vec/recursio`.
- **The spine is exactly seven modules:** `children, interpreter, memory, audit, proposals, telemetry, policy_search`. Do not add an eighth. Adding `agent` or `assembly` breaks import closure and drags in nine rubric-shaped modules.
- **Package name is `recursio`**, never `sentinelprime` — two packages of the same name collide in any venv holding both, and Task 5 requires exactly that.
- **Code is copied, not moved.** All seven modules remain live in sentinel-prime; `agent`, `assembly`, and `harness` depend on them. Never delete them from the source repo.
- **`reference/` imports are NOT rewritten.** Leaving them unresolvable is what guarantees no live module imports it by accident.
- **Commit on `main` in the main tree root.** Branches only where this plan explicitly creates one (Task 1, which is a PR).
- **Two gated steps require explicit user confirmation at the moment of execution:** Task 9 (first push to a new remote) and Task 10 (discards uncommitted work). Stop and ask; do not batch through them.
- **Python invocations use `.venv/bin/...` in sentinel-prime** and `.venv/bin/...` in recursio's own venv. There is no runner script.

---

### Task 0: Put the evidence under version control

`git ls-files lab_runs` and `git ls-files lab_tasks` both return 0. 10.3MB of judged corpus across 278 files — every run behind the 12.7% noise floor, the taxonomy counts, and the arm_f null — exists only on this disk. A fresh clone gives `replay.py` nothing to replay. The split is the moment this becomes load-bearing, because sentinel-prime's entire claim to keeping the repo identity is that it holds this evidence.

**Files:**
- Modify: `/Users/aadityasrivathsan/Desktop/rlm_project/.gitignore` (last two lines)
- Modify: `/Users/aadityasrivathsan/Desktop/rlm_project/docs/superpowers/specs/2026-09-12-repo-split-design.md` (§1, §2)

**Interfaces:**
- Consumes: nothing.
- Produces: a tracked `lab_tasks/` and `lab_runs/` that Task 11's README pointer can honestly reference.

- [ ] **Step 1: Confirm the corpus is untracked and measure it**

```bash
cd /Users/aadityasrivathsan/Desktop/rlm_project
git ls-files lab_runs lab_tasks | wc -l        # expect: 0
find lab_tasks lab_runs -type f | wc -l        # expect: 278
du -sh lab_tasks lab_runs                      # expect: 4.7M, 5.6M
```

Expected: `0` tracked, `278` files on disk, 10.3MB total.

- [ ] **Step 2: Remove the two ignore lines**

In `.gitignore`, delete these two lines (currently the last two in the file, appended below the `.worktrees/` block with no section comment):

```
lab_tasks/
lab_runs/
```

Leave every other line unchanged. In particular `.env` and `config.yaml` MUST remain ignored — `config.yaml` names a provider endpoint and `.env` holds the key.

- [ ] **Step 3: Verify no secret is about to be committed**

```bash
git status --short lab_tasks lab_runs | head
git check-ignore -v .env config.yaml          # both must still print a match
grep -rlI "sk-\|api[_-]key\|OPENAI_API_KEY" lab_tasks lab_runs | head
```

Expected: `.env` and `config.yaml` still ignored; the grep prints nothing. If the grep prints any file, stop and report it — do not commit.

- [ ] **Step 4: Commit the corpus**

```bash
git add .gitignore lab_tasks lab_runs
git commit -m "$(cat <<'EOF'
chore: track the judged corpus instead of gitignoring it

lab_tasks/ and lab_runs/ were ignored, so 10.3MB of evidence lived only on
one disk. A fresh clone got replay.py with nothing to replay and taxonomy.py
with nothing to classify — including the runs behind the 12.7% noise floor,
the 29/17/0 failure-mode counts, and the arm_f null result.

The repo's claim to this work is the evidence, not just the commit messages.
EOF
)"
```

- [ ] **Step 5: Correct the spec's overstatement**

In `docs/superpowers/specs/2026-09-12-repo-split-design.md` §2, the sentence

> Rationale: the commit history *is* this project's evidence.

is now true only because Task 0 made it so. Replace it with:

> Rationale: this repository holds the evidence — the judged corpus in `lab_tasks/`
> and `lab_runs/` (tracked as of 2026-09-12) plus the commits that produced it. A new
> repository can copy the code but cannot manufacture the record of the runs.

- [ ] **Step 6: Commit the spec correction**

```bash
git add docs/superpowers/specs/2026-09-12-repo-split-design.md
git commit -m "docs: correct the split spec — the corpus is tracked as of Task 0"
```

---

### Task 1: Land `feat/replay-scoring` before splitting

`main` is 2 commits behind this branch (`replay`, `taxonomy`) and `origin/main` is at `fb86667`. Splitting first leaves sentinel-prime's main branch without its most recent asset — and `replay.py` is the module that makes the whole check-synthesis direction affordable.

**Files:** none modified; this is a merge.

**Interfaces:**
- Consumes: Task 0's two commits (they were made on `feat/replay-scoring`, which is the current branch).
- Produces: a `main` containing `replay.py`, `taxonomy.py`, the corpus, and the spec/plan docs.

- [ ] **Step 1: Confirm the branch state**

```bash
cd /Users/aadityasrivathsan/Desktop/rlm_project
git branch --show-current                      # expect: feat/replay-scoring
git rev-list --left-right --count main...HEAD  # expect: 0  <N>
git log --oneline main..HEAD
```

Expected: `0` on the left (main is not ahead), and the log lists `replay`, `taxonomy`, the design spec, this plan, and Task 0's commits.

- [ ] **Step 2: Run the full suite before merging**

```bash
.venv/bin/pytest -q
```

Expected: `346 passed`. If the count differs, stop — Task 0 should not have changed any test.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin feat/replay-scoring
```

- [ ] **Step 4: Open the PR**

`gh` is not authenticated; pass the keychain credential as `GH_TOKEN`.

```bash
GH_TOKEN=$(security find-internet-password -s github.com -w) gh pr create \
  --title "Replay scoring, failure taxonomy, and the tracked corpus" \
  --body "$(cat <<'EOF'
## Summary
- `replay.py` scores a check against runs that already happened, turning the agent run into a fixed cost amortised over every candidate
- `taxonomy.py` classifies how deliverables failed: 29 omission / 17 partial / ~0 fabrication across 50 failed criteria
- `lab_tasks/` and `lab_runs/` are now tracked rather than gitignored

## Test plan
- [ ] `.venv/bin/pytest -q` passes (346 tests)
- [ ] `git ls-files lab_runs | wc -l` is non-zero after merge
EOF
)"
```

- [ ] **Step 5: Merge and return to main**

Merge via the PR, then:

```bash
git checkout main && git pull && git log --oneline -1
.venv/bin/pytest -q                            # expect: 346 passed
```

---

### Task 2: Scaffold the recursio repository

**Files:**
- Create: `/Users/aadityasrivathsan/Desktop/recursio/pyproject.toml`
- Create: `/Users/aadityasrivathsan/Desktop/recursio/.gitignore`
- Create: `/Users/aadityasrivathsan/Desktop/recursio/recursio/` (directory)
- Create: `/Users/aadityasrivathsan/Desktop/recursio/tests/__init__.py`
- Create: `/Users/aadityasrivathsan/Desktop/recursio/reference/` (directory)
- Create: `/Users/aadityasrivathsan/Desktop/recursio/docs/` (directory)

**Interfaces:**
- Consumes: nothing.
- Produces: `pyproject.toml` declaring package `recursio`, `testpaths = ["tests"]`, and the `integration` marker that `tests/test_policy_search.py` uses.

- [ ] **Step 1: Create the directory tree**

```bash
test -e /Users/aadityasrivathsan/Desktop/recursio && echo "COLLISION - stop" || \
  mkdir -p /Users/aadityasrivathsan/Desktop/recursio/{recursio,tests,reference,docs}
cd /Users/aadityasrivathsan/Desktop/recursio && git init -b main && pwd
```

Expected: `Initialized empty Git repository`. If `COLLISION - stop` prints, halt and report.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "recursio"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["dspy>=2.6"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
# `reference/` is a read-only historical artifact with deliberately unrewritten
# imports. It has no __init__.py and is excluded here so it can never be packaged
# or imported by a live module.
include = ["recursio*"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: exercises the real dspy.RLM loop (scripted LM, still no network)"]
```

- [ ] **Step 3: Write `.gitignore`**

This is sentinel-prime's file minus the `lab_tasks/`/`lab_runs/` lines (removed in Task 0) and minus `config.yaml` (recursio has no config yet — see spec §9).

```
# Secrets
.env

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# OS
.DS_Store

# SDD scratch workspace (git-ignored per skill)
.superpowers/

# Git worktrees (git-ignored per skill)
.worktrees/
```

- [ ] **Step 4: Create the empty tests package marker**

```bash
touch /Users/aadityasrivathsan/Desktop/recursio/tests/__init__.py
```

- [ ] **Step 5: Verify the tree**

```bash
cd /Users/aadityasrivathsan/Desktop/recursio && find . -not -path './.git/*' | sort
```

Expected: `./pyproject.toml`, `./.gitignore`, `./recursio`, `./tests`, `./tests/__init__.py`, `./reference`, `./docs`.

---

### Task 3: Copy the seven spine modules and rewrite imports

**Files:**
- Create: `recursio/{children,interpreter,memory,audit,proposals,telemetry,policy_search}.py`
- Create: `recursio/__init__.py`

**Interfaces:**
- Consumes: Task 2's package directory.
- Produces: importable `recursio.children.ChildSessionManager`, `recursio.interpreter.LocalInterpreter`, `recursio.memory.JsonMemoryBackend`, `recursio.audit.AuditLog`/`AuditRecord`/`machinery_fingerprint`, `recursio.proposals.ProposalLog`/`ProposalRecord`/`proposal_id`, `recursio.telemetry.RunLog`/`UsageMeter`, `recursio.policy_search.Policy`/`Budget`/`PolicyArchive`.

- [ ] **Step 1: Copy the seven files**

```bash
SRC=/Users/aadityasrivathsan/Desktop/rlm_project/sentinelprime
DST=/Users/aadityasrivathsan/Desktop/recursio/recursio
for m in children interpreter memory audit proposals telemetry policy_search; do
  cp "$SRC/$m.py" "$DST/$m.py"
done
ls "$DST"
```

Expected: exactly seven `.py` files.

- [ ] **Step 2: Rewrite the internal imports**

Only `proposals.py` imports another spine module (`audit`); the other six have no internal imports. Rewrite all of them anyway so the command is uniform and future-proof:

```bash
cd /Users/aadityasrivathsan/Desktop/recursio
grep -rl "sentinelprime" recursio/ | xargs sed -i '' 's/\bsentinelprime\./recursio./g; s/\bfrom sentinelprime\b/from recursio/g'
grep -rn "sentinelprime" recursio/ || echo "CLEAN: no sentinelprime references remain"
```

Expected: `CLEAN: no sentinelprime references remain`.

- [ ] **Step 3: Write a fresh `__init__.py`**

Do NOT copy sentinel-prime's — it imports `harness`, `agent`, and `assembly`, none of which exist here.

```python
"""Recursio: substrate for predictive reasoning with executable recursion.

This package is the seven-module spine copied from sentinel-prime at the 2026-09-12
split: a spawn/collect child manager, an in-process REPL, a versioned memory backend,
an append-only audit log, a proposal log, run telemetry, and a bounded policy search.

Nothing here implements the architecture in docs/PRD.md. These are the domain-neutral
pieces that survived the split because they assume nothing about rubrics, judges, or
legal documents — see docs/PLAN.md T01 for where the actual work starts.
"""

__all__: list[str] = []
```

Deliberately empty: re-exporting the spine would invite code to depend on it as an API before the typed interfaces in PLAN T03 exist.

- [ ] **Step 4: Verify each module imports standalone**

```bash
cd /Users/aadityasrivathsan/Desktop/recursio
python3 -c "
import sys; sys.path.insert(0, '.')
for m in ['children','interpreter','memory','audit','proposals','telemetry','policy_search']:
    __import__('recursio.' + m); print('ok', m)
"
```

Expected: seven `ok` lines. `interpreter` and `policy_search` need `dspy` — if it is not on the system python, defer this check to Task 5 (which installs into a venv) and note it.

---

### Task 4: Copy the seven test files and trim the 12 out-of-spine tests

The module closure is clean but the *test* closure is not. `tests/test_audit.py` imports `harness` and `feedback` at module scope because 10 of its 18 tests are really `refine()` tests asserting on the audit record it emits. `tests/test_policy_search.py`'s last two tests reach `policy_experiment` and `scripts.policy_lab`. Both deletion blocks are contiguous.

These tests are correctly written and correctly placed — they test modules that are not coming. They stay in sentinel-prime and are deleted here, not ported.

**Files:**
- Create: `tests/test_{children,interpreter,memory,audit,proposals,telemetry,policy_search}.py`
- Modify: `tests/test_audit.py` (delete lines 72–247 and two imports)
- Modify: `tests/test_policy_search.py` (delete from line 148 to EOF)

**Interfaces:**
- Consumes: Task 3's `recursio.*` modules.
- Produces: a 69-test suite (8 audit + 12 policy_search + 6 children + 16 interpreter + 7 memory + 9 proposals + 11 telemetry).

- [ ] **Step 1: Copy the seven test files**

```bash
SRC=/Users/aadityasrivathsan/Desktop/rlm_project/tests
DST=/Users/aadityasrivathsan/Desktop/recursio/tests
for m in children interpreter memory audit proposals telemetry policy_search; do
  cp "$SRC/test_$m.py" "$DST/test_$m.py"
done
```

- [ ] **Step 2: Trim `tests/test_audit.py`**

Delete the two module-level imports of excluded modules (originally lines 4 and 6):

```python
from sentinelprime.harness import ContinualHarness
from sentinelprime.feedback import parse_lab_result
```

Then delete the contiguous block from `def test_harness_explain_delegates` through the end of `test_machinery_fingerprint_of_something_with_no_predictors_is_empty` — originally lines 72–247. That block contains these 10 tests and 2 helper classes:

```
test_harness_explain_delegates
class _StubPropose
test_refine_emits_audit_record
test_refine_without_log_is_unchanged
class _StubVerifier
test_refine_rejects_unverified_edit
test_refine_writes_verification_into_record
test_refine_reports_rejected_edit_ids_for_ablation
test_refine_reports_no_rejections_when_ungated
test_machinery_fingerprint_is_stable_for_the_same_module
test_machinery_fingerprint_changes_when_a_prompt_is_rewritten
test_machinery_fingerprint_of_something_with_no_predictors_is_empty
```

`_StubPropose` and `_StubVerifier` are used only by the deleted tests, and `_StubVerifier` carries the only `verifier` import. `test_machinery_fingerprint_of_something_with_no_predictors_is_empty` is in this block because it imports `verifier.GroundingProbe` — it looks pure but is not.

These 8 tests must remain: `test_append_and_roundtrip`, `test_content_hash_dedup`, `test_by_version_filters_on_window_end`, `test_explain_renders_causal_chain`, `test_explain_unknown_version`, `test_audit_record_defaults_machinery_to_empty_for_old_logs`, `test_explain_renders_the_machinery_that_admitted_the_edit`, `test_explain_omits_the_machinery_line_when_unstamped`. Keep the `_record(**over)` helper at line 9 — the survivors use it.

- [ ] **Step 3: Trim `tests/test_policy_search.py`**

Delete from the `@pytest.mark.integration` decorator preceding `def test_real_rlm_uses_full_assembler_and_check_feedback_to_repair` (originally ~line 148) through end of file. That removes exactly:

```
test_real_rlm_uses_full_assembler_and_check_feedback_to_repair   # imports sentinelprime (assembly)
test_scripted_experiment_completes_and_preserves_outputs          # imports scripts.policy_lab
```

The 12 tests above them stay, including `test_fresh_holdout_promotes_and_final_test_never_reaches_proposer` and `test_restart_cannot_reuse_reserved_evaluation_families` — the evaluation-boundary tests are the reason `policy_search` is in the spine at all.

- [ ] **Step 4: Rewrite imports in the tests**

```bash
cd /Users/aadityasrivathsan/Desktop/recursio
grep -rl "sentinelprime" tests/ | xargs sed -i '' 's/\bsentinelprime\./recursio./g; s/\bfrom sentinelprime\b/from recursio/g'
grep -rn "sentinelprime\|scripts\.policy_lab" tests/ || echo "CLEAN: tests reference nothing outside recursio"
```

Expected: `CLEAN: tests reference nothing outside recursio`. If anything prints, a trim was incomplete — fix before moving on.

---

### Task 5: Prove the closure in a clean virtualenv

This is the step that *proves* spec §3's claim rather than asserting it. If it fails, the seven-module closure is wrong and the split design needs revisiting.

**Files:** none modified.

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: a green 69-test suite with no `sentinelprime` importable.

- [ ] **Step 1: Build a fresh venv and install**

```bash
cd /Users/aadityasrivathsan/Desktop/recursio
python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]"
```

- [ ] **Step 2: Prove `sentinelprime` is not importable**

```bash
.venv/bin/python -c "
import importlib.util
assert importlib.util.find_spec('sentinelprime') is None, 'sentinelprime is on the path - the test is invalid'
print('confirmed: sentinelprime is not importable')
"
```

Expected: `confirmed: sentinelprime is not importable`. If it *is* importable the whole verification is meaningless — find and remove the stray path entry before continuing.

- [ ] **Step 3: Run the suite**

```bash
.venv/bin/pytest -q
```

Expected: `69 passed`. A `ModuleNotFoundError` naming any of `harness`, `feedback`, `verifier`, `credit`, `planner`, `monitor`, `reuse`, `subcache`, `grounding`, `agent`, `assembly`, `policy_experiment`, or `scripts` means Task 4's trim missed something — fix the trim, do not add the module.

- [ ] **Step 4: Commit the spine**

```bash
git add pyproject.toml .gitignore recursio/ tests/
git commit -m "$(cat <<'EOF'
feat: the seven-module spine, copied from sentinel-prime

children, interpreter, memory, audit, proposals, telemetry, policy_search —
the import-closed subset that assumes nothing about rubrics or judges.

Not nine: including agent.py and assembly.py is not import-closed and drags in
credit, planner, monitor, verifier, harness, reuse, subcache, feedback and
grounding, all shaped around a rubric judge this repo will not have. They would
not sit inert; they would quietly hand recursio a rubric.

12 tests were dropped rather than ported — 10 audit tests that are really
refine() tests, and 2 policy_search tests that reach the assembler and the lab
runner. They test modules that stayed behind. 69 tests pass with sentinelprime
provably absent from the path.
EOF
)"
```

---

### Task 6: Park `agent.py` and `assembly.py` in `reference/`

**Files:**
- Create: `reference/agent.py`, `reference/assembly.py`, `reference/README.md`

**Interfaces:**
- Consumes: Task 2's `reference/` directory.
- Produces: read-only prose. Nothing imports this.

- [ ] **Step 1: Copy the two files unmodified**

```bash
SRC=/Users/aadityasrivathsan/Desktop/rlm_project/sentinelprime
cp "$SRC/agent.py" "$SRC/assembly.py" /Users/aadityasrivathsan/Desktop/recursio/reference/
```

Do NOT rewrite their imports. They must stay `from sentinelprime.harness import ...` — unresolvable on purpose.

- [ ] **Step 2: Write `reference/README.md`**

```markdown
# reference/ — read-only, not importable

`agent.py` and `assembly.py` as they stood in sentinel-prime at the 2026-09-12 split.
They are here for their docstrings, not their code.

Their `sentinelprime.` imports are **deliberately not rewritten**. They do not resolve,
and that is the point: nothing live can import this directory by accident. It is
excluded from `testpaths` and from the packaged distribution.

Three arguments worth carrying into recursio's own assembler (PLAN T03–T04):

- **One assembly point.** Before `assembly.py` existed, six entry points each hand-built
  an agent and had begun to drift apart. Every entry point calls the assembler, so a
  candidate policy is expressible only as assembler arguments and cannot reach around it.
- **The default is the configuration the project claims.** A thinner default is how five
  of six call sites ended up under-wired — nobody chose it, it was just the shortest
  constructor.
- **Ablation stays a flag, not a branch.** Switching one mechanism must not take a
  different assembly path that might differ in some second way nobody controlled for.
  And a combination that would report a mechanism as *on while doing nothing* is refused
  rather than degraded — a null result that looks like evidence is worse than a crash.

Do not restore these files. recursio's agent and assembler are written against the typed
action/evidence interfaces in PLAN T03, which these predate.
```

- [ ] **Step 3: Verify `reference/` is inert**

```bash
cd /Users/aadityasrivathsan/Desktop/recursio
.venv/bin/pytest -q                              # expect: 69 passed (unchanged)
grep -rn "reference" recursio/ tests/ || echo "CLEAN: nothing live references reference/"
test -f reference/__init__.py && echo "ERROR: reference must not be a package" || echo "ok: not a package"
```

Expected: `69 passed`, `CLEAN: ...`, `ok: not a package`.

- [ ] **Step 4: Commit**

```bash
git add reference/
git commit -m "docs: park agent.py and assembly.py as read-only reference

Their value is the design argument in the docstrings, which survives a copy.
Imports left unrewritten so nothing live can import them by accident."
```

---

### Task 7: Move the PRD and PLAN

**Files:**
- Create: `docs/PRD.md`, `docs/PLAN.md` (in recursio)
- Modify: both title lines; prepend one note to PLAN §2

**Interfaces:**
- Consumes: the uncommitted `docs/PRD.md` and `docs/PLAN.md` in sentinel-prime's working tree.
- Produces: recursio's specification. Task 10 deletes the originals — this task must complete first.

- [ ] **Step 1: Copy both documents**

```bash
SRC=/Users/aadityasrivathsan/Desktop/rlm_project/docs
cp "$SRC/PRD.md" "$SRC/PLAN.md" /Users/aadityasrivathsan/Desktop/recursio/docs/
wc -l /Users/aadityasrivathsan/Desktop/recursio/docs/*.md
```

Expected: `PRD.md` 198 lines, `PLAN.md` 210 lines.

- [ ] **Step 2: Rename in exactly two lines**

`docs/PRD.md` line 1:

```markdown
# Recursio — Predictive Reasoning with Executable Recursion
```

`docs/PLAN.md` line 1:

```markdown
# Recursio — Execution Plan for Predictive Recursive Reasoning
```

Change nothing else. These are the only two occurrences of "SentinelPrime" in either file (verified by grep).

- [ ] **Step 3: Prepend the §2 accuracy note**

PLAN §2's table names `harness.py` and `lab.py` in its reuse column and §T04 names `agent.py`/`assembly.py` as files to extend. Under the split those are not present, so the table is a false map of this repository. Insert this immediately below the `## 2. Existing substrate and necessary changes` heading, above the table. Leave the table itself verbatim.

```markdown
> **Note added at the 2026-09-12 split.** This table was written when one repository held
> both directions. recursio carries a seven-module spine — `children`, `interpreter`,
> `memory`, `audit`, `proposals`, `telemetry`, `policy_search` — and three rows below do
> not describe files present here:
>
> - `agent.py` / `assembly.py` — read-only in `reference/`, not importable. Their gap
>   column ("new typed action/evidence interfaces") is now the whole job, not an edit.
> - `harness.py` — not carried. T09's "use the continual harness" means building one
>   against the typed interfaces, not extending the one in sentinel-prime.
> - `lab.py` — not carried, and its judges were never to be reused as truth anyway.
>
> The rest of the table stands. The reasoning is unchanged; only the file map moved.
```

- [ ] **Step 4: Verify and commit**

```bash
cd /Users/aadityasrivathsan/Desktop/recursio
grep -n "SentinelPrime" docs/*.md || echo "CLEAN: no SentinelPrime references remain"
head -1 docs/PRD.md docs/PLAN.md
git add docs/
git commit -m "docs: the PRD and execution plan for predictive recursion

Moved verbatim from sentinel-prime, where they had been sitting uncommitted while
every module in the repo implemented the other direction. Two title lines renamed;
one note added to PLAN section 2 because three rows of its file map do not describe
this repository."
```

Expected: `CLEAN: no SentinelPrime references remain`.

---

### Task 8: Write recursio's README, CLAUDE.md, and AGENTS.md

**Files:**
- Create: `README.md`, `CLAUDE.md`, `AGENTS.md`

**Interfaces:**
- Consumes: Tasks 3–7.
- Produces: the honesty statement spec §5.4 requires.

- [ ] **Step 1: Write `README.md`**

```markdown
# recursio

**Predictive reasoning with executable recursion.** A predictive model guides where to
investigate, recursive execution produces observations, and structured memory keeps
evidence separable from predictions.

## Status: no results

This repository contains a specification and a substrate. It does not contain the
architecture the specification describes.

- There is no environment, no predictor, and no coupling.
- The predictive coupling is **unproven**. Whether predictor-directed investigation beats
  an uncoupled control is the open research question, not a property of this code.
- The seven modules in `recursio/` were copied from
  [sentinel-prime](https://github.com/aadi2vec/sentinel-prime) at the 2026-09-12 split.
  They are domain-neutral infrastructure — a child manager, a REPL, a versioned memory
  backend, two append-only logs, telemetry, and a bounded search with family-disjoint
  evaluation splits. None of them implements anything in the PRD.

Read [`docs/PRD.md`](docs/PRD.md) for the thesis and [`docs/PLAN.md`](docs/PLAN.md) for
the execution sequence. Work starts at T01: define one partially observed environment and
its action/observation boundary.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q          # 69 tests, hermetic, no network
```

## What is not here

`reference/` holds `agent.py` and `assembly.py` from sentinel-prime, read-only, with
unresolvable imports on purpose. They predate the typed interfaces in PLAN T03 and are
kept for their design arguments, not their code.

The sibling repository keeps the RLM research frontier: failure-driven self-improvement,
the admission ladder, credit assignment, and a judged corpus with a measured noise floor.
```

- [ ] **Step 2: Write `CLAUDE.md`**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # setup (Python 3.10+)
.venv/bin/pytest -q                                          # full suite (69 tests, hermetic)
```

Always use `.venv/bin/...` — there is no runner script. No linter or formatter is configured.

## What this repository is

A **specification plus a substrate**. `docs/PRD.md` and `docs/PLAN.md` define proposed
research: predictive models guiding executable recursive investigation, with structured
memory and separately evaluated predictive and procedural self-improvement.

None of it is implemented. Do not describe this repo as having predictive fidelity,
recursive-investigation benefits, or continual-learning gains — it has a test suite and
seven infrastructure modules. Read `docs/PLAN.md` first; T01 is where work begins.

## The seven-module spine

`children` (spawn/collect over a thread pool), `interpreter` (in-process REPL),
`memory` (versioned KV with snapshot/rollback), `audit` (append-only records),
`proposals` (JSONL proposal log), `telemetry` (tailable events + token accounting),
`policy_search` (bounded search with family-disjoint splits and staged promotion).

These were copied from sentinel-prime because they are import-closed and assume nothing
about rubrics or judges. **Do not restore an eighth module from sentinel-prime.** The
nine that were left behind — `harness`, `verifier`, `planner`, `credit`, `monitor`,
`reuse`, `subcache`, `feedback`, `grounding` — are shaped around a rubric judge this repo
does not have (`planner.criterion_ids` regexes `[c1]` out of judge text; `credit.py`
scores against rubric criterion ids). Importing one does not cost a file; it quietly
gives the new architecture a rubric it was designed not to need.

`recursio/__init__.py` is deliberately empty of re-exports. The spine is not an API until
PLAN T03 defines the typed interfaces.

## Invariants worth preserving from the substrate

- **`policy_search` is the evaluation boundary as a type.** `Policy.parse` rejects unknown
  fields; development, validation, and final-test families are disjoint and reserved before
  work begins so a restart cannot re-spend the test set. A candidate that can redefine
  success improves its score without improving its problem-solving.
- **`audit` is append-only.** Records are never edited or deleted; duplicates collapse by
  `content_hash`. A rolled-back edit is still a fact that happened.
- **`audit` and `proposals` are two logs with two purposes.** The audit log records what
  entered the ledger (compliance, all-positive). The proposal log records what a verifier
  was asked about, admitted or not (training data). A gate fit to the audit log learns to
  admit everything. Do not conflate them.
- **`telemetry.UsageMeter` refuses partial totals.** A partial cost gets read as the whole
  bill, so an unpriced model produces no dollar figure at all.

## Security note

`interpreter.py` is an **in-process** Python REPL with real filesystem and subprocess
access. It is *not* a sandbox; `_confine_path` is a convenience, not a boundary. PLAN T03
requires replacing it with an isolated worker before any agent executes over untrusted
evidence. Never point it at untrusted input in the meantime.

## Conventions

- Tests are hermetic: no network, no real LM. Stub the LM seam by assigning over it.
- Module docstrings carry the *why*; inline comments mark invariants. Match that density.
- Citations and analogies belong in `docs/`, never in identifiers.
- `reference/` is read-only with deliberately unresolvable imports. Do not fix them.
- Models are swappable `dspy.LM` instances passed in by the caller; nothing is hardcoded.
```

- [ ] **Step 3: Generate `AGENTS.md`**

`AGENTS.md` is a mechanical mirror of `CLAUDE.md` — only the title and the tool line differ.

```bash
cd /Users/aadityasrivathsan/Desktop/recursio
sed 's/^# CLAUDE\.md$/# AGENTS.md/; s/^This file provides guidance to Claude Code (claude\.ai\/code) when working/This file provides guidance to Codex (Codex.ai\/code) when working/' \
  CLAUDE.md > AGENTS.md
diff CLAUDE.md AGENTS.md
```

Expected: exactly two differing lines.

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md AGENTS.md
git commit -m "docs: README, CLAUDE.md, AGENTS.md

The README leads with 'no results' because a repo containing a confident PRD and a
working test suite reads like a working system, and it is not one."
```

---

### Task 9: Push recursio — GATED

**This creates a public artifact. Stop and ask for explicit confirmation before running Step 2.**

**Files:** none modified.

- [ ] **Step 1: Review what is about to become public**

```bash
cd /Users/aadityasrivathsan/Desktop/recursio
git log --oneline
git ls-files | sort
grep -rlI "sk-\|api[_-]key\|OPENAI_API_KEY\|BASE_URL" $(git ls-files) | head
```

Expected: 5 commits; no `.env`, no `config.yaml`, no secret matches. If the grep prints anything, stop.

- [ ] **Step 2: Ask the user, then create the remote**

Ask: *"recursio is ready — 5 commits, 69 tests passing, no secrets. Create `aadi2vec/recursio` and push? Public or private?"* Wait for an answer. Then:

```bash
GH_TOKEN=$(security find-internet-password -s github.com -w) \
  gh repo create aadi2vec/recursio --private --source=. --remote=origin --push
```

Use `--public` instead of `--private` if that is what the user chose.

- [ ] **Step 3: Confirm**

```bash
git remote -v && git log --oneline origin/main -1
```

---

### Task 10: Revert the pivot in sentinel-prime — GATED

**This discards uncommitted work. Task 7 must have completed and Task 9 must have pushed, so the content exists in two places. Stop and ask for explicit confirmation before running Step 2.**

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/PLAN.md` (restore to committed state)
- Delete: `docs/PRD.md`
- Modify: `AGENTS.md` (regenerate)

- [ ] **Step 1: Show exactly what will be lost**

```bash
cd /Users/aadityasrivathsan/Desktop/rlm_project
git status --short
git diff --stat CLAUDE.md README.md docs/PLAN.md
REC=/Users/aadityasrivathsan/Desktop/recursio
diff <(tail -n +2 docs/PRD.md) <(tail -n +2 "$REC/docs/PRD.md") && echo "PRD body preserved in recursio"
# PLAN differs by design (title + the section-2 note), so compare a line the note cannot touch:
grep -c . docs/PLAN.md "$REC/docs/PLAN.md"
cd "$REC" && git log --oneline -1 && git status --short && cd -
```

`PRD body preserved in recursio` must print (only line 1 differs, by design). recursio's
`git status` must be clean and its log must show the docs commit from Task 7. If either
check fails, **stop** — the content is not safe yet.

- [ ] **Step 2: Ask the user, then revert**

Ask: *"About to discard the uncommitted pivot edits in sentinel-prime: `CLAUDE.md`, `README.md`, `docs/PLAN.md` restored to `fb86667`'s state, `docs/PRD.md` deleted. Both docs are preserved in recursio and pushed. Proceed?"* Wait for an answer. Then:

```bash
git restore --source=HEAD -- CLAUDE.md README.md docs/PLAN.md
rm docs/PRD.md
git status --short
```

Expected: only `AGENTS.md` remains untracked.

- [ ] **Step 3: Regenerate `AGENTS.md` from the restored `CLAUDE.md`**

```bash
sed 's/^# CLAUDE\.md$/# AGENTS.md/; s/^This file provides guidance to Claude Code (claude\.ai\/code) when working/This file provides guidance to Codex (Codex.ai\/code) when working/' \
  CLAUDE.md > AGENTS.md
diff CLAUDE.md AGENTS.md          # expect exactly two differing lines
```

- [ ] **Step 4: Verify nothing broke**

```bash
.venv/bin/pytest -q               # expect: 346 passed
grep -rn "PRD.md" README.md CLAUDE.md docs/ || echo "CLEAN: no dangling PRD references"
```

---

### Task 11: Point sentinel-prime at the fork

**Files:**
- Modify: `README.md` (in sentinel-prime)

- [ ] **Step 1: Add the pointer**

Insert immediately below the `# sentinel-prime` heading:

```markdown
> **The predictive-recursion direction moved to [recursio](https://github.com/aadi2vec/recursio)
> on 2026-09-12.** Its PRD and execution plan live there, along with a copy of seven
> domain-neutral modules from this repo. This repository keeps the RLM research frontier:
> failure-driven self-improvement, the admission ladder, credit assignment, replay scoring,
> and the judged corpus in `lab_tasks/` and `lab_runs/`.
```

- [ ] **Step 2: Commit and verify the final state**

```bash
git add README.md AGENTS.md
git commit -m "docs: point at recursio, where the predictive direction went

The PRD that was sitting uncommitted in this tree now lives in its own repo.
This one keeps the corpus, the noise floor, and the null result."
git status --short                # expect: clean
.venv/bin/pytest -q               # expect: 346 passed
```

- [ ] **Step 3: Push**

```bash
git push origin main
```

---

## Done when

- `recursio`: 5+ commits, `69 passed` in a venv with `sentinelprime` provably absent, no secrets, `reference/` inert and unpackaged.
- `sentinel-prime`: `346 passed`, working tree clean, `lab_tasks/`+`lab_runs/` tracked, no dangling `PRD.md` references, README points at the fork.
- Neither repository's `docs/` references a file it does not contain.

## Deliberately not in this plan

- **Rewriting sentinel-prime's `docs/PLAN.md`.** Task 10 restores the execution-policy plan, which matches the code but is not the "research frontier" direction and does not reflect the arm_f null or the adaptive-compute gap. That is a separate brainstorm. A truthful old plan beats a new thesis smuggled in through a migration commit.
- **Any code change.** Modules are copied and imports rewritten. No refactoring, no new interfaces. recursio's first commits contain no new logic.
- **PLAN T01–T05.** The environment, predictor, and typed interfaces begin after the split, in recursio, against its own plan.
