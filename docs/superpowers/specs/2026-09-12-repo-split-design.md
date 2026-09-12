# Repo split: sentinel-prime and recursio

Date: 2026-09-12 · Owner: aadi · Status: design approved, not yet executed

## 1. Why split

Two incompatible bets were sharing one repository.

The code's accumulated asset is failure-driven self-improvement behind a write barrier,
evaluated against a real judged corpus: `lab_tasks/` + `lab_runs/`, a measured noise floor
(12.7% agent variance, 0 judge variance), the failure-mode counts in `taxonomy.py` (29
omission / 17 partial / ~0 fabrication across 50 failed criteria), and the `arm_f` null
result (empty ledger 31 and 38 of 55; oracle ledger 29 and 34 — the guidance arm was
*worse*, n=2, inside the noise floor).

The uncommitted PRD and PLAN dated 2026-09-09 specify something else: an action-conditioned
predictor guiding recursive investigation in a synthetic partially-observed environment.
Under that plan the first four weeks (T01–T02) build an environment, and none of the corpus
above carries forward.

Both are defensible. Neither is served by sharing a `docs/PLAN.md`, and the pivot edits
sitting uncommitted in the working tree are the symptom: `CLAUDE.md` currently describes the
predictive direction while every module in `sentinelprime/` implements the other one.

## 2. The two repositories

### `aadi2vec/sentinel-prime` — RLM research frontier

Keeps its identity, remote, all 25 commits, all 24 modules, 346 tests, `lab_tasks/` (4.7M),
`lab_runs/` (5.6M), `graveyard/`, `scripts/`.

Rationale: the commit history *is* this project's evidence. A new repo can copy the code but
cannot manufacture the record of the runs that produced the noise floor and the null result.

### `aadi2vec/recursio` — predictive recursion

New repository, clean initial commit. Package name `recursio` (a second package named
`sentinelprime` would collide in any virtualenv holding both, and the split procedure
requires exactly that).

Contents at first commit:

```
recursio/            children.py interpreter.py memory.py audit.py
                     proposals.py telemetry.py policy_search.py
reference/           agent.py assembly.py            (read-only, not collected)
tests/               test_children.py test_interpreter.py test_memory.py
                     test_audit.py test_proposals.py test_telemetry.py
                     test_policy_search.py
docs/                PRD.md PLAN.md
                     pyproject.toml README.md CLAUDE.md AGENTS.md .gitignore
```

No results, no environment, no predictor. The first commit honestly represents that.

## 3. The spine, and why it is seven modules

The initially proposed spine was nine modules including `agent` and `assembly`. It is not
import-closed. `assembly.py` is by design "the one assembly point" and imports everything;
`agent.py` imports `credit`, `grounding`, `subcache`, and `harness`. The transitive closure
of those nine is **eighteen** — it silently drags in `credit`, `feedback`, `grounding`,
`harness`, `monitor`, `planner`, `reuse`, `subcache`, `verifier`.

Dropping `agent` and `assembly` yields a closure of exactly seven, with zero drag:

```
children  interpreter  memory  audit  proposals  telemetry  policy_search
```

Each is domain-neutral: a thread-pool spawn/collect manager, an in-process REPL, a versioned
KV store with snapshot/rollback, an append-only record log, a JSONL proposal log, a tailable
event log with token accounting, and a bounded search with family-disjoint splits and staged
promotion. None knows what a rubric criterion is.

The nine excluded modules are shaped around a rubric judge that `recursio` will not have.
`planner.criterion_ids` regexes `[c1]` out of judge failure text; `credit.py` scores lessons
against rubric criterion ids. Carried into a repo with no judge, they do not sit inert — they
quietly define the interfaces, and the ecosystem environment acquires a rubric because a
module was already expecting one.

Dropping `agent` and `assembly` costs little that the plan was keeping: PLAN §2's own gap
column requires "new typed action/evidence interfaces" for both, "a genuinely global budget"
for `children`, and "replace or contain" for `interpreter`. Copying `agent.py` intact buys a
starting point already committed to rewriting, at the price of nine rubric-shaped modules.

`agent.py` and `assembly.py` are copied to `reference/` instead — read-only, excluded from
`testpaths`, the convention `graveyard/` already establishes in sentinel-prime. Their value
is the prose: the one-assembly-point rule, "ablation stays a flag, not a branch", and the two
refused configurations. That prose survives a copy; the imports do not need to.

## 4. What moves where

| Artifact | sentinel-prime | recursio |
|---|---|---|
| `children, interpreter, memory, audit, proposals, telemetry, policy_search` | stays | **copied** (live spine) |
| `agent, assembly` | stays | **copied to `reference/`** |
| `harness, verifier, planner, credit, monitor, reuse, subcache, feedback, grounding` | stays | — |
| `lab, replay, taxonomy, policy_experiment` | stays | — |
| `lab_tasks/`, `lab_runs/`, `graveyard/`, `scripts/` | stays | — |
| `docs/PRD.md` (uncommitted) | **deleted** | **moves** |
| `docs/PLAN.md` (predictive, uncommitted) | **reverted** | **moves** |
| `docs/PLAN.md` (execution-policy, committed at `fb86667`) | restored | — |
| tests for the seven spine modules | stays | **copied** |

Code is **copied, not moved**. The spine stays live in sentinel-prime because `agent`,
`assembly`, and `harness` depend on it. The two copies will diverge, and that is correct:
these are two projects testing different bets, not one project with a shared library.
Extracting a common package would couple them precisely where independence is the point.

## 5. Document handling

### 5.1 Titles

`PRD.md:1` and `PLAN.md:1` both read "SentinelPrime". That name stays with the other
repository. Exactly these two lines change to "Recursio". Every other word is preserved
verbatim, per the instruction to keep both documents unchanged.

### 5.2 The §2 accuracy note

PLAN §2's "Existing code" table names `harness.py` and `lab.py` in its reuse column, and §T04
(line 85) names `agent.py` / `assembly.py` as files to extend. Under this split, `harness.py`
and `lab.py` are not carried at all, and `agent.py` / `assembly.py` arrive as read-only
reference. On arrival the table is a false map of the repository.

Resolution: prepend one short note to §2 stating which rows the split did not carry and
where `agent`/`assembly` live. The table itself is left verbatim. A note added is compatible
with "keep the plan the same"; shipping a file map that does not match the filesystem is not.

### 5.3 CLAUDE.md / AGENTS.md

`AGENTS.md` is a mechanical mirror of `CLAUDE.md` — only the title line and the tool name
differ. Each repository gets its own pair, generated from its own `CLAUDE.md`. In
sentinel-prime, `AGENTS.md` is regenerated from the *restored* `CLAUDE.md` after step 5.

### 5.4 READMEs

Both repositories are newly easy to misread, so both state their position explicitly.

- **recursio**: no results; a specification plus a substrate copied from sentinel-prime; the
  predictive coupling is unproven and no environment exists yet. The PRD already says this.
  The README must not contradict it by reading like a working system.
- **sentinel-prime**: a pointer to the fork naming what left, so a reader of the history is
  not left with an unexplained deleted PRD.

## 6. Order of operations

The destructive step is last, and gated on the content already existing elsewhere.

0. **Precondition — land `feat/replay-scoring`.** It is 2 commits ahead of `main` (`replay`,
   `taxonomy`) and `origin/main` is at `fb86667`. Splitting first would leave the RLM repo's
   main branch without its most recent asset. PR and merge before anything else.
1. **Create `recursio` locally** — `git init`, copy the seven modules, their seven test files,
   `reference/`, and the two docs. Write its `pyproject.toml`, `README.md`, `CLAUDE.md`,
   `AGENTS.md`, `.gitignore`.
2. **Rewrite imports** — `sentinelprime.` → `recursio.` across the copied modules and tests.
   `reference/` is deliberately **not** rewritten: it is a historical artifact, and its
   original imports are part of what it records. Leaving them unresolvable is also what
   guarantees nothing live can import it by accident.
3. **Verify standalone** — fresh virtualenv, `pip install -e ".[dev]"`, `pytest -q` green with
   no `sentinelprime` on `sys.path`. This is what *proves* the closure claim of §3 rather
   than asserting it.
4. **Push `recursio`.** The content now exists in two places.
5. **Revert the pivot in sentinel-prime** — restore `CLAUDE.md`, `README.md`, `docs/PLAN.md`
   to their committed state; delete `docs/PRD.md`; regenerate `AGENTS.md`. **This discards
   uncommitted work and requires explicit confirmation at the moment of execution**, which
   step 4 has by then made recoverable.
6. **Commit the pointer** in sentinel-prime's README.

## 7. Verification

The split is done when all of the following hold:

- `recursio` installs into a clean virtualenv and its test suite passes with no
  `sentinelprime` importable. Any failure here falsifies §3's closure claim.
- `sentinel-prime` still passes 346 tests after the revert.
- `grep -r sentinelprime recursio/ tests/` in the new repo returns nothing outside
  `reference/` and prose.
- `reference/` is excluded from `testpaths` and is not imported by any live module.
- Neither repository's `docs/` references a file it does not contain.

## 8. Out of scope

**The RLM repo's new plan.** Step 5 restores the *execution-policy* plan, which matches the
code but is not the "research frontier" direction, and does not reflect the `arm_f` null or
the adaptive-compute gap. Rewriting it is a separate brainstorm. Leaving sentinel-prime in a
truthful old state is preferable to introducing a new thesis through a migration commit.

**Any code change to the spine.** The modules are copied and their imports rewritten. Nothing
is refactored, and no new interface is written. The first `recursio` commit contains no new
logic.

**The environment, predictor, and typed interfaces (PLAN T01–T05).** Those begin after the
split, in `recursio`, against its own plan.

## 9. Open items

- Whether `recursio` should carry a `.env.example` and `config.yaml` analogue, or defer
  configuration until T02 gives it something to configure. Deferring is the default.
- The stale memory recording "work directly on main, no feature branches" contradicts five
  merged PRs and the current `feat/replay-scoring` branch. The branch-and-PR pattern the
  repository actually exhibits is followed here, and the memory is corrected separately.
