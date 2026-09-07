"""run_lab.py — drive SentinelPrime's online self-improvement loop over a task set.

Two modes:

  (default) Scripted mechanism demo — fully deterministic, no network. A *simulated*
      agent and a keyword grader exercise the real loop end to end: gated `read()` ->
      run -> grade -> `refine()` (propose -> apply -> audit) -> `ProgressMonitor`, and
      the pass-rate curve rising across epochs as the ledger accumulates gated lessons.
      This is a simulation of the *mechanism* (the ledger/audit/monitor are the real
      objects); it is not the LM doing the reasoning.

  --live  Real stack — `PrimeAgent` (dspy.RLM + workdir-confined interpreter) against a
      real LM (provider auto-detected from the API key present). The grader is a keyword
      rubric over the deliverable. Point --tasks at the real Harvey LAB M&A slice when
      available; the built-in fixtures make the loop runnable today.

Both modes drive the *same* wired loop, so a benchmark can ablate one component at a
time by switching it off and re-reading the curve:

    --no-verifier        drop admission control (edits enter the ledger unjudged)
    --no-monitor         drop thrash detection (no `replan` audit events)
    --no-context         drop reuse gating (read() stops seeing live task state)
    --no-semantic-cache  drop semantic sub-query dedup (live; exact dedup still runs)

Usage:
    .venv/bin/python scripts/run_lab.py            # scripted, 2 epochs
    .venv/bin/python scripts/run_lab.py --epochs 3
    .venv/bin/python scripts/run_lab.py --live
    .venv/bin/python scripts/run_lab.py --no-verifier   # ablate the admission gate
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile

from sentinelprime.audit import AuditLog, digest
from sentinelprime.harness import ContinualHarness
from sentinelprime.memory import JsonMemoryBackend
from sentinelprime.monitor import ProgressMonitor
from sentinelprime.feedback import parse_lab_result
from sentinelprime.verifier import GroundingProbe, LadderVerifier, VerifierLevel


# ---- Built-in fixture set (stand-in for the LAB M&A slice) -------------------------
# Each task: a document, a prompt, and a rubric of (criterion_id, needle, lesson-reason).
# The needle is the fact the deliverable must contain; the reason is the lesson text the
# proposer will store when the criterion fails (it contains the needle, so learned
# guidance later surfaces it).
FIXTURES = [
    {
        "id": "ma-001",
        "filename": "msa.txt",
        "content": (
            "MASTER SERVICES AGREEMENT\n"
            "Section 3. Term begins 2026-01-01, twelve (12) months, auto-renew unless "
            "30 days notice.\n"
            "Section 7. Governing Law: State of Delaware.\n"
            "Section 9. Change of Control: assignment permitted on merger.\n"
        ),
        "prompt": "Summarize the key diligence facts of msa.txt.",
        "rubric": [
            ("c1", "change of control", "always extract the change of control clause"),
            ("c2", "governing law", "always report the governing law"),
        ],
    },
    {
        "id": "ma-002",
        "filename": "spa.txt",
        "content": (
            "STOCK PURCHASE AGREEMENT\n"
            "Section 2. Governing Law: State of New York.\n"
            "Section 5. Change of Control: consent required on transfer.\n"
            "Section 8. Indemnification cap: 10% of purchase price.\n"
        ),
        "prompt": "Summarize the key diligence facts of spa.txt.",
        "rubric": [
            ("c1", "change of control", "always extract the change of control clause"),
            ("c2", "governing law", "always report the governing law"),
        ],
    },
]


def task_context(task: dict) -> dict:
    """The live state `read()` gates reuse on: which matter, which document revision.

    A lesson recorded against `doc_sha` X is withheld once the document moves to Y —
    that is the whole point of threading context through `run_task`.
    """
    return {
        "matter": task["id"],
        "document": task["filename"],
        "doc_sha": digest(task["content"])[:12],
    }


def amend(task: dict, epoch: int) -> dict:
    """From epoch 2 on, ma-001's document is amended, so its `doc_sha` moves.

    This is the event reuse gating exists for: a lesson derived from the prior revision
    is no longer *provably* current, so `read(context=…)` must withhold it. Without a
    moving source in the fixture set, ablating the gate would show nothing.
    """
    if epoch < 2 or task["id"] != "ma-001":
        return task
    amended = dict(task)
    amended["content"] = task["content"] + "Section 12. Amendment No. 1 dated 2026-03-01.\n"
    return amended


def build_verifier(live: bool) -> LadderVerifier:
    """The admission ladder: cheap deterministic rung first, generative judge second.

    The planner orders by cost / P(fail), so the LM rung only runs on edits the free
    grounding probe could not already reject. Scripted mode keeps the deterministic rung
    alone so the demo stays hermetic.

    `adaptive=True` lets the ladder re-derive P(fail) from its own observed rejections, so
    the ordering sharpens over a run instead of resting on the declared costs alone.
    """
    levels = [VerifierLevel("grounding_probe", cost=1.0, verifier=GroundingProbe())]
    if live:
        from sentinelprime.verifier import PredictVerifier
        levels.append(VerifierLevel("llm_verifier", cost=100.0,
                                    verifier=PredictVerifier(min_score=0.0)))
    return LadderVerifier(levels, adaptive=True)


def print_config(mode: str, verifier, monitor, use_context: bool, embedder) -> None:
    on = lambda flag: "on " if flag else "off"
    print(f"[run_lab] {mode} | verifier={on(verifier is not None)} "
          f"monitor={on(monitor is not None)} context={on(use_context)} "
          f"semantic_cache={on(embedder is not None)}\n")


def print_summary(curve: list[float], audit_log: AuditLog, rejected: int,
                  cache_stats: dict | None = None, withheld: int | None = None) -> None:
    replans = [r for r in audit_log.records() if r.op == "replan"]
    print(f"\n[run_lab] self-improvement curve: "
          f"{' -> '.join(f'{r:.0%}' for r in curve)}")
    print(f"[run_lab] audit records: {len(audit_log.records())} "
          f"| replan events: {len(replans)} | edits rejected by the verifier: {rejected}")
    if withheld is not None:
        print(f"[run_lab] stale lessons withheld by reuse gating: {withheld}")
    if cache_stats:
        print(f"[run_lab] sub-query cache: {cache_stats}")


def print_prefix_report(harness: ContinualHarness, tasks: list[dict],
                        use_context: bool) -> None:
    """Measure where the prompt first diverges between tasks under the final ledger.

    Provider prompt caching bills the shared leading segment, so the number that matters
    is how much of the prompt is identical across tasks — and whether the ledger block is
    inside it. This needs no provider: it renders the real adapter messages and takes the
    longest common prefix.
    """
    from sentinelprime.agent import cacheable_prefix

    ctx = task_context(tasks[-1]) if use_context else None
    ledger = harness.read(context=ctx)
    prefix = cacheable_prefix(ledger or "(no learned guidance yet)",
                              [t["prompt"] for t in tasks])
    if not ledger:
        inside = "n/a (empty ledger)"
    else:
        inside = str(ledger.strip() in prefix)
    print(f"[run_lab] cacheable prompt prefix: {len(prefix)} chars shared across "
          f"{len(tasks)} tasks | ledger block inside it: {inside}")


def grade(deliverable: str, task: dict) -> dict:
    """Keyword rubric -> a LAB-style result dict `parse_lab_result` understands."""
    text = (deliverable or "").lower()
    criteria = []
    for cid, needle, reason in task["rubric"]:
        passed = needle in text
        criteria.append({"id": cid, "passed": passed, "reason": "" if passed else reason})
    return {"task_id": task["id"], "criteria": criteria}


class _SimulatedAgent:
    """Deterministic stand-in: reports a needle only once learned guidance surfaces it.

    Models the difficulty that the base agent misses hidden clauses until a lesson tells
    it to look — so the pass-rate rises only *because* the gated ledger accumulates.

    It also models *how* it fails, in three ways the gates are supposed to catch:

      1. with no guidance it spins, re-reading the same document with near-identical
         reasoning — the thrash `ProgressMonitor` detects;
      2. it *follows* the ungrounded lesson if one reaches it, reformatting as board
         minutes and losing the governing-law fact — what admission control prevents;
      3. it *trusts* a revision-specific lesson after the document has been amended,
         reading the wrong section and losing the change-of-control fact — what reuse
         gating prevents.

    Points 2 and 3 are the reason ablating a gate moves the pass-rate here at all. They
    are also a *stipulation*: this agent is defined to be misled by bad guidance. The
    ablation deltas below therefore demonstrate what each gate protects against; they are
    not evidence about how a real LM responds to a polluted ledger. Only the live path
    against the LAB slice can say that.
    """

    def __init__(self):
        self.last_cache_stats = {"hits": 0, "misses": 0, "calls": 0, "semantic_hits": 0}

    def run(self, task: dict, guidance: str):
        g = (guidance or "").lower()
        found = [needle for _, needle, _ in task["rubric"] if needle in g]
        # Misled by an ungrounded lesson that survived admission control.
        if "board minutes" in g:
            found = [n for n in found if n != "governing law"]
        # Misled by a stale lesson that survived reuse gating: it points at section 9 of
        # a revision this document no longer is.
        if "sits in section 9" in g and "Amendment No. 1" in task["content"]:
            found = [n for n in found if n != "change of control"]
        deliverable = "\n".join(found)
        if found:
            trajectory = [{"reasoning": f"extract facts from {task['filename']}"}]
        else:
            # unguided: three near-identical re-reads -> a reasoning stall
            trajectory = [
                {"reasoning": f"re-read {task['filename']} for the clause i am missing"}
                for _ in range(3)
            ]
        return deliverable, trajectory


def _stub_proposer(harness: ContinualHarness, context_holder: dict | None = None):
    """Deterministic proposer: one create-note per failed criterion, text = the lesson.

    It also emits one *ungrounded* lesson on every round — a plausible-sounding note no
    observed failure supports. That is the proposer's real label-free failure mode, and it
    is what the admission ladder exists to stop; without it in the fixture set, ablating
    the verifier would show nothing.

    One lesson it emits is *session*-scoped and declares the document revision it was
    derived from, so it becomes provably stale the moment that document is amended.
    """
    context_holder = context_holder if context_holder is not None else {}

    def propose(**kw):
        failures = kw.get("rubric_failures", "")
        edits = []
        if failures.strip() and "All rubric criteria passed" not in failures:
            edits.append({
                "op": "create", "id": "lesson.hallucinated", "kind": "note",
                "text": "prefer tabular output for board minutes", "scope": "global",
            })
            edits.append({
                "op": "create", "id": "lesson.revision", "kind": "note",
                "text": "for this revision the change of control clause sits in section 9",
                "scope": "session",
                "meta": {"depends_on": {"doc_sha": context_holder.get("doc_sha", "")}},
            })
        for task in FIXTURES:
            for cid, _needle, reason in task["rubric"]:
                if f"[{cid}]" in failures and reason in failures:
                    edits.append({
                        "op": "create", "id": f"lesson.{cid}", "kind": "note",
                        "text": reason, "scope": "global",
                    })
        # dedup by id (a criterion may recur across tasks)
        seen, uniq = set(), []
        for e in edits:
            if e["id"] not in seen:
                seen.add(e["id"])
                uniq.append(e)
        return type("P", (), {"edits": json.dumps(uniq)})()

    harness.propose = propose


def run_scripted(epochs: int, *, use_verifier: bool = True, use_monitor: bool = True,
                 use_context: bool = True) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        backend = JsonMemoryBackend(str(tmp / "ledger.json"))
        audit_log = AuditLog(str(tmp / "audit.json"))
        verifier = build_verifier(live=False) if use_verifier else None
        harness = ContinualHarness(backend, audit_log=audit_log, verifier=verifier)
        # Mutable holder: the proposer stamps each session-scoped lesson with the document
        # revision it was derived from, which is what makes it gateable later.
        context_holder: dict = {}
        _stub_proposer(harness, context_holder)
        monitor = ProgressMonitor() if use_monitor else None
        agent = _SimulatedAgent()

        print("[run_lab] scripted mechanism demo (deterministic, no network)")
        print_config("scripted", verifier, monitor, use_context, None)
        curve: list[float] = []
        rejected_total = withheld_total = 0
        for epoch in range(1, epochs + 1):
            passed_criteria = total_criteria = 0
            for task in FIXTURES:
                task = amend(task, epoch)
                # Reuse gating sees the live task state; None -> ungated (the ablation).
                ctx = task_context(task) if use_context else None
                context_holder.clear()
                context_holder.update(task_context(task))
                withheld = (len(backend.read()) -
                            len(harness.admissible_items(context=ctx)))
                withheld_total += withheld
                guidance = harness.read(context=ctx)
                deliverable, trajectory = agent.run(task, guidance)
                result = grade(deliverable, task)
                fb = parse_lab_result(result)
                version = backend.current_version().number
                if monitor is not None:
                    monitor.check_and_record(trajectory, agent.last_cache_stats,
                                             audit_log, version, task["id"])
                refine = harness.refine(trajectory, fb)
                rejected_total += len(refine.rejected)
                passed_criteria += sum(1 for c in result["criteria"] if c["passed"])
                total_criteria += len(result["criteria"])
                print(f"  epoch {epoch} {task['id']}: score={fb.score:.2f} "
                      f"created={refine.created} updated={refine.updated} "
                      f"rejected={refine.rejected} withheld_by_context={withheld}")
            rate = passed_criteria / total_criteria if total_criteria else 0.0
            curve.append(rate)
            print(f"  epoch {epoch} pass-rate: {rate:.0%}\n")

        print_summary(curve, audit_log, rejected_total, withheld=withheld_total)
        print_prefix_report(harness, FIXTURES, use_context)
        # Show the compliance chain for the first learned edit.
        first_edit_version = min(
            (r.to_version for r in audit_log.records() if r.op == "create"),
            default=None,
        )
        if first_edit_version is not None:
            print("\n[run_lab] explain() for the first learned edit:\n")
            print(harness.explain(first_edit_version))


def run_live(epochs: int, *, use_verifier: bool = True, use_monitor: bool = True,
             use_context: bool = True, semantic_cache: bool = True) -> None:
    import dspy
    from sentinelprime.agent import PrimeAgent

    model = _pick_model()
    lm = dspy.LM(model, **_lm_kwargs())
    # The harness's proposer and the ladder's LM rung are dspy.Predicts that run in
    # learn()/refine() outside the RLM's own lm-context, so give DSPy a default LM.
    dspy.configure(lm=lm)
    embedder = _build_embedder() if semantic_cache else None

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        backend = JsonMemoryBackend(str(tmp / "ledger.json"))
        audit_log = AuditLog(str(tmp / "audit.json"))
        verifier = build_verifier(live=True) if use_verifier else None
        harness = ContinualHarness(backend, audit_log=audit_log, verifier=verifier)
        monitor = ProgressMonitor() if use_monitor else None
        agent = PrimeAgent(harness, root_lm=lm, sub_lm=lm, monitor=monitor,
                           subquery_embedder=embedder)

        print_config(f"live ({model})", verifier, monitor, use_context, embedder)
        curve: list[float] = []
        rejected_total = 0
        for epoch in range(1, epochs + 1):
            passed = total = 0
            for task in FIXTURES:
                workdir = tmp / f"{task['id']}-e{epoch}"
                workdir.mkdir()
                (workdir / task["filename"]).write_text(task["content"])
                # run_task now owns the gated read *and* the progress monitor, so the
                # live path exercises the same seams the scripted loop does.
                pred = agent.run_task(
                    task["prompt"], workdir=str(workdir),
                    context=task_context(task) if use_context else None,
                    task_id=task["id"],
                )
                deliverable = getattr(pred, "deliverable", str(pred))
                result = grade(deliverable, task)
                fb = parse_lab_result(result)
                # Learn from the *real* RLM trajectory, not an empty list.
                refine = agent.learn(agent.last_trajectory, fb)
                rejected_total += len(refine.rejected)
                decision = agent.last_monitor_decision
                if decision is not None and decision.replan:
                    print(f"    replan detected: {'; '.join(decision.reasons)}")
                passed += sum(1 for c in result["criteria"] if c["passed"])
                total += len(result["criteria"])
                print(f"  epoch {epoch} {task['id']}: score={fb.score:.2f} "
                      f"created={refine.created} rejected={refine.rejected}")
            curve.append(passed / total if total else 0.0)
        print_summary(curve, audit_log, rejected_total, agent.last_cache_stats)
        print_prefix_report(harness, FIXTURES, use_context)


def _build_embedder():
    """Semantic sub-query dedup for the live path.

    Returns None if an embedder cannot even be constructed. If the endpoint later refuses
    the request, `SubQueryCache` disables its semantic tier with a warning and the run
    continues on exact dedup — so enabling this can cost a run its dedup *rate*, never the
    run itself.
    """
    import dspy

    from sentinelprime.subcache import make_embedder

    model = os.environ.get("EMBED_MODEL", "openai/text-embedding-3-small")
    try:
        embedder = make_embedder(dspy.Embedder(model))
    except Exception as exc:
        print(f"[run_lab] semantic sub-query cache unavailable ({exc!r}); exact dedup only")
        return None
    print(f"[run_lab] semantic sub-query cache embedder: {model}")
    return embedder


def _pick_model() -> str:
    _load_dotenv()
    if os.environ.get("LAB_MODEL"):
        return os.environ["LAB_MODEL"]
    # OpenAI first: this repo is configured for the gpt-5.6 "luna" endpoint.
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("OPENAI_MODEL", "openai/gpt-5.6-luna")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-5-20250929"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini/gemini-2.5-flash"
    raise SystemExit("No provider API key found in environment (.env).")


def _lm_kwargs() -> dict:
    # Custom endpoint (e.g. the luna proxy) via OPENAI_BASE_URL; blank -> api.openai.com.
    kwargs = {"max_tokens": 4000}
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        kwargs["api_base"] = base
    return kwargs


def _load_dotenv(path: str = ".env") -> None:
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        os.environ.setdefault(key.strip(), val)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="run the real PrimeAgent stack")
    ap.add_argument("--epochs", type=int, default=2, help="passes over the task set")
    # Ablation switches: turn one component off and re-read the curve.
    ap.add_argument("--no-verifier", action="store_true",
                    help="ablate admission control (edits enter the ledger unjudged)")
    ap.add_argument("--no-monitor", action="store_true",
                    help="ablate thrash detection (no replan audit events)")
    ap.add_argument("--no-context", action="store_true",
                    help="ablate reuse gating (read() stops seeing live task state)")
    ap.add_argument("--no-semantic-cache", action="store_true",
                    help="ablate semantic sub-query dedup (live only; exact dedup remains)")
    args = ap.parse_args()
    common = dict(use_verifier=not args.no_verifier, use_monitor=not args.no_monitor,
                  use_context=not args.no_context)
    if args.live:
        run_live(args.epochs, semantic_cache=not args.no_semantic_cache, **common)
    else:
        if args.no_semantic_cache:
            print("[run_lab] --no-semantic-cache applies to --live only; ignoring.")
        run_scripted(args.epochs, **common)


if __name__ == "__main__":
    main()
