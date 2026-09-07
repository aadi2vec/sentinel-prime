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

Usage:
    .venv/bin/python scripts/run_lab.py            # scripted, 2 epochs
    .venv/bin/python scripts/run_lab.py --epochs 3
    .venv/bin/python scripts/run_lab.py --live
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile

from sentinelprime.audit import AuditLog
from sentinelprime.harness import ContinualHarness
from sentinelprime.memory import JsonMemoryBackend
from sentinelprime.monitor import ProgressMonitor
from sentinelprime.feedback import parse_lab_result


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
    """

    def __init__(self):
        self.last_cache_stats = {"hits": 0, "misses": 0, "calls": 0, "semantic_hits": 0}

    def run(self, task: dict, guidance: str):
        g = (guidance or "").lower()
        found = [needle for _, needle, _ in task["rubric"] if needle in g]
        deliverable = "\n".join(found)
        # a benign trajectory (healthy) — the monitor should stay quiet here
        trajectory = [{"reasoning": f"extract facts from {task['filename']}"}]
        return deliverable, trajectory


def _stub_proposer(harness: ContinualHarness):
    """Deterministic proposer: one create-note per failed criterion, text = the lesson."""

    def propose(**kw):
        failures = kw.get("rubric_failures", "")
        edits = []
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


def run_scripted(epochs: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        backend = JsonMemoryBackend(str(tmp / "ledger.json"))
        audit_log = AuditLog(str(tmp / "audit.json"))
        harness = ContinualHarness(backend, audit_log=audit_log)
        _stub_proposer(harness)
        monitor = ProgressMonitor()
        agent = _SimulatedAgent()

        print("[run_lab] scripted mechanism demo (deterministic, no network)\n")
        curve = []
        for epoch in range(1, epochs + 1):
            passed_criteria = total_criteria = 0
            for task in FIXTURES:
                guidance = harness.read()
                deliverable, trajectory = agent.run(task, guidance)
                result = grade(deliverable, task)
                fb = parse_lab_result(result)
                version = backend.current_version().number
                monitor.check_and_record(trajectory, agent.last_cache_stats,
                                         audit_log, version, task["id"])
                refine = harness.refine(trajectory, fb)
                passed_criteria += sum(1 for c in result["criteria"] if c["passed"])
                total_criteria += len(result["criteria"])
                print(f"  epoch {epoch} {task['id']}: score={fb.score:.2f} "
                      f"created={refine.created} updated={refine.updated}")
            rate = passed_criteria / total_criteria if total_criteria else 0.0
            curve.append(rate)
            print(f"  epoch {epoch} pass-rate: {rate:.0%}\n")

        print(f"[run_lab] self-improvement curve: "
              f"{' -> '.join(f'{r:.0%}' for r in curve)}")
        print(f"[run_lab] audit records: {len(audit_log.records())}")
        # Show the compliance chain for the first learned edit.
        first_edit_version = min(
            (r.to_version for r in audit_log.records() if r.op == "create"),
            default=None,
        )
        if first_edit_version is not None:
            print("\n[run_lab] explain() for the first learned edit:\n")
            print(harness.explain(first_edit_version))


def run_live(epochs: int) -> None:
    import dspy
    from sentinelprime.agent import PrimeAgent

    model = _pick_model()
    print(f"[run_lab] live mode using model: {model}\n")
    lm = dspy.LM(model, max_tokens=4000)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        backend = JsonMemoryBackend(str(tmp / "ledger.json"))
        audit_log = AuditLog(str(tmp / "audit.json"))
        harness = ContinualHarness(backend, audit_log=audit_log)
        agent = PrimeAgent(harness, root_lm=lm, sub_lm=lm)
        monitor = ProgressMonitor()

        curve = []
        for epoch in range(1, epochs + 1):
            passed = total = 0
            for task in FIXTURES:
                workdir = tmp / f"{task['id']}-e{epoch}"
                workdir.mkdir()
                (workdir / task["filename"]).write_text(task["content"])
                pred = agent.run_task(task["prompt"], workdir=str(workdir))
                deliverable = getattr(pred, "deliverable", str(pred))
                result = grade(deliverable, task)
                fb = parse_lab_result(result)
                version = backend.current_version().number
                monitor.check_and_record([], agent.last_cache_stats,
                                         audit_log, version, task["id"])
                agent.learn([], fb)
                passed += sum(1 for c in result["criteria"] if c["passed"])
                total += len(result["criteria"])
                print(f"  epoch {epoch} {task['id']}: score={fb.score:.2f}")
            curve.append(passed / total if total else 0.0)
        print(f"\n[run_lab] self-improvement curve: "
              f"{' -> '.join(f'{r:.0%}' for r in curve)}")


def _pick_model() -> str:
    _load_dotenv()
    if os.environ.get("LAB_MODEL"):
        return os.environ["LAB_MODEL"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-5-20250929"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini/gemini-2.5-flash"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-4o-mini"
    raise SystemExit("No provider API key found in environment (.env).")


def _load_dotenv(path: str = ".env") -> None:
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="run the real PrimeAgent stack")
    ap.add_argument("--epochs", type=int, default=2, help="passes over the task set")
    args = ap.parse_args()
    if args.live:
        run_live(args.epochs)
    else:
        run_scripted(args.epochs)


if __name__ == "__main__":
    main()
