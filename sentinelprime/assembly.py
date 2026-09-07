"""One place that wires a SentinelPrime agent together.

Every collaborator in this repo is optional by design — that is invariant 7, and it is what
makes ablation possible. The cost of that design is that *assembling* the system is real
work, and for a while six entry points each did it by hand: `paired_ab.py` built the full
stack, `run_lab.py` hand-rolled its own ladder, and `lab_eval.py` / `smoke_live.py` quietly
ran with no gate and no training record at all. Nothing was broken, but "what is a wired
SentinelPrime agent?" had six answers and they had begun to drift — `run_lab`'s private
ladder could not load a compiled program or explore, because it predated both.

So the wiring lives here, once, and the scripts pass flags. Two properties follow:

  1. **The default is the configuration the project claims.** Gate on, audit on, credit on,
     monitor on, proposals recorded. A thinner default is exactly how five of six call sites
     ended up under-wired — nobody chose that, it was just what the shortest constructor
     gave them.
  2. **Ablation stays a flag, not a rewrite.** Each mechanism is still individually
     switchable, so `--no-verifier` and friends remain one argument rather than a different
     assembly path that might differ in some second way nobody controlled for.

One thing is refused rather than silently degraded: `credit=True, audit=False`.
`CreditAssigner` reads an edit's targets off its `AuditRecord`, so without the audit log it
would run and never attribute anything. An ablation that reports a mechanism as *on* while
it does nothing is worse than a crash — it produces a null result that looks like evidence.

This is the seam a compiled program re-enters through: `program=` loads a GEPA-tuned judge
into the ladder, and every `AuditRecord` the resulting harness writes carries that prompt's
fingerprint. `describe()` renders the whole configuration in one line, fingerprint included,
so a run log says exactly which system produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sentinelprime.agent import PrimeAgent
from sentinelprime.audit import AuditLog, machinery_fingerprint
from sentinelprime.credit import CreditAssigner
from sentinelprime.harness import ContinualHarness
from sentinelprime.memory import JsonMemoryBackend
from sentinelprime.monitor import ProgressMonitor
from sentinelprime.proposals import ProposalLog
from sentinelprime.verifier import LadderVerifier, build_ladder

# gate= accepts these. "probe" is the hermetic rung set: deterministic, no LM call.
GATES = (None, "probe", "ladder")


@dataclass
class Assembly:
    """The wired system, with every part still reachable for inspection or ablation."""

    agent: PrimeAgent
    harness: ContinualHarness
    backend: JsonMemoryBackend
    audit_log: AuditLog | None
    proposal_log: ProposalLog | None
    verifier: LadderVerifier | None
    gate: str | None
    program: str | None

    def describe(self) -> str:
        """One line naming the configuration that produced a run.

        Includes the machinery fingerprint, because two runs of "the same" arm with
        different compiled judges are not the same experiment, and the arm name alone does
        not say which one this was.
        """
        def on(flag) -> str:
            return "on" if flag else "off"

        gate = self.gate or "none"
        if self.program:
            gate += f"({Path(self.program).name})"
        return (f"gate={gate} audit={on(self.audit_log)} "
                f"credit={on(self.harness.credit_assigner)} "
                f"monitor={on(self.agent.monitor)} record={on(self.proposal_log)} "
                f"explore={getattr(self.verifier, 'explore', 0.0):g} "
                f"machinery={machinery_fingerprint(self.harness)[:12] or 'none'}")


def assemble(*, lm, root: str | Path, sub_lm=None, gate: str | None = "ladder",
             program: str | None = None, explore: float = 0.0, seed: int = 0,
             audit: bool = True, credit: bool = True, credit_min_exposures: int = 3,
             credit_min_success_rate: float = 0.5, monitor: bool = True,
             record: bool = True, enable_children: bool = False,
             subquery_embedder=None) -> Assembly:
    """Build a wired agent under `root`, with each mechanism individually switchable.

    `root` holds the run's durable state: `ledger.json`, `audit.json`, `_proposals.jsonl`.
    Point two arms at two roots and they cannot contaminate each other.
    """
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    if credit and not audit:
        raise ValueError(
            "credit=True needs audit=True: CreditAssigner reads an edit's targets off its "
            "AuditRecord, so without the audit log it would observe outcomes and attribute "
            "none of them — a mechanism reported as on while doing nothing.")
    if program and gate != "ladder":
        raise ValueError(
            f"program= needs gate='ladder' (got {gate!r}); there is no judge to load a "
            f"compiled prompt into, and ignoring it would report a tuned run that used the "
            f"untuned admission bar.")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    backend = JsonMemoryBackend(str(root / "ledger.json"))
    audit_log = AuditLog(str(root / "audit.json")) if audit else None
    proposal_log = ProposalLog(str(root / "_proposals.jsonl")) if record else None
    verifier = None
    if gate is not None:
        verifier = build_ladder(program_path=program, explore=explore, seed=seed,
                                generative=(gate == "ladder"))

    harness = ContinualHarness(
        backend,
        audit_log=audit_log,
        verifier=verifier,
        credit_assigner=(CreditAssigner(audit_log,
                                        min_exposures=credit_min_exposures,
                                        min_success_rate=credit_min_success_rate)
                         if credit else None),
        proposal_log=proposal_log,
    )
    agent = PrimeAgent(harness, root_lm=lm, sub_lm=sub_lm,
                       monitor=ProgressMonitor() if monitor else None,
                       enable_children=enable_children,
                       subquery_embedder=subquery_embedder)
    return Assembly(agent=agent, harness=harness, backend=backend, audit_log=audit_log,
                    proposal_log=proposal_log, verifier=verifier, gate=gate,
                    program=program)
