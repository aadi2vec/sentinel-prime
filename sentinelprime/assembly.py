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

Two things are refused rather than silently degraded, for the same reason: an ablation that
reports a mechanism as *on* while it does nothing is worse than a crash, because it produces
a null result that looks like evidence.

  - `credit=True, audit=False` — `CreditAssigner` reads an edit's targets off its
    `AuditRecord`, so without the audit log it would run and attribute nothing.
  - `error_budget=True, audit=False` — the freeze is recorded as an audit event, and an
    unrecorded freeze is a silent outage: the run reads as having had nothing to learn
    rather than as one that was halted.

`editable_context` is the one mechanism here that defaults *off*, and not for lack of
confidence in it: it changes what the prompt is mid-run, which is the prefix determinism
invariant 5 protects, so flipping the default would silently reclassify every existing run.
It is wired, reachable, and fingerprinted like everything else — just opt-in.

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
from sentinelprime.contextgate import build_context_ladder
from sentinelprime.credit import CreditAssigner
from sentinelprime.errorbudget import ErrorBudget, online_objectives
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
    error_budget: ErrorBudget | None
    gate: str | None
    program: str | None
    # The write path's ladder, held here for the same reason `verifier` is: a rung set is
    # inspectable state, and an arm whose context gate had two rungs rather than three is a
    # different arm.
    context_verifier: LadderVerifier | None = None

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
        # Three states, not two: a run that was *halted* and a run that was free are not
        # the same experiment, and `budget=on` alone does not say which this was.
        budget = "off" if self.error_budget is None else (
            "frozen" if self.error_budget.frozen else "on")
        # Three states again, and for the ablation the thesis turns on: a write path running
        # *ungated* is the CLM baseline this repo argues against, and `ctx=on` alone cannot
        # tell it apart from the gated arm.
        ctx = "off" if self.agent.context is None else (
            "gated" if self.context_verifier else "open")
        return (f"gate={gate} audit={on(self.audit_log)} "
                f"credit={on(self.harness.credit_assigner)} "
                f"monitor={on(self.agent.monitor)} record={on(self.proposal_log)} "
                f"budget={budget} ctx={ctx} "
                f"explore={getattr(self.verifier, 'explore', 0.0):g} "
                f"machinery={machinery_fingerprint(self.harness)[:12] or 'none'}")


def assemble(*, lm, root: str | Path, sub_lm=None, gate: str | None = "ladder",
             program: str | None = None, explore: float = 0.0, seed: int = 0,
             audit: bool = True, credit: bool = True, credit_min_exposures: int = 3,
             credit_min_success_rate: float = 0.5, monitor: bool = True,
             record: bool = True, error_budget: bool = True,
             editable_context: bool = True, context_gate: bool | None = None,
             enable_children: bool = False, subquery_embedder=None) -> Assembly:
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
    if error_budget and not audit:
        raise ValueError(
            "error_budget=True needs audit=True: the freeze is recorded as an audit event, "
            "and a freeze nobody records is a silent outage — the run reads as having had "
            "nothing to learn rather than as one that was halted.")
    if program and gate != "ladder":
        raise ValueError(
            f"program= needs gate='ladder' (got {gate!r}); there is no judge to load a "
            f"compiled prompt into, and ignoring it would report a tuned run that used the "
            f"untuned admission bar.")
    if context_gate and not editable_context:
        raise ValueError(
            "context_gate=True needs editable_context=True: there are no context edits to "
            "admit or refuse, so the gate would be built, reported as on by describe(), and "
            "never consulted — a mechanism reporting as on while doing nothing.")
    # Tri-state deliberately. `None` means "whatever the write path is", so switching the
    # write path off stays one flag; only an explicit `True` alongside it is the
    # contradiction refused above, and an explicit `False` is the real ablation — the write
    # path running ungated, which is the CLM baseline this repo's thesis is an argument
    # against and therefore has to be expressible.
    if context_gate is None:
        context_gate = editable_context

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    backend = JsonMemoryBackend(str(root / "ledger.json"))
    audit_log = AuditLog(str(root / "audit.json")) if audit else None
    proposal_log = ProposalLog(str(root / "_proposals.jsonl")) if record else None
    verifier = None
    if gate is not None:
        verifier = build_ladder(program_path=program, explore=explore, seed=seed,
                                generative=(gate == "ladder"))
    # The write path's own ladder. Separate object from the ledger's, same class and same
    # planner: the two gates see different ops at different times (one mid-task against the
    # task statement, one post-grading against rubric failures), so sharing one instance
    # would pool two different P(fail) distributions into one ordering. `gate` still decides
    # whether the generative rung is present, because a no-network run is a no-network run
    # on both paths.
    ctx_ladder = build_context_ladder(generative=(gate == "ladder"), explore=explore,
                                      seed=seed) if context_gate else None

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
    # On by default, and only with the objectives the live loop actually feeds: a
    # registered objective nobody observes never reaches min_observations, so it never
    # breaches, and the budget then reports itself as guarding a metric it is structurally
    # blind to. That is the `credit=True, audit=False` failure one level up. §5's primary
    # metric needs a graded corpus, which `assemble()` has none of — see
    # `errorbudget.check_set_objectives` for the replay caller.
    budget = (ErrorBudget(online_objectives(), audit_log=audit_log)
              if error_budget else None)
    agent = PrimeAgent(harness, root_lm=lm, sub_lm=sub_lm,
                       monitor=ProgressMonitor() if monitor else None,
                       error_budget=budget,
                       # On by default: the write path is part of the configuration the
                       # project claims, so it is not an opt-in extra. It does break
                       # invariant 5 by construction, which is why every edit is priced
                       # (`context.reprefill_chars`) and why `describe()` prints `ctx` —
                       # runs with it on and off are different experiments, and the run log
                       # has to say which this was rather than leaving it to be inferred.
                       editable_context=editable_context,
                       context_gate=ctx_ladder,
                       enable_children=enable_children,
                       subquery_embedder=subquery_embedder)
    return Assembly(agent=agent, harness=harness, backend=backend, audit_log=audit_log,
                    proposal_log=proposal_log, verifier=verifier, error_budget=budget,
                    gate=gate, program=program, context_verifier=ctx_ladder)
