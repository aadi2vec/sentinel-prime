"""An editable live context: the write path, as a mechanism the model drives.

`monitor.py` already names the failure this exists to address — the live context "appends to
a monotone history with no progress signal and no backtrack, so a wrong strategy gets
*elaborated* until max_iters". `dspy.RLM`'s `REPLHistory` is append-only by construction
(frozen pydantic, `append()` returns a new instance), so a model that has gone down the wrong
path cannot clear the evidence of it; it can only add more. This module gives the model three
in-place operations over its own history — compact a turn, evict a turn, keep a sketchpad —
and accounts for what each one costs.

**Two views of the same run, and conflating them is a credit-assignment bug.** This is the
repo's two-logs split (`AuditLog` vs `ProposalLog`) one level down:

  - `render()` is the **prompt view**: what the model sees next turn, after its edits.
  - `full_trajectory()` is the **grounding view**: every turn that ever executed, evicted or
    not, in the order it ran.

`credit.grounding_from_trajectory` reads only REPL `output`. Hand it the prompt view and an
evicted turn takes its evidence with it, so a criterion the run genuinely derived reads as
ungrounded and `credit.observe` silently drops the credit — a lesson is then retired for a
failure that did not happen. The grounding view is append-only for exactly that reason, and
`EvictionRecord` keeps the dropped text rather than a count of it.

**The floor is structural, not a rule.** §3a's third constraint wants the task statement and
guidance block non-evictable, the live-context analogue of invariant 1. They are — not
because eviction checks for them, but because they are never addressable: `task` and
`guidance` are `PrimeTask` signature fields rendered by the adapter, and the only thing an
edit operation can name is a history step. A check could be removed by a later refactor; an
operation with no way to express the illegal edit cannot. Individual turns can also be
`pin`ned, which is a weaker, checked guarantee for evidence the caller knows it needs.

**Every edit is priced, because the obvious win is a cache miss.** §3a's first constraint:
an in-place edit is exactly what invalidates the prompt prefix that `read()`'s sort order and
`PrimeTask`'s field order exist to protect. Compacting a 10k-character turn looks like a
saving and is one *only* if the turns before it are untouched; edit step 1 of 9 and every
subsequent byte is re-sent at full price. `reprefill_chars` measures that directly, by
rendering before and after and finding where the two diverge — the same longest-common-prefix
measurement `agent.cacheable_prefix` makes across tasks, applied within one run.

Honest limits. The accounting is in **characters, not tokens or dollars** — the same limit
the policy `Budget` carries, and for the same reason: a provider-independent unit that no
price table has to be trusted for. It states where the prefix broke, not what the break cost.
And an edit's *benefit* is not measured here at all; only its price is.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from sentinelprime.contextgate import TaskStatement


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


@dataclass
class ContextBlock:
    """One executed turn, as the prompt may show it.

    `step` is assigned at append and **never renumbered**, including after an eviction.
    Renumbering would make the model's own addressing unstable — it evicts step 3, and the
    step it meant to compact next silently becomes a different turn — which is the editable
    context equivalent of the unstable read() ordering invariant 5 forbids.
    """

    step: int
    reasoning: str
    code: str
    output: str
    pinned: bool = False
    # Set by compact(): what the prompt shows instead of `output`. None -> show `output`.
    # The original is kept rather than overwritten, so the grounding view is unaffected by
    # a compaction and `full_trajectory` does not need to reconstruct anything.
    note: str | None = None

    @property
    def compacted(self) -> bool:
        return self.note is not None

    @property
    def shown(self) -> str:
        """What the prompt currently carries for this turn — the note once compacted.

        The retention rung has to ask what an edit would *remove from the prompt*, which
        after a compaction is the note, not the original output. Grounding reads `output`;
        admission reads this. Same two views, one field apart.
        """
        return self.output if self.note is None else self.note

    def render(self) -> str:
        shown = self.shown
        reasoning = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        label = " (compacted by you)" if self.compacted else ""
        return (f"=== Step {self.step + 1}{label} ===\n{reasoning}"
                f"Code:\n```python\n{self.code}\n```\n"
                f"Output ({len(shown):,} chars):\n{shown}")


@dataclass(frozen=True)
class EvictionRecord:
    """What an eviction dropped, kept in full.

    A count would be cheaper and useless: the point is that `grounding_from_trajectory` can
    still read the `output` of a turn the prompt no longer shows.
    """

    step: int
    reason: str
    reasoning: str
    code: str
    output: str
    evicted_at: str


@dataclass(frozen=True)
class ReprefillEvent:
    """One in-place edit, and the prompt prefix it invalidated.

    `chars` is measured from the *rendered* context, so it counts the real consequence of the
    edit rather than the size of the thing edited. Editing the last turn of nine is nearly
    free; editing the first invalidates everything after it.
    """

    op: str
    step: int
    chars: int          # rendered characters after the divergence point — re-sent at full price
    before_chars: int   # size of the rendered context before the edit
    after_chars: int
    at: str

    @property
    def fraction(self) -> float | None:
        """Share of the pre-edit context that was invalidated. None when it was empty."""
        return self.chars / self.before_chars if self.before_chars else None


class ContextEditError(Exception):
    """A refused edit. Raised into the REPL so the model sees why and can choose again.

    Raising rather than returning an error string is what makes a cell of several edits stop
    at the first bad one: ordinary Python semantics, which is what the model already expects,
    and the message names the offending call. Edits before the failure stand — the partial
    application is visible in `reprefills`, and swallowing the error instead would leave the
    model unable to tell which of its edits took effect.
    """


@dataclass(frozen=True)
class ContextCheckpoint:
    """A restore point for the **prompt view only**.

    Deliberately not a snapshot of the whole object. `executed` and `reprefills` are not in
    here and are never restored, which is invariant 4's reasoning one level down: a turn that
    ran is a fact, and characters that were re-sent were really billed. Rewinding undoes what
    the model will *see*, not what the run *did* — restoring the grounding view would let a
    rewind launder a turn out of existence, and restoring the price ledger would make
    backtracking look free.
    """

    name: str
    blocks: tuple[ContextBlock, ...]
    evictions: tuple[EvictionRecord, ...]
    sketchpad: str
    next_step: int
    at: str


@dataclass
class EditableContext:
    """The model's own history, editable in place, with every edit recorded and priced.

    Not thread-safe, matching the rest of `PrimeAgent`: one RLM loop, sequential turns.
    """

    blocks: list[ContextBlock] = field(default_factory=list)
    evictions: list[EvictionRecord] = field(default_factory=list)
    reprefills: list[ReprefillEvent] = field(default_factory=list)
    # Append-only record of every turn that ever executed, in execution order. This is the
    # grounding view and nothing removes from it.
    executed: list[dict] = field(default_factory=list)
    sketchpad: str = ""
    # Named restore points the model can rewind to. `monitor.py` names "no backtrack" as
    # half the thrash failure; eviction alone does not fix that, because a model that has
    # gone down the wrong path needs to *undo* its own context surgery, not add more of it.
    checkpoints: dict[str, ContextCheckpoint] = field(default_factory=dict)
    # Optional admission gate over the model's own edits, shaped exactly like the ledger's:
    # anything exposing `verify(op, feedback, trajectory) -> VerifierVerdict`. None -> every
    # well-formed edit applies, which is the behaviour before the gate existed (invariant 7).
    gate: object | None = None
    # What the task is, for the gate to ground an edit in. Set per task by the agent.
    task: str = ""
    # Refused edits, with the ladder's reasoning. The analogue of `RefineResult.rejected`:
    # a refused context edit leaves no reprefill event, so this is the only record of it.
    refusals: list[dict] = field(default_factory=list)
    _next_step: int = 0

    # --- what the harness writes -----------------------------------------------------

    def reset(self) -> None:
        """Clear everything for a new task, in place.

        In place rather than by replacement because the model's edit tools are bound to one
        instance at RLM construction, and rebinding them per task is how a tool ends up
        editing the previous task's context. Invariant 6's analogue: a run starts from a
        clean context, and nothing carries across the task boundary.
        """
        self.blocks.clear()
        self.evictions.clear()
        self.reprefills.clear()
        self.executed.clear()
        self.checkpoints.clear()
        self.refusals.clear()
        self.sketchpad = ""
        self.task = ""
        self._next_step = 0

    def append(self, reasoning: str, code: str, output: str) -> ContextBlock:
        """Record an executed turn. Pure append — this is not an edit and is never priced."""
        block = ContextBlock(step=self._next_step, reasoning=reasoning, code=code,
                             output=output)
        self._next_step += 1
        self.blocks.append(block)
        self.executed.append({"reasoning": reasoning, "code": code, "output": output})
        return block

    def pin(self, step: int) -> None:
        """Make one turn non-evictable. A checked guarantee, weaker than the structural one."""
        self._block(step).pinned = True

    # --- reversibility: invariant 2 for the write path --------------------------------

    def checkpoint(self, name: str) -> ContextCheckpoint:
        """Take a named restore point the model can `rewind` to."""
        cp = self._snapshot(name)
        self.checkpoints[name] = cp
        return cp

    def _snapshot(self, name: str) -> ContextCheckpoint:
        """A restore point that is *not* registered under a name.

        The gate needs one per edit so it can undo a refusal, and those must not appear in
        the model's checkpoint namespace: a rewind target it never marked is a tool whose
        behaviour depends on how many edits the gate happened to refuse.
        """
        return ContextCheckpoint(
            name=name,
            # Copies, not references: `compact()` mutates a block in place, so storing the
            # live objects would make a checkpoint silently track the edits it exists to
            # undo — a restore point that restores to wherever the run has got to.
            blocks=tuple(replace(b) for b in self.blocks),
            evictions=tuple(self.evictions),
            sketchpad=self.sketchpad,
            next_step=self._next_step,
            at=_now())

    def restore(self, cp: ContextCheckpoint, *, price: bool = True) -> None:
        """Put the prompt view back. `executed` and `reprefills` are untouched by design."""
        before = self.render()
        self.blocks[:] = [replace(b) for b in cp.blocks]
        self.evictions[:] = list(cp.evictions)
        self.sketchpad = cp.sketchpad
        # `_next_step` is restored so step numbers stay dense, but only ever downward: a
        # rewind followed by new turns must not reuse a number the run already issued, or
        # the model's addressing stops identifying a turn uniquely. Turns appended after
        # the checkpoint keep their numbers spent.
        self._next_step = max(self._next_step, cp.next_step)
        if price:
            self._price("rewind", -1, before)

    def rewind(self, name: str) -> str:
        """Undo every context edit and turn since the named checkpoint."""
        cp = self.checkpoints.get(name)
        if cp is None:
            raise ContextEditError(
                f"no checkpoint named {name!r}; marked: {sorted(self.checkpoints)}")
        dropped = len(self.blocks) - len(cp.blocks)
        self.restore(cp)
        return (f"rewound to {name!r}: prompt view back to {len(cp.blocks)} step(s) "
                f"({dropped:+d}). Nothing you actually ran was forgotten.")

    # --- what the model drives -------------------------------------------------------

    def evict(self, step: int, reason: str = "") -> str:
        """Drop a turn from the prompt view, keeping its text in the eviction record."""
        block = self._block(step)
        if block.pinned:
            raise ContextEditError(
                f"step {step + 1} is pinned and cannot be evicted; compact it instead")

        def apply() -> None:
            self.blocks.remove(block)
            self.evictions.append(EvictionRecord(
                step=block.step, reason=reason, reasoning=block.reasoning,
                code=block.code,
                # The original output, never the compacted note: a turn that was compacted
                # and then evicted would otherwise lose its evidence in two steps.
                output=block.output, evicted_at=_now()))

        self._edit("evict", block.step, apply,
                   op={"op": "evict", "step": block.step + 1, "reason": reason,
                       "text": block.shown, "code": block.code})
        return f"evicted step {step + 1} ({len(block.output):,} chars); recorded"

    def compact(self, step: int, note: str) -> str:
        """Replace a turn's output with a shorter note, keeping the original for grounding."""
        if not note.strip():
            raise ContextEditError(
                "a compaction needs a non-empty note; to remove a turn entirely use evict")
        block = self._block(step)
        dropped = block.shown
        self._edit("compact", block.step, lambda: setattr(block, "note", note),
                   op={"op": "compact", "step": block.step + 1, "note": note,
                       "text": dropped, "code": block.code})
        return f"compacted step {step + 1} to {len(note):,} chars"

    def write_sketchpad(self, text: str) -> str:
        """Set the scratch block that survives every turn, replacing its prior contents.

        Deliberately replace, not append: a scratchpad that only grows is the monotone
        history this module exists to escape, one level in.

        Gated like the other two, and for a reason that is not obvious: replacing the
        sketchpad is the one operation that can drop a figure the model had already chosen
        to carry forward, which is precisely the evidence loss the retention rung exists to
        refuse.
        """
        self._edit("sketchpad", -1, lambda: setattr(self, "sketchpad", text),
                   op={"op": "sketchpad", "step": 0, "text": self.sketchpad,
                       "note": text, "code": ""})
        return f"sketchpad now {len(text):,} chars"

    # --- the two views ---------------------------------------------------------------

    def render(self) -> str:
        """The prompt view: what the model sees next turn.

        Ordered most-stable-first, which is the whole prefix-cache argument and is not
        obvious: the tempting layout puts the sketchpad at the top where the model will
        read it, but the sketchpad is the element rewritten most often, and anything above
        the blocks invalidates all of them every time it changes. Measured on a three-turn
        run, leading with the sketchpad charged 771 re-prefill characters for a 9-character
        note — 99% of the prefix — which would have made every editable-context win an
        accounting artifact. Appending instead costs the sketchpad's own length, and so
        does a new turn.
        """
        parts = [block.render() for block in self.blocks]
        if self.evictions:
            # The model is told what it dropped. An eviction it cannot see is one it will
            # redo the work for. After the blocks: this list grows, and a growing header
            # would re-charge for every surviving block each time.
            dropped = ", ".join(f"step {r.step + 1}" for r in self.evictions)
            parts.append(f"=== Evicted by you ===\n{dropped} "
                         f"(recorded; ask again if you need the detail)")
        if self.sketchpad:
            parts.append(f"=== Sketchpad (yours, editable) ===\n{self.sketchpad}")
        if not parts:
            return "You have not interacted with the REPL environment yet."
        return "\n".join(parts)

    def full_trajectory(self) -> list[dict]:
        """The grounding view: every turn that executed, evicted or not, in order."""
        return [dict(row) for row in self.executed]

    # --- accounting ------------------------------------------------------------------

    @property
    def reprefill_chars(self) -> int:
        """Total rendered characters invalidated by edits over the run."""
        return sum(event.chars for event in self.reprefills)

    def stats(self) -> dict:
        return {
            "turns": len(self.executed),
            "blocks_shown": len(self.blocks),
            "evicted": len(self.evictions),
            "compacted": sum(1 for b in self.blocks if b.compacted),
            "edits": len(self.reprefills),
            "reprefill_chars": self.reprefill_chars,
            "rendered_chars": len(self.render()),
            "sketchpad_chars": len(self.sketchpad),
            # Reported alongside the admitted edits for the same reason `RefineResult`
            # carries `rejected`: a run where the gate refused half the edits and one where
            # it refused none are different runs, and `edits` alone cannot tell them apart.
            "refused": len(self.refusals),
            "checkpoints": len(self.checkpoints),
        }

    def describe(self) -> str:
        s = self.stats()
        return (f"turns={s['turns']} shown={s['blocks_shown']} evicted={s['evicted']} "
                f"compacted={s['compacted']} edits={s['edits']} "
                f"refused={s['refused']} reprefill={s['reprefill_chars']}c")

    # --- internals -------------------------------------------------------------------

    def _block(self, step: int) -> ContextBlock:
        for block in self.blocks:
            if block.step == step:
                return block
        shown = sorted(b.step + 1 for b in self.blocks)
        raise ContextEditError(
            f"no step {step + 1} in your context; steps present: {shown}")

    def _edit(self, kind: str, step: int, apply, op: dict) -> None:
        """Apply one edit through the gate, pricing it only if it survives.

        The edit is judged on the context it would actually produce — applied, rendered,
        then undone if refused. Asking a rung to *simulate* the edit instead would give the
        gate a second rendering path, and a gate judging a layout the prompt never sees is
        the `credit=True, audit=False` failure in miniature: it reports on something other
        than the thing that shipped.

        A refusal is not priced. Nothing was sent to the provider, so no prefix was
        re-billed, and charging for it would make the gate look like it costs cache.
        """
        before = self.render()
        if self.gate is None:
            apply()
            self._price(kind, step, before)
            return
        undo = self._snapshot("_gate")
        apply()
        verdict = self.gate.verify({**op, "before": before, "after": self.render()},
                                   TaskStatement(self.task), self.full_trajectory())
        if not verdict.admitted:
            self.restore(undo, price=False)
            # The only record a refused edit leaves. `reprefills` has no entry for it by
            # construction, which is the same reason `ProposalLog` exists one level up:
            # the admitted set alone is an all-positive dataset.
            self.refusals.append({"kind": kind, "step": step + 1, "score": verdict.score,
                                  "why": verdict.justification, "at": _now()})
            raise ContextEditError(verdict.justification)
        self._price(kind, step, before)

    def _price(self, op: str, step: int, before: str) -> None:
        after = self.render()
        self.reprefills.append(ReprefillEvent(
            op=op, step=step,
            # Everything after the divergence point is re-sent at full price. Measured on
            # the rendered text because that is what the provider actually bills a prefix on.
            chars=len(before) - _common_prefix_len(before, after),
            before_chars=len(before), after_chars=len(after), at=_now()))


def context_tools(context: EditableContext) -> list:
    """The edit operations, as dspy tools bound to one run's context.

    Returned as functions rather than methods so the docstrings the model reads describe the
    *operation* rather than the class, and so nothing in the tool surface can address
    anything but a history step — which is what makes the task statement and guidance block
    structurally non-evictable rather than non-evictable by a check.

    dspy re-signs each tool with the wrapped function's own signature, so in the REPL these
    are called by name with keyword arguments: `evict_step(step=3)`.

    Three destructive edits and two reversibility operations. The pair is deliberate and is
    the same lesson `spawn_child`/`collect_child` taught: an edit tool without an undo is a
    tool the model cannot afford to use on anything it is unsure about, which is exactly the
    case where editing would help most.
    """

    def evict_step(step: int, reason: str = "") -> str:
        """Remove step N from your context. Its text is recorded, not lost.

        Call with keyword arguments: evict_step(step=3, reason="wrong approach").
        Use this when a turn's output is no longer worth carrying — a failed approach, a
        large dump you have already summarised. `step` is the number shown in the header.
        """
        return context.evict(step - 1, reason)

    def compact_step(step: int, note: str) -> str:
        """Replace step N's output with a short note of your own.

        Call with keyword arguments: compact_step(step=2, note="found 3 CoC clauses").
        Prefer this over evict_step when the turn's *conclusion* still matters but its bulk
        does not.
        """
        return context.compact(step - 1, note)

    def write_sketchpad(text: str) -> str:
        """Replace your sketchpad — a scratch block that survives every turn.

        Call with keyword arguments: write_sketchpad(text="...").
        Use it for running state: what you have established, what is left. It replaces the
        previous contents rather than adding to them, so keep it current rather than long.
        """
        return context.write_sketchpad(text)

    def mark_context(name: str) -> str:
        """Mark the current context so you can return to it later.

        Call with keyword arguments: mark_context(name="before-indemnities").
        Mark before you start an approach you may want to abandon. Marking is free — it
        changes nothing about what you see, so it never costs you a re-send.
        """
        cp = context.checkpoint(name)
        return (f"marked {name!r} at {len(cp.blocks)} step(s); "
                f"rewind_context(name={name!r}) returns here")

    def rewind_context(name: str) -> str:
        """Return your context to a mark, undoing every edit and turn since.

        Call with keyword arguments: rewind_context(name="before-indemnities").
        Use this when an approach did not work out: rather than evicting its turns one by
        one, put the context back where it was. Everything you actually ran is still
        recorded — this changes what you see, not what happened.
        """
        return context.rewind(name)

    return [evict_step, compact_step, write_sketchpad, mark_context, rewind_context]
