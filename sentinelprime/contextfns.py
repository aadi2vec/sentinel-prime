"""Named context operations the model invokes by name and parameter, never by code.

CLM's third borrow is *model-defined context functions*: rather than issuing one edit at a
time, the model names a reusable operation over its own context. The obvious reading of that
is code synthesis — let the model write the function — and this module deliberately does not
do it. `interpreter.py` is an in-process REPL and explicitly not a sandbox, and §8 excludes
code synthesis from the mutable surface for the same reason `Policy.parse` rejects unknown
fields: a candidate that can supply the implementation can redefine what the operation
means, and then the operation is no longer the thing being measured.

So a context function here is a **parameterised template**: a fixed plan over the three
primitives in `context.py`, with the model supplying arguments. `Policy` has exactly this
shape one level up — it selects from a registry and binds a bounded parameter, and cannot
reach the implementation. The vocabulary is what a template adds, not the capability.

Two properties make this worth having rather than sugar:

  - **Every expansion is gated and priced individually.** A template is not a bypass. It
    produces primitive calls, each of which goes through `EditableContext._edit`, so the
    retention and re-prefill rungs see each one. A template that could evict in bulk without
    the ladder seeing it would be the admission gate's `credit=True, audit=False` failure:
    on, reported, and routed around.
  - **A composite is atomic, which the primitives are not.** A REPL cell holding three edit
    calls stops at the first refusal and the earlier two stand — ordinary Python semantics,
    and the model can see which took effect. One tool call is different: the model asked for
    one operation, and half of a fold is worse than none of it, because the note has been
    written and the turns it was summarising are still there. `ContextCheckpoint` from the
    reversibility layer is what makes the rollback possible, which is why that layer came
    first.

The honest limit: `drop_failures` matches a substring, so what counts as a failed turn is a
pattern the caller supplies and not a judgement. It will catch a turn that printed the word
"Error" while succeeding. That is a refusal-shaped error — the turn is still in
`full_trajectory()` and still in the eviction record — but it is a real imprecision and not
one the ladder can catch, because an edit that drops a successfully-handled error message
loses no figure and invalidates little.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sentinelprime.context import ContextEditError, EditableContext

# One planned primitive call: the method name on `EditableContext` and its arguments.
# Named rather than bound so a plan is inspectable before it runs — a plan that was a list
# of closures could not be printed back to the model or counted in a refusal message.
Step = tuple[str, tuple]


@dataclass(frozen=True)
class ContextFn:
    """A named plan over the context primitives, with the model supplying the parameters."""

    name: str
    doc: str
    plan: Callable[[EditableContext, dict], list[Step]]


def _drop_failures(ctx: EditableContext, args: dict) -> list[Step]:
    pattern = str(args["pattern"]).lower()
    return [("evict", (b.step, f"failed attempt matching {args['pattern']!r}"))
            for b in list(ctx.blocks) if pattern in b.shown.lower()]


def _drop_duplicates(ctx: EditableContext, args: dict) -> list[Step]:
    seen: set[str] = set()
    steps: list[Step] = []
    for block in list(ctx.blocks):
        key = block.shown.strip()
        if key and key in seen:
            steps.append(("evict", (block.step, "duplicate of an earlier turn")))
        seen.add(key)
    return steps


def _fold_range(ctx: EditableContext, args: dict) -> list[Step]:
    first, last = int(args["first"]) - 1, int(args["last"]) - 1
    if last < first:
        raise ContextEditError(
            f"fold_range needs last >= first; got first={first + 1} last={last + 1}")
    inside = [b.step for b in ctx.blocks if first <= b.step <= last]
    if not inside:
        shown = sorted(b.step + 1 for b in ctx.blocks)
        raise ContextEditError(
            f"no steps between {first + 1} and {last + 1}; steps present: {shown}")
    # The note lands on the *first* step of the range and the rest are evicted, rather than
    # the other way round. Two reasons, both about the prefix: the note replaces the
    # earliest-diverging block either way, so putting it there costs the same re-prefill as
    # putting it last, and leaving it first keeps the context in the order the run happened
    # in — a summary that appears after the turns it summarises reads as a later conclusion.
    return [("compact", (inside[0], args["note"]))] + [
        ("evict", (step, f"folded into step {inside[0] + 1}")) for step in inside[1:]]


REGISTRY: dict[str, ContextFn] = {
    "drop_failures": ContextFn(
        "drop_failures",
        "Evict every turn whose visible output contains `pattern`. For clearing a run of "
        "failed attempts in one call instead of one eviction at a time.",
        _drop_failures),
    "drop_duplicates": ContextFn(
        "drop_duplicates",
        "Evict every turn whose visible output repeats an earlier turn's exactly, keeping "
        "the first. Takes no parameters.",
        _drop_duplicates),
    "fold_range": ContextFn(
        "fold_range",
        "Replace steps `first`..`last` with a single note of your own on the first of them. "
        "For collapsing a finished line of work into its conclusion.",
        _fold_range),
}


def apply_context_fn(ctx: EditableContext, name: str, **args) -> str:
    """Run one named context function, all-or-nothing.

    Each planned primitive still passes the gate on its own; what the checkpoint adds is
    that a refusal part-way through leaves the context exactly as it was, so the model's one
    call has one outcome. The refusal it then sees is the rung's own reasoning about the
    specific edit that failed — not a generic "the template was refused", which would tell
    it nothing about what to do differently.
    """
    fn = REGISTRY.get(name)
    if fn is None:
        raise ContextEditError(
            f"no context function named {name!r}; available: {sorted(REGISTRY)}")
    steps = fn.plan(ctx, args)
    if not steps:
        # Not an error: "nothing matched" is a useful answer, and raising would make the
        # model treat a clean context as a failure and go looking for something to fix.
        return f"{name}: nothing matched; context unchanged"
    undo = ctx._snapshot(f"_fn:{name}")
    for i, (op, call_args) in enumerate(steps):
        try:
            getattr(ctx, op)(*call_args)
        except ContextEditError as exc:
            ctx.restore(undo, price=False)
            raise ContextEditError(
                f"{name} refused at edit {i + 1} of {len(steps)} "
                f"({op} step {call_args[0] + 1}): {exc}. Nothing was changed.") from exc
    return f"{name}: applied {len(steps)} edit(s)"


def context_fn_tools(ctx: EditableContext) -> list:
    """The templates as dspy tools, one per entry, bound to one run's context.

    One tool per function rather than a single `run_context_fn(name=..., **args)` dispatcher.
    The dispatcher is tidier and worse: dspy builds the tool surface the model reads from
    each function's own signature and docstring, so a dispatcher would present one tool whose
    real parameters are described only in prose — and a parameter the model has to infer from
    a docstring is one it will get wrong.
    """

    def drop_failures(pattern: str = "Traceback") -> str:
        """Evict every turn whose output contains `pattern`. Their text is recorded.

        Call with keyword arguments: drop_failures(pattern="Traceback").
        Use after a run of failed attempts, instead of evicting each one. Refused as a whole
        if any of the turns it would drop holds the last copy of a figure.
        """
        return apply_context_fn(ctx, "drop_failures", pattern=pattern)

    def drop_duplicates() -> str:
        """Evict turns whose output exactly repeats an earlier turn's, keeping the first.

        Call with no arguments: drop_duplicates().
        """
        return apply_context_fn(ctx, "drop_duplicates")

    def fold_range(first: int, last: int, note: str) -> str:
        """Collapse steps `first`..`last` into one note of your own.

        Call with keyword arguments: fold_range(first=2, last=6, note="reviewed all 8 MSAs").
        The note lands on step `first` and the rest are evicted. Carry any figures you still
        need into the note — the whole fold is refused if it would drop the last copy of one.
        """
        return apply_context_fn(ctx, "fold_range", first=first, last=last, note=note)

    return [drop_failures, drop_duplicates, fold_range]
