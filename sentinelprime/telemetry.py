"""Run telemetry: a tailable event log and token accounting.

Two jobs, both learned from running the first LAB sweeps blind:

  1. **A sweep you cannot watch is a sweep you debug afterwards.** The first warmup spent
     194s producing nothing, and the reason (the interpreter ignoring its workdir) was only
     visible after the fact by noticing a stray file in the repo root. Every meaningful
     step now emits an event as it happens: ledger edits, verifier verdicts, sub-query
     cache stats, replan decisions, judge verdicts. One JSON object per line, flushed
     immediately, so `tail -f` works and a killed run keeps everything up to the kill.

  2. **Cost was the largest unknown before committing budget.** `UsageMeter` aggregates
     DSPy's per-model token counts. It deliberately refuses to produce a dollar figure for
     a model it has no price for, and refuses to produce a *partial* total when any model
     is unpriced — a partial cost gets read as the whole bill.

Telemetry must never be the thing that kills a two-hour run, so event writing swallows
serialization problems rather than raising.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any


class RunLog:
    """Append-only JSONL event log, flushed per event, optionally echoed to stdout."""

    def __init__(self, path: str | Path, echo: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        # Judge calls and sub-agents run on threads; one lock keeps lines whole.
        self._lock = Lock()

    def event(self, kind: str, **fields: Any) -> None:
        record = {"ts": time.time(), "kind": kind, **fields}
        try:
            line = json.dumps(record, default=str)
        except Exception:
            line = json.dumps({"ts": record["ts"], "kind": kind,
                               "note": "unserializable payload dropped"})
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            if self.echo:
                print(self.render(record), flush=True)

    @staticmethod
    def render(record: dict) -> str:
        """A one-line human form. Keep it scannable — this is what a tail shows."""
        kind = record.get("kind", "?")
        skip = {"ts", "kind"}
        parts = []
        for key, value in record.items():
            if key in skip:
                continue
            text = str(value)
            if len(text) > 90:
                text = text[:87] + "..."
            parts.append(f"{key}={text}")
        stamp = time.strftime("%H:%M:%S", time.localtime(record.get("ts", time.time())))
        return f"[{stamp}] {kind:<18} " + " ".join(parts)


class UsageMeter:
    """Aggregate DSPy's per-model token usage, and price it only when we actually can."""

    def __init__(self, prices: dict[str, dict[str, float]] | None = None) -> None:
        # prices: {model: {"input": $/1M prompt tokens, "output": $/1M completion tokens}}
        self.prices = prices or {}
        self._by_model: dict[str, dict[str, int]] = {}
        self._lock = Lock()

    def add(self, usage: dict | None) -> None:
        """Accept a `Prediction.get_lm_usage()` mapping of {model: {token counts}}."""
        if not usage:
            return
        with self._lock:
            for model, counts in usage.items():
                if not isinstance(counts, dict):
                    continue
                slot = self._by_model.setdefault(
                    model, {"prompt_tokens": 0, "completion_tokens": 0})
                slot["prompt_tokens"] += int(counts.get("prompt_tokens") or 0)
                slot["completion_tokens"] += int(counts.get("completion_tokens") or 0)

    def by_model(self) -> dict[str, dict[str, int]]:
        return {m: dict(c) for m, c in self._by_model.items()}

    def totals(self) -> dict[str, int]:
        prompt = sum(c["prompt_tokens"] for c in self._by_model.values())
        completion = sum(c["completion_tokens"] for c in self._by_model.values())
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def unpriced(self) -> list[str]:
        return sorted(m for m in self._by_model if m not in self.prices)

    def cost(self) -> float | None:
        """Total USD, or None if any model in play has no price.

        None rather than a partial sum: a number that silently omits one model reads as
        the whole bill, and the whole point of measuring cost is to decide whether to
        spend more.
        """
        if not self._by_model or self.unpriced():
            return None
        total = 0.0
        for model, counts in self._by_model.items():
            price = self.prices[model]
            total += counts["prompt_tokens"] / 1e6 * price.get("input", 0.0)
            total += counts["completion_tokens"] / 1e6 * price.get("output", 0.0)
        return total

    def summary(self) -> str:
        t = self.totals()
        cost = self.cost()
        money = f" | ${cost:.2f}" if cost is not None else " | cost unknown (no price table)"
        return (f"{t['total_tokens']:,} tokens "
                f"({t['prompt_tokens']:,} in / {t['completion_tokens']:,} out)"
                f"{money}")


def load_prices(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Price table from JSON, if one exists. Absent file -> no prices, so cost is None."""
    path = Path(path or os.environ.get("LM_PRICES", "lm_prices.json"))
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
