"""SubQueryCache: content-hash dedup for RLM sub-LLM calls.

A distinct cache layer from the audit ledger. Inside a single RLM run,
`llm_query`/`llm_query_batched` re-hit the sub-LM for overlapping document
chunks with zero memory. This wraps those tools with an exact content-hash
cache and hit/miss counters, so a run can report a call/token reduction — the
measurement that later justifies a cost-based planner.

Scope: exact-match dedup only. A `near_dup` semantic hook is stubbed but a
no-op; pulling in an embedding dependency (and the correctness question of when
a near-match is safe to reuse) belongs with the auditable ReuseController, not
here.
"""
from __future__ import annotations

import hashlib
from typing import Callable


def _key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class SubQueryCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._hits = 0
        self._misses = 0
        self._calls = 0

    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "calls": self._calls}

    def _near_dup(self, prompt: str) -> str | None:
        # Placeholder for embedding-based near-duplicate reuse. Intentionally a
        # no-op: safe near-match reuse needs the ReuseController's scope/currency
        # gates, not a bare cosine threshold. TODO: wire once that lands.
        return None

    def wrap(self, tools: dict[str, Callable]) -> dict[str, Callable]:
        """Return a tools dict with llm_query/llm_query_batched cache-wrapped.

        Any other entries pass through untouched.
        """
        wrapped = dict(tools)
        if "llm_query" in tools:
            wrapped["llm_query"] = self._wrap_single(tools["llm_query"])
        if "llm_query_batched" in tools:
            wrapped["llm_query_batched"] = self._wrap_batched(tools["llm_query_batched"])
        return wrapped

    def _wrap_single(self, fn: Callable[[str], str]) -> Callable[[str], str]:
        def llm_query(prompt: str) -> str:
            self._calls += 1
            key = _key(prompt)
            if key in self._store:
                self._hits += 1
                return self._store[key]
            self._misses += 1
            result = fn(prompt)
            self._store[key] = result
            return result

        return llm_query

    def _wrap_batched(self, fn: Callable[[list], list]) -> Callable[[list], list]:
        def llm_query_batched(prompts: list) -> list:
            self._calls += len(prompts)
            results: dict[int, str] = {}
            missing: dict[str, int] = {}   # key -> first index needing an underlying call
            to_query: list[str] = []

            for i, prompt in enumerate(prompts):
                key = _key(prompt)
                if key in self._store:
                    self._hits += 1
                    results[i] = self._store[key]
                elif key in missing:
                    # duplicate within this batch — count as a hit, fill after fetch
                    self._hits += 1
                else:
                    self._misses += 1
                    missing[key] = i
                    to_query.append(prompt)

            if to_query:
                fetched = fn(to_query)
                for prompt, value in zip(to_query, fetched):
                    self._store[_key(prompt)] = value

            # Fill every position (including in-batch dups) from the store.
            for i, prompt in enumerate(prompts):
                if i not in results:
                    results[i] = self._store[_key(prompt)]
            return [results[i] for i in range(len(prompts))]

        return llm_query_batched
