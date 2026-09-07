"""SubQueryCache: dedup for RLM sub-LLM calls — exact by default, semantic on request.

A distinct cache layer from the audit ledger. Inside a single RLM run,
`llm_query`/`llm_query_batched` re-hit the sub-LM for overlapping document chunks with
zero memory. This wraps those tools with a cache and hit/miss counters so a run can report
a call/token reduction — the measurement that justifies the cost-based planner.

Two tiers:

  1. Exact content-hash dedup (always on, dependency-free): identical prompts collapse.
  2. Semantic near-duplicate reuse (opt-in): pass an ``embedder`` and prompts whose
     embeddings are within ``similarity_threshold`` cosine reuse a prior answer. This is
     bounded to a *single run's* sub-queries, where near-duplicate chunk questions are the
     dominant waste and the blast radius is one trajectory — so a cosine threshold is a
     reasonable intra-run heuristic. It is deliberately NOT the mechanism for *cross-task*
     reuse: that carries compliance risk (a near-match can be wrong, and the world moves),
     which is exactly what the auditable ReuseController's scope/currency/verifier gates
     exist to adjudicate. Keep the two separate.
"""
from __future__ import annotations

import hashlib
import math
from typing import Callable


def _key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SubQueryCache:
    def __init__(self, embedder: Callable[[str], list[float]] | None = None,
                 similarity_threshold: float = 0.9) -> None:
        self._store: dict[str, str] = {}
        # (embedding, value) pairs for semantic near-duplicate lookup; only populated
        # when an embedder is configured.
        self._embeds: list[tuple[list[float], str]] = []
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self._hits = 0
        self._misses = 0
        self._calls = 0
        self._semantic_hits = 0

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "calls": self._calls,
            "semantic_hits": self._semantic_hits,
        }

    def _get_cached(self, prompt: str) -> tuple[str | None, str | None]:
        # Returns (kind, value): kind is "exact", "semantic", or None. Exact is tried
        # first (cheapest and unambiguous); semantic only when an embedder is configured.
        key = _key(prompt)
        if key in self._store:
            return "exact", self._store[key]
        if self.embedder is not None and self._embeds:
            vec = self.embedder(prompt)
            best_val, best_sim = None, 0.0
            for stored_vec, value in self._embeds:
                sim = _cosine(vec, stored_vec)
                if sim > best_sim:
                    best_val, best_sim = value, sim
            if best_val is not None and best_sim >= self.similarity_threshold:
                return "semantic", best_val
        return None, None

    def _put(self, prompt: str, value: str) -> None:
        self._store[_key(prompt)] = value
        if self.embedder is not None:
            self._embeds.append((self.embedder(prompt), value))

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
            kind, value = self._get_cached(prompt)
            if kind is not None:
                self._hits += 1
                if kind == "semantic":
                    self._semantic_hits += 1
                return value
            self._misses += 1
            result = fn(prompt)
            self._put(prompt, result)
            return result

        return llm_query

    def _wrap_batched(self, fn: Callable[[list], list]) -> Callable[[list], list]:
        def llm_query_batched(prompts: list) -> list:
            self._calls += len(prompts)
            results: dict[int, str] = {}
            pending_key: dict[str, int] = {}  # exact key -> first index awaiting a fetch
            to_query: list[str] = []

            for i, prompt in enumerate(prompts):
                kind, value = self._get_cached(prompt)
                if kind is not None:
                    self._hits += 1
                    if kind == "semantic":
                        self._semantic_hits += 1
                    results[i] = value
                    continue
                key = _key(prompt)
                if key in pending_key:
                    # exact duplicate within this batch — count as a hit, fill after fetch
                    self._hits += 1
                else:
                    self._misses += 1
                    pending_key[key] = i
                    to_query.append(prompt)

            if to_query:
                fetched = fn(to_query)
                for prompt, value in zip(to_query, fetched):
                    self._put(prompt, value)

            # Fill every remaining position (in-batch exact dups) from the store.
            for i, prompt in enumerate(prompts):
                if i not in results:
                    results[i] = self._store[_key(prompt)]
            return [results[i] for i in range(len(prompts))]

        return llm_query_batched
