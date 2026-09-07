from sentinelprime.subcache import SubQueryCache


def test_wraps_and_caches_llm_query():
    calls = []

    def llm_query(prompt):
        calls.append(prompt)
        return f"ans:{prompt}"

    cache = SubQueryCache()
    wrapped = cache.wrap({"llm_query": llm_query})
    q = wrapped["llm_query"]

    assert q("hello") == "ans:hello"
    assert q("hello") == "ans:hello"   # served from cache
    assert calls == ["hello"]          # underlying called once
    assert cache.stats() == {"hits": 1, "misses": 1, "calls": 2, "semantic_hits": 0}


def test_distinct_prompts_miss():
    def llm_query(prompt):
        return f"ans:{prompt}"

    cache = SubQueryCache()
    q = cache.wrap({"llm_query": llm_query})["llm_query"]
    q("a")
    q("b")
    assert cache.stats() == {"hits": 0, "misses": 2, "calls": 2, "semantic_hits": 0}


def test_batched_dedups_within_and_across_calls():
    underlying = []

    def llm_query_batched(prompts):
        underlying.extend(prompts)
        return [f"ans:{p}" for p in prompts]

    cache = SubQueryCache()
    qb = cache.wrap({"llm_query_batched": llm_query_batched})["llm_query_batched"]

    # repeated prompt within the batch collapses to one underlying query
    out = qb(["x", "y", "x"])
    assert out == ["ans:x", "ans:y", "ans:x"]
    assert sorted(underlying) == ["x", "y"]

    # fully-cached batch makes no underlying call
    underlying.clear()
    out2 = qb(["x", "y"])
    assert out2 == ["ans:x", "ans:y"]
    assert underlying == []
    assert cache.stats()["hits"] >= 2


def test_wrap_preserves_unknown_tools():
    cache = SubQueryCache()
    marker = object()
    wrapped = cache.wrap({"llm_query": lambda p: p, "other": marker})
    assert wrapped["other"] is marker


def _coc_embedder(text):
    # Toy 2-D embedder: near-synonymous prompts about the same clause map to the
    # same direction, orthogonal topics map to an orthogonal axis.
    return [1.0, 0.0] if "change-of-control" in text else [0.0, 1.0]


def test_semantic_hit_reuses_near_duplicate():
    calls = []

    def llm_query(prompt):
        calls.append(prompt)
        return f"ans:{prompt}"

    cache = SubQueryCache(embedder=_coc_embedder, similarity_threshold=0.9)
    q = cache.wrap({"llm_query": llm_query})["llm_query"]

    a = q("find the change-of-control clause")
    b = q("locate the change-of-control terms")   # different string, same meaning
    assert a == "ans:find the change-of-control clause"
    assert b == a                                  # served from the near-duplicate
    assert calls == ["find the change-of-control clause"]  # underlying called once
    stats = cache.stats()
    assert stats["semantic_hits"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_dissimilar_prompts_are_not_semantic_hits():
    def llm_query(prompt):
        return f"ans:{prompt}"

    cache = SubQueryCache(embedder=_coc_embedder, similarity_threshold=0.9)
    q = cache.wrap({"llm_query": llm_query})["llm_query"]
    q("the change-of-control clause")
    q("the governing law section")   # orthogonal vector -> cosine 0
    stats = cache.stats()
    assert stats["semantic_hits"] == 0
    assert stats["misses"] == 2


def test_no_embedder_means_exact_only():
    def llm_query(prompt):
        return f"ans:{prompt}"

    cache = SubQueryCache()  # no embedder -> exact-hash dedup only
    q = cache.wrap({"llm_query": llm_query})["llm_query"]
    q("the change-of-control clause")
    q("the change-of-control terms")
    assert cache.stats()["semantic_hits"] == 0
    assert cache.stats()["misses"] == 2


# ---- make_embedder: adapt a batch embedder (dspy.Embedder) to the cache's contract ----

def test_make_embedder_calls_the_batch_embedder_with_a_single_prompt():
    from sentinelprime.subcache import make_embedder

    calls = []

    def batch_embed(texts):
        calls.append(list(texts))
        return [[1.0, 2.0] for _ in texts]

    assert make_embedder(batch_embed)("hello") == [1.0, 2.0]
    assert calls == [["hello"]]


def test_make_embedder_coerces_the_row_to_a_plain_float_list():
    """dspy.Embedder returns numpy rows; SubQueryCache's cosine needs plain floats."""
    from sentinelprime.subcache import make_embedder

    vec = make_embedder(lambda texts: [(1, 2, 3)])("hello")
    assert vec == [1.0, 2.0, 3.0]
    assert all(isinstance(x, float) for x in vec)


def test_semantic_cache_end_to_end_through_make_embedder():
    from sentinelprime.subcache import SubQueryCache, make_embedder

    def batch_embed(texts):
        return [[1.0, 0.0] if "governing law" in t.lower() else [0.0, 1.0] for t in texts]

    cache = SubQueryCache(embedder=make_embedder(batch_embed), similarity_threshold=0.9)
    calls = []
    wrapped = cache.wrap({"llm_query": lambda p: calls.append(p) or "Delaware"})

    wrapped["llm_query"]("What is the governing law?")
    wrapped["llm_query"]("Governing law of this agreement?")

    assert calls == ["What is the governing law?"]
    assert cache.stats()["semantic_hits"] == 1


# ---- Degradation: a broken embedder must not take down a run that exact dedup serves --

def test_cache_falls_back_to_exact_dedup_when_the_embedder_fails():
    import pytest
    from sentinelprime.subcache import SubQueryCache

    def boom(_text):
        raise RuntimeError("embeddings endpoint unavailable")

    calls = []
    cache = SubQueryCache(embedder=boom)
    wrapped = cache.wrap({"llm_query": lambda p: calls.append(p) or "ANS"})

    with pytest.warns(RuntimeWarning, match="semantic sub-query cache disabled"):
        wrapped["llm_query"]("what is the governing law?")
    wrapped["llm_query"]("what is the governing law?")

    assert calls == ["what is the governing law?"]   # exact tier still dedups
    assert cache.stats()["hits"] == 1
    assert cache.stats()["semantic_hits"] == 0
    assert cache.semantic_enabled is False


def test_cache_warns_once_not_on_every_query():
    import pytest, warnings
    from sentinelprime.subcache import SubQueryCache

    def boom(_text):
        raise RuntimeError("nope")

    cache = SubQueryCache(embedder=boom)
    wrapped = cache.wrap({"llm_query": lambda p: "ANS"})
    with pytest.warns(RuntimeWarning):
        wrapped["llm_query"]("a")
    with warnings.catch_warnings(record=True) as later:
        warnings.simplefilter("always")
        wrapped["llm_query"]("b")
    assert later == []


def test_cache_stats_shape_is_unchanged_by_the_fallback():
    """run_lab and the agent tests compare stats dicts exactly; keep the keys stable."""
    from sentinelprime.subcache import SubQueryCache

    assert set(SubQueryCache().stats()) == {"hits", "misses", "calls", "semantic_hits"}
