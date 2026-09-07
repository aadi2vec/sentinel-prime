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
