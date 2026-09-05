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
    assert cache.stats() == {"hits": 1, "misses": 1, "calls": 2}


def test_distinct_prompts_miss():
    def llm_query(prompt):
        return f"ans:{prompt}"

    cache = SubQueryCache()
    q = cache.wrap({"llm_query": llm_query})["llm_query"]
    q("a")
    q("b")
    assert cache.stats() == {"hits": 0, "misses": 2, "calls": 2}


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
