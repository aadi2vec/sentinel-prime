import pytest
import dspy
from sentinelprime.agent import PrimeAgent, PrimeTask
from sentinelprime.harness import ContinualHarness
from sentinelprime.memory import JsonMemoryBackend
from sentinelprime.feedback import parse_lab_result


class StubRLM:
    """Records inputs; returns a canned Prediction. No LM involved."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return dspy.Prediction(deliverable="STUB", trajectory=[])


def _harness(tmp_path):
    import os
    os.makedirs(tmp_path, exist_ok=True)
    return ContinualHarness(JsonMemoryBackend(str(tmp_path / "state.json")))


def test_run_task_injects_frozen_guidance(tmp_path):
    harness = _harness(tmp_path)
    rlm = StubRLM()
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=rlm)
    pred = agent.run_task("summarize the SPA", workdir=str(tmp_path))
    assert pred.deliverable == "STUB"
    assert rlm.calls[0]["task"] == "summarize the SPA"
    # empty ledger -> placeholder guidance, base task untouched
    assert rlm.calls[0]["guidance"] == "(no learned guidance yet)"


def test_guidance_reflects_ledger_but_only_between_tasks(tmp_path):
    harness = _harness(tmp_path)
    # Seed the ledger directly via the backend so read() is non-empty.
    from sentinelprime.memory import MemoryItem
    harness.backend.write([MemoryItem(id="n1", scope="global", kind="note",
                                      text="check change-of-control", created_at="t")])
    rlm = StubRLM()
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=rlm)
    agent.run_task("t1", workdir=str(tmp_path))
    assert "change-of-control" in rlm.calls[0]["guidance"]


def test_learn_delegates_to_refine(tmp_path, monkeypatch):
    harness = _harness(tmp_path)
    seen = {}

    def fake_refine(trajectory, feedback):
        seen["called"] = (trajectory, feedback)
        return "REFINED"

    monkeypatch.setattr(harness, "refine", fake_refine)
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=StubRLM())
    fb = parse_lab_result({"task_id": "t", "criteria": [{"id": "c1", "passed": False, "reason": "x"}]})
    out = agent.learn(trajectory=[{"a": 1}], feedback=fb)
    assert out == "REFINED"
    assert seen["called"][0] == [{"a": 1}]


def test_default_rlm_is_named_predictor(tmp_path):
    harness = _harness(tmp_path)
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"))
    names = dict(agent.named_predictors())
    # RLM's inner predictors and the harness proposer are all GEPA-visible.
    assert any("generate_action" in n for n in names)
    assert any("propose" in n for n in names)


def test_default_rlm_dedups_subqueries_and_reports_stats(tmp_path):
    """Real CachingRLM + LocalInterpreter: identical llm_query calls hit the cache.

    Scripted root LM issues two identical llm_query('same') calls in one REPL
    turn, then SUBMITs. The sub-LM is a bare callable returning a list; the
    SubQueryCache should serve the second call from cache so the sub-LM runs
    once, and the per-run stats surface on the agent.
    """
    from dspy.utils.dummies import DummyLM

    sub_calls = []

    def sub_lm(prompt):
        sub_calls.append(prompt)
        return ["ANS"]

    lm = DummyLM([
        {"reasoning": "query twice",
         "code": "```python\na = llm_query('same')\nb = llm_query('same')\nprint(a, b)\n```"},
        {"reasoning": "submit",
         "code": "```python\nSUBMIT(deliverable='done')\n```"},
    ])

    harness = _harness(tmp_path)
    agent = PrimeAgent(harness=harness, root_lm=lm, sub_lm=sub_lm)
    pred = agent.run_task("dedup demo", workdir=str(tmp_path))

    assert pred.deliverable == "done"
    assert sub_calls == ["same"]  # underlying sub-LM hit once, not twice
    assert agent.last_cache_stats == {"hits": 1, "misses": 1, "calls": 2, "semantic_hits": 0}


@pytest.mark.integration
def test_end_to_end_with_scripted_lm(tmp_path, monkeypatch):
    """Real dspy.RLM + LocalInterpreter driven by a scripted LM.

    The scripted LM returns fixed reasoning+code so the RLM reads a file
    and SUBMITs — exercising the interpreter's fs access and
    SUBMIT->FinalOutput path without any network call.

    Implementation notes:
    - DummyLM accepts a list of dicts keyed by generate_action's output field
      names: 'reasoning' and 'code' (confirmed from dspy.predict.rlm source).
    - LocalInterpreter executes code in the process CWD, not in workdir
      (workdir is stored but no chdir occurs). We use monkeypatch.chdir so
      open('note.txt') resolves against tmp_path.
    - DummyLM is consumed as an iterator: first call returns the read-file
      code, second call returns SUBMIT. The persistent REPL namespace carries
      the 'text' variable across turns.
    """
    doc = tmp_path / "note.txt"
    doc.write_text("change-of-control clause present")

    from sentinelprime.harness import ContinualHarness
    from sentinelprime.memory import JsonMemoryBackend

    # A scripted LM: first turn reads the file, second turn SUBMITs.
    # Field names match generate_action's output fields: 'reasoning' and 'code'.
    from dspy.utils.dummies import DummyLM
    lm = DummyLM([
        {"reasoning": "read the doc",
         "code": "```python\nwith open('note.txt') as f: text = f.read()\nprint(text)\n```"},
        {"reasoning": "submit",
         "code": "```python\nSUBMIT(deliverable=text)\n```"},
    ])

    harness = ContinualHarness(JsonMemoryBackend(str(tmp_path / "state.json")))
    agent = PrimeAgent(harness=harness, root_lm=lm, sub_lm=lm)
    # chdir so open('note.txt') resolves to tmp_path/note.txt inside the REPL
    monkeypatch.chdir(tmp_path)
    pred = agent.run_task("summarize note", workdir=str(tmp_path))
    assert "change-of-control" in pred.deliverable


# ---- Integration seams: context threading, progress monitoring, semantic cache -------

class TrajectoryRLM:
    """StubRLM variant returning a caller-supplied trajectory."""

    def __init__(self, trajectory):
        self.calls = []
        self._trajectory = trajectory

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return dspy.Prediction(deliverable="STUB", trajectory=self._trajectory)


def _audited_harness(tmp_path, *, depends_on, score=1.0):
    """Harness whose ledger holds one external-scope lesson with a declared dependency."""
    from sentinelprime.audit import AuditLog, AuditRecord
    from sentinelprime.memory import MemoryItem

    backend = JsonMemoryBackend(str(tmp_path / "state.json"))
    backend.write([MemoryItem(id="ext1", scope="session", kind="note",
                              text="counterparty playbook says escrow 10%", created_at="t")])
    audit_log = AuditLog(str(tmp_path / "audit.json"))
    audit_log.append(AuditRecord(
        edit_id="ext1", op="create", scope="external", from_version=0, to_version=1,
        cause_task_id="ma-001", cause_failures="- [c1] missed escrow", trajectory_digest="d",
        score=score, created_at="t", content_hash="h", depends_on=depends_on,
    ))
    return ContinualHarness(backend, audit_log=audit_log)


def test_run_task_withholds_stale_lesson_when_context_source_moved(tmp_path):
    harness = _audited_harness(tmp_path, depends_on={"playbook_version": "v1"})
    rlm = TrajectoryRLM([])
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=rlm)
    agent.run_task("t1", workdir=str(tmp_path), context={"playbook_version": "v2"})
    assert "escrow 10%" not in rlm.calls[0]["guidance"]


def test_run_task_keeps_lesson_when_context_dependency_still_current(tmp_path):
    harness = _audited_harness(tmp_path, depends_on={"playbook_version": "v1"})
    rlm = TrajectoryRLM([])
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=rlm)
    agent.run_task("t1", workdir=str(tmp_path), context={"playbook_version": "v1"})
    assert "escrow 10%" in rlm.calls[0]["guidance"]


def test_run_task_exposes_rlm_trajectory_for_learning(tmp_path):
    traj = [{"reasoning": "read the doc", "code": "open(...)", "output": "..."}]
    harness = _harness(tmp_path)
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"),
                       rlm=TrajectoryRLM(traj))
    agent.run_task("t1", workdir=str(tmp_path))
    assert agent.last_trajectory == traj


def test_run_task_records_replan_audit_event_on_stalled_trajectory(tmp_path):
    from sentinelprime.audit import AuditLog
    from sentinelprime.monitor import ProgressMonitor

    stalled = [{"reasoning": "re-read the msa for change of control"} for _ in range(3)]
    audit_log = AuditLog(str(tmp_path / "audit.json"))
    harness = ContinualHarness(JsonMemoryBackend(str(tmp_path / "state.json")),
                               audit_log=audit_log)
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"),
                       rlm=TrajectoryRLM(stalled), monitor=ProgressMonitor())
    agent.run_task("t1", workdir=str(tmp_path), task_id="ma-001")
    replans = [r for r in audit_log.records() if r.op == "replan"]
    assert len(replans) == 1
    assert replans[0].cause_task_id == "ma-001"


def test_run_task_leaves_audit_log_clean_when_trajectory_is_healthy(tmp_path):
    from sentinelprime.audit import AuditLog
    from sentinelprime.monitor import ProgressMonitor

    healthy = [{"reasoning": "read the msa"}, {"reasoning": "extract governing law"},
               {"reasoning": "draft the summary"}]
    audit_log = AuditLog(str(tmp_path / "audit.json"))
    harness = ContinualHarness(JsonMemoryBackend(str(tmp_path / "state.json")),
                               audit_log=audit_log)
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"),
                       rlm=TrajectoryRLM(healthy), monitor=ProgressMonitor())
    agent.run_task("t1", workdir=str(tmp_path), task_id="ma-001")
    assert [r for r in audit_log.records() if r.op == "replan"] == []


def test_default_rlm_reuses_paraphrased_subqueries_with_embedder(tmp_path):
    """Semantic tier in the live agent path: paraphrased sub-queries hit the sub-LM once."""
    from dspy.utils.dummies import DummyLM

    sub_calls = []

    def sub_lm(prompt):
        sub_calls.append(prompt)
        return ["Delaware"]

    def embed(text: str) -> list[float]:
        # deterministic stand-in for a real embedder: same topic -> same vector
        return [1.0, 0.0] if "governing law" in text.lower() else [0.0, 1.0]

    lm = DummyLM([
        {"reasoning": "ask twice, paraphrased",
         "code": ("```python\na = llm_query('What is the governing law?')\n"
                  "b = llm_query('Governing law of this agreement?')\nprint(a, b)\n```")},
        {"reasoning": "submit", "code": "```python\nSUBMIT(deliverable='done')\n```"},
    ])

    harness = _harness(tmp_path)
    agent = PrimeAgent(harness=harness, root_lm=lm, sub_lm=sub_lm, subquery_embedder=embed)
    agent.run_task("paraphrase demo", workdir=str(tmp_path))

    assert len(sub_calls) == 1  # second, paraphrased query served from cache
    assert agent.last_cache_stats["semantic_hits"] == 1


# ---- Prefix-cache structuring: the ledger must sit inside the shared prompt prefix ----

def test_ledger_block_sits_inside_the_prefix_shared_across_tasks():
    """Provider prefix caching only pays if the stable ledger precedes the variable task."""
    from sentinelprime.agent import cacheable_prefix

    prefix = cacheable_prefix("LEDGER-BLOCK-XYZ",
                              ["summarize the SPA", "summarize the MSA"])
    assert "LEDGER-BLOCK-XYZ" in prefix


def test_cacheable_prefix_is_byte_stable_across_repeated_reads():
    from sentinelprime.agent import cacheable_prefix

    tasks = ["summarize the SPA", "summarize the MSA"]
    assert cacheable_prefix("LEDGER", tasks) == cacheable_prefix("LEDGER", tasks)


def test_changing_the_ledger_invalidates_the_shared_prefix():
    """The ledger is *inside* the cached prefix, so editing it must move the boundary."""
    from sentinelprime.agent import cacheable_prefix

    tasks = ["summarize the SPA", "summarize the MSA"]
    assert "OLD-LESSON" not in cacheable_prefix("NEW-LESSON", tasks)


def test_cacheable_prefix_ends_before_the_per_task_content():
    from sentinelprime.agent import cacheable_prefix

    prefix = cacheable_prefix("LEDGER", ["summarize the SPA", "summarize the MSA"])
    assert "summarize the SPA" not in prefix


# ---- Subagents reaching the live path -------------------------------------------------

def test_children_are_off_unless_asked_for(tmp_path):
    agent = PrimeAgent(harness=_harness(tmp_path), root_lm=dspy.LM("openai/gpt-4o-mini"))
    assert agent.rlm.tools == {}


def test_enabling_children_exposes_spawn_and_collect_tools(tmp_path):
    """Spawning without a way to collect is why subagents were dead weight before."""
    agent = PrimeAgent(harness=_harness(tmp_path), root_lm=dspy.LM("openai/gpt-4o-mini"),
                       enable_children=True)
    assert set(agent.rlm.tools) == {"spawn_child", "collect_child"}


def test_child_runs_the_task_and_its_result_is_collectable(tmp_path):
    """Real RLM + interpreter: the model spawns a child and reads its deliverable back."""
    from dspy.utils.dummies import DummyLM

    lm = DummyLM([
        {"reasoning": "delegate the contract read",
         "code": ("```python\ncid = spawn_child(task='summarize spa.txt')\n"
                  "answer = collect_child(child_id=cid)\nprint(answer)\n```")},
        {"reasoning": "submit", "code": "```python\nSUBMIT(deliverable=answer)\n```"},
    ])
    child_tasks = []

    def run_child(task, name, session_dir):
        child_tasks.append(task)
        return "CHILD-FOUND-CHANGE-OF-CONTROL"

    agent = PrimeAgent(harness=_harness(tmp_path), root_lm=lm, enable_children=True,
                       child_runner=run_child)
    pred = agent.run_task("delegate", workdir=str(tmp_path))

    assert child_tasks == ["summarize spa.txt"]
    assert pred.deliverable == "CHILD-FOUND-CHANGE-OF-CONTROL"


def test_default_child_runner_reads_the_ledger_but_never_writes_to_it(tmp_path):
    """Children run on a thread pool; only the parent may edit the ledger, between tasks."""
    from dspy.utils.dummies import DummyLM
    from sentinelprime.memory import MemoryItem

    harness = _harness(tmp_path)
    harness.backend.write([MemoryItem(id="n1", scope="global", kind="note",
                                      text="check change-of-control", created_at="t")])
    seen = {}

    lm = DummyLM([
        {"reasoning": "delegate", "code": "```python\ncid = spawn_child(task='sub')\nr = collect_child(child_id=cid)\n```"},
        {"reasoning": "child works", "code": "```python\nSUBMIT(deliverable='child done')\n```"},
        {"reasoning": "submit", "code": "```python\nSUBMIT(deliverable=r)\n```"},
    ])
    agent = PrimeAgent(harness=harness, root_lm=lm, enable_children=True)
    versions_before = harness.backend.current_version().number
    agent.run_task("delegate", workdir=str(tmp_path))
    seen["after"] = harness.backend.current_version().number
    assert seen["after"] == versions_before  # no child wrote a ledger version


def test_run_task_drains_children_so_no_worker_outlives_the_task(tmp_path):
    from dspy.utils.dummies import DummyLM

    finished = []

    def run_child(task, name, session_dir):
        finished.append(task)
        return "ok"

    lm = DummyLM([
        # spawned but deliberately never collected by the model
        {"reasoning": "fire and forget", "code": "```python\nspawn_child(task='orphan')\n```"},
        {"reasoning": "submit", "code": "```python\nSUBMIT(deliverable='done')\n```"},
    ])
    agent = PrimeAgent(harness=_harness(tmp_path), root_lm=lm, enable_children=True,
                       child_runner=run_child)
    agent.run_task("t", workdir=str(tmp_path))
    assert finished == ["orphan"]


# ---- async: concurrency across tasks, not within one -----------------------------------

def test_async_run_task_returns_the_same_shape_as_sync(tmp_path):
    """A single RLM loop is inherently sequential; async exists to overlap whole tasks."""
    import asyncio

    class _AsyncRLM:
        def __init__(self):
            self.calls = []

        async def acall(self, **kw):
            self.calls.append(kw)
            return dspy.Prediction(deliverable="ASYNC", trajectory=[{"reasoning": "r"}])

    rlm = _AsyncRLM()
    agent = PrimeAgent(harness=_harness(tmp_path), root_lm=dspy.LM("openai/gpt-4o-mini"),
                       rlm=rlm)
    pred = asyncio.run(agent.arun_task("t", workdir=str(tmp_path)))
    assert pred.deliverable == "ASYNC"
    assert agent.last_trajectory == [{"reasoning": "r"}]
    assert rlm.calls[0]["guidance"] == "(no learned guidance yet)"


def test_async_run_task_still_gates_guidance_and_monitors(tmp_path):
    import asyncio
    from sentinelprime.audit import AuditLog
    from sentinelprime.monitor import ProgressMonitor

    class _AsyncRLM:
        async def acall(self, **kw):
            stalled = [{"reasoning": "re-read the msa for change of control"}] * 3
            return dspy.Prediction(deliverable="X", trajectory=stalled)

    audit_log = AuditLog(str(tmp_path / "audit.json"))
    harness = ContinualHarness(JsonMemoryBackend(str(tmp_path / "s.json")),
                               audit_log=audit_log)
    agent = PrimeAgent(harness=harness, root_lm=dspy.LM("openai/gpt-4o-mini"),
                       rlm=_AsyncRLM(), monitor=ProgressMonitor())
    asyncio.run(agent.arun_task("t", workdir=str(tmp_path), task_id="ma-001"))
    assert [r.op for r in audit_log.records()] == ["replan"]


def test_two_tasks_overlap_when_run_concurrently(tmp_path):
    """The point of async: task B's LM latency overlaps task A's."""
    import asyncio

    peak = {"active": 0, "max": 0}

    class _SlowAsyncRLM:
        async def acall(self, **kw):
            peak["active"] += 1
            peak["max"] = max(peak["max"], peak["active"])
            await asyncio.sleep(0.05)
            peak["active"] -= 1
            return dspy.Prediction(deliverable="ok", trajectory=[])

    async def both():
        a = PrimeAgent(harness=_harness(tmp_path / "a"), root_lm=dspy.LM("openai/gpt-4o-mini"),
                       rlm=_SlowAsyncRLM())
        b = PrimeAgent(harness=_harness(tmp_path / "b"), root_lm=dspy.LM("openai/gpt-4o-mini"),
                       rlm=_SlowAsyncRLM())
        (tmp_path / "a").mkdir(exist_ok=True); (tmp_path / "b").mkdir(exist_ok=True)
        return await asyncio.gather(a.arun_task("t1", workdir=str(tmp_path / "a")),
                                    b.arun_task("t2", workdir=str(tmp_path / "b")))

    results = asyncio.run(both())
    assert len(results) == 2
    assert peak["max"] == 2       # they really did overlap


class TrajectoryRLM(StubRLM):
    """StubRLM that returns a fixed REPL trajectory."""

    def __init__(self, trajectory):
        super().__init__()
        self._trajectory = trajectory

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return dspy.Prediction(deliverable="STUB", trajectory=self._trajectory)


def _workdir_with_documents(tmp_path, *names):
    docs = tmp_path / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    for name in names:
        (docs / name).write_text("body")
    return tmp_path


def test_run_task_reports_which_documents_went_unread(tmp_path):
    """Not opening a document is a failure with no reasoning in it — measure it free."""
    workdir = _workdir_with_documents(tmp_path, "merger.txt", "escrow.txt")
    rlm = TrajectoryRLM([{"code": "open('documents/merger.txt').read()", "output": "..."}])
    agent = PrimeAgent(harness=_harness(tmp_path / "h"),
                       root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=rlm)
    agent.run_task("review the documents", workdir=str(workdir))
    assert agent.last_coverage.touched == ["merger.txt"]
    assert agent.last_coverage.untouched == ["escrow.txt"]
    assert agent.last_coverage.rate == 0.5


def test_coverage_is_none_when_the_workdir_has_no_documents(tmp_path):
    """Optional collaborator: a workspace with no documents/ dir reports nothing."""
    rlm = TrajectoryRLM([{"code": "print(1)", "output": "1"}])
    agent = PrimeAgent(harness=_harness(tmp_path / "h"),
                       root_lm=dspy.LM("openai/gpt-4o-mini"), rlm=rlm)
    agent.run_task("no documents here", workdir=str(tmp_path))
    assert agent.last_coverage is None
