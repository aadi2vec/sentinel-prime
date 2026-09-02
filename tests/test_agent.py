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
