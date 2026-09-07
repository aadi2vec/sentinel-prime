"""Run telemetry: a tailable event log and token/cost accounting."""
import json

import pytest


# ---- RunLog ---------------------------------------------------------------------------

def test_event_is_written_as_one_json_object_per_line(tmp_path):
    from sentinelprime.telemetry import RunLog

    log = RunLog(tmp_path / "run.jsonl", echo=False)
    log.event("task_start", task_id="ma-001", arm="control")
    log.event("task_done", task_id="ma-001", pooled=0.5)

    lines = (tmp_path / "run.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["kind"] == "task_start"
    assert first["task_id"] == "ma-001"
    assert "ts" in first


def test_events_are_flushed_immediately_so_a_tail_sees_them(tmp_path):
    """A sweep is hours long and gets killed; buffered events would be lost."""
    from sentinelprime.telemetry import RunLog

    path = tmp_path / "run.jsonl"
    log = RunLog(path, echo=False)
    log.event("task_start", task_id="t")
    assert "task_start" in path.read_text()   # visible before close()


def test_echo_renders_a_human_line(tmp_path, capsys):
    from sentinelprime.telemetry import RunLog

    RunLog(tmp_path / "r.jsonl", echo=True).event(
        "ledger_edit", op="create", edit_id="lesson.c1", text="always check CoC")
    out = capsys.readouterr().out
    assert "ledger_edit" in out
    assert "lesson.c1" in out


def test_reopening_appends_rather_than_truncating(tmp_path):
    from sentinelprime.telemetry import RunLog

    p = tmp_path / "r.jsonl"
    RunLog(p, echo=False).event("a", i=1)
    RunLog(p, echo=False).event("b", i=2)
    assert len(p.read_text().strip().split("\n")) == 2


def test_unserializable_values_do_not_kill_the_run(tmp_path):
    """Telemetry must never be the thing that crashes a two-hour sweep."""
    from sentinelprime.telemetry import RunLog

    log = RunLog(tmp_path / "r.jsonl", echo=False)
    log.event("weird", obj=object())
    rec = json.loads((tmp_path / "r.jsonl").read_text().strip())
    assert rec["kind"] == "weird"


# ---- UsageMeter -----------------------------------------------------------------------

def test_usage_totals_accumulate_across_calls():
    from sentinelprime.telemetry import UsageMeter

    m = UsageMeter()
    m.add({"gpt-x": {"prompt_tokens": 100, "completion_tokens": 20}})
    m.add({"gpt-x": {"prompt_tokens": 50, "completion_tokens": 5}})
    t = m.totals()
    assert t["prompt_tokens"] == 150
    assert t["completion_tokens"] == 25
    assert t["total_tokens"] == 175


def test_usage_is_kept_per_model():
    from sentinelprime.telemetry import UsageMeter

    m = UsageMeter()
    m.add({"big": {"prompt_tokens": 100, "completion_tokens": 10}})
    m.add({"small": {"prompt_tokens": 7, "completion_tokens": 1}})
    assert set(m.by_model()) == {"big", "small"}
    assert m.by_model()["small"]["prompt_tokens"] == 7


def test_cost_is_none_without_a_price_table():
    """Never invent a dollar figure for a model whose pricing we do not know."""
    from sentinelprime.telemetry import UsageMeter

    m = UsageMeter()
    m.add({"mystery": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})
    assert m.cost() is None


def test_cost_uses_the_supplied_price_table():
    from sentinelprime.telemetry import UsageMeter

    m = UsageMeter(prices={"gpt-x": {"input": 1.0, "output": 10.0}})
    m.add({"gpt-x": {"prompt_tokens": 2_000_000, "completion_tokens": 100_000}})
    assert m.cost() == pytest.approx(2.0 + 1.0)


def test_unknown_model_makes_the_total_cost_unknown():
    """A partial cost is worse than none — it reads as the whole bill."""
    from sentinelprime.telemetry import UsageMeter

    m = UsageMeter(prices={"known": {"input": 1.0, "output": 1.0}})
    m.add({"known": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})
    m.add({"unknown": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})
    assert m.cost() is None
    assert m.unpriced() == ["unknown"]


def test_usage_tolerates_missing_fields():
    from sentinelprime.telemetry import UsageMeter

    m = UsageMeter()
    m.add({"m": {}})
    m.add({})
    assert m.totals()["total_tokens"] == 0
