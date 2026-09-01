import json
from sentinelprime.memory import JsonMemoryBackend, MemoryItem
from sentinelprime.feedback import parse_lab_result
from sentinelprime.harness import ContinualHarness, RefineResult


def _backend(tmp_path):
    return JsonMemoryBackend(str(tmp_path / "h.json"))


def _seed(be, id="n1", kind="note", text="always check change-of-control", scope="global"):
    be.write([MemoryItem(id=id, scope=scope, kind=kind, text=text, created_at="2026-01-01", meta={})])


def test_read_empty_ledger_returns_empty_string(tmp_path):
    h = ContinualHarness(_backend(tmp_path))
    assert h.read() == ""


def test_read_serializes_items_grouped_by_kind(tmp_path):
    be = _backend(tmp_path)
    _seed(be, id="n1", kind="note", text="note-body")
    be.write([MemoryItem(id="s1", scope="global", kind="sub_agent_spec",
                         text="extract CoC clauses", created_at="2026-01-01",
                         meta={"name": "coc_extractor", "when_to_use": "M&A docs"})])
    block = ContinualHarness(be).read()
    assert "note-body" in block
    assert "coc_extractor" in block
    assert "M&A docs" in block


def test_apply_edits_create_update_delete(tmp_path):
    be = _backend(tmp_path)
    _seed(be, id="keep", text="keep me")
    _seed(be, id="old", text="delete me")
    h = ContinualHarness(be)
    created, updated, deleted = h._apply_edits([
        {"op": "create", "id": "new1", "kind": "note", "text": "fresh insight", "scope": "global"},
        {"op": "update", "id": "keep", "kind": "note", "text": "updated body", "scope": "global"},
        {"op": "delete", "id": "old"},
    ])
    assert created == ["new1"]
    assert updated == ["keep"]
    assert deleted == ["old"]
    ids = {i.id: i.text for i in be.read()}
    assert ids == {"keep": "updated body", "new1": "fresh insight"}


def test_refine_is_reversible(tmp_path, monkeypatch):
    be = _backend(tmp_path)
    _seed(be, id="n1", text="original")
    h = ContinualHarness(be)

    # Stub the LM-calling predictor to return deterministic edits.
    edits = [{"op": "create", "id": "n2", "kind": "note", "text": "learned", "scope": "global"}]
    monkeypatch.setattr(h, "propose",
                        lambda **kw: type("P", (), {"edits": json.dumps(edits)})())

    fb = parse_lab_result({"task_id": "t", "criteria": [
        {"id": "c1", "passed": False, "reason": "missed CoC clause"}]})
    result = h.refine(trajectory=[{"role": "action", "code": "..."}], feedback=fb)

    assert isinstance(result, RefineResult)
    assert result.created == ["n2"]
    assert result.from_version != result.to_version
    assert {i.id for i in be.read()} == {"n1", "n2"}

    h.rollback(result.from_version)
    assert {i.id for i in be.read()} == {"n1"}


def test_refine_exposes_predictor_to_gepa(tmp_path):
    h = ContinualHarness(_backend(tmp_path))
    names = dict(h.named_predictors())
    assert any("propose" in n for n in names)
