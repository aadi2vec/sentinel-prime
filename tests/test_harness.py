import json
from sentinelprime.memory import JsonMemoryBackend, MemoryItem
from sentinelprime.feedback import parse_lab_result
from sentinelprime.harness import ContinualHarness, RefineResult
from sentinelprime.audit import AuditLog, AuditRecord
from sentinelprime.reuse import ReuseController


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


def _audit_rec(edit_id, scope, depends_on=None, to_version=2):
    return AuditRecord(
        edit_id=edit_id, op="create", scope=scope, from_version=1, to_version=to_version,
        cause_task_id="t", cause_failures="", trajectory_digest="d", score=1.0,
        created_at="t", content_hash=edit_id + str(to_version), depends_on=depends_on or {},
    )


def _harness_with_audit(tmp_path):
    be = _backend(tmp_path)
    log = AuditLog(str(tmp_path / "audit.json"))
    h = ContinualHarness(be, audit_log=log, reuse_controller=ReuseController())
    return be, log, h


def test_read_without_context_is_ungated(tmp_path):
    be, log, h = _harness_with_audit(tmp_path)
    _seed(be, id="ext1", text="external note", scope="session")
    log.append(_audit_rec("ext1", "external", depends_on={"playbook_version": 5}))
    # no context -> current behavior, item retained even though source could be stale
    assert "external note" in h.read()


def test_read_gates_stale_external_but_keeps_intrinsic(tmp_path):
    be, log, h = _harness_with_audit(tmp_path)
    _seed(be, id="ext1", text="external note", scope="session")
    _seed(be, id="intr1", text="intrinsic note", scope="global")
    log.append(_audit_rec("ext1", "external", depends_on={"playbook_version": 5}))
    log.append(_audit_rec("intr1", "intrinsic"))

    block = h.read(context={"playbook_version": 6})  # source moved 5 -> 6
    assert "external note" not in block
    assert "intrinsic note" in block


def test_read_keeps_external_when_source_current(tmp_path):
    be, log, h = _harness_with_audit(tmp_path)
    _seed(be, id="ext1", text="external note", scope="session")
    log.append(_audit_rec("ext1", "external", depends_on={"playbook_version": 5}))
    assert "external note" in h.read(context={"playbook_version": 5})


def test_read_keeps_items_without_provenance(tmp_path):
    be, log, h = _harness_with_audit(tmp_path)
    _seed(be, id="seeded", text="no audit record", scope="session")
    # no audit record for 'seeded' -> not blocked
    assert "no audit record" in h.read(context={"playbook_version": 6})


def test_read_uses_latest_record_per_item(tmp_path):
    be, log, h = _harness_with_audit(tmp_path)
    _seed(be, id="ext1", text="external note", scope="session")
    # older record depended on v5; newer refinement re-derived against v6
    log.append(_audit_rec("ext1", "external", depends_on={"playbook_version": 5}, to_version=2))
    log.append(_audit_rec("ext1", "external", depends_on={"playbook_version": 6}, to_version=4))
    assert "external note" in h.read(context={"playbook_version": 6})


def test_refine_exposes_predictor_to_gepa(tmp_path):
    h = ContinualHarness(_backend(tmp_path))
    names = dict(h.named_predictors())
    assert any("propose" in n for n in names)


def test_read_is_deterministic_regardless_of_insertion_order(tmp_path):
    # Prefix-cache guardrail: the read() block must be a stable prefix. Two ledgers
    # with the same items inserted in different orders must serialize byte-identically.
    be1 = JsonMemoryBackend(str(tmp_path / "a.json"))
    be2 = JsonMemoryBackend(str(tmp_path / "b.json"))
    for id in ["n3", "n1", "n2"]:
        _seed(be1, id=id, text=f"body-{id}")
    for id in ["n2", "n3", "n1"]:
        _seed(be2, id=id, text=f"body-{id}")
    block1 = ContinualHarness(be1).read()
    block2 = ContinualHarness(be2).read()
    assert block1 == block2
    # items appear sorted by id within their kind
    assert block1.index("body-n1") < block1.index("body-n2") < block1.index("body-n3")


def test_read_of_unchanged_ledger_is_byte_identical(tmp_path):
    be = _backend(tmp_path)
    _seed(be, id="n1", text="one")
    _seed(be, id="n2", text="two")
    h = ContinualHarness(be)
    assert h.read() == h.read()
