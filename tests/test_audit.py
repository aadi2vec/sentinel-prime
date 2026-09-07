import dspy

from sentinelprime.audit import AuditRecord, AuditLog
from sentinelprime.harness import ContinualHarness
from sentinelprime.memory import JsonMemoryBackend
from sentinelprime.feedback import parse_lab_result


def _record(**over):
    base = dict(
        edit_id="checklist.change_of_control",
        op="create",
        scope="intrinsic",
        from_version=4,
        to_version=5,
        cause_task_id="ma-001",
        cause_failures="- [c1] missed change-of-control clause",
        trajectory_digest="abc",
        score=0.88,
        created_at="t",
        content_hash="h1",
    )
    base.update(over)
    return AuditRecord(**base)


def test_append_and_roundtrip(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    log.append(_record())
    reloaded = AuditLog(str(tmp_path / "audit.json"))
    assert reloaded.records()[0].edit_id == "checklist.change_of_control"
    assert reloaded.by_version(5)[0].op == "create"


def test_content_hash_dedup(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    assert log.seen("h1") is False
    log.append(_record(content_hash="h1"))
    assert log.seen("h1") is True
    # a second identical-hash edit is not re-appended
    log.append(_record(content_hash="h1"))
    assert len(log.records()) == 1


def test_by_version_filters_on_window_end(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    log.append(_record(content_hash="h1", to_version=5))
    log.append(_record(content_hash="h2", edit_id="k2", to_version=7))
    assert [r.edit_id for r in log.by_version(7)] == ["k2"]


def test_explain_renders_causal_chain(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    log.append(_record(edit_id="checklist.change_of_control", op="create",
                        scope="intrinsic", from_version=4, to_version=5,
                        cause_task_id="ma-001",
                        cause_failures="missed change-of-control clause",
                        content_hash="9da22988"))
    out = log.explain(5)
    assert "ma-001" in out
    assert "create" in out
    assert "v4->v5" in out
    assert "scope=intrinsic" in out
    assert "rollback(from_version=4)" in out


def test_explain_unknown_version(tmp_path):
    log = AuditLog(str(tmp_path / "audit.json"))
    assert "no audit records" in log.explain(99).lower()


def test_harness_explain_delegates(tmp_path):
    backend = JsonMemoryBackend(str(tmp_path / "state.json"))
    log = AuditLog(str(tmp_path / "audit.json"))
    harness = ContinualHarness(backend, audit_log=log)
    harness.propose = _StubPropose(
        '[{"op":"create","id":"k1","kind":"note","text":"t","scope":"global"}]'
    )
    fb = parse_lab_result(
        {"task_id": "ma-002", "criteria": [{"id": "c1", "passed": False, "reason": "boom"}]}
    )
    result = harness.refine(trajectory=[], feedback=fb)
    out = harness.explain(result.to_version)
    assert "ma-002" in out
    assert "k1" in out


class _StubPropose:
    """Stands in for the dspy.Predict proposer: returns canned edit ops."""

    def __init__(self, edits_json):
        self._edits_json = edits_json

    def __call__(self, **kwargs):
        return dspy.Prediction(edits=self._edits_json)


def test_refine_emits_audit_record(tmp_path):
    backend = JsonMemoryBackend(str(tmp_path / "state.json"))
    log = AuditLog(str(tmp_path / "audit.json"))
    harness = ContinualHarness(backend, audit_log=log)
    harness.propose = _StubPropose(
        '[{"op":"create","id":"checklist.change_of_control",'
        '"kind":"note","text":"check change-of-control","scope":"global"}]'
    )
    fb = parse_lab_result(
        {"task_id": "ma-001",
         "criteria": [{"id": "c1", "passed": False,
                       "reason": "missed change-of-control clause"}]}
    )
    result = harness.refine(trajectory=[{"step": 1}], feedback=fb)

    recs = log.records()
    assert len(recs) == 1
    rec = recs[0]
    assert rec.edit_id == "checklist.change_of_control"
    assert rec.op == "create"
    assert rec.from_version == result.from_version
    assert rec.to_version == result.to_version
    assert rec.cause_task_id == "ma-001"
    assert "change-of-control" in rec.cause_failures
    assert rec.score == fb.score


def test_refine_without_log_is_unchanged(tmp_path):
    backend = JsonMemoryBackend(str(tmp_path / "state.json"))
    harness = ContinualHarness(backend)  # no audit_log
    harness.propose = _StubPropose(
        '[{"op":"create","id":"n1","kind":"note","text":"x","scope":"global"}]'
    )
    fb = parse_lab_result({"task_id": "t", "criteria": []})
    # must not raise, returns a normal RefineResult
    result = harness.refine(trajectory=[], feedback=fb)
    assert result.created == ["n1"]


class _StubVerifier:
    """Admits an edit iff its text contains `needle`; carries a justification."""

    def __init__(self, needle, score=1.0):
        self._needle = needle
        self._score = score

    def verify(self, op, feedback, trajectory):
        from sentinelprime.verifier import VerifierVerdict

        admitted = self._needle in op.get("text", "")
        return VerifierVerdict(
            admitted=admitted,
            score=self._score,
            justification=f"{'admit' if admitted else 'reject'}: looked for {self._needle!r}",
        )


def test_refine_rejects_unverified_edit(tmp_path):
    backend = JsonMemoryBackend(str(tmp_path / "state.json"))
    log = AuditLog(str(tmp_path / "audit.json"))
    harness = ContinualHarness(backend, audit_log=log, verifier=_StubVerifier("keep"))
    harness.propose = _StubPropose(
        '[{"op":"create","id":"good","kind":"note","text":"keep this","scope":"global"},'
        ' {"op":"create","id":"bad","kind":"note","text":"drop this","scope":"global"}]'
    )
    fb = parse_lab_result({"task_id": "t", "criteria": []})
    result = harness.refine(trajectory=[], feedback=fb)

    # only the admitted edit is applied
    assert result.created == ["good"]
    assert {i.id for i in backend.read()} == {"good"}
    # and only the admitted edit produced an audit record
    assert [r.edit_id for r in log.records()] == ["good"]


def test_refine_writes_verification_into_record(tmp_path):
    backend = JsonMemoryBackend(str(tmp_path / "state.json"))
    log = AuditLog(str(tmp_path / "audit.json"))
    harness = ContinualHarness(backend, audit_log=log, verifier=_StubVerifier("keep"))
    harness.propose = _StubPropose(
        '[{"op":"create","id":"good","kind":"note","text":"keep this","scope":"global"}]'
    )
    fb = parse_lab_result({"task_id": "t", "criteria": []})
    result = harness.refine(trajectory=[], feedback=fb)

    rec = log.records()[0]
    assert "admit" in rec.verification
    out = harness.explain(result.to_version)
    assert "verification" in out
    assert "admit" in out
