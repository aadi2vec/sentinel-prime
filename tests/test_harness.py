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


def test_admissible_items_reports_what_context_gating_withheld(tmp_path):
    """read() renders a string; a benchmark needs to count what the gate dropped."""
    be = _backend(tmp_path)
    _seed(be, id="intrinsic1", text="global lesson", scope="global")
    be.write([MemoryItem(id="ext1", scope="session", kind="note", text="revision-specific",
                         created_at="t", meta={})])
    log = AuditLog(str(tmp_path / "audit.json"))
    log.append(AuditRecord(
        edit_id="ext1", op="create", scope="external", from_version=0, to_version=1,
        cause_task_id="t", cause_failures="f", trajectory_digest="d", score=1.0,
        created_at="t", content_hash="h", depends_on={"doc_sha": "aaa"},
    ))
    h = ContinualHarness(be, audit_log=log)

    assert {i.id for i in h.admissible_items(context={"doc_sha": "aaa"})} == {"intrinsic1", "ext1"}
    assert {i.id for i in h.admissible_items(context={"doc_sha": "bbb"})} == {"intrinsic1"}


def test_admissible_items_is_ungated_without_context(tmp_path):
    be = _backend(tmp_path)
    _seed(be, id="n1")
    h = ContinualHarness(be)
    assert [i.id for i in h.admissible_items()] == ["n1"]


# --- proposal recording + machinery provenance ---------------------------------------

def _stub_propose(h, edits):
    # Stub the LM *call*, not the predictor object: replacing h.propose outright would
    # delete the dspy.Predict that named_predictors() (and the machinery fingerprint)
    # depend on, which is exactly the seam under test here.
    h.propose.forward = lambda **kw: type("P", (), {"edits": json.dumps(edits)})()


def _fb2(passed=False):
    return parse_lab_result({"task_id": "ma-001", "criteria": [
        {"id": "c1", "passed": passed, "reason": "missed the change of control clause"}]})


class _GateOn:
    """Admits only edits whose text contains 'good'."""
    def verify(self, op, feedback, trajectory):
        from sentinelprime.verifier import VerifierVerdict
        ok = "good" in op.get("text", "")
        return VerifierVerdict(ok, 1.0 if ok else 0.0, "ok" if ok else "nope")


def test_refine_records_rejected_proposals_the_audit_log_never_sees(tmp_path):
    from sentinelprime.proposals import ProposalLog
    plog = ProposalLog(str(tmp_path / "p.jsonl"))
    h = ContinualHarness(_backend(tmp_path), audit_log=AuditLog(str(tmp_path / "a.json")),
                         verifier=_GateOn(), proposal_log=plog)
    _stub_propose(h, [
        {"op": "create", "id": "g", "kind": "note", "text": "good lesson", "scope": "global"},
        {"op": "create", "id": "b", "kind": "note", "text": "bad lesson", "scope": "global"},
    ])
    result = h.refine([], _fb2())
    assert result.rejected == ["b"]
    # the audit log holds only the admitted edit; the proposal log holds both
    assert [r.edit_id for r in h.audit_log.records()] == ["g"]
    assert {r.op["id"]: r.admitted for r in plog.records()} == {"g": True, "b": False}


def test_recorded_proposal_carries_the_text_needed_to_replay_the_verification(tmp_path):
    from sentinelprime.proposals import ProposalLog
    plog = ProposalLog(str(tmp_path / "p.jsonl"))
    h = ContinualHarness(_backend(tmp_path), verifier=_GateOn(), proposal_log=plog)
    _stub_propose(h, [{"op": "create", "id": "b", "kind": "note",
                       "text": "bad lesson", "scope": "global"}])
    h.refine([{"output": "obs"}], _fb2())
    rec = plog.records()[0]
    assert rec.op["text"] == "bad lesson"
    assert "change of control" in rec.rubric_failures
    assert rec.task_id == "ma-001"


def test_proposals_are_recorded_even_with_no_verifier_configured(tmp_path):
    # The probe supplies the label at training time, so an ungated round is still data.
    from sentinelprime.proposals import ProposalLog
    plog = ProposalLog(str(tmp_path / "p.jsonl"))
    h = ContinualHarness(_backend(tmp_path), proposal_log=plog)
    _stub_propose(h, [{"op": "create", "id": "x", "kind": "note", "text": "t",
                       "scope": "global"}])
    h.refine([], _fb2())
    assert [r.admitted for r in plog.records()] == [True]


def test_ladder_rung_verdicts_reach_the_proposal_record(tmp_path):
    from sentinelprime.proposals import ProposalLog
    from sentinelprime.verifier import GroundingProbe, LadderVerifier, VerifierLevel
    plog = ProposalLog(str(tmp_path / "p.jsonl"))
    ladder = LadderVerifier([VerifierLevel("grounding_probe", 1, GroundingProbe())])
    h = ContinualHarness(_backend(tmp_path), verifier=ladder, proposal_log=plog)
    _stub_propose(h, [{"op": "create", "id": "x", "kind": "note",
                       "text": "missed the change of control clause", "scope": "global"}])
    h.refine([], _fb2())
    assert "grounding_probe" in plog.records()[0].rungs


def test_audit_records_are_stamped_with_the_machinery_that_admitted_them(tmp_path):
    from sentinelprime.audit import machinery_fingerprint
    h = ContinualHarness(_backend(tmp_path), audit_log=AuditLog(str(tmp_path / "a.json")))
    _stub_propose(h, [{"op": "create", "id": "x", "kind": "note", "text": "t",
                       "scope": "global"}])
    h.refine([], _fb2())
    assert h.audit_log.records()[0].machinery == machinery_fingerprint(h)


def test_a_tuned_proposer_produces_a_different_machinery_stamp(tmp_path):
    # The point of the stamp: two records that look identical came from different bars.
    stamps = []
    for n, instructions in enumerate(("original", "tuned by GEPA")):
        h = ContinualHarness(JsonMemoryBackend(str(tmp_path / f"led-{n}.json")),
                             audit_log=AuditLog(str(tmp_path / f"a-{n}.json")))
        h.propose.signature = h.propose.signature.with_instructions(instructions)
        _stub_propose(h, [{"op": "create", "id": "x", "kind": "note", "text": "t",
                           "scope": "global"}])
        h.refine([], _fb2())
        stamps.append(h.audit_log.records()[0].machinery)
    assert stamps[0] != stamps[1]


def test_no_proposal_log_configured_leaves_refine_unchanged(tmp_path):
    # Invariant 7: optional collaborators degrade to base behavior.
    h = ContinualHarness(_backend(tmp_path))
    _stub_propose(h, [{"op": "create", "id": "x", "kind": "note", "text": "t",
                       "scope": "global"}])
    assert h.refine([], _fb2()).created == ["x"]


def test_proposal_records_the_ledger_the_proposer_actually_saw(tmp_path):
    # current_ledger is the proposer's third input. Without it the offline trainset trains
    # on a different prompt than the online loop ran, which is not a replay.
    from sentinelprime.proposals import ProposalLog
    be = _backend(tmp_path)
    _seed(be, id="n1", text="already learned this")
    plog = ProposalLog(str(tmp_path / "p.jsonl"))
    h = ContinualHarness(be, proposal_log=plog)
    _stub_propose(h, [{"op": "create", "id": "x", "kind": "note", "text": "t",
                       "scope": "global"}])
    h.refine([], _fb2())
    assert "already learned this" in plog.records()[0].current_ledger


def test_credit_writes_the_downstream_outcome_back_to_the_proposal(tmp_path):
    # Stratum 2 of the trainset. Without this the only labels are probe rejections, and
    # the judge is never trained on the distribution it actually faces.
    from sentinelprime.credit import CreditAssigner
    from sentinelprime.proposals import ProposalLog, ProposalRecord, proposal_id
    audit = AuditLog(str(tmp_path / "a.json"))
    audit.append(AuditRecord(
        edit_id="x", op="create", scope="intrinsic", from_version=1, to_version=2,
        cause_task_id="ma-001", cause_failures="- [c1] missed it", trajectory_digest="d",
        score=0.0, created_at="t", content_hash="h", targets=["c1"]))
    op = {"op": "create", "id": "x", "kind": "note", "text": "t", "scope": "global"}
    plog = ProposalLog(str(tmp_path / "p.jsonl"))
    plog.append(ProposalRecord(
        proposal_id=proposal_id(op, "ma-001"), task_id="ma-001", op=op,
        rubric_failures="- [c1] missed it", trajectory_digest="d", trajectory_summary="[]",
        admitted=True, justification="j"))
    h = ContinualHarness(_backend(tmp_path), audit_log=audit,
                         credit_assigner=CreditAssigner(audit), proposal_log=plog)
    h.credit(["x"], parse_lab_result({"task_id": "ma-002", "criteria": [
        {"id": "c1", "passed": True, "reason": "ok"}]}))
    outcome = plog.outcomes()[proposal_id(op, "ma-001")]
    assert (outcome.exposures, outcome.successes) == (1, 1)


def test_credit_without_a_proposal_log_is_unchanged(tmp_path):
    from sentinelprime.credit import CreditAssigner
    audit = AuditLog(str(tmp_path / "a.json"))
    h = ContinualHarness(_backend(tmp_path), audit_log=audit,
                         credit_assigner=CreditAssigner(audit))
    h.credit(["x"], _fb2(passed=True))   # must not raise


def test_credit_ignores_exposed_items_that_were_never_proposed_here(tmp_path):
    # Externally seeded ledger items have no proposal record; that is not an error.
    from sentinelprime.credit import CreditAssigner
    from sentinelprime.proposals import ProposalLog
    audit = AuditLog(str(tmp_path / "a.json"))
    plog = ProposalLog(str(tmp_path / "p.jsonl"))
    h = ContinualHarness(_backend(tmp_path), audit_log=audit,
                         credit_assigner=CreditAssigner(audit), proposal_log=plog)
    h.credit(["seeded"], _fb2(passed=True))
    assert plog.outcomes() == {}


def test_shadow_verdicts_reach_the_proposal_record_marked_as_shadow(tmp_path):
    # The judge's opinion on a probe-rejected edit is the row that makes stratum 1 less
    # off-distribution — but it must never be mistaken for part of the decision.
    from sentinelprime.proposals import ProposalLog
    from sentinelprime.verifier import (GroundingProbe, LadderVerifier, VerifierLevel,
                                        VerifierVerdict)

    class _Judge:
        def verify(self, op, feedback, trajectory):
            return VerifierVerdict(True, 0.9, "the judge would have admitted it")

    plog = ProposalLog(str(tmp_path / "p.jsonl"))
    ladder = LadderVerifier([VerifierLevel("grounding_probe", 1, GroundingProbe()),
                             VerifierLevel("llm_verifier", 100, _Judge())], explore=1.0)
    h = ContinualHarness(_backend(tmp_path), verifier=ladder, proposal_log=plog)
    _stub_propose(h, [{"op": "create", "id": "cake", "kind": "note",
                       "text": "bake a sponge cake at 180 degrees", "scope": "global"}])
    h.refine([], _fb2())
    rungs = plog.records()[0].rungs
    assert rungs["grounding_probe"]["shadow"] is False
    assert rungs["llm_verifier"]["shadow"] is True
    assert rungs["llm_verifier"]["admitted"] is True


def test_without_exploration_no_shadow_rungs_are_recorded(tmp_path):
    from sentinelprime.proposals import ProposalLog
    from sentinelprime.verifier import GroundingProbe, LadderVerifier, VerifierLevel

    class _Boom:
        def verify(self, op, feedback, trajectory):
            raise AssertionError("must not run without exploration")

    plog = ProposalLog(str(tmp_path / "p.jsonl"))
    ladder = LadderVerifier([VerifierLevel("grounding_probe", 1, GroundingProbe()),
                             VerifierLevel("llm_verifier", 100, _Boom())], explore=0.0)
    h = ContinualHarness(_backend(tmp_path), verifier=ladder, proposal_log=plog)
    _stub_propose(h, [{"op": "create", "id": "cake", "kind": "note",
                       "text": "bake a sponge cake at 180 degrees", "scope": "global"}])
    h.refine([], _fb2())
    assert list(plog.records()[0].rungs) == ["grounding_probe"]
