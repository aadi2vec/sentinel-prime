"""The proposal log: the training record the compliance ledger deliberately refuses to be.

`AuditLog` records only *admitted* edits — a rejected edit is not a fact about the ledger,
so it produces no audit record. That invariant is what makes the audit trail meaningful,
and it is also what makes the audit log useless as a trainset: every row is a positive.
`ProposalLog` is the other half — every proposal the verifier judged, admitted or not.
"""
from sentinelprime.proposals import ProposalLog, ProposalRecord, proposal_id


def _rec(**over):
    base = dict(
        proposal_id="p1",
        task_id="ma-001",
        op={"op": "create", "id": "lesson.coc", "kind": "note",
            "text": "always extract the change of control clause"},
        rubric_failures="- [c1] missed the change of control clause",
        trajectory_digest="abc",
        trajectory_summary="[]",
        admitted=True,
        justification="grounding: 0.42 overlap",
        rungs={"grounding_probe": {"admitted": True, "score": 0.42,
                                   "justification": "grounding: 0.42 overlap"}},
        created_at="t",
    )
    base.update(over)
    return ProposalRecord(**base)


def test_append_and_roundtrip(tmp_path):
    log = ProposalLog(str(tmp_path / "proposals.jsonl"))
    log.append(_rec())
    reloaded = ProposalLog(str(tmp_path / "proposals.jsonl"))
    assert reloaded.records()[0].op["text"].startswith("always extract")


def test_records_rejections_which_the_audit_log_never_sees(tmp_path):
    # The reason this class exists: a trainset needs the negatives.
    log = ProposalLog(str(tmp_path / "p.jsonl"))
    log.append(_rec(proposal_id="p1", admitted=True))
    log.append(_rec(proposal_id="p2", admitted=False,
                    justification="grounding: 0.05 overlap, below 0.30"))
    assert [r.admitted for r in log.records()] == [True, False]


def test_dedups_by_proposal_id(tmp_path):
    log = ProposalLog(str(tmp_path / "p.jsonl"))
    log.append(_rec(proposal_id="p1"))
    log.append(_rec(proposal_id="p1"))
    assert len(log.records()) == 1


def test_proposal_id_is_content_addressed_per_task(tmp_path):
    op = {"op": "create", "id": "x", "text": "same text"}
    assert proposal_id(op, "ma-001") == proposal_id(dict(op), "ma-001")
    assert proposal_id(op, "ma-001") != proposal_id(op, "ma-002")


def test_survives_a_truncated_trailing_line(tmp_path):
    # A sweep killed mid-write leaves a partial line. The trainset must still load.
    path = tmp_path / "p.jsonl"
    log = ProposalLog(str(path))
    log.append(_rec(proposal_id="p1"))
    with open(path, "a") as f:
        f.write('{"proposal_id": "p2", "task_id": "ma-0')
    assert len(ProposalLog(str(path)).records()) == 1


def test_outcomes_are_appended_not_edited_and_last_wins(tmp_path):
    # Append-only: a later outcome for the same proposal supersedes without rewriting.
    log = ProposalLog(str(tmp_path / "p.jsonl"))
    log.append(_rec(proposal_id="p1"))
    log.append_outcome("p1", edit_id="lesson.coc", targets=["c1"], exposures=2, successes=1)
    log.append_outcome("p1", edit_id="lesson.coc", targets=["c1"], exposures=4, successes=3)
    outcomes = ProposalLog(str(tmp_path / "p.jsonl")).outcomes()
    assert outcomes["p1"].exposures == 4
    assert outcomes["p1"].successes == 3


def test_outcomes_do_not_appear_as_proposals(tmp_path):
    log = ProposalLog(str(tmp_path / "p.jsonl"))
    log.append(_rec(proposal_id="p1"))
    log.append_outcome("p1", edit_id="e", targets=[], exposures=1, successes=1)
    assert len(log.records()) == 1


def test_for_edit_returns_the_most_recent_proposal_for_a_ledger_id(tmp_path):
    log = ProposalLog(str(tmp_path / "p.jsonl"))
    log.append(_rec(proposal_id="p1", task_id="ma-001"))
    log.append(_rec(proposal_id="p2", task_id="ma-002"))
    assert log.for_edit("lesson.coc").task_id == "ma-002"


def test_for_edit_is_none_for_an_unknown_ledger_id(tmp_path):
    log = ProposalLog(str(tmp_path / "p.jsonl"))
    assert log.for_edit("never-proposed") is None
