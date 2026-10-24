"""Phase 4: valid-time provenance on ledger items.

The claim under test is the one CLAUDE.md's invariant 8 already makes — "prove the guidance
was valid at the time" — which the ledger could not actually answer, because retirement
deleted the row. These tests pin the two halves of the fix: an out-of-force lesson is
invisible to the online loop (no behavioural regression against delete), and it is still
answerable to a historical query (the thing delete made impossible).
"""
import json

import dspy

from sentinelprime.audit import AuditLog, AuditRecord
from sentinelprime.credit import CreditAssigner
from sentinelprime.feedback import parse_lab_result
from sentinelprime.harness import ContinualHarness
from sentinelprime.memory import JsonMemoryBackend, MemoryItem


def _backend(tmp_path):
    return JsonMemoryBackend(str(tmp_path / "h.json"))


def _item(id="n1", text="always check change-of-control", **kw):
    base = dict(id=id, scope="global", kind="note", text=text,
                created_at="2026-01-01", meta={})
    base.update(kw)
    return MemoryItem(**base)


# --- the fields themselves -------------------------------------------------------------

def test_item_defaults_to_in_force_with_no_valid_time_set():
    item = _item()
    assert item.valid_at == ""
    assert item.invalid_at == ""


def test_item_with_no_valid_at_falls_back_to_created_at():
    # An item seeded before this field existed has no valid_at, and must not thereby
    # become out of force everywhere: created_at is the best available start.
    item = _item(created_at="2026-01-01")
    assert item.in_force_at("2026-06-01")
    assert not item.in_force_at("2025-06-01")


def test_ledger_persisted_without_valid_time_still_loads(tmp_path):
    # Back-compat, same shape as AuditRecord.machinery: the fields are additive, so a
    # ledger written before Phase 4 must round-trip rather than raise.
    path = str(tmp_path / "old.json")
    with open(path, "w") as f:
        json.dump({"version": 1, "parent_version": 0, "history": {}, "items": [
            {"id": "n1", "scope": "global", "kind": "note", "text": "old",
             "created_at": "2026-01-01", "meta": {}}]}, f)
    items = JsonMemoryBackend(path).read()
    assert [i.id for i in items] == ["n1"]
    assert items[0].invalid_at == ""


def test_interval_is_half_open_so_consecutive_intervals_do_not_overlap():
    # In force at valid_at, out of force at invalid_at. Without this an item and its
    # replacement are both in force for one instant and a historical read sees two.
    item = _item(valid_at="2026-03-01", invalid_at="2026-06-01")
    assert item.in_force_at("2026-03-01")
    assert item.in_force_at("2026-05-31")
    assert not item.in_force_at("2026-06-01")
    assert not item.in_force_at("2026-02-28")


# --- reader 1: the online loop cannot see an out-of-force lesson -----------------------

def test_read_omits_an_out_of_force_item(tmp_path):
    be = _backend(tmp_path)
    be.write([_item(id="live", text="live-guidance"),
              _item(id="dead", text="dead-guidance", invalid_at="2026-02-01")])
    block = ContinualHarness(be).read()
    assert "live-guidance" in block
    assert "dead-guidance" not in block


def test_admissible_items_omits_out_of_force_regardless_of_context(tmp_path):
    # Valid time is not reuse gating: an out-of-force lesson is out of force with no
    # context supplied at all, so the filter cannot live behind the `context is None`
    # early return.
    be = _backend(tmp_path)
    be.write([_item(id="dead", invalid_at="2026-02-01")])
    h = ContinualHarness(be)
    assert h.admissible_items() == []
    assert h.admissible_items(context={"task_id": "t1"}) == []


def test_read_returns_empty_string_when_every_item_is_out_of_force(tmp_path):
    # Invariant 1: an empty ledger emits "" so nothing is prepended. A ledger whose rows
    # all sit outside their validity window is an empty ledger as far as the prompt goes.
    be = _backend(tmp_path)
    be.write([_item(id="dead", invalid_at="2026-02-01")])
    assert ContinualHarness(be).read() == ""


# --- reader 2: the historical query delete made impossible ----------------------------

def test_in_force_at_recovers_guidance_that_has_since_been_invalidated(tmp_path):
    be = _backend(tmp_path)
    be.write([_item(id="dead", text="was-in-force", valid_at="2026-01-01",
                    invalid_at="2026-06-01")])
    h = ContinualHarness(be)
    assert [i.id for i in h.in_force_at("2026-03-01")] == ["dead"]
    assert h.in_force_at("2026-09-01") == []


def test_in_force_at_omits_guidance_that_had_not_come_into_force_yet(tmp_path):
    be = _backend(tmp_path)
    be.write([_item(id="later", valid_at="2026-06-01")])
    h = ContinualHarness(be)
    assert h.in_force_at("2026-03-01") == []
    assert [i.id for i in h.in_force_at("2026-07-01")] == ["later"]


def test_in_force_at_does_not_mutate_the_live_ledger(tmp_path):
    # The whole point: answering "what was in force then" used to require rollback(),
    # which mutates the ledger to answer a read-only question.
    be = _backend(tmp_path)
    be.write([_item(id="live"), _item(id="dead", invalid_at="2026-02-01")])
    h = ContinualHarness(be)
    h.in_force_at("2026-01-15")
    assert {i.id for i in be.read()} == {"live", "dead"}
    assert [i.id for i in h.admissible_items()] == ["live"]


# --- retirement closes the interval; delete still deletes -----------------------------

class _StubPropose:
    def __init__(self, edits):
        self.edits = edits

    def __call__(self, **kwargs):
        return dspy.Prediction(edits=json.dumps(self.edits))


def _failing_feedback(task_id="t1"):
    return parse_lab_result({"task_id": task_id, "score": 0.0, "criteria": [
        {"id": "c1", "passed": False, "note": "missed the indemnity cap"}]})


def _retirable_harness(tmp_path, item_id="stale"):
    be = _backend(tmp_path)
    be.write([_item(id=item_id, text="stale-guidance")])
    audit = AuditLog(str(tmp_path / "audit.json"))
    audit.append(AuditRecord(
        edit_id=item_id, op="create", scope="intrinsic", from_version=0, to_version=1,
        cause_task_id="t0", cause_failures="c1 failed", trajectory_digest="",
        score=0.0, created_at="2026-01-01",
        content_hash="h", targets=["c1"]))
    credit = CreditAssigner(audit, window=10, min_exposures=2, min_success_rate=0.5)
    h = ContinualHarness(be, audit_log=audit, credit_assigner=credit)
    h.propose = _StubPropose([])
    # Two exposures, both failures -> retirable.
    for i in range(2):
        credit.observe([item_id], _failing_feedback(f"t{i}"))
    return be, h, credit


def test_retirement_invalidates_rather_than_deleting(tmp_path):
    be, h, _ = _retirable_harness(tmp_path)
    result = h.refine([], _failing_feedback())
    assert result.retired == ["stale"]
    # The row survives -- that is the whole change.
    rows = {i.id: i for i in be.read()}
    assert "stale" in rows
    assert rows["stale"].invalid_at != ""
    # ...and is invisible to the online loop, exactly as deleting it was.
    assert h.read() == ""


def test_retirement_is_undone_by_rolling_back_the_round(tmp_path):
    # Invariant 2 one level down: closing an interval is as reversible as deleting was.
    be, h, _ = _retirable_harness(tmp_path)
    result = h.refine([], _failing_feedback())
    be.rollback(result.from_version)
    rows = {i.id: i for i in be.read()}
    assert rows["stale"].invalid_at == ""
    assert "stale-guidance" in h.read()


def test_retirement_does_not_re_retire_an_already_out_of_force_item(tmp_path):
    # `present` used to mean "in the ledger"; with invalidation that is no longer the same
    # as "in force", and retiring twice would emit a second audit record — under a new
    # cause_task_id, so a new content_hash, so no dedup — for one departure.
    #
    # `retirable()` is stubbed to name the lesson unconditionally rather than driven through
    # observations, because the guard under test is `_retire`'s and must hold however the
    # assigner came to name the id. Driving it through `observe()` tests nothing here: the
    # first retirement makes the retire record the latest one for that id, which changes what
    # `targets()` resolves to, so the second round declines for an unrelated reason.
    be, h, credit = _retirable_harness(tmp_path)
    credit.retirable = lambda: ["stale"]
    assert h.refine([], _failing_feedback("first")).retired == ["stale"]
    assert h.refine([], _failing_feedback("second")).retired == []
    assert [r.edit_id for r in h.audit_log.records() if r.op == "retire"] == ["stale"]


def test_proposed_delete_still_deletes(tmp_path):
    # Retirement and delete are different claims and must stay different operations:
    # retirement says "this stopped working over an interval", delete says "this is
    # wrong", which has no interval to close.
    be = _backend(tmp_path)
    be.write([_item(id="wrong")])
    h = ContinualHarness(be)
    h._apply_edits([{"op": "delete", "id": "wrong"}])
    assert be.read() == []


def test_update_to_an_out_of_force_item_brings_it_back_into_force(tmp_path):
    # The mirror of credit's forget()-on-retire rule. A retired lesson now leaves a row
    # behind and forget() clears its blame, so a rewrite gets a fresh trial -- but if the
    # rewrite inherited the closed interval it would sit in the ledger, unblamed, and
    # never be read again.
    be = _backend(tmp_path)
    be.write([_item(id="n1", text="old-text", invalid_at="2026-02-01")])
    h = ContinualHarness(be)
    created, updated, deleted = h._apply_edits([
        {"op": "update", "id": "n1", "kind": "note", "text": "rewritten", "scope": "global"}])
    assert updated == ["n1"]
    rows = {i.id: i for i in be.read()}
    assert rows["n1"].invalid_at == ""
    assert rows["n1"].valid_at != ""
    assert "rewritten" in h.read()


# --- the compliance chain can now name the guidance, not just the edit ----------------

def test_explain_names_the_guidance_that_was_in_force_at_the_edit(tmp_path):
    be = _backend(tmp_path)
    be.write([_item(id="standing", text="standing-guidance")])
    audit = AuditLog(str(tmp_path / "audit.json"))
    h = ContinualHarness(be, audit_log=audit)
    h.propose = _StubPropose([{"op": "create", "id": "fresh", "kind": "note",
                               "text": "new lesson", "scope": "global"}])
    result = h.refine([], _failing_feedback())
    chain = h.explain(result.to_version)
    assert "[guidance" in chain
    # The *text*, not the id: an id already appears on the `[ledger_edit]` line, so
    # asserting on it would pass without the guidance line existing at all.
    assert "standing-guidance" in chain


def test_explain_still_names_guidance_that_has_since_left_force(tmp_path):
    # The payoff of the whole phase. Under deletion this was unanswerable: the row that
    # would prove what the run was told had been destroyed by the time anyone asked.
    be, h, credit = _retirable_harness(tmp_path)
    credit.retirable = lambda: ["stale"]
    result = h.refine([], _failing_feedback())
    # It is out of force now...
    assert h.read() == ""
    # ...and the compliance chain for that round still quotes the guidance itself. The id
    # alone would not show this: `[ledger_edit] retire 'stale'` carries it either way, and
    # under deletion the *text* was exactly what became unrecoverable.
    assert "stale-guidance" in h.explain(result.to_version)


def test_explain_degrades_to_its_old_output_with_no_audit_log(tmp_path):
    # Invariant 7: the audit log is optional, so the guidance line cannot make it required.
    h = ContinualHarness(_backend(tmp_path))
    assert h.explain(1) == "(no audit log configured)"


def test_in_force_at_reports_a_gap_rather_than_the_wrong_text_before_a_rewrite(tmp_path):
    # The honest limit of storing valid time on the row: one interval per id, not a
    # history of them. A rewrite overwrites the text and opens a new interval, so the
    # period before it becomes unanswerable *here*.
    #
    # What matters is which way it fails. Because the rewrite also moves `valid_at`
    # forward, a query before it returns nothing rather than returning the new text
    # labelled as having been in force then — a gap, not a false statement. The
    # transaction history is still in the audit log, which is the half of the pair that
    # keeps every version.
    be = _backend(tmp_path)
    be.write([_item(id="n1", text="first-text", valid_at="2026-01-01")])
    h = ContinualHarness(be)
    assert [i.text for i in h.in_force_at("2026-02-01")] == ["first-text"]
    h._apply_edits([{"op": "update", "id": "n1", "kind": "note", "text": "second-text",
                     "scope": "global"}])
    assert [i.text for i in h.in_force_at("2026-02-01")] == []
    assert [i.text for i in h.in_force_at("2099-01-01")] == ["second-text"]


def test_a_lesson_rewritten_and_retired_in_one_round_leaves_an_empty_interval(tmp_path):
    # `refine()` runs `_apply_edits` before `_retire`, so a lesson the proposer rewrote and
    # credit retired in the same round comes into force and leaves it within that round.
    # Pinned because reordering those two calls would invert it: retiring first and then
    # rewriting would leave the lesson in force with its blame already forgotten.
    be, h, credit = _retirable_harness(tmp_path)
    credit.retirable = lambda: ["stale"]
    h.propose = _StubPropose([{"op": "update", "id": "stale", "kind": "note",
                               "text": "rewritten", "scope": "global"}])
    result = h.refine([], _failing_feedback())
    assert result.updated == ["stale"]
    assert result.retired == ["stale"]
    row = be.read()[0]
    assert row.text == "rewritten"
    assert row.valid_at <= row.invalid_at != ""
    assert h.read() == ""


def test_created_item_records_when_it_came_into_force(tmp_path):
    be = _backend(tmp_path)
    h = ContinualHarness(be)
    h._apply_edits([{"op": "create", "id": "n1", "kind": "note", "text": "t",
                     "scope": "global"}])
    item = be.read()[0]
    assert item.valid_at != ""
    assert item.in_force_at(item.valid_at)
