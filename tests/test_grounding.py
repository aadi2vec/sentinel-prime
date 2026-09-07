"""Grounding: did the run actually *read* the evidence, or did it guess well?"""
import pytest

from sentinelprime.grounding import (CoverageReport, anchors, criterion_grounding,
                                     document_coverage, document_names, grounding_map)


def _step(reasoning="", code="", output=""):
    return {"reasoning": reasoning, "code": code, "output": output}


# --- document coverage -------------------------------------------------------------

def test_document_named_in_repl_code_counts_as_touched():
    traj = [_step(code="open('documents/merger.txt').read()", output="...")]
    report = document_coverage(traj, ["merger.txt", "escrow.txt"])
    assert report.touched == ["merger.txt"]
    assert report.untouched == ["escrow.txt"]


def test_document_appearing_only_in_repl_output_counts_as_touched():
    """Name-mention is the proxy, so a listing counts too — permissive by design."""
    traj = [_step(code="import os; os.listdir('documents')",
                  output="['merger.txt', 'escrow.txt']")]
    report = document_coverage(traj, ["merger.txt", "escrow.txt"])
    assert report.touched == ["merger.txt", "escrow.txt"]


def test_untouched_documents_are_reported_with_a_rate():
    traj = [_step(code="read('a.txt')")]
    report = document_coverage(traj, ["a.txt", "b.txt", "c.txt"])
    assert report.untouched == ["b.txt", "c.txt"]
    assert report.rate == pytest.approx(1 / 3)


def test_coverage_of_no_documents_is_complete_not_a_division_error():
    report = document_coverage([], [])
    assert report.rate == 1.0
    assert report.untouched == []


def test_document_names_reads_a_directory(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    assert document_names(tmp_path) == ["a.txt", "b.txt"]


def test_document_names_of_a_missing_directory_is_empty(tmp_path):
    assert document_names(tmp_path / "nope") == []


# --- anchors -----------------------------------------------------------------------

def test_anchors_extract_amounts_sections_and_durations():
    found = anchors("Identifies the $12.5M escrow cap in § 7.3 and the 30-day cure period")
    assert "$12.5m" in found
    assert "§7.3" in found
    assert "30day" in found


def test_anchors_normalize_separators_so_formatting_does_not_break_matching():
    assert anchors("$1,200,000") == anchors("$1200000")
    assert anchors("Section 7.3") == anchors("§ 7.3")


def test_prose_without_checkable_facts_yields_no_anchors():
    assert anchors("The analysis is well reasoned and clearly written") == set()


# --- criterion grounding -----------------------------------------------------------

def test_criterion_is_grounded_when_its_anchors_appear_in_repl_output():
    traj = [_step(code="print(open('merger.txt').read())",
                  output="Escrow Amount. The parties shall deposit $12,500,000 ...")]
    report = criterion_grounding(traj, "identifies the $12,500,000 escrow cap")
    assert report.grounded is True
    assert report.missing == []


def test_criterion_is_ungrounded_when_the_anchor_only_appears_in_model_text():
    """The lucky guess: the number is in the model's reasoning, never in what it read."""
    traj = [_step(reasoning="the escrow is probably $12,500,000",
                  code="print('escrow cap is $12,500,000')",
                  output="escrow cap is $12,500,000".replace("$12,500,000", "..."))]
    report = criterion_grounding(traj, "identifies the $12,500,000 escrow cap")
    assert report.grounded is False
    assert report.missing == ["$12500000"]


def test_criterion_with_no_checkable_anchors_is_not_judged():
    """Unknown is not the same as ungrounded — it must not cost a lesson its credit."""
    report = criterion_grounding([_step(output="anything")], "the memo is well organized")
    assert report.grounded is None


def test_grounding_ignores_an_empty_trajectory_but_still_reports_the_anchors():
    report = criterion_grounding([], "cites § 7.3")
    assert report.grounded is False
    assert report.anchors == ["§7.3"]


def test_grounding_map_keys_criteria_by_id():
    traj = [_step(output="the cure period is 30 days")]
    m = grounding_map(traj, {"c1": "notes the 30-day cure period",
                             "c2": "cites § 9.1",
                             "c3": "the memo reads clearly"})
    assert m == {"c1": True, "c2": False, "c3": None}
