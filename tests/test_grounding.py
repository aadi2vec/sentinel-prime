"""Document coverage: which documents did the run never open?"""
import pytest

from sentinelprime.grounding import document_coverage, document_names


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
