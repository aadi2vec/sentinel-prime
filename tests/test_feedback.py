from sentinelprime.feedback import parse_lab_result, Feedback, CriterionResult


SAMPLE = {
    "task_id": "ma-001",
    "criteria": [
        {"id": "c1", "passed": True, "reason": "cited clause 4.2"},
        {"id": "c2", "passed": False, "reason": "missed change-of-control provision"},
        {"id": "c3", "passed": False, "reason": "no source for indemnity cap"},
    ],
}


def test_parse_produces_feedback_with_fractional_score():
    fb = parse_lab_result(SAMPLE)
    assert isinstance(fb, Feedback)
    assert fb.task_id == "ma-001"
    assert abs(fb.score - (1 / 3)) < 1e-9
    assert len(fb.criteria) == 3


def test_failures_returns_only_failed_criteria():
    fb = parse_lab_result(SAMPLE)
    failed_ids = [c.id for c in fb.failures]
    assert failed_ids == ["c2", "c3"]
    assert all(isinstance(c, CriterionResult) for c in fb.failures)


def test_as_text_lists_failure_reasons():
    fb = parse_lab_result(SAMPLE)
    text = fb.as_text()
    assert "change-of-control" in text
    assert "indemnity cap" in text
    assert "cited clause 4.2" not in text  # passing criteria excluded


def test_empty_criteria_scores_zero_without_error():
    fb = parse_lab_result({"task_id": "x", "criteria": []})
    assert fb.score == 0.0
    assert fb.failures == []
