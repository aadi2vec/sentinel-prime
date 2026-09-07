import dspy

from sentinelprime.verifier import PredictVerifier, VerifierVerdict
from sentinelprime.feedback import parse_lab_result


class _StubVerify:
    """Stands in for the dspy.Predict verifier: returns a canned verdict."""

    def __init__(self, admit: str, score: str, justification: str):
        self._admit = admit
        self._score = score
        self._justification = justification

    def __call__(self, **kwargs):
        return dspy.Prediction(
            admit=self._admit, score=self._score, justification=self._justification
        )


def _fb():
    return parse_lab_result(
        {"task_id": "t", "criteria": [{"id": "c1", "passed": False, "reason": "boom"}]}
    )


def _op():
    return {"op": "create", "id": "n1", "kind": "note", "text": "learned", "scope": "global"}


def test_verifier_admits_edit_above_threshold():
    v = PredictVerifier(min_score=0.5)
    v.verify_predict = _StubVerify("yes", "0.9", "edit is grounded in the failure")
    verdict = v.verify(_op(), _fb(), trajectory=[])
    assert isinstance(verdict, VerifierVerdict)
    assert verdict.admitted is True
    assert verdict.score == 0.9
    assert "grounded" in verdict.justification


def test_verifier_rejects_edit_below_threshold():
    v = PredictVerifier(min_score=0.5)
    v.verify_predict = _StubVerify("yes", "0.2", "weak support for this edit")
    verdict = v.verify(_op(), _fb(), trajectory=[])
    assert verdict.admitted is False
    assert verdict.score == 0.2


def test_verifier_rejects_when_admit_is_no():
    v = PredictVerifier(min_score=0.0)
    v.verify_predict = _StubVerify("no", "0.9", "edit contradicts the task")
    verdict = v.verify(_op(), _fb(), trajectory=[])
    assert verdict.admitted is False


def test_verifier_exposes_predictor_to_gepa():
    v = PredictVerifier()
    names = dict(v.named_predictors())
    assert any("verify" in n for n in names)
