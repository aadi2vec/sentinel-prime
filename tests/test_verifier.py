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


# ---- LadderVerifier: cost-ladder ordering over multiple verifier levels --------------

class _RecordingLevel:
    """A verifier level that records that it ran and returns a fixed verdict."""

    def __init__(self, admitted: bool, calls: list, name: str, score: float = 1.0):
        self._verdict = VerifierVerdict(admitted, score, f"{name} says {admitted}")
        self._calls = calls
        self._name = name

    def verify(self, op, feedback, trajectory):
        self._calls.append(self._name)
        return self._verdict


def _ladder(levels):
    from sentinelprime.verifier import LadderVerifier
    return LadderVerifier(levels)


def test_ladder_runs_cheapest_detection_first_and_short_circuits():
    from sentinelprime.verifier import VerifierLevel

    calls: list[str] = []
    # Declared expensive-first; the ladder must still run the cheap probe first
    # and never reach the expensive level once the cheap one rejects.
    levels = [
        VerifierLevel("llm_verifier", cost=100.0,
                      verifier=_RecordingLevel(True, calls, "llm_verifier")),
        VerifierLevel("span_probe", cost=1.0,
                      verifier=_RecordingLevel(False, calls, "span_probe")),
    ]
    verdict = _ladder(levels).verify(_op(), _fb(), trajectory=[])

    assert calls == ["span_probe"]
    assert verdict.admitted is False


def test_ladder_records_executed_order_in_the_justification():
    from sentinelprime.verifier import VerifierLevel

    calls: list[str] = []
    levels = [
        VerifierLevel("llm_verifier", cost=100.0,
                      verifier=_RecordingLevel(True, calls, "llm_verifier")),
        VerifierLevel("span_probe", cost=1.0,
                      verifier=_RecordingLevel(True, calls, "span_probe")),
    ]
    verdict = _ladder(levels).verify(_op(), _fb(), trajectory=[])

    assert verdict.admitted is True
    # the audit trail must show which checks ran, in which order
    assert "span_probe" in verdict.justification
    assert verdict.justification.index("span_probe") < verdict.justification.index("llm_verifier")


def test_ladder_names_the_level_that_rejected():
    from sentinelprime.verifier import VerifierLevel

    calls: list[str] = []
    levels = [
        VerifierLevel("span_probe", cost=1.0,
                      verifier=_RecordingLevel(True, calls, "span_probe")),
        VerifierLevel("llm_verifier", cost=100.0,
                      verifier=_RecordingLevel(False, calls, "llm_verifier")),
    ]
    verdict = _ladder(levels).verify(_op(), _fb(), trajectory=[])

    assert calls == ["span_probe", "llm_verifier"]
    assert verdict.admitted is False
    assert "llm_verifier" in verdict.justification


def test_ladder_scores_the_admitted_edit_by_its_weakest_level():
    from sentinelprime.verifier import VerifierLevel

    calls: list[str] = []
    levels = [
        VerifierLevel("span_probe", cost=1.0,
                      verifier=_RecordingLevel(True, calls, "span_probe", score=0.9)),
        VerifierLevel("llm_verifier", cost=100.0,
                      verifier=_RecordingLevel(True, calls, "llm_verifier", score=0.6)),
    ]
    verdict = _ladder(levels).verify(_op(), _fb(), trajectory=[])
    assert verdict.score == 0.6


def test_ladder_orders_by_detection_per_dollar_not_raw_cost():
    """A pricier check that catches far more failures should still run first."""
    from sentinelprime.planner import CostLadderPlanner
    from sentinelprime.verifier import LadderVerifier, VerifierLevel

    calls: list[str] = []
    levels = [
        # cost 1, almost never catches anything -> cost/P(fail) = 1/0.01 = 100
        VerifierLevel("weak_probe", cost=1.0,
                      verifier=_RecordingLevel(True, calls, "weak_probe")),
        # cost 10, catches most failures -> cost/P(fail) = 10/0.9 = 11.1
        VerifierLevel("strong_probe", cost=10.0,
                      verifier=_RecordingLevel(True, calls, "strong_probe")),
    ]
    planner = CostLadderPlanner(stats={"weak_probe": 0.01, "strong_probe": 0.9})
    LadderVerifier(levels, planner=planner).verify(_op(), _fb(), trajectory=[])
    assert calls == ["strong_probe", "weak_probe"]


def test_ladder_exposes_nested_predictors_to_gepa():
    from sentinelprime.verifier import LadderVerifier, VerifierLevel

    ladder = LadderVerifier([VerifierLevel("llm_verifier", cost=10.0,
                                           verifier=PredictVerifier())])
    names = dict(ladder.named_predictors())
    assert any("verify" in n for n in names)


# ---- GroundingProbe: the deterministic, LM-free cheap rung of the ladder --------------

def _probe(**kw):
    from sentinelprime.verifier import GroundingProbe
    return GroundingProbe(**kw)


def _failure_fb(reason="missed change of control clause"):
    return parse_lab_result(
        {"task_id": "t", "criteria": [{"id": "c1", "passed": False, "reason": reason}]}
    )


def test_grounding_probe_admits_an_edit_that_reuses_the_failure_vocabulary():
    op = {"op": "create", "id": "n1", "kind": "note", "scope": "global",
          "text": "always extract the change of control clause"}
    verdict = _probe().verify(op, _failure_fb(), trajectory=[])
    assert verdict.admitted is True


def test_grounding_probe_rejects_an_edit_unrelated_to_any_observed_failure():
    op = {"op": "create", "id": "n1", "kind": "note", "scope": "global",
          "text": "prefer tabular output for board minutes"}
    verdict = _probe().verify(op, _failure_fb(), trajectory=[])
    assert verdict.admitted is False
    assert "grounding" in verdict.justification


def test_grounding_probe_rejects_a_malformed_edit_with_no_text():
    op = {"op": "create", "id": "n1", "kind": "note", "scope": "global", "text": "   "}
    assert _probe().verify(op, _failure_fb(), trajectory=[]).admitted is False


def test_grounding_probe_passes_delete_ops_through_to_the_next_rung():
    """A delete carries no text to ground; judging it is the generative rung's job."""
    verdict = _probe().verify({"op": "delete", "id": "n1"}, _failure_fb(), trajectory=[])
    assert verdict.admitted is True


def test_grounding_probe_costs_no_lm_call():
    """It must be usable as the cheap rung: pure computation, no predictor."""
    assert dict(_probe().named_predictors()) == {}


# ---- Adaptive ordering: the ladder's own rejections are its statistics catalog --------

def test_ladder_reports_observed_rejection_rates_per_rung():
    from sentinelprime.verifier import LadderVerifier, VerifierLevel

    calls: list[str] = []
    lad = LadderVerifier([
        VerifierLevel("permissive", cost=1.0,
                      verifier=_RecordingLevel(True, calls, "permissive")),
        VerifierLevel("strict", cost=1.0,
                      verifier=_RecordingLevel(False, calls, "strict")),
    ])
    lad.verify(_op(), _fb(), trajectory=[])
    assert lad.observed_stats() == {"permissive": 0.0, "strict": 1.0}


def test_adaptive_ladder_promotes_the_rung_that_actually_rejects():
    """Equal cost, so only observed detection rate can reorder them."""
    from sentinelprime.verifier import LadderVerifier, VerifierLevel

    calls: list[str] = []
    lad = LadderVerifier([
        VerifierLevel("permissive", cost=1.0,
                      verifier=_RecordingLevel(True, calls, "permissive")),
        VerifierLevel("strict", cost=1.0,
                      verifier=_RecordingLevel(False, calls, "strict")),
    ], adaptive=True)

    lad.verify(_op(), _fb(), trajectory=[])      # round 1: declared order
    assert calls == ["permissive", "strict"]

    calls.clear()
    lad.verify(_op(), _fb(), trajectory=[])      # round 2: strict has earned the front
    assert calls == ["strict"]


def test_non_adaptive_ladder_keeps_its_declared_ordering():
    from sentinelprime.verifier import LadderVerifier, VerifierLevel

    calls: list[str] = []
    lad = LadderVerifier([
        VerifierLevel("permissive", cost=1.0,
                      verifier=_RecordingLevel(True, calls, "permissive")),
        VerifierLevel("strict", cost=1.0,
                      verifier=_RecordingLevel(False, calls, "strict")),
    ])
    lad.verify(_op(), _fb(), trajectory=[])
    calls.clear()
    lad.verify(_op(), _fb(), trajectory=[])
    assert calls == ["permissive", "strict"]


def test_adaptive_ladder_still_respects_cost_when_detection_is_equal():
    """Detection-per-dollar, not detection alone: equal rejection rates -> cheapest first."""
    from sentinelprime.verifier import LadderVerifier, VerifierLevel

    calls: list[str] = []
    lad = LadderVerifier([
        VerifierLevel("expensive", cost=100.0,
                      verifier=_RecordingLevel(True, calls, "expensive")),
        VerifierLevel("cheap", cost=1.0,
                      verifier=_RecordingLevel(True, calls, "cheap")),
    ], adaptive=True)
    lad.verify(_op(), _fb(), trajectory=[])
    calls.clear()
    lad.verify(_op(), _fb(), trajectory=[])
    assert calls == ["cheap", "expensive"]
