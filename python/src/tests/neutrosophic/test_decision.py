import pytest

from agent_squad.neutrosophic import DecisionAction, DecisionThresholds, Triplet, decide


def test_decide_clarify_has_priority_over_reject():
    value = Triplet(T=0.9, I=0.7, F=0.9)

    assert decide(value) == DecisionAction.CLARIFY


def test_decide_reject_when_falsity_is_high():
    value = Triplet(T=0.9, I=0.2, F=0.6)

    assert decide(value) == DecisionAction.REJECT


def test_decide_confidence_when_truth_is_high():
    value = Triplet(T=0.8, I=0.2, F=0.1)

    assert decide(value) == DecisionAction.CONFIDENCE


def test_decide_caveat_for_low_confidence_middle_state():
    value = Triplet(T=0.5, I=0.2, F=0.1)

    assert decide(value) == DecisionAction.CAVEAT


def test_decide_accepts_custom_thresholds():
    value = Triplet(T=0.65, I=0.2, F=0.1)
    thresholds = DecisionThresholds(truth=0.6)

    assert decide(value, thresholds) == DecisionAction.CONFIDENCE


def test_decision_thresholds_validate_component_bounds():
    with pytest.raises(ValueError, match="between 0 and 1"):
        DecisionThresholds(indeterminacy=1.2)
