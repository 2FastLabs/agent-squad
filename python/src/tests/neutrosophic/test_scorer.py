import pytest

from agent_squad.neutrosophic import Triplet, score_classifier_confidence, score_text_response


def test_score_text_response_rejects_non_string_input():
    with pytest.raises(TypeError, match="text must be a string"):
        score_text_response(None)


def test_score_text_response_marks_empty_text_as_indeterminate():
    assert score_text_response("") == Triplet(T=0.0, I=1.0, F=0.0)


def test_score_text_response_returns_bounded_triplet():
    score = score_text_response("Maybe this could be unclear and failed because of an error.")

    assert 0 <= score.T <= 1
    assert 0 <= score.I <= 1
    assert 0 <= score.F <= 1


def test_score_text_response_increases_indeterminacy_for_hedging():
    direct = score_text_response("The request should be routed to the billing agent for invoice help.")
    hedged = score_text_response("Maybe it could be routed to billing, but it depends and is unclear.")

    assert hedged.I > direct.I


def test_score_text_response_increases_falsity_for_errors():
    direct = score_text_response("The task completed successfully with a clear answer.")
    failed = score_text_response("The task failed with an invalid response and an error.")

    assert failed.F > direct.F


def test_score_text_response_scores_direct_substantive_answer_with_more_truth():
    short = score_text_response("Maybe.")
    substantive = score_text_response(
        "The billing agent is the correct destination because the user asks about invoices, refunds, "
        "and account balance details that match the billing agent description."
    )

    assert substantive.T > short.T


def test_score_classifier_confidence_maps_selected_agent_to_truth_and_indeterminacy():
    score = score_classifier_confidence(0.8, selected=True)

    assert score.T == 0.8
    assert score.I == pytest.approx(0.2)
    assert score.F == 0.0


def test_score_classifier_confidence_maps_no_agent_to_clarification_indeterminacy():
    assert score_classifier_confidence(0.95, selected=False) == Triplet(T=0.0, I=0.7, F=0.0)


def test_score_classifier_confidence_rejects_boolean_confidence():
    with pytest.raises(TypeError, match="confidence must be a number"):
        score_classifier_confidence(True, selected=True)
