import pytest

from agent_squad.neutrosophic import Triplet, score_text_response


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
