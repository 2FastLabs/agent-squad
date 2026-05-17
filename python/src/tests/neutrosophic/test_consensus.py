import pytest

from agent_squad.neutrosophic import Triplet, neutrosophic_consensus, neutrosophic_evidence_consensus


def test_neutrosophic_consensus_fuses_with_n_conorm():
    responses = [
        Triplet(T=0.3, I=0.8, F=0.6),
        Triplet(T=0.9, I=0.5, F=0.4),
        Triplet(T=0.7, I=0.2, F=0.1),
    ]

    assert neutrosophic_consensus(responses) == Triplet(T=0.9, I=0.2, F=0.1)


def test_neutrosophic_consensus_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one triplet"):
        neutrosophic_consensus([])


def test_neutrosophic_evidence_consensus_preserves_conflicting_falsity():
    responses = [
        Triplet(T=0.9, I=0.1, F=0.0),
        Triplet(T=0.2, I=0.2, F=0.8),
    ]

    consensus = neutrosophic_evidence_consensus(responses)

    assert consensus.T == 0.9
    assert consensus.I == 0.8
    assert consensus.F == 0.8


def test_neutrosophic_evidence_consensus_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one triplet"):
        neutrosophic_evidence_consensus([])
