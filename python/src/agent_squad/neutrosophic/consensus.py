from collections.abc import Iterable

from agent_squad.neutrosophic.operators import n_conorm
from agent_squad.neutrosophic.triplet import Triplet


def neutrosophic_consensus(responses: Iterable[Triplet]) -> Triplet:
    """Fuse agent response scores using repeated N-conorm aggregation."""
    iterator = iter(responses)
    try:
        consensus = next(iterator)
    except StopIteration as exc:
        raise ValueError("responses must contain at least one triplet") from exc

    for response in iterator:
        consensus = n_conorm(consensus, response)

    return consensus


def neutrosophic_evidence_consensus(responses: Iterable[Triplet]) -> Triplet:
    """Fuse agent evidence while preserving contradiction as indeterminacy.

    N-conorm is still the formal OR operator. For multi-agent response fusion,
    however, strong counter-evidence should not disappear just because another
    response has low falsity. This consensus keeps the strongest truth, keeps
    the strongest falsity, and raises indeterminacy when the evidence disagrees.
    """
    triplets = list(responses)
    if not triplets:
        raise ValueError("responses must contain at least one triplet")

    formal_union = neutrosophic_consensus(triplets)
    truth_values = [response.T for response in triplets]
    falsity_values = [response.F for response in triplets]

    truth_spread = max(truth_values) - min(truth_values)
    falsity_spread = max(falsity_values) - min(falsity_values)
    contradiction = min(max(truth_values), max(falsity_values))
    conflict = max(truth_spread, falsity_spread, contradiction)

    return Triplet(
        T=formal_union.T,
        I=max(formal_union.I, conflict),
        F=max(falsity_values),
    )
