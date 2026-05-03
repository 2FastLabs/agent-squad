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
