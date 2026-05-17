from dataclasses import dataclass
from enum import Enum

from agent_squad.neutrosophic.triplet import Triplet


class DecisionAction(str, Enum):
    CLARIFY = "CLARIFY"
    CONFIDENCE = "CONFIDENCE"
    CAVEAT = "CAVEAT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class DecisionThresholds:
    indeterminacy: float = 0.6
    falsity: float = 0.5
    truth: float = 0.7

    def __post_init__(self) -> None:
        Triplet(
            T=self.truth,
            I=self.indeterminacy,
            F=self.falsity,
        )


def decide(value: Triplet, thresholds: DecisionThresholds | None = None) -> DecisionAction:
    """Choose a response action from a neutrosophic triplet."""
    thresholds = thresholds or DecisionThresholds()

    if value.I > thresholds.indeterminacy:
        return DecisionAction.CLARIFY
    if value.F > thresholds.falsity:
        return DecisionAction.REJECT
    if value.T > thresholds.truth:
        return DecisionAction.CONFIDENCE
    return DecisionAction.CAVEAT
