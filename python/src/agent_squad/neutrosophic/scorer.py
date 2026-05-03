import re

from agent_squad.neutrosophic.triplet import Triplet

HEDGING_PATTERNS = (
    "maybe",
    "might",
    "possibly",
    "probably",
    "not sure",
    "unclear",
    "unknown",
    "ambiguous",
    "insufficient",
    "could be",
    "it depends",
)

ERROR_PATTERNS = (
    "error",
    "failed",
    "failure",
    "cannot",
    "can't",
    "unable",
    "invalid",
    "contradiction",
    "conflict",
    "not possible",
    "i do not know",
    "i don't know",
)

REFUSAL_PATTERNS = (
    "i can't help",
    "i cannot help",
    "i cannot comply",
    "i can't comply",
    "sorry, but i can't",
    "sorry, but i cannot",
)


def score_text_response(text: str) -> Triplet:
    """Score plain text with deterministic baseline heuristics.

    This scorer is intentionally lightweight. It provides a dependency-free
    baseline until framework integrations can inject model-specific scorers.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = " ".join(text.lower().split())
    if not normalized:
        return Triplet(T=0.0, I=1.0, F=0.0)

    hedge_count = _count_matches(normalized, HEDGING_PATTERNS)
    error_count = _count_matches(normalized, ERROR_PATTERNS)
    refusal_count = _count_matches(normalized, REFUSAL_PATTERNS)
    word_count = len(re.findall(r"\b\w+\b", normalized))

    substance = min(word_count / 80, 1.0)
    directness = 1.0 if hedge_count == 0 else max(0.0, 1.0 - (hedge_count * 0.2))

    truth = _clamp((0.35 + (0.45 * substance) + (0.20 * directness)) - (0.20 * error_count))
    indeterminacy = _clamp((0.15 if word_count >= 8 else 0.45) + (0.18 * hedge_count))
    falsity = _clamp((0.22 * error_count) + (0.35 * refusal_count))

    return Triplet(T=truth, I=indeterminacy, F=falsity)


def score_classifier_confidence(confidence: float, selected: bool) -> Triplet:
    """Map legacy classifier confidence into a neutrosophic triplet."""
    if isinstance(confidence, bool):
        raise TypeError("confidence must be a number")

    confidence_value = _clamp(float(confidence))
    if selected:
        return Triplet(T=confidence_value, I=1 - confidence_value, F=0)

    return Triplet(T=0, I=max(1 - confidence_value, 0.7), F=0)


def _count_matches(text: str, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns if pattern in text)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
