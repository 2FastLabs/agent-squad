from dataclasses import dataclass
from math import isfinite
from numbers import Real


@dataclass(frozen=True)
class Triplet:
    """Neutrosophic truth, indeterminacy, and falsity values."""

    T: float
    I: float
    F: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "T", self._validate_component("T", self.T))
        object.__setattr__(self, "I", self._validate_component("I", self.I))
        object.__setattr__(self, "F", self._validate_component("F", self.F))

    @staticmethod
    def _validate_component(name: str, value: Real) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a number")

        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise ValueError(f"{name} must be finite")
        if not 0 <= numeric_value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

        return numeric_value
