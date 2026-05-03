import pytest

from agent_squad.neutrosophic import Triplet


def test_triplet_accepts_valid_values():
    triplet = Triplet(T=1, I=0.5, F=0)

    assert triplet.T == 1.0
    assert triplet.I == 0.5
    assert triplet.F == 0.0


@pytest.mark.parametrize("field", ["T", "I", "F"])
def test_triplet_rejects_out_of_range_values(field):
    values = {"T": 0.2, "I": 0.3, "F": 0.4}
    values[field] = 1.1

    with pytest.raises(ValueError, match="between 0 and 1"):
        Triplet(**values)


@pytest.mark.parametrize("bad_value", [True, "0.5", None])
def test_triplet_rejects_non_numeric_values(bad_value):
    with pytest.raises(TypeError, match="must be a number"):
        Triplet(T=bad_value, I=0.2, F=0.3)
