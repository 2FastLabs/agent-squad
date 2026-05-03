from agent_squad.neutrosophic import Triplet, n_conorm, n_norm, negate


def test_n_norm_uses_min_truth_max_indeterminacy_max_falsity():
    left = Triplet(T=0.8, I=0.2, F=0.1)
    right = Triplet(T=0.4, I=0.7, F=0.5)

    assert n_norm(left, right) == Triplet(T=0.4, I=0.7, F=0.5)


def test_n_conorm_uses_max_truth_min_indeterminacy_min_falsity():
    left = Triplet(T=0.8, I=0.2, F=0.1)
    right = Triplet(T=0.4, I=0.7, F=0.5)

    assert n_conorm(left, right) == Triplet(T=0.8, I=0.2, F=0.1)


def test_negate_swaps_truth_and_falsity():
    assert negate(Triplet(T=0.8, I=0.2, F=0.1)) == Triplet(T=0.1, I=0.2, F=0.8)
