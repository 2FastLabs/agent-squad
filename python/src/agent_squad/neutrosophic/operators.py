from agent_squad.neutrosophic.triplet import Triplet


def n_norm(left: Triplet, right: Triplet) -> Triplet:
    """Smarandache N-norm: AND over two neutrosophic triplets."""
    return Triplet(
        T=min(left.T, right.T),
        I=max(left.I, right.I),
        F=max(left.F, right.F),
    )


def n_conorm(left: Triplet, right: Triplet) -> Triplet:
    """Smarandache N-conorm: OR over two neutrosophic triplets."""
    return Triplet(
        T=max(left.T, right.T),
        I=min(left.I, right.I),
        F=min(left.F, right.F),
    )


def negate(value: Triplet) -> Triplet:
    """Neutrosophic negation: swap truth and falsity, keep indeterminacy."""
    return Triplet(T=value.F, I=value.I, F=value.T)
