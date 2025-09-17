"""Coherence lower bounds.

All functions return a numerical lower bound on the minimal achievable coherence
for a (d, n) frame/packing. Implementations are independent of evomof.core/optim.
"""

import math

__all__ = [
    "welch",
    "levenstein",
    "orthoplex",
    "bukhcox",
    "max_lower_bound",
]


def _validate(d: int, n: int) -> None:
    """Validate that d and n are positive integers."""
    if d < 1 or n < 1:
        raise ValueError("Both d and n must be positive integers.")


def welch(d: int, n: int) -> float:
    """Welch bound: sqrt((n - d) / (d * (n - 1))).

    Returns 0.0 if n <= d.
    """
    _validate(d, n)

    if n <= d:
        return 0.0

    return math.sqrt((n - d) / (d * (n - 1)))


def levenstein(d: int, n: int) -> float:
    """Levenstein-type bound with m=1 (piecewise, active for n ≥ d**2).

    Returns 0.0 if n < d**2 or if denominator or radicand is non-positive (inactive region).
    """
    _validate(d, n)
    m = 1

    if n <= d**2:
        return 0.0

    denominator = (n - d) * (m * d + 1)
    radicand = (n * (m + 1) - d * (m * d + 1)) / denominator
    return math.sqrt(radicand)


def orthoplex(d: int, n: int) -> float:
    """Orthoplex/Rankin case: 1/sqrt(d) when n > d**2, else 0.0."""
    _validate(d, n)

    if n <= d**2:
        return 0.0

    return 1.0 / math.sqrt(d)


def bukhcox(d: int, n: int) -> float:
    """Bukh–Cox bound with m=1 (algebraic form as implemented).

    Returns 0.0 if n <= d or denominator is non-positive (inactive region).
    """
    _validate(d, n)
    m = 1

    if n <= d:
        return 0.0

    denominator = n * (1 + m * (n - d - 1) * math.sqrt(1 / m + n - d)) - (n - d) ** 2
    return (n - d) ** 2 / denominator


def max_lower_bound(d: int, n: int) -> float:
    """Compute the maximum of all implemented lower bounds for given (d, n)."""
    bounds = [
        welch(d, n),
        levenstein(d, n),
        orthoplex(d, n),
        bukhcox(d, n),
    ]

    return max(bounds)
