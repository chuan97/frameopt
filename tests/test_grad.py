"""Finite-difference checks for analytic gradients.

For each energy / gradient pair:
    • Generate a random frame.
    • Pick a random *tangent* direction U (‖U‖ = 1).
    • Compare directional derivative
        (E(F ⊕ εU) − E(F)) / ε
      with  ⟨∇E(F), U⟩.
"""

from __future__ import annotations

import numpy as np
import pytest

from frameopt.core.energy import (
    diff_coherence,
    frame_potential,
    grad_diff_coherence,
    grad_frame_potential,
)
from frameopt.core.frame import Frame
from frameopt.core.manifold import PRODUCT_CP

# ---------------------------- helpers ---------------------------------


def fd_directional_derivative(
    energy_fn, frame: Frame, tangent: np.ndarray, eps: float
) -> float:
    """Central finite difference in direction *tangent*."""
    f_plus = PRODUCT_CP.retract(frame, eps * tangent)
    f_minus = PRODUCT_CP.retract(frame, -eps * tangent)
    return (energy_fn(f_plus) - energy_fn(f_minus)) / (2 * eps)


def inner_product(grad: np.ndarray, tangent: np.ndarray) -> float:
    """Real Frobenius inner product (metric on the manifold)."""
    return float(np.real(np.sum(grad.conj() * tangent)))


# -------------------------- parametrised cases -------------------------

TEST_CASES = [
    # (energy_fn, grad_fn, kwargs)
    (
        lambda F, p=4: frame_potential(F, p=p),
        lambda F, p=4: grad_frame_potential(F, p=p),
        {"p": 4},
    ),
    (
        lambda F, p=10: diff_coherence(F, p=p),
        lambda F, p=10: grad_diff_coherence(F, p=p),
        {"p": 10},
    ),
    (
        lambda F, p=1000: diff_coherence(F, p=p),
        lambda F, p=1000: grad_diff_coherence(F, p=p),
        {"p": 1000},
    ),
]


@pytest.mark.parametrize("energy_fn_raw, grad_fn_raw, kwargs", TEST_CASES)
def test_gradient_finite_difference(energy_fn_raw, grad_fn_raw, kwargs):

    # Small frame for quick CI; more sizes covered elsewhere.
    n, d = 6, 3
    rng = np.random.default_rng(123)
    F = Frame.random(n, d, rng=rng)

    # Build random unit tangent vector via the geometry policy.
    U = PRODUCT_CP.random_tangent(F, rng=rng, unit=True)

    eps = 1e-6

    def energy_fn(G):
        return energy_fn_raw(G, **kwargs)

    def grad_fn(G):
        return grad_fn_raw(G, **kwargs)

    fd = fd_directional_derivative(energy_fn, F, U, eps)
    ana = inner_product(grad_fn(F), U)

    assert abs(fd - ana) < 1e-6, f"Finite diff {fd} vs analytic {ana}"
