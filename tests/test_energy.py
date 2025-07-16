import numpy as np
import pytest

from evomof.core.energy import coherence, diff_coherence, frame_potential, riesz_energy
from evomof.core.frame import Frame

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def orthonormal_frame() -> Frame:
    """d × d identity columns → perfect orthonormal frame."""
    d = 4
    id = np.eye(d, dtype=np.complex128)
    return Frame.from_array(id, copy=False)


@pytest.fixture(scope="module")
def random_frame() -> Frame:
    return Frame.random(8, 5, rng=np.random.default_rng(123))


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_frame_potential_orthonormal(orthonormal_frame: Frame) -> None:
    """All off-diagonal inner products are zero ⇒ Φ_p = 0."""
    assert frame_potential(orthonormal_frame, p=4) == 0.0


def test_coherence_vs_diff(random_frame: Frame) -> None:
    """diff_coherence(p) ≥ true coherence."""
    mu = coherence(random_frame)
    mu_diff = diff_coherence(random_frame, p=64)
    assert mu_diff >= mu


def test_riesz_energy_clamp(random_frame: Frame) -> None:
    """
    Riesz energy must remain finite even with identical rows,
    thanks to the eps clamp.
    """
    duplicate = random_frame.copy()
    duplicate.vectors[0] = duplicate.vectors[1]  # make two rows identical
    energy = riesz_energy(duplicate, s=2, eps=1e-6)
    assert np.isfinite(energy)
