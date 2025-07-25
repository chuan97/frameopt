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


def test_diff_coherence_monotone(random_frame: Frame) -> None:
    """diff_coherence should be non-increasing as p grows."""
    ps = [2, 4, 8, 16, 32, 64]
    vals = [diff_coherence(random_frame, p=p) for p in ps]
    # Monotonic non-increasing
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    # Final value still above (or equal to) true coherence
    assert vals[-1] >= coherence(random_frame) - 1e-12


def test_diff_coherence_high_p_no_underflow(random_frame: Frame) -> None:
    """For very large p the surrogate should approach coherence without underflow."""
    p = 2000
    val = diff_coherence(random_frame, p=p)
    mu = coherence(random_frame)
    assert val > 0.0
    # Within 0.1% relative error of true coherence
    assert abs(val - mu) / mu < 1e-3


def test_riesz_energy_clamp(random_frame: Frame) -> None:
    """
    Riesz energy must remain finite even with identical rows,
    thanks to the eps clamp.
    """
    duplicate = random_frame.copy()
    duplicate.vectors[0] = duplicate.vectors[1]  # make two rows identical
    energy = riesz_energy(duplicate, s=2, eps=1e-6)
    assert np.isfinite(energy)


def test_diff_coherence_multiple_maxima() -> None:
    """
    When several pairs attain the maximum inner product (duplicates),
    diff_coherence should remain stable and approach that maximum for large p.
    """
    # Construct a frame with three identical columns and one orthogonal column.
    # Columns: e0, e0, e0, e1 in R^3
    arr = np.array(
        [
            [1.0, 0.0, 0.0],  # e0
            [1.0, 0.0, 0.0],  # e0 duplicate
            [1.0, 0.0, 0.0],  # e0 duplicate
            [0.0, 1.0, 0.0],  # e1 orthogonal
        ],
        dtype=np.complex128,
    )
    frame = Frame.from_array(arr, copy=False)

    # True coherence: maximum |<fi,fj>| is 1
    mu = coherence(frame)
    assert mu == pytest.approx(1.0)

    # Large p: surrogate should approach 1 without underflow.
    # Exact expected value under current definition:
    # Φ_p = sum_{i≠j} |g_ij|^{2p} = m*(m-1) for m identical
    # columns (counts both (i,j),(j,i))
    # Here m=3 ⇒ Φ_p = 3*2 = 6 → diff_coherence = 6^{1/(2p)}.
    p = 2000
    expected = 6 ** (1.0 / (2 * p))
    val = diff_coherence(frame, p=p)
    assert val > 0.0
    # Check against closed-form and allow small numerical drift
    assert abs(val - expected) < 1e-10
    # And ensure it is within a reasonable distance to the
    # true coherence (tends to 1 as p→∞)
    assert abs(val - mu) < 5e-4  # theoretical gap ≈ ln(6)/(2p) ≈ 4.48e-4
