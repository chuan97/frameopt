import numpy as np
import pytest

from frameopt.core.energy import (
    coherence,
    frame_potential,
    mellowmax_coherence,
    pnormmax_coherence,
)
from frameopt.core.frame import Frame


@pytest.fixture(scope="module")
def orthonormal_frame() -> Frame:
    """d × d identity columns → perfect orthonormal frame."""
    d = 4
    id = np.eye(d, dtype=np.complex128)
    return Frame(id, copy=False)


@pytest.fixture(scope="module")
def random_frame() -> Frame:
    return Frame.random(8, 5, rng=np.random.default_rng(123))


@pytest.fixture(scope="module")
def degenerate_frame() -> Frame:
    """Frame with three identical columns and one orthogonal column in R^3."""
    arr = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    return Frame(arr, copy=False)


def test_frame_potential_orthonormal(orthonormal_frame: Frame) -> None:
    """All off-diagonal inner products are zero ⇒ Φ_p = 0."""
    assert frame_potential(orthonormal_frame, p=4) == 0.0


def test_coherence_vs_pnorm(random_frame: Frame) -> None:
    """pnormmax_coherence(p) ≥ true coherence."""
    mu = coherence(random_frame)
    mu_pnorm = pnormmax_coherence(random_frame, p=64)
    assert mu_pnorm >= mu


def test_pnormmax_coherence_monotone(random_frame: Frame) -> None:
    """pnormmax_coherence should be non-increasing as p grows."""
    ps = [2, 4, 8, 16, 32, 64]
    vals = [pnormmax_coherence(random_frame, p=p) for p in ps]
    # Monotonic non-increasing
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    # Final value still above (or equal to) true coherence
    assert vals[-1] >= coherence(random_frame) - 1e-12


def test_pnormmax_coherence_high_p_no_underflow(random_frame: Frame) -> None:
    """For very large p the surrogate should approach coherence without underflow."""
    p = 2000
    val = pnormmax_coherence(random_frame, p=p)
    mu = coherence(random_frame)
    assert val > 0.0
    # Within 0.1% relative error of true coherence
    assert abs(val - mu) / mu < 1e-3


def test_pnormmax_coherence_multiple_maxima(degenerate_frame: Frame) -> None:
    """
    When several pairs attain the maximum inner product (duplicates),
    pnormmax_coherence should remain stable and approach that maximum for large p.
    """
    frame = degenerate_frame

    # True coherence: maximum |<fi,fj>| is 1
    mu = coherence(frame)
    assert mu == pytest.approx(1.0)

    # Large p: surrogate should approach 1 without underflow.
    # Exact expected value under current definition:
    # Φ_p = sum_{i≠j} |g_ij|^{2p} = m*(m-1) for m identical
    # columns (counts both (i,j),(j,i))
    # Here m=3 ⇒ Φ_p = 3*2 = 6 → pnormmax_coherence = 6^{1/(2p)}.
    p = 2000
    expected = 6 ** (1.0 / (2 * p))
    val = pnormmax_coherence(frame, p=p)
    assert val > 0.0
    # Check against closed-form and allow small numerical drift
    assert abs(val - expected) < 1e-10
    # And ensure it is within a reasonable distance to the
    # true coherence (tends to 1 as p→∞)
    assert abs(val - mu) < 5e-4  # theoretical gap ≈ ln(6)/(2p) ≈ 4.48e-4


def test_mellowmax_coherence_orthonormal(orthonormal_frame: Frame) -> None:
    """All off-diagonal inner products are zero ⇒ mellowmax = 0 for any ω."""
    assert mellowmax_coherence(orthonormal_frame, omega=0.0) == 0.0
    assert mellowmax_coherence(orthonormal_frame, omega=10.0) == 0.0


def test_mellowmax_coherence_monotone(random_frame: Frame) -> None:
    """mellowmax should be non-decreasing with ω and bounded above by coherence."""
    omegas = [0.0, 1.0, 2.0, 5.0, 10.0, 50.0, 1000.0]
    vals = [mellowmax_coherence(random_frame, omega=w) for w in omegas]

    # Monotonic non-decreasing in ω
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    mu = coherence(random_frame)
    # Always ≤ true coherence (up to tiny numerical wiggle)
    assert vals[-1] <= mu + 1e-12

    # At large ω, mellowmax approaches from below with gap ≈ (log M)/ω
    n = random_frame.shape[0]
    M = n * (n - 1) // 2
    w = omegas[-1]
    expected_lower = mu - np.log(M) / w
    assert expected_lower - 1e-12 <= vals[-1] <= mu + 1e-12


def test_mellowmax_coherence_high_omega_no_underflow(random_frame: Frame) -> None:
    """For very large ω the surrogate should approach coherence without collapsing to 0."""
    w = 2000.0
    val = mellowmax_coherence(random_frame, omega=w)
    mu = coherence(random_frame)
    assert val > 0.0
    # Large-ω asymptotics: mm_ω ≈ μ − (log M)/ω (from below)
    n = random_frame.shape[0]
    M = n * (n - 1) // 2
    expected_lower = mu - np.log(M) / w
    assert expected_lower - 1e-12 <= val <= mu + 1e-12


def test_mellowmax_coherence_multiple_maxima(degenerate_frame: Frame) -> None:
    """
    When several pairs attain the maximum inner product (duplicates),
    mellowmax should remain stable and approach that maximum for large ω.
    Also checks the ω→0 limit equals the mean of off-diagonal overlaps.
    """
    frame = degenerate_frame

    # True coherence: maximum |<fi,fj>| is 1
    mu = coherence(frame)
    assert mu == pytest.approx(1.0)

    # ω → 0 limit: mean of off-diagonal magnitudes.
    # There are M=6 unique pairs; 3 have overlap 1, 3 have overlap 0 ⇒ mean = 0.5
    mm0 = mellowmax_coherence(frame, omega=0.0)
    assert mm0 == pytest.approx(0.5)

    # Large ω: mellowmax ≈ 1 − (log 2)/ω for this construction
    w = 2000.0
    val = mellowmax_coherence(frame, omega=w)
    expected_upper = 1.0  # never exceeds true coherence
    expected_lower = 1.0 - np.log(2.0) / w

    assert expected_lower <= val <= expected_upper + 1e-12
    # And it should be close to coherence
    assert abs(val - mu) < 1e-3
