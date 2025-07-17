import numpy as np

from evomof import Frame


def test_random_shape_norms():
    f = Frame.random(n=12, d=6, rng=np.random.default_rng(0))
    assert f.shape == (12, 6)
    norms = np.linalg.norm(f.vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_phase_fixed():
    f = Frame.random(4, 3, rng=np.random.default_rng(1))
    for vec in f:
        first = vec[np.flatnonzero(vec)][0]
        assert np.isclose(np.angle(first), 0.0, atol=1e-12)


def test_log_retract_inverse():
    rng = np.random.default_rng(2)
    f1 = Frame.random(8, 4, rng=rng)
    tang = 1e-3 * rng.standard_normal(f1.shape) + 1e-3j * rng.standard_normal(f1.shape)
    tang -= (
        np.einsum("nd,nd->n", tang.conj(), f1.vectors)[:, None] * f1.vectors
    )  # tangent
    f2 = f1.retract(tang)
    ξ = f1.log_map(f2)
    f3 = f1.retract(ξ)
    np.testing.assert_allclose(f2.vectors, f3.vectors, atol=1e-8, rtol=0)


def test_log_retract_large_angle():
    f1 = Frame.random(10, 5, rng=np.random.default_rng(4))

    # Build a deterministic tangent: swap real/imag, project, normalise.
    swaps = 1j * f1.vectors  # swap real<->imag parts; certainly not collinear
    swaps -= np.einsum("nd,nd->n", swaps.conj(), f1.vectors)[:, None] * f1.vectors
    tang = swaps / np.linalg.norm(swaps, axis=1, keepdims=True)
    tang *= 1.0  # 1-radian step

    f2 = f1.retract(tang)
    xi = f1.log_map(f2)
    f3 = f1.retract(xi)

    np.testing.assert_allclose(f2.vectors, f3.vectors, atol=1e-8, rtol=0)


def test_project_tangent():
    """`Frame.project` should return a tangent vector: Re⟨f_i,ξ_i⟩ = 0."""
    rng = np.random.default_rng(5)
    f = Frame.random(6, 3, rng=rng)

    # Build an arbitrary ambient perturbation.
    ambient = rng.standard_normal(f.shape) + 1j * rng.standard_normal(f.shape)

    proj = f.project(ambient)

    # 1. Shape consistency
    assert proj.shape == f.shape

    # 2. Tangency: real part of row‑wise inner products must vanish.
    radial = np.real(np.sum(proj.conj() * f.vectors, axis=1))
    np.testing.assert_allclose(radial, 0.0, atol=1e-12)
