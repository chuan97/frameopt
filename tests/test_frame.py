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
    tang = f1.project(tang)
    f2 = f1.retract(tang)
    ξ = f1.log_map(f2)
    f3 = f1.retract(ξ)
    np.testing.assert_allclose(f2.vectors, f3.vectors, atol=1e-8, rtol=0)


def test_log_retract_large_angle():
    f1 = Frame.random(10, 5, rng=np.random.default_rng(4))

    # Build a deterministic tangent: swap real/imag, project, normalise.
    swaps = 1j * f1.vectors  # swap real<->imag parts; certainly not collinear
    tang = f1.project(swaps)
    tang /= np.linalg.norm(swaps, axis=1, keepdims=True)

    f2 = f1.retract(tang)
    xi = f1.log_map(f2)
    f3 = f1.retract(xi)

    np.testing.assert_allclose(f2.vectors, f3.vectors, atol=1e-8, rtol=0)


def test_project_tangent():
    """`Frame.project` should return a tangent vector: ⟨f_i,ξ_i⟩ = 0."""
    rng = np.random.default_rng(5)
    f = Frame.random(6, 3, rng=rng)

    # Build an arbitrary ambient perturbation.
    ambient = rng.standard_normal(f.shape) + 1j * rng.standard_normal(f.shape)

    proj = f.project(ambient)

    # 1. Shape consistency
    assert proj.shape == f.shape

    # 2. Tangency: full row‑wise inner products must vanish.
    radial = np.sum(f.vectors.conj() * proj, axis=1)
    np.testing.assert_allclose(radial, 0.0 + 0.0j, atol=1e-12)


def test_project_kills_radial_and_phase():
    rng = np.random.default_rng(0)
    f = Frame.random(5, 3, rng=rng)

    # Radial direction projects to zero
    radial = f.vectors.copy()
    np.testing.assert_allclose(f.project(radial), 0.0, atol=1e-12)

    # Phase (vertical) direction projects to zero
    vertical = 1j * f.vectors
    np.testing.assert_allclose(f.project(vertical), 0.0, atol=1e-12)


def test_transport_tangent_and_norm():
    """Parallel transport preserves tangency at the target and the tangent norm."""
    rng = np.random.default_rng(7)
    X = Frame.random(8, 4, rng=rng)

    # Tangent at X
    U = rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)
    U = X.project(U)

    # Small geodesic step to Y
    eta = 0.1 * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
    eta = X.project(eta)
    Y = X.retract(eta)

    V = X.transport(Y, U)

    # 1) V is tangent at Y
    radial_Y = np.sum(Y.vectors.conj() * V, axis=1)
    np.testing.assert_allclose(radial_Y, 0.0 + 0.0j, atol=1e-12)

    # 2) Norm preservation (product metric → ambient Frobenius)
    np.testing.assert_allclose(
        np.linalg.norm(U), np.linalg.norm(V), rtol=1e-10, atol=1e-12
    )


def test_transport_roundtrip():
    """Transport forth and back along the same geodesic is (numerically) identity."""
    rng = np.random.default_rng(8)
    X = Frame.random(6, 3, rng=rng)

    U = rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)
    U = X.project(U)

    eta = 0.2 * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
    eta = X.project(eta)
    Y = X.retract(eta)

    V = X.transport(Y, U)
    U_back = Y.transport(X, V)

    np.testing.assert_allclose(U_back, U, atol=1e-10, rtol=0.0)


def test_transport_identity_when_same_point():
    """Transport to the same base point should be the identity on the tangent."""
    rng = np.random.default_rng(9)
    X = Frame.random(5, 3, rng=rng)
    U = rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)
    U = X.project(U)

    V = X.transport(X, U)

    np.testing.assert_allclose(V, U, atol=1e-12, rtol=0.0)
