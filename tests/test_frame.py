import numpy as np

from evomof.core.frame import Frame


def test_random_shape():
    f = Frame.random(n=12, d=6, rng=np.random.default_rng(0))
    assert f.shape == (12, 6)


def test_random_norms():
    f = Frame.random(n=12, d=6, rng=np.random.default_rng(1))
    norms = np.linalg.norm(f.vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_phase_fixed():
    """Gauge invariance: multiplying rows by a global phase should be undone by normalize()."""
    rng = np.random.default_rng(1)
    f = Frame.random(8, 5, rng=rng)

    # Random global U(1) phase per row
    phi = rng.uniform(0.0, 2 * np.pi, size=f.shape[0])
    phases = np.exp(1j * phi)

    V_phase = f.vectors * phases[:, None]
    g = Frame(V_phase, normalize=True)

    np.testing.assert_allclose(g.vectors, f.vectors, atol=1e-12)
