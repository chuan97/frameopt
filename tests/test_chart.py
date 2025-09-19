import numpy as np
import pytest

from evomof.core.frame import Frame
from evomof.core.manifold import PRODUCT_CP, Chart


def test_dim_matches_formula():
    rng = np.random.default_rng(0)
    f = Frame.random(n=4, d=5, rng=rng)
    chart = Chart.at(f)
    assert chart.dim() == 2 * 4 * (5 - 1)


def test_encode_decode_inverse_on_tangent():
    rng = np.random.default_rng(1)
    f = Frame.random(n=3, d=6, rng=rng)
    chart = Chart.at(f)

    # random tangent at f
    U = PRODUCT_CP.random_tangent(f, rng=rng, unit=False)

    y = chart.encode(U)
    U_dec = chart.decode(y)

    # Should recover the same tangent (encode/decode isometry)
    np.testing.assert_allclose(U_dec, U, rtol=0, atol=1e-12)

    # Tangency check: ⟨f_i, U_i⟩ = 0 for each row
    inner = np.sum(f.vectors.conj() * U_dec, axis=1)
    np.testing.assert_allclose(inner, 0.0, atol=1e-12)


def test_encode_type_and_length():
    rng = np.random.default_rng(2)
    f = Frame.random(n=5, d=4, rng=rng)
    chart = Chart.at(f)
    U = PRODUCT_CP.random_tangent(f, rng=rng, unit=False)

    y = chart.encode(U)
    assert y.dtype == np.float64
    assert y.ndim == 1
    assert y.size == chart.dim()


def test_decode_length_mismatch_raises():
    rng = np.random.default_rng(3)
    f = Frame.random(n=2, d=5, rng=rng)
    chart = Chart.at(f)
    bad = np.zeros(chart.dim() - 1, dtype=np.float64)
    with pytest.raises(ValueError):
        chart.decode(bad)


def test_transport_coords_identity_same_point():
    rng = np.random.default_rng(4)
    f = Frame.random(n=3, d=5, rng=rng)
    chart = Chart.at(f)
    U = PRODUCT_CP.random_tangent(f, rng=rng, unit=False)
    y = chart.encode(U)

    y_back = chart.transport_coords(chart, y)
    np.testing.assert_allclose(y_back, y, atol=1e-12)


def test_transport_preserves_norm_small_step():
    rng = np.random.default_rng(5)
    f = Frame.random(n=3, d=5, rng=rng)
    chart_X = Chart.at(f)

    # small step to nearby frame
    eta = PRODUCT_CP.random_tangent(f, rng=rng, unit=True)
    Y = PRODUCT_CP.retract(f, 1e-3 * eta)
    chart_Y = Chart.at(Y)

    # random direction in coords
    U = PRODUCT_CP.random_tangent(f, rng=rng, unit=False)
    y = chart_X.encode(U)

    y_to = chart_X.transport_coords(chart_Y, y)

    # Parallel transport is (near-)isometric
    np.testing.assert_allclose(
        np.linalg.norm(y_to), np.linalg.norm(y), rtol=0, atol=1e-10
    )


def test_chart_at_raises_for_d_lt_2():
    rng = np.random.default_rng(8)
    f = Frame.random(n=3, d=1, rng=rng)
    with pytest.raises(ValueError):
        Chart.at(f)


def test_encode_shape_mismatch_raises():
    rng = np.random.default_rng(9)
    f = Frame.random(n=3, d=4, rng=rng)
    chart = Chart.at(f)
    bad = rng.standard_normal((f.shape[0] + 1, f.shape[1])) + 1j * rng.standard_normal(
        (f.shape[0] + 1, f.shape[1])
    )
    with pytest.raises(ValueError):
        chart.encode(bad)
