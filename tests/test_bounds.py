import math

from evomof.bounds import bukhcox, levenstein, max_lower_bound, orthoplex, welch


def test_bukhcox():
    assert math.isclose(bukhcox(5, 7), 0.26447408, abs_tol=1e-7)


def test_levenstein():
    assert math.isclose(levenstein(4, 40), 0.57735027, abs_tol=1e-7)


def test_orthoplex():
    assert math.isclose(orthoplex(3, 10), 0.57735027, abs_tol=1e-7)


def test_welch():
    assert math.isclose(welch(2, 4), 0.57735027, abs_tol=1e-7)


def test_max_lower_bound():
    assert math.isclose(max_lower_bound(6, 9), 0.25000000, abs_tol=1e-7)
    assert math.isclose(max_lower_bound(6, 19), 0.34694433, abs_tol=1e-7)
    assert math.isclose(max_lower_bound(6, 37), 0.40824829, abs_tol=1e-7)
    assert math.isclose(max_lower_bound(6, 49), 0.43133109, abs_tol=1e-7)
