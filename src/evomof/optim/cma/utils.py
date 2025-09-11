"""
Helpers to flatten/unflatten Frame objects for pycma.
"""

from __future__ import annotations

import numpy as np

from evomof.core.frame import Frame


def frame_to_realvec(frame: Frame) -> np.ndarray:
    """
    Flatten a Frame into a 1-D float64 vector (blocked layout).

    Layout (blocked):
      x = [ Re(V).ravel(order="C"),  Im(V).ravel(order="C") ]  concatenated

    Returns
    -------
    np.ndarray
        1-D float64 vector of length 2 * n * d.
    """
    re = frame.vectors.real.ravel(order="C")
    im = frame.vectors.imag.ravel(order="C")
    return np.concatenate([re, im]).astype(np.float64, copy=False)


def realvec_to_frame(vec: np.ndarray, n: int, d: int) -> Frame:
    """
    Inverse of `frame_to_realvec` for the blocked layout.

    Parameters
    ----------
    vec :
        1-D float64 array with length exactly 2 * n * d.
    n, d :
        Frame shape (rows × cols).

    Returns
    -------
    Frame
        New frame constructed from the real/imag blocks, normalised.
    """
    v = np.asarray(vec, dtype=np.float64)
    m = n * d
    expected = 2 * m
    if v.size != expected:
        raise ValueError(f"Vector length {v.size} inconsistent with 2·n·d={expected}")

    re = v[:m].reshape(n, d)
    im = v[m:].reshape(n, d)
    z = (re + 1j * im).astype(np.complex128, copy=False)

    # Normalise & gauge-fix via Frame constructor.
    return Frame(z, normalize=True, copy=False)
