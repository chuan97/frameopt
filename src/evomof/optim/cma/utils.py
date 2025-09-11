"""
Helpers to flatten/unflatten Frame objects for pycma.
"""

from __future__ import annotations

import numpy as np

from evomof.core._types import Complex128Array
from evomof.core.frame import Frame


def frame_to_realvec(frame: Frame) -> np.ndarray:
    """
    Interleave real and imaginary parts row-wise into a 1-D ``float64`` vector.

    This is the representation expected by ``cma.CMAEvolutionStrategy``,
    which works purely in real Euclidean space.
    """
    V = np.ascontiguousarray(frame.vectors, dtype=np.complex128)
    return V.ravel().view(np.float64)


def realvec_to_frame(vec: np.ndarray, n: int, d: int) -> Frame:
    """
    Convert a flattened real-valued vector back into a :class:`Frame`.

    The input ``vec`` must have been produced by
    :func:`frame_to_realvec`, i.e. real and imaginary parts interleaved
    row-wise.  Therefore its length must equal ``2 * n * d``.

    Parameters
    ----------
    vec :
        1-D NumPy array of ``float64`` holding real/imag pairs.
    n, d :
        Original frame dimensions (rows × columns).

    Returns
    -------
    Frame
        A new frame whose ``.vectors`` array *views* the memory of ``vec``
        (no copy).
    """
    # Ensure vec is contiguous float64 memory before viewing as complex128
    vec_f64 = np.ascontiguousarray(vec, dtype=np.float64)

    expected_len = 2 * n * d
    if vec_f64.size != expected_len:
        raise ValueError(
            f"Vector length {vec_f64.size} inconsistent with 2·n·d={expected_len}"
        )

    z: Complex128Array = vec_f64.view(np.complex128).reshape(n, d)
    return Frame.from_array(z, copy=False)
