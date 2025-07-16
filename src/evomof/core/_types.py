"""
Common NumPy typing aliases used throughout *evomof*.

Import with::

    from evomof.core._types import Complex128Array, Float64Array
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Float64Array: TypeAlias = NDArray[np.float64]
"""Shorthand for an ndarray of float64."""

Complex128Array: TypeAlias = NDArray[np.complex128]
"""Shorthand for an ndarray of complex128."""
