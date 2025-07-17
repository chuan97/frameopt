"""
Riemannian conjugate gradients utilities for *evomof*.

Public API
----------
polish_with_cg
    Convenience wrapper that performs a limited number of Riemannian
    conjugate gradients iterations to polish a frame produced by a global optimiser.

Any future conjugate gradients-related helpers can live in this sub-package without
cluttering the top-level ``evomof.optim`` namespace.
"""

from .cg import polish_with_cg

__all__: list[str] = ["polish_with_cg"]
