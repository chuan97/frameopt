"""
Riemannian conjugate gradients utilities for *evomof*.

Public API
----------
cg_minimize
    Convenience wrapper that performs a limited number of Riemannian
    conjugate gradients iterations to polish a frame produced by a global optimiser.

Any future conjugate gradients-related helpers can live in this sub-package without
cluttering the top-level ``evomof.optim`` namespace.
"""

from .cg import minimize as cg_minimize

__all__: list[str] = ["cg_minimize"]
