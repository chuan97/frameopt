"""
Riemannian conjugate gradients utilities for *evomof*.

Public API
----------
polish_with_conjgrad
    Convenience wrapper that performs a limited number of Riemannian
    conjugate gradients iterations to polish a frame produced by a global optimiser.

Any future conjugate gradients-related helpers can live in this sub-package without
cluttering the top-level ``evomof.optim`` namespace.
"""

from .polish import polish_with_conjgrad

__all__: list[str] = ["polish_with_conjgrad"]
