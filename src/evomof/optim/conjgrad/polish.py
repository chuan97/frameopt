from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
from pymanopt import Problem, function
from pymanopt.optimizers.conjugate_gradient import ConjugateGradient

from evomof.core._types import Complex128Array
from evomof.core.frame import Frame
from evomof.core.manifold import FrameManifold  # see §3


def polish_with_conjgrad(
    frame0: Frame,
    energy_fn: Callable[[Frame], float],
    grad_fn: Callable[[Frame], Complex128Array],
    maxiter: int = 50,
    **solver_kw: Any,
) -> Frame:
    """
    One‑shot Riemannian conjugate‑gradient **polish**.

    Parameters
    ----------
    frame0 :
        Initial frame (e.g. the best candidate from CMA‑ES).
    energy_fn :
        Callable that returns the real‑valued cost at a frame.
    grad_fn :
        Callable returning the **Riemannian gradient** at a frame.
        If the supplied gradient is Euclidean, it will be projected
        internally.
    maxiter :
        Maximum number of CG iterations (must be positive).
    **solver_kw :
        Extra keyword arguments forwarded to
        :class:`pymanopt.optimizers.conjugate_gradient.ConjugateGradient`.
        ``verbosity`` (default 0) can be passed here.

    Returns
    -------
    Frame
        Polished frame whose energy is no worse than the input.

    Raises
    ------
    ValueError
        If ``maxiter`` is not positive.
    TypeError
        If ``max_iterations`` is supplied via ``solver_kw`` instead of
        using the ``maxiter`` parameter.
    """
    if maxiter <= 0:
        raise ValueError("maxiter must be positive.")

    manifold = FrameManifold(*frame0.shape)

    # --- cost & gradient wrapped with pymanopt decorator ----------------
    @function.numpy(manifold)  # type: ignore[misc]
    def cost(x: Frame) -> float:
        return energy_fn(x)

    @function.numpy(manifold)  # type: ignore[misc]
    def grad(x: Frame) -> np.ndarray:
        # Ensure the provided gradient is tangent
        return manifold.projection(x, grad_fn(x))

    problem = Problem(manifold, cost=cost, riemannian_gradient=grad)
    if "max_iterations" in solver_kw:
        raise TypeError("Pass 'maxiter' positional, not in solver_kw")
    verbosity = solver_kw.pop("verbosity", 0)
    solver = ConjugateGradient(
        max_iterations=maxiter,
        verbosity=verbosity,
        **solver_kw,
    )
    return cast(Frame, solver.run(problem, initial_point=frame0).point)
