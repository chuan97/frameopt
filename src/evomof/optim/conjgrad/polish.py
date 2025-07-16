from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
from pymanopt import Problem, function
from pymanopt.optimizers.conjugate_gradient import ConjugateGradient

from evomof.core.frame import Frame
from evomof.optim.manifold import FrameManifold  # see §3


def polish_with_conjgrad(
    frame0: Frame,
    energy_fn: Callable[[Frame], float],
    grad_fn: Callable[[Frame], np.ndarray],
    maxiter: int = 50,
    **solver_kw: Any,
) -> Frame:
    """
    One-shot Riemannian conjugate-gradient polish starting from *frame0*.
    Returns the optimised frame.
    """
    manifold = FrameManifold(*frame0.shape)

    @function.numpy(manifold)  # type: ignore[misc]
    def cost(x: Frame) -> float:
        return energy_fn(x)

    @function.numpy(manifold)  # type: ignore[misc]
    def grad(x: Frame) -> np.ndarray:
        return grad_fn(x)

    problem = Problem(manifold, cost=cost, riemannian_gradient=grad)
    solver = ConjugateGradient(max_iterations=maxiter, verbosity=0, **solver_kw)
    return cast(Frame, solver.run(problem, initial_point=frame0).point)
