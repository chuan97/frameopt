from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pymanopt import Problem, function
from pymanopt.optimizers.trust_regions import TrustRegions

from frameopt.core._types import Complex128Array
from frameopt.core.frame import Frame
from frameopt.core.manifold import PRODUCT_CP
from frameopt.optim.local._pymanopt_adapters import PymanoptProductCP

__all__ = ["minimize"]


def minimize(
    frame0: Frame,
    energy_fn: Callable[[Frame], float],
    grad_fn: Callable[[Frame], Complex128Array],
    maxiter: int = 50,
    **solver_kw: Any,
) -> Frame:
    """
    Riemannian trust‑regions.

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
        Maximum number of trust‑regions iterations (must be positive).
    **solver_kw :
        Extra keyword arguments forwarded to
        :class:`pymanopt.optimizers.trust_regions.TrustRegions`.
        Trust‑regions accepts parameters like ``miniter``, ``kappa``, ``theta``, ``rho_prime``, ``use_rand``, ``rho_regularization``, as well as generic optimizer options.
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

    manifold = PymanoptProductCP(*frame0.shape, geom=PRODUCT_CP)

    # --- cost & gradient wrapped with pymanopt decorator ----------------
    @function.numpy(manifold)  # type: ignore[misc]
    def cost(x: Frame) -> float:
        return energy_fn(x)

    @function.numpy(manifold)  # type: ignore[misc]
    def grad(x: Frame) -> np.ndarray:
        return grad_fn(x)

    # --- Hessian-vector product via central finite differences of Riemannian grad ---
    @function.numpy(manifold)  # type: ignore[misc]
    def rhess(x: Frame, eta: np.ndarray) -> np.ndarray:
        # Use a symmetric FD along the retraction; project gradients back to T_x.
        # Note: this is a nonlinear approximation (RTR-FD). TR handles safeguards.
        h = 1e-8
        gx = grad(x)
        x_f = manifold.retraction(x, h * eta)
        x_b = manifold.retraction(x, -h * eta)
        gf = grad(x_f)
        gb = grad(x_b)
        # Map gradients to T_x to keep the operator in the correct space.
        Pf = manifold.to_tangent_space(x, gf)
        Pb = manifold.to_tangent_space(x, gb)
        hx = (Pf - Pb) / (2.0 * h)
        return hx

    problem = Problem(
        manifold, cost=cost, riemannian_gradient=grad, riemannian_hessian=rhess
    )
    if "max_iterations" in solver_kw:
        raise TypeError("Pass 'maxiter' positional, not in solver_kw")

    verbosity = solver_kw.pop("verbosity", 0)

    kwargs = {
        "max_time": np.inf,
        "min_gradient_norm": 0.0,
        "min_step_size": 0.0,
        "max_cost_evaluations": np.inf,
    }
    solver_kw = kwargs | solver_kw

    solver = TrustRegions(
        max_iterations=maxiter,
        verbosity=verbosity,
        **solver_kw,
    )
    result = solver.run(problem, initial_point=frame0)
    sc = result.stopping_criterion
    assert sc.startswith("Terminated - max iterations reached"), sc
    final_frame: Frame = result.point

    return final_frame
