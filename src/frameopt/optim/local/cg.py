from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pymanopt import Problem, function
from pymanopt.manifolds.manifold import Manifold
from pymanopt.optimizers.conjugate_gradient import ConjugateGradient

from frameopt.core._types import Complex128Array
from frameopt.core.frame import Frame
from frameopt.core.manifold import PRODUCT_CP, ProductCP


def minimize(
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

    manifold = _PymanoptProductCP(*frame0.shape, geom=PRODUCT_CP)

    # --- cost & gradient wrapped with pymanopt decorator ----------------
    @function.numpy(manifold)  # type: ignore[misc]
    def cost(x: Frame) -> float:
        return energy_fn(x)

    @function.numpy(manifold)  # type: ignore[misc]
    def grad(x: Frame) -> np.ndarray:
        return grad_fn(x)

    problem = Problem(manifold, cost=cost, riemannian_gradient=grad)
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

    solver = ConjugateGradient(
        max_iterations=maxiter,
        verbosity=verbosity,
        **solver_kw,
    )
    result = solver.run(problem, initial_point=frame0)
    assert result.stopping_criterion[:35] == "Terminated - max iterations reached"
    final_frame: Frame = result.point

    return final_frame


class _PymanoptProductCP(Manifold):  # type: ignore[misc]
    """
    Minimal Pymanopt manifold wrapper for :class:`Frame` on (CP^{d−1})ⁿ.

    Delegates all geometry to :class:`frameopt.core.manifold.ProductCP`.
    """

    def __init__(self, n: int, d: int, geom: ProductCP | None = None):
        if n <= 0 or d <= 0:
            raise ValueError("Both n and d must be positive integers.")

        self._n, self._d = n, d
        self._dim: int = 2 * n * d - 2 * n
        self._geom = PRODUCT_CP if geom is None else geom

        super().__init__(dimension=self._dim, name="FrameManifold")

    @property
    def dimension(self) -> int:
        """Real dimension of the manifold (number of free parameters)."""
        return self._dim

    # -- basic ops -------------------------------------------------------
    def random_point(self, rng: np.random.Generator | None = None) -> Frame:
        """
        Random CP point as a :class:`Frame` (delegates to :meth:`Frame.random`).
        """
        return Frame.random(self._n, self._d, rng=rng)

    def projection(self, X: Frame, U: np.ndarray) -> Complex128Array:
        """
        Orthogonal projection onto the CP tangent at ``X``; delegates to :class:`ProductCP`.
        """
        return self._geom.project_to_tangent(X, U)

    def exp(self, X: Frame, U: np.ndarray) -> Frame:
        """
        Exponential map via :meth:`ProductCP.retract`.
        """
        return self._geom.retract(X, U)

    def retraction(self, X: Frame, U: np.ndarray) -> Frame:
        """
        Retraction per :class:`ProductCP` policy ("sphere" or "normalize").
        """
        return self._geom.retract(X, U)

    def log(self, X: Frame, Y: Frame) -> Complex128Array:
        """
        Log map via :meth:`ProductCP.log_map`.
        """
        return self._geom.log_map(X, Y)

    def inner_product(self, X: Frame, U: np.ndarray, V: np.ndarray) -> float:
        """
        Real part of the Frobenius inner product.
        """
        return float(np.real(np.sum(U.conj() * V)))

    def zero_vector(self, X: Frame) -> Complex128Array:
        """
        Zero element of T_X.
        """
        return np.zeros_like(X.vectors)

    def norm(self, X: Frame, U: np.ndarray) -> float:
        """
        ‖U‖ induced by :meth:`inner_product`.
        """
        return float(np.sqrt(self.inner_product(X, U, U)))

    def random_tangent_vector(
        self, X: Frame, rng: np.random.Generator | None = None
    ) -> Complex128Array:
        """
        Random unit tangent via :meth:`ProductCP.random_tangent`.
        """
        return self._geom.random_tangent(X, rng=rng, unit=True)

    def transport(self, X: Frame, Y: Frame, U: np.ndarray) -> Complex128Array:
        """
        Transport via :meth:`ProductCP.transport`.
        """
        return self._geom.transport(X, Y, U)

    def dist(self, X: Frame, Y: Frame) -> float:
        """
        Geodesic distance via ‖log_map(X, Y)‖.
        """
        xi = self._geom.log_map(X, Y)
        return float(np.linalg.norm(xi))
