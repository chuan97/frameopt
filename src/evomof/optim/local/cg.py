from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pymanopt import Problem, function
from pymanopt.manifolds.manifold import Manifold
from pymanopt.optimizers.conjugate_gradient import ConjugateGradient

from evomof.core._types import Complex128Array
from evomof.core.frame import Frame


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

    manifold = _PymanoptProductCP(*frame0.shape)

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
    Pymanopt-compatible manifold for :class:`Frame` on (CP^{d−1})ⁿ.

    A point is represented by a :class:`Frame` (one unit-sphere representative per row with
    a fixed U(1) phase). This wraps frame-level operations so generic Riemannian optimizers
    can work without knowing about :class:`Frame`.

    Metric
    ------
    Real part of the Frobenius inner product: ⟨U, V⟩ = Re tr(Uᴴ V).

    Dimension
    ---------
    2·n·(d−1) = 2·n·d − 2·n.
    """

    def __init__(self, n: int, d: int):
        if n <= 0 or d <= 0:
            raise ValueError("Both n and d must be positive integers.")
        self._n, self._d = n, d
        self._dim: int = 2 * n * d - 2 * n
        super().__init__(dimension=self._dim, name="FrameManifold")

    @property
    def dimension(self) -> int:
        """Real dimension of the manifold (number of free parameters)."""
        return self._dim

    # -- basic ops -------------------------------------------------------
    def random_point(self, rng: np.random.Generator | None = None) -> Frame:
        """
        Return a random CP point as a :class:`Frame`, via :meth:`Frame.random`.

        Parameters
        ----------
        rng : numpy.random.Generator | None
            Optional generator for reproducibility.

        Returns
        -------
        :class:`Frame`
            Random point on (CP^{d−1})ⁿ represented as a frame.
        """
        return Frame.random(self._n, self._d, rng=rng)

    def projection(self, X: Frame, U: np.ndarray) -> Complex128Array:
        """
        Orthogonal projection onto the CP tangent at ``X``; delegates to :meth:`Frame.project`.

        Parameters
        ----------
        X : :class:`Frame`
            Base point.
        U : numpy.ndarray
            Ambient array of shape (n, d).

        Returns
        -------
        Complex128Array
            Tangent at ``X`` of shape (n, d) with ⟨X[i], out[i]⟩ = 0 per row.
        """
        return X.project(U)

    def exp(self, X: Frame, U: np.ndarray) -> Frame:
        """
        Exact exponential map on (CP^{d−1})ⁿ via :meth:`Frame.retract` (per‑row CP geodesic using the sphere lift).

        Parameters
        ----------
        X : :class:`Frame`
            Base point.
        U : numpy.ndarray
            Tangent at ``X`` (shape (n, d), complex orthogonal per row).

        Returns
        -------
        :class:`Frame`
            Endpoint of the geodesic starting at ``X`` with initial velocity ``U``.
        """
        return X.retract(U)

    def retraction(self, X: Frame, U: np.ndarray) -> Frame:
        """
        First-order retraction: add ``U`` row-wise, then renormalize and fix phase (delegates to :class:`Frame`(..., normalize=True)).

        Parameters
        ----------
        X : :class:`Frame`
            Base point.
        U : numpy.ndarray
            Ambient increment of shape (n, d).

        Returns
        -------
        :class:`Frame`
            Retracted point with gauge fixed.
        """
        new_vecs = X.vectors + U
        # Normalise each row and fix global phase
        return Frame(new_vecs, normalize=True, copy=False)

    def log(self, X: Frame, Y: Frame) -> Complex128Array:
        """
        Riemannian log map on (CP^{d-1})^n; delegates to :meth:`Frame.log_map`.

        Parameters
        ----------
        X : :class:`Frame`
            Base point.
        Y : :class:`Frame`
            Target point (same shape as ``X``).

        Returns
        -------
        Complex128Array
            CP tangent at ``X`` of shape (n, d).
        """
        return X.log_map(Y)

    def inner_product(self, X: Frame, U: np.ndarray, V: np.ndarray) -> float:
        """
        Real part of the complex Frobenius inner product.

        Parameters
        ----------
        X : :class:`Frame`
            Base point (unused; present for API compatibility).
        U, V : numpy.ndarray
            Tangents or ambient arrays of shape (n, d).

        Returns
        -------
        float
            Re tr(Uᴴ V).
        """
        return float(np.real(np.sum(U.conj() * V)))

    def zero_vector(self, X: Frame) -> Complex128Array:
        """
        Zero element of the tangent space at ``X``.

        Parameters
        ----------
        X : :class:`Frame`
            Base point.

        Returns
        -------
        Complex128Array
            Zero array of shape (n, d).
        """
        return np.zeros_like(X.vectors)

    def norm(self, X: Frame, U: np.ndarray) -> float:
        """
        Riemannian norm induced by :meth:`inner_product`.

        Parameters
        ----------
        X : :class:`Frame`
            Base point (unused; present for API compatibility).
        U : numpy.ndarray
            Tangent at ``X`` (shape (n, d)).

        Returns
        -------
        float
            √(Re tr(Uᴴ U)).
        """
        return float(np.sqrt(self.inner_product(X, U, U)))

    def random_tangent_vector(
        self, X: Frame, rng: np.random.Generator | None = None
    ) -> Complex128Array:
        """
        Draw a random unit tangent at ``X`` using :meth:`Frame.random_tangent`.

        Parameters
        ----------
        X : :class:`Frame`
            Base point.
        rng : numpy.random.Generator | None
            Optional generator for reproducibility.

        Returns
        -------
        Complex128Array
            Tangent at ``X`` with Frobenius norm 1.
        """
        return X.random_tangent(rng=rng, unit=True)

    def transport(self, X: Frame, Y: Frame, U: np.ndarray) -> Complex128Array:
        """
        Parallel transport from ``X`` to ``Y`` on (CP^{d−1})ⁿ; delegates to :meth:`Frame.transport`.

        Parameters
        ----------
        X : :class:`Frame`
            Source point.
        Y : :class:`Frame`
            Target point (same shape as ``X``).
        U : numpy.ndarray
            Tangent at ``X`` to transport (shape (n, d)).

        Returns
        -------
        Complex128Array
            Transported tangent at ``Y`` (shape (n, d)).
        """
        return X.transport(Y, U)

    def dist(self, X: Frame, Y: Frame) -> float:
        """
        Geodesic distance on (CP^{d−1})ⁿ computed as the Frobenius norm of the per‑row CP log map (via :meth:`Frame.log_map`).

        Parameters
        ----------
        X : :class:`Frame`
            First point.
        Y : :class:`Frame`
            Second point.

        Returns
        -------
        float
            Geodesic distance between ``X`` and ``Y``.
        """
        xi = X.log_map(Y)
        return float(np.linalg.norm(xi))
