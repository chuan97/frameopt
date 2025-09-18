import numpy as np
from pymanopt.manifolds.manifold import Manifold

from evomof.core._types import Complex128Array
from evomof.core.frame import Frame

__all__: list[str] = ["FrameManifold"]


class FrameManifold(Manifold):  # type: ignore[misc]
    """
    Pymanopt-compatible manifold for :class:`Frame` on (CP^{d-1})^n.

    A point is represented by a :class:`Frame` (one unit-sphere representative per row with
    a fixed U(1) phase). This wraps frame-level operations so generic Riemannian optimizers
    can work without knowing about :class:`Frame`.

    Metric
    ------
    Real part of the Frobenius inner product: <U, V> = Re tr(U^H V).

    Dimension
    ---------
    2*n*(d - 1) = 2*n*d - 2*n.
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
            Random point on (CP^{d-1})^n represented as a frame.
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
            Tangent at ``X`` of shape (n, d) with <X[i], out[i]> = 0 per row.
        """
        return X.project(U)

    def exp(self, X: Frame, U: np.ndarray) -> Frame:
        """
        Exact exponential map on (CP^{d-1})^n via :meth:`Frame.retract` (per-row CP geodesic using the sphere lift).

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
            Re tr(U^H V).
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
            sqrt( Re tr(U^H U) ).
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
        Parallel transport from ``X`` to ``Y`` on (CP^{d-1})^n; delegates to :meth:`Frame.transport`.

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
        Geodesic distance on (CP^{d-1})^n computed as the Frobenius norm of the per-row CP log map (via :meth:`Frame.log_map`).

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
