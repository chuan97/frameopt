import numpy as np
from pymanopt.manifolds.manifold import Manifold

from frameopt.core._types import Complex128Array
from frameopt.core.frame import Frame
from frameopt.core.manifold import PRODUCT_CP, ProductCP

__all__ = ["PymanoptProductCP"]


class PymanoptProductCP(Manifold):  # type: ignore[misc]
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

    def to_tangent_space(self, X: Frame, U: np.ndarray) -> Complex128Array:
        """
        Ensure ``U`` lies in T_X by projecting to the tangent space (used by TR).
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
