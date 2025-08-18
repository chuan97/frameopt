import typing

import numpy as np
from pymanopt.manifolds.manifold import Manifold

from evomof.core._types import Complex128Array
from evomof.core.frame import Frame


class FrameManifold(Manifold):  # type: ignore[misc]
    """
    Thin wrapper exposing a Pymanopt-compatible manifold for `Frame`.

    The underlying manifold is the product
    \(\mathcal M = (S^{2d-1})^{\times n}\) of complex unit spheres where a
    point is represented by a :class:`~evomof.core.frame.Frame` (rows have unit
    \ell_2 norm). This class only forwards operations to the corresponding
    `Frame` methods so generic Riemannian optimisers can work without knowing
    about `Frame`.

    * **Metric** – real part of the Frobenius inner product
      \(\langle U,V\rangle_X = \operatorname{Re}\,\mathrm{tr}(U^\dagger V)\).
    * **Dimension** – \(\dim\mathcal M = 2nd - 2n\).
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
        Return a Haar‑random frame by calling :meth:`Frame.random`.
        """
        return Frame.random(self._n, self._d, rng=rng)

    def projection(self, X: Frame, U: np.ndarray) -> Complex128Array:
        """
        Orthogonal projection onto \(T_X\mathcal M\), delegates to
        :meth:`Frame.project`.
        """
        return X.project(U)

    def exp(self, X: Frame, U: np.ndarray) -> Frame:
        """
        Exact exponential map via :meth:`Frame.retract` (geodesic per row).
        """
        return X.retract(U)

    def retraction(self, X: Frame, U: np.ndarray) -> Frame:
        """
        First‑order retraction implemented by renormalising rows via
        :meth:`Frame.from_array`.
        """
        new_vecs = X.vectors + U
        # Normalise each row and fix global phase
        return Frame.from_array(new_vecs, copy=False)

    def log(self, X: Frame, Y: Frame) -> Complex128Array:
        """
        Riemannian log map, delegates to :meth:`Frame.log_map`.
        """
        return X.log_map(Y)

    def inner_product(self, X: Frame, U: np.ndarray, V: np.ndarray) -> float:
        """
        Real part of the complex Frobenius inner product.
        """
        return float(np.real(np.sum(U.conj() * V)))

    def zero_vector(self, X: Frame) -> Complex128Array:
        """
        Zero element of \(T_X\mathcal M\).
        """
        return typing.cast(Complex128Array, np.zeros_like(X.vectors))

    def norm(self, X: Frame, U: np.ndarray) -> float:
        """
        Riemannian norm induced by :meth:`inner_product`.
        """
        return float(np.sqrt(self.inner_product(X, U, U)))

    def random_tangent_vector(
        self, X: Frame, rng: np.random.Generator | None = None
    ) -> Complex128Array:
        """
        Draw a random unit tangent at ``X`` using :meth:`Frame.random_tangent`.
        """
        return X.random_tangent(rng=rng, unit=True)

    def transport(self, X: Frame, Y: Frame, U: np.ndarray) -> Complex128Array:
        """
        Parallel transport from ``X`` to ``Y``, delegates to :meth:`Frame.transport`.
        """
        return X.transport(Y, U)

    def dist(self, X: Frame, Y: Frame) -> float:
        """
        Geodesic distance computed as the Frobenius norm of the per‑row log map,
        via :meth:`Frame.log_map`.
        """
        xi = X.log_map(Y)
        return float(np.linalg.norm(xi))


__all__: list[str] = ["FrameManifold"]
