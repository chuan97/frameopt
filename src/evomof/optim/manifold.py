import typing

import numpy as np
import numpy.linalg as npl
from pymanopt.manifolds.manifold import Manifold

from evomof.core._types import Complex128Array
from evomof.core.frame import Frame


class FrameManifold(Manifold):  # type: ignore[misc]
    """Product of unit spheres represented by our Frame object."""

    def __init__(self, n: int, d: int):
        self._n, self._d = n, d
        real_dim = 2 * n * d - 2 * n
        # Pass both keyword arguments to satisfy any Manifold signature variant
        super().__init__(dimension=real_dim, name="FrameManifold")

    @property
    def dimension(self) -> int:
        """Real dimension of the manifold (number of free parameters)."""
        return 2 * self._n * self._d - 2 * self._n

    # -- basic ops -------------------------------------------------------
    def random_point(self, rng: np.random.Generator | None = None) -> Frame:
        """
        Draw a Haar‑random point on the product sphere.

        Parameters
        ----------
        rng :
            Optional NumPy RNG for reproducibility.  If *None*, the global
            default generator is used.

        Returns
        -------
        Frame
            Sampled frame of shape ``(n, d)`` with unit‑norm rows.
        """
        return Frame.random(self._n, self._d, rng=rng)

    def projection(self, X: Frame, U: np.ndarray) -> Complex128Array:
        """
        Project an ambient perturbation onto the tangent space T_X𝓜.

        Computes ``U − Re⟨f_i,U_i⟩ f_i`` row‑wise so the result is
        orthogonal to each reference vector.

        Returns
        -------
        Complex128Array
            Tangent array with the same shape as ``U``.
        """
        radial = np.real(np.sum(U.conj() * X.vectors, axis=1, keepdims=True))
        projected = U - radial * X.vectors
        return typing.cast(Complex128Array, projected)

    def exp(self, X: Frame, U: np.ndarray) -> Frame:
        """
        Riemannian exponential map based on exact sphere geodesic.

        Equivalent to :pymeth:`Frame.retract`.

        Returns
        -------
        Frame
            New frame reached by moving along the geodesic with initial
            velocity ``U`` for unit time.
        """
        return X.retract(U)

    def retraction(self, X: Frame, U: np.ndarray) -> Frame:
        return self.exp(X, U)

    def log(self, X: Frame, Y: Frame) -> Complex128Array:
        """
        Riemannian logarithmic map (inverse of ``exp``).

        Parameters
        ----------
        X, Y :
            Frames of identical shape.

        Returns
        -------
        Complex128Array
            Tangent vector at ``X`` that reaches ``Y`` via ``exp``.
        """
        return X.log_map(Y)

    def inner_product(self, X: Frame, U: np.ndarray, V: np.ndarray) -> float:
        """
        Canonical inner product on the product sphere.

        Simply the real part of the complex Euclidean inner product.

        Returns
        -------
        float
            ⟨U, V⟩ₓ.
        """
        return float(np.real(np.sum(U.conj() * V)))

    # ---- zero vector ---------------------------------------------------
    def zero_vector(self, X: Frame) -> Complex128Array:
        """Return the zero element of T_X𝓜."""
        return typing.cast(Complex128Array, np.zeros_like(X.vectors))

    # ---- norm ----------------------------------------------------------
    def norm(self, X: Frame, U: np.ndarray) -> float:
        """Riemannian norm induced by the inner product."""
        return float(np.sqrt(self.inner_product(X, U, U)))

    # ---- random tangent -----------------------------------------------
    def random_tangent_vector(
        self, X: Frame, rng: np.random.Generator | None = None
    ) -> Complex128Array:
        """
        Draw a random unit tangent vector at ``X``.

        Sampling is i.i.d. complex normal followed by projection and
        normalisation.
        """
        rng = np.random.default_rng() if rng is None else rng
        Z = (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)).astype(
            np.complex128
        )
        tangent = self.projection(X, Z)
        tangent /= npl.norm(tangent)  # normalise
        return tangent

    # ---- vector transport ---------------------------------------------
    def transport(
        self, X: Frame, Y: Frame, U: np.ndarray
    ) -> Complex128Array:  # noqa: D401
        """
        Transport tangent vector ``U`` at ``X`` to the tangent space at ``Y``.

        For the product‑sphere manifold, parallel transport followed by
        projection reduces to simply projecting the same ambient vector
        onto ``T_Y𝓜``.
        """
        return self.projection(Y, U)

    # ---- distance ------------------------------------------------------
    def dist(self, X: Frame, Y: Frame) -> float:
        """
        Geodesic distance induced by the 2‑norm on the product sphere.

        Each row lives on a unit 2‑sphere, so the distance is the Frobenius
        norm of the per‑row log map.
        """
        xi = X.log_map(Y)
        return float(np.linalg.norm(xi))
