import typing

import numpy as np
import numpy.linalg as npl
from pymanopt.manifolds.manifold import Manifold

from evomof.core._types import Complex128Array
from evomof.core.frame import Frame


class FrameManifold(Manifold):  # type: ignore[misc]
    r"""
    Product manifold :math:`\mathcal M = (S^{2d-1})^{\times n}` of *n*
    complex unit spheres, where each point is represented by an
    :class:`~evomof.core.frame.Frame` whose rows have unit ℓ₂‑norm.

    * **Metric** – real part of the Frobenius inner product
      :math:`\langle U,V\rangle_X = \operatorname{Re}\,\mathrm{tr}(U^\dagger V)`.
    * **Dimension** – :math:`\dim\mathcal M = 2nd - 2n`.

    The manifold provides the minimal interface required by **Pymanopt**
    (random point, projection, retraction, log/exp, transport, …) so that
    generic Riemannian optimisers can operate directly on
    :class:`~evomof.core.frame.Frame` objects.
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
        Orthogonally project an ambient perturbation onto the tangent
        space :math:`T_X\mathcal M`.

        Parameters
        ----------
        X :
            Base frame at which the projection is taken.
        U :
            Ambient array of shape ``X.shape``.

        Returns
        -------
        Complex128Array
            Tangent array orthogonal to every row of *X*.
        """
        return X.project(U)

    def exp(self, X: Frame, U: np.ndarray) -> Frame:
        """
        Exact exponential map on the product sphere.

        For each row the update is
        :math:`f \;\mapsto\; \cos\|u\|\,f + \sin\|u\|\,\tfrac{u}{\|u\|}`,
        which is implemented by :pymeth:`Frame.retract`.

        Parameters
        ----------
        X :
            Base frame.
        U :
            Tangent perturbation at *X*.

        Returns
        -------
        Frame
            Point reached by following the geodesic emanating from ``X``
            with initial velocity ``U`` for unit time.
        """
        return X.retract(U)

    def retraction(self, X: Frame, U: np.ndarray) -> Frame:
        """
        Cheaper first‑order retraction: renormalise rows directly.

        Parameters
        ----------
        X :
            Base frame.
        U :
            Tangent perturbation at *X*.

        Returns
        -------
        Frame
            Retracted frame lying back on the manifold.
        """
        new_vecs = X.vectors + U
        # Normalise each row and fix global phase
        return Frame.from_array(new_vecs, copy=False)

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
            Tangent vector (same shape as the frames) in :math:`T_X\mathcal M`.
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

    def zero_vector(self, X: Frame) -> Complex128Array:
        """Return the zero element of T_X𝓜."""
        return typing.cast(Complex128Array, np.zeros_like(X.vectors))

    def norm(self, X: Frame, U: np.ndarray) -> float:
        """Riemannian norm induced by the inner product."""
        return float(np.sqrt(self.inner_product(X, U, U)))

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

    def transport(self, X: Frame, Y: Frame, U: np.ndarray) -> Complex128Array:
        """
        Parallel–transport a tangent vector from ``X`` to ``Y``.

        For each row we use the exact formula on the unit 2‑sphere
        (complex version):

        .. math::

            \operatorname{PT}_{X\to Y}(U)
            = U -
              \frac{\langle Y,U\rangle}{1 + \langle X,Y\rangle}\,(X + Y).

        The result lives in :math:`T_Y\mathcal M`.  A final projection is
        applied for numerical safety.
        """
        # Inner products row‑wise (complex)
        dot_yu = np.sum(Y.vectors.conj() * U, axis=1, keepdims=True)  # ⟨Y,U⟩
        dot_xy = np.sum(X.vectors.conj() * Y.vectors, axis=1, keepdims=True)  # ⟨X,Y⟩
        factor = dot_yu / (1.0 + dot_xy)

        transported = U - factor * (X.vectors + Y.vectors)
        return self.projection(Y, transported)

    def dist(self, X: Frame, Y: Frame) -> float:
        """
        Geodesic distance induced by the 2‑norm on the product sphere.

        Each row lives on a unit 2‑sphere, so the distance is the Frobenius
        norm of the per‑row log map.
        """
        xi = X.log_map(Y)
        return float(np.linalg.norm(xi))


__all__: list[str] = ["FrameManifold"]
