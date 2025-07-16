# src/evomof/core/frame.py
from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Final, Iterator, Tuple

import numpy as np

from evomof.core._types import Complex128Array, Float64Array

__all__: Final = ["Frame"]


@dataclass(slots=True)
class Frame:
    """
    A collection of `n` complex d-dimensional unit vectors.

    The first non-zero component of every vector is made real-positive to
    quotient out the irrelevant global U(1) phase.
    """

    # ------------------------------------------------------------------ #
    # Dataclass validation                                               #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """Ensure internal array is complex128 and 2‑D (n, d)."""
        if self.vectors.ndim != 2:
            raise ValueError("Frame.vectors must be 2‑D (n, d)")
        # Promote to complex128 if necessary (no copy when already correct)
        if not np.iscomplexobj(self.vectors) or self.vectors.dtype != np.complex128:
            self.vectors = self.vectors.astype(np.complex128, copy=False)
        # We rely on normalisation/phase fix elsewhere; don't enforce here

    vectors: np.ndarray  # shape (n, d), complex128/complex64

    # --------------------------------------------------------------------- #
    # Constructors
    # --------------------------------------------------------------------- #

    @classmethod
    def from_array(cls, arr: np.ndarray, *, copy: bool = True) -> "Frame":
        """
        Wrap an existing `(n, d)` complex array.

        Normalises and fixes phases in-place unless `copy=False` is chosen.
        """
        if arr.ndim != 2:
            raise ValueError("`arr` must be 2-D (n, d)")

        vecs = arr.copy() if copy else arr
        frame = cls(vecs.astype(np.complex128, copy=False))
        frame.renormalise()
        return frame

    @classmethod
    def random(
        cls,
        n: int,
        d: int,
        rng: np.random.Generator | None = None,
    ) -> "Frame":
        """
        Return a frame whose rows are sampled uniformly from the unit sphere
        ``S^{2d-1}`` and then gauge‑fixed.

        Uniformity is achieved by normalising i.i.d. complex‑Gaussian vectors,
        which is equivalent to taking the first column of a Haar‑random
        unitary.  The subsequent phase fix (first non‑zero entry real‑positive)
        chooses one representative per projective equivalence class and does
        **not** bias the distribution.

        Parameters
        ----------
        n :
            Number of vectors (rows).
        d :
            Ambient complex dimension.
        rng :
            Optional :class:`numpy.random.Generator` for reproducibility.  If
            *None*, a fresh default generator is used.

        Returns
        -------
        Frame
            A new random frame with shape ``(n, d)``.
        """
        rng = rng or np.random.default_rng()
        z = rng.standard_normal((n, d)) + 1j * rng.standard_normal((n, d))
        return cls.from_array(z, copy=False)

    # ------------------------------------------------------------------ #
    # Public geometry helpers
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> Tuple[int, int]:
        return self.vectors.shape

    @property
    def gram(self) -> Complex128Array:
        """Return the complex Gram matrix ``G = V V†`` of shape ``(n, n)``."""
        g = self.vectors @ self.vectors.conj().T
        return typing.cast(Complex128Array, g)

    def chordal_distances(self) -> Float64Array:
        """
        Pair‑wise **chordal distances** between frame vectors.

        We define the chordal distance via the overlap magnitude
        :math:`x = |\\langle f_i, f_j \\rangle|` as

        .. math::

            D(x) \;=\; 2 \\sqrt{1 - x^{2}}.

        The returned array has shape ``(n, n)`` with zeros on the diagonal.
        """
        g = np.abs(self.gram) ** 2
        np.fill_diagonal(g, 1.0)
        dist = 2 * np.sqrt(np.maximum(1.0 - g, 0.0))
        return typing.cast(Float64Array, dist)

    # -------------------------------------------------------------- #
    # Manifold operations (sphere product ≅ CP^{d-1})               #
    # -------------------------------------------------------------- #

    def retract(self, tang: np.ndarray) -> "Frame":
        """
        Exact exponential-map retraction (per‑row great‑circle step).

        Given a base frame ``self`` and a tangent perturbation ``tang`` lying
        in the product tangent space
        :math:`T_{\\text{self}}(S^{2d-1})^{n}`, this method returns a *new*
        :class:`Frame` whose rows are obtained by moving along the geodesic
        starting at each original vector:

        .. math::

            f_i' \;=\; \cos\\lVert\\xi_i\\rVert \, f_i \;+\;
                     \\frac{\\sin\\lVert\\xi_i\\rVert}{\\lVert\\xi_i\\rVert}\,
                     \\xi_i,

        where :math:`\\xi_i` is the *i*-th row of ``tang``.  For
        :math:`\\lVert\\xi_i\\rVert \\to 0` the Taylor expansion reduces to the
        familiar first‑order update ``f_i + xi_i`` followed by re‑normalisation.

        Parameters
        ----------
        tang :
            Complex array of shape ``self.shape`` representing a tangent vector
            field.  It **must** satisfy
            :math:`\\operatorname{Re}\\langle f_i, \\xi_i \\rangle = 0`
            for every row, i.e. it lies in the orthogonal complement of each
            original vector.

        Returns
        -------
        Frame
            A new frame with the same shape as ``self`` whose rows remain
            unit‑norm and have the same fixed global phase convention.

        Raises
        ------
        ValueError
            If the shape of ``tang`` is different from that of the base frame.
        """
        if tang.shape != self.shape:
            raise ValueError("Tangent array shape mismatch.")

        norms = np.linalg.norm(tang, axis=1, keepdims=True)
        # Where ‖ξ‖ very small, fall back to first‑order update
        small = norms < 1e-12
        scale_sin = np.zeros_like(norms)
        scale_cos = np.zeros_like(norms)

        scale_sin[~small] = np.sin(norms[~small]) / norms[~small]
        scale_cos[~small] = np.cos(norms[~small])

        # First‑order Taylor for tiny norms: cos≈1, sin≈norm
        scale_sin[small] = 1.0
        scale_cos[small] = 1.0 - 0.5 * norms[small] ** 2

        new_vecs = scale_cos * self.vectors + scale_sin * tang
        return Frame.from_array(new_vecs, copy=False)

    def log_map(self, other: "Frame") -> Complex128Array:
        """
        Compute the exact Riemannian logarithmic map on the product sphere.

        Given two frames with identical shape, this returns a tangent array
        ``xi`` such that::

            self.retract(xi) == other      (up to numerical precision)

        Each row‐pair ``(f_i, g_i)`` is treated independently:

        * θ = arccos Re⟨f_i, g_i⟩  is the great‑circle distance.
        * xi_i = (θ / sin θ) · (g_i − cos θ · f_i)

        The result lives in the tangent space ``T_self M`` and satisfies
        ``Re⟨f_i, xi_i⟩ = 0`` for every i.

        Parameters
        ----------
        other :
            Target frame with the same ``(n, d)`` shape.

        Returns
        -------
        np.ndarray
            Tangent array of shape ``self.shape`` (complex128).

        Raises
        ------
        ValueError
            If ``other`` does not have the same shape as ``self``.
        """
        if self.shape != other.shape:
            raise ValueError("Frame shapes mismatch")

        inner = np.real(np.sum(self.vectors.conj() * other.vectors, axis=1))
        inner = np.clip(inner, -1.0, 1.0)  # numerical safety
        theta = np.arccos(inner)  # angle on the sphere

        # Avoid division by zero for identical vectors
        mask = theta > 1e-12
        scale = np.zeros_like(theta)
        scale[mask] = theta[mask] / np.sin(theta[mask])

        diff = other.vectors - inner[:, None] * self.vectors
        tang = scale[:, None] * diff
        return typing.cast(Complex128Array, tang.astype(np.complex128))

    # -------------------------------------------------------------- #
    # Private helpers                                               #
    # -------------------------------------------------------------- #

    def renormalise(self) -> None:
        """
        In‑place normalisation and gauge fix.

        * Each row is scaled to unit L2 norm.
        * The global U(1) phase is removed by rotating every vector so that
          its first non‑zero component becomes real‑positive.

        Idempotent: calling this method multiple times leaves ``vectors``
        unchanged.
        """
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors /= norms

        for vec in self.vectors:
            nz = np.flatnonzero(vec)
            if nz.size:
                phase = np.angle(vec[nz[0]])
                vec *= np.exp(-1j * phase)

    # ------------------------------------------------------------------ #
    # Convenience & dunder methods                                       #
    # ------------------------------------------------------------------ #

    def copy(self) -> "Frame":
        return Frame.from_array(self.vectors, copy=True)

    def __iter__(self) -> Iterator[Complex128Array]:
        return iter(self.vectors)

    def __repr__(self) -> str:  # pragma: no cover
        n, d = self.shape
        return f"Frame(n={n}, d={d})"
