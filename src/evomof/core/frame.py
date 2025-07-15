# src/evomof/core/frame.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Tuple

import numpy as np

__all__: Final = ["Frame"]


@dataclass(slots=True)
class Frame:
    """
    A collection of `n` complex d-dimensional unit vectors.

    The first non-zero component of every vector is made real-positive to
    quotient out the irrelevant global U(1) phase.
    """

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
        frame._renormalise()
        return frame

    @classmethod
    def random(
        cls,
        n: int,
        d: int,
        rng: np.random.Generator | None = None,
    ) -> "Frame":
        """Haar-uniform random frame on the unit sphere."""
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
    def gram(self) -> np.ndarray:
        """`(n, n)` matrix of inner-products 〈f_i, f_j〉."""
        return self.vectors @ self.vectors.conj().T

    def chordal_distances(self) -> np.ndarray:
        """Pairwise chordal distances ∥f_i − f_j∥₂ between *rows*."""
        g = np.abs(self.gram) ** 2
        np.fill_diagonal(g, 1.0)
        return np.sqrt(2.0 - 2.0 * np.sqrt(g))

    # -------------------------------------------------------------- #
    # Manifold operations (sphere product ≅ CP^{d-1})               #
    # -------------------------------------------------------------- #

    def retract(self, tang: np.ndarray) -> "Frame":
        """
        Exact exponential-map retraction (sphere geodesic).

        `tang` must be tangent:  Re⟨f_i, t_i⟩ = 0  for every row.
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

    def log_map(self, other: "Frame") -> np.ndarray:
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
        return tang.astype(np.complex128)

    # -------------------------------------------------------------- #
    # Private helpers                                               #
    # -------------------------------------------------------------- #

    def _renormalise(self) -> None:
        """Unit-normalise rows and fix first non-zero component’s phase."""
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors /= norms

        for vec in self.vectors:
            nz = np.flatnonzero(vec)
            if nz.size:
                phase = np.angle(vec[nz[0]])
                vec *= np.exp(-1j * phase)

    # ------------------------------------------------------------------ #
    # Dunder methods for convenience                                     #
    # ------------------------------------------------------------------ #

    def copy(self) -> "Frame":
        return Frame.from_array(self.vectors, copy=True)

    def __iter__(self):  # type: ignore [return-value]
        return iter(self.vectors)

    def __repr__(self) -> str:  # pragma: no cover
        n, d = self.shape
        return f"Frame(n={n}, d={d})"
