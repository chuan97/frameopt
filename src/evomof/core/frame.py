# src/evomof/core/frame.py
from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Final, Iterator, Tuple

import numpy as np

from evomof.core._types import Complex128Array

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
    # Tangent‑space helper                                               #
    # ------------------------------------------------------------------ #
    def project(self, arr: np.ndarray) -> Complex128Array:
        """
        Orthogonally project an ambient array onto the tangent space
        at this frame.

        For each row ``i`` the projection subtracts the
        inner product with the base vector so that the result satisfies

        ``⟨f_i, ξ_i⟩ = 0``.

        Parameters
        ----------
        arr :
            Complex array of shape ``self.shape``.  It does **not** need to
            be tangent already.

        Returns
        -------
        Complex128Array
            Tangent array of the same shape as the frame.
        """
        inner = np.sum(self.vectors.conj() * arr, axis=1, keepdims=True)
        return typing.cast(Complex128Array, arr - inner * self.vectors)

    # -------------------------------------------------------------- #
    # Manifold operations (sphere product ≅ CP^{d-1})                #
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

    def transport(
        self, other: "Frame", tang: np.ndarray, eps: float = 1e-12
    ) -> Complex128Array:
        """
        Exact parallel transport on the CP lift (row-wise), using phase alignment.
        For each row:
        - Let s = <x,y>. Set phase = s/|s| (or 1 if |s|≈0),
            y_tilde = y / phase so <x,y_tilde>∈R_{>=0}.
        - Apply sphere PT with y_tilde:
            xi_tilde = U - (<y_tilde,U>/(1+<x,y_tilde>)) (x + y_tilde).
        - Rotate back: xi = phase * xi_tilde.

        This preserves tangency (⟨y,xi⟩=0)
        and the tangent norm exactly in exact arithmetic.
        """
        if other.shape != self.shape:
            raise ValueError("Frame shapes mismatch")
        if tang.shape != self.shape:
            raise ValueError("Tangent array shape mismatch.")

        x = self.vectors  # (n, d)
        y = other.vectors  # (n, d)

        # Row-wise complex inner product s = <x, y>
        s = np.sum(x.conj() * y, axis=1, keepdims=True)  # (n,1) complex
        abs_s = np.abs(s)
        # Phase to align <x, y_tilde> to be real nonnegative
        phase = np.ones_like(s, dtype=np.complex128)
        nz = abs_s > eps
        phase[nz] = s[nz] / abs_s[nz]  # e^{i φ}
        y_tilde = y / phase  # y * e^{-i φ}

        # Inner products with aligned target
        dot_yu = np.sum(y_tilde.conj() * tang, axis=1, keepdims=True)  # <y_tilde, U>
        dot_xy = np.sum(
            x.conj() * y_tilde, axis=1, keepdims=True
        )  # <x, y_tilde> ∈ ℝ ideally
        # Use the real part for numerical robustness (should be real after alignment)
        dot_xy_real = np.real(dot_xy)

        denom = 1.0 + dot_xy_real  # (n,1) real
        bad = np.abs(denom) < eps  # antipodal guard
        denom_safe = denom.copy()
        denom_safe[bad] = 1.0

        transported_tilde = tang - (dot_yu / denom_safe) * (x + y_tilde)
        transported = transported_tilde * phase  # rotate back

        # Fallback for (near) antipodal rows: projection transport
        if np.any(bad):
            idx = bad.ravel()
            transported[idx, :] = other.project(tang[idx, :])

        return typing.cast(Complex128Array, transported.astype(np.complex128))

    def random_tangent(
        self,
        rng: np.random.Generator | None = None,
        *,
        unit: bool = True,
    ) -> Complex128Array:
        """
        Draw a random tangent array at this Frame.

        Sampling: i.i.d. complex normal in ambient, projected to the tangent space.
        If `unit=True`, the result is normalised to Frobenius norm 1.

        Returns
        -------
        Complex128Array
            Shape == self.shape, tangent at `self`.
        """
        rng = rng or np.random.default_rng()
        Z = rng.standard_normal(self.shape) + 1j * rng.standard_normal(self.shape)
        U = self.project(Z.astype(np.complex128, copy=False))
        if unit:
            n = np.linalg.norm(U)
            if n > 0:
                U = U / n
        return U

    # ------------------------------------------------------------------ #
    # Convenience & dunder methods                                       #
    # ------------------------------------------------------------------ #

    def copy(self) -> "Frame":
        return Frame.from_array(self.vectors, copy=True)

    def __iter__(self) -> Iterator[Complex128Array]:
        return iter(self.vectors)

    def save_npy(self, path: str) -> None:
        """
        Save this frame's vectors to a NumPy .npy file.

        Parameters
        ----------
        path : str
            Path where the .npy file will be written.
        """
        # Save the complex array directly
        np.save(path, self.vectors)

    @classmethod
    def load_npy(cls, path: str) -> "Frame":
        """
        Load a Frame from a NumPy .npy file containing a complex array.

        Parameters
        ----------
        path : str
            Path to the .npy file to load.

        Returns
        -------
        Frame
            A Frame initialized from the loaded array.
        """
        arr = np.load(path)
        return cls.from_array(arr, copy=False)

    def export_txt(self, path: str) -> None:
        """
        Export this frame to a text file in the submission format:
        - First all real parts (row-major), one per line, then all imaginary parts.
        - Each number formatted with 18-digit exponential notation.

        Parameters
        ----------
        path : str
            Path where the .txt file will be written. Filename
            should follow `<d>x<n>_<tag>.txt` convention externally.
        """
        # Flatten row-major: rows are vectors
        flat_real = self.vectors.real.ravel(order="C")
        flat_imag = self.vectors.imag.ravel(order="C")
        with open(path, "w") as f:
            for val in flat_real:
                f.write(f"{val:.15e}\n")
            for val in flat_imag:
                f.write(f"{val:.15e}\n")

    def __repr__(self) -> str:  # pragma: no cover
        n, d = self.shape
        return f"Frame(n={n}, d={d})"
