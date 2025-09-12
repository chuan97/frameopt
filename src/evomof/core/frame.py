# src/evomof/core/frame.py
from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np

from evomof.core._types import Complex128Array

__all__ = ["Frame"]


class Frame:
    """
    A collection of `n` complex d-dimensional unit vectors.

    The first non-zero component of every vector is made real-positive to
    quotient out the irrelevant global U(1) phase.
    """

    __slots__ = ("_vectors", "_is_normalized")

    def __init__(
        self,
        vectors: Complex128Array | np.ndarray,
        *,
        normalize: bool = True,
        copy: bool = True,
    ) -> None:
        """
        Create a Frame from an array of shape (n, d).

        Parameters
        ----------
        vectors : np.ndarray
            Complex array of shape (n, d).
        normalize : bool
            If True, call normalize() after storing the array.
        copy : bool
            If True (default), always make a private copy (safe).
            If False, require that `vectors` is already complex128,
            C-contiguous, and owned. Otherwise raises ValueError.

        Notes
        -----
        Internally, vectors are always stored as C-contiguous np.complex128.
        """
        if vectors.ndim != 2:
            raise ValueError("`vectors` must be 2-D (n, d)")

        if copy:
            arr = np.array(vectors, dtype=np.complex128, order="C", copy=True)
        else:
            if not (
                vectors.dtype == np.complex128
                and vectors.flags["C_CONTIGUOUS"]
                and vectors.flags["OWNDATA"]
            ):
                raise ValueError(
                    "When copy=False, input must be complex128, C-contiguous, and own its data"
                )
            arr = vectors

        self._vectors = arr
        self._is_normalized = False
        if normalize:
            self.normalize()

    @property
    def vectors(self) -> Complex128Array:
        """Read-only view of the underlying array."""
        v = self._vectors.view()
        v.setflags(write=False)
        return v

    @property
    def is_normalized(self) -> bool:
        """Whether this frame has been normalized (rows unit-norm and gauge-fixed)."""
        return self._is_normalized

    @classmethod
    def random(
        cls,
        n: int,
        d: int,
        rng: np.random.Generator | None = None,
    ) -> Frame:
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
        return cls(z, normalize=True, copy=False)

    # ------------------------------------------------------------------ #
    # Public geometry helpers
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> tuple[int, int]:
        return self._vectors.shape

    @property
    def gram(self) -> Complex128Array:
        """Return the complex Gram matrix ``G = V V†`` of shape ``(n, n)``."""
        return self._vectors @ self._vectors.conj().T

    def normalize(self) -> None:
        """
        In‑place normalisation and gauge fix.

        * Each row is scaled to unit L2 norm.
        * The global U(1) phase is removed by rotating every vector so that
            its first non‑zero component becomes real‑positive.

        Idempotent: calling this method multiple times leaves ``vectors``
        unchanged.
        """
        if self._is_normalized:
            return

        V = self._vectors
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("normalize(): zero or non-finite norm in frame vectors")

        V /= norms

        # Phase fix: rotate each row so its **first non-zero** entry is real‑positive
        n = self.shape[0]
        nz_mask = np.abs(V) > 0
        first_idx = np.argmax(nz_mask, axis=1)  # first True per row
        pivots = V[np.arange(n), first_idx]  # (n,)
        phases = np.conj(pivots / np.abs(pivots))  # e^{-i arg(pivot)} without trig
        V *= phases[:, None]

        self._is_normalized = True

    # ------------------------------------------------------------------ #
    # Tangent‑space helper                                               #
    # ------------------------------------------------------------------ #
    def project(self, arr: Complex128Array | np.ndarray) -> Complex128Array:
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
        inner = np.sum(self._vectors.conj() * arr, axis=1, keepdims=True)
        out: Complex128Array = arr - inner * self._vectors

        return out

    # -------------------------------------------------------------- #
    # Manifold operations (sphere product ≅ CP^{d-1})                #
    # -------------------------------------------------------------- #

    def retract(self, tang: Complex128Array | np.ndarray) -> Frame:
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

        new_vecs = scale_cos * self._vectors + scale_sin * tang

        return Frame(new_vecs, normalize=True, copy=False)

    def log_map(self, other: Frame) -> Complex128Array:
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

        inner = np.real(np.sum(self._vectors.conj() * other._vectors, axis=1))
        inner = np.clip(inner, -1.0, 1.0)  # numerical safety
        theta = np.arccos(inner)  # angle on the sphere

        # Avoid division by zero for identical vectors
        mask = theta > 1e-12
        scale = np.zeros_like(theta)
        scale[mask] = theta[mask] / np.sin(theta[mask])

        diff = other._vectors - inner[:, None] * self._vectors
        tang: Complex128Array = scale[:, None] * diff

        return tang

    def transport(
        self, other: Frame, tang: Complex128Array | np.ndarray, eps: float = 1e-12
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

        x = self._vectors  # (n, d)
        y = other._vectors  # (n, d)

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
        transported: Complex128Array = transported_tilde * phase  # rotate back

        # Fallback for (near) antipodal rows: projection transport
        if np.any(bad):
            idx = bad.ravel()
            transported[idx, :] = other.project(tang[idx, :])

        return transported

    def random_tangent(
        self,
        rng: np.random.Generator | None = None,
        *,
        unit: bool = True,
    ) -> Complex128Array:
        """
        Draw a random tangent array at this Frame.

        Sampling: i.i.d. complex normal in ambient, projected to the tangent space.
        If `unit=True`, the result is normalized to Frobenius norm 1.

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

    def copy(self) -> Frame:
        f = Frame(self._vectors, normalize=False, copy=True)
        f._is_normalized = self._is_normalized

        return f

    def __iter__(self) -> Iterator[Complex128Array]:
        return iter(self._vectors)

    def save_npy(self, path: str) -> None:
        """
        Save this frame's vectors to a NumPy .npy file.

        Parameters
        ----------
        path : str
            Path where the .npy file will be written.
        """
        # Save the complex array directly
        np.save(path, self._vectors)

    @classmethod
    def load_npy(cls, path: str) -> Frame:
        """
        Load a Frame from a NumPy .npy file containing a complex array.

        Parameters
        ----------
        path : str
            Path to the .npy file to load.

        Returns
        -------
        Frame
            A normalized Frame initialized from the loaded array.

        Notes
        -----
        If the stored array is not already in canonical normalized form
        (unit-norm rows with the first non-zero entry real-positive),
        a RuntimeWarning is emitted and the loaded data are normalized.
        """
        raw = np.asarray(np.load(path), dtype=np.complex128)
        # Build normalized copy using the standard constructor
        f_norm = cls(raw, normalize=True, copy=True)
        # Warn if the on-disk data were not already normalized
        if not np.allclose(f_norm.vectors, raw, rtol=1e-12, atol=1e-15):
            warnings.warn(
                f"Loaded frame from {path!s} was not normalized; normalizing on load.",
                RuntimeWarning,
                stacklevel=2,
            )
        return f_norm

    def export_txt(self, path: str) -> None:
        """
        Export this frame to a text file in the submission format:
        - First all real parts (row-major), one per line, then all imaginary parts.
        - Each number formatted with 15-digit exponential notation.

        Parameters
        ----------
        path : str
            Path where the .txt file will be written.
        """
        # Flatten row-major: rows are vectors
        flat_real = self._vectors.real.ravel(order="C")
        flat_imag = self._vectors.imag.ravel(order="C")

        with open(path, "w") as f:
            for val in flat_real:
                f.write(f"{val:.15e}\n")
            for val in flat_imag:
                f.write(f"{val:.15e}\n")

    def __repr__(self) -> str:  # pragma: no cover
        n, d = self.shape

        return f"Frame(n={n}, d={d})"

    @classmethod
    def load_txt(cls, path: str, n: int, d: int) -> Frame:
        """
        Load a Frame from a text file in the submission format:
        - File contains a newline-separated list of all real parts (row-major),
          followed by all imaginary parts (row-major), one number per line.
        - The file must contain exactly 2*n*d entries.

        Parameters
        ----------
        path : str
            Path to the .txt file to load.
        n : int
            Number of vectors (rows).
        d : int
            Dimension of each vector (columns).

        Returns
        -------
        Frame
            A normalized Frame initialized from the loaded array.

        Notes
        -----
        If the stored data are not already in canonical normalized form
        (unit-norm rows with the first non-zero entry real-positive),
        a RuntimeWarning is emitted and the loaded data are normalized.
        """
        with open(path) as f:
            lines = f.readlines()

        nums = np.array([float(line.strip()) for line in lines], dtype=np.float64)

        if nums.size != 2 * n * d:
            raise ValueError(
                f"File does not contain 2*n*d={2*n*d} entries (got {nums.size})"
            )

        reals = nums[: n * d].reshape((n, d))
        imags = nums[n * d :].reshape((n, d))
        raw = (reals + 1j * imags).astype(np.complex128, copy=False)

        # Build normalized copy using the standard constructor
        f_norm = cls(raw, normalize=True, copy=True)
        # Warn if the on-disk data were not already normalized
        if not np.allclose(f_norm.vectors, raw, rtol=1e-12, atol=1e-15):
            warnings.warn(
                f"Loaded frame from {path!s} was not normalized; normalizing on load.",
                RuntimeWarning,
                stacklevel=2,
            )
        return f_norm
