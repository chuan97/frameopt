# src/evomof/core/frame.py
from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np

from evomof.core._types import Complex128Array

__all__ = ["Frame"]


class Frame:
    """
    A frame represents an element of (CP^{d-1})^n.

    Internally we store one unit-sphere representative per projective vector
    (one row per vector) and fix the U(1) phase by making the first nonzero
    component real and positive. Different representatives of the same projective
    point are therefore identified by construction.
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
        """Read-only view of the underlying array.

        Returns
        -------
        Complex128Array
            Complex128 array of shape (n, d).
        """
        v = self._vectors.view()
        v.setflags(write=False)
        return v

    @property
    def is_normalized(self) -> bool:
        """Whether this frame has been normalized (rows unit-norm and gauge-fixed).

        Returns
        -------
        bool
            True if normalized; False otherwise.
        """
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
        S^{2d-1} and then gauge-fixed, i.e., a random element of (CP^{d-1})^n
        represented by its unit-sphere lifts.

        Uniformity is achieved by normalizing i.i.d. complex-Gaussian vectors,
        which is equivalent to taking the first column of a Haar-random unitary.
        The subsequent phase fix (first nonzero entry real and positive) chooses one
        representative per projective class and does not bias the distribution.

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
        """Return the dimensions of the frame.

        Returns
        -------
        tuple[int, int]
            (n, d) = (number of rows, complex dimension).
        """
        return self._vectors.shape

    @property
    def gram(self) -> Complex128Array:
        """Return the complex Gram matrix G = V V^H (Hermitian transpose).

        Returns
        -------
        Complex128Array
            Hermitian (n, n) Gram matrix.
        """
        return self._vectors @ self._vectors.conj().T

    def normalize(self) -> None:
        """
        In-place normalization and phase gauge fix.

        * Each row is scaled to unit L2 norm.
        * The global U(1) phase is removed by rotating every vector so that
            its first nonzero component becomes real and positive.

        Idempotent: calling this method multiple times leaves ``vectors``
        unchanged.

        Returns
        -------
        None
        """
        if self._is_normalized:
            return

        V = self._vectors
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("normalize(): zero or non-finite norm in frame vectors")

        V /= norms

        # Phase fix: rotate each row so its **first nonzero** entry is real and positive
        n = self.shape[0]
        nz_mask = np.abs(V) > 0
        first_idx = np.argmax(nz_mask, axis=1)  # first True per row
        pivots = V[np.arange(n), first_idx]  # (n,)
        phases = np.conj(pivots / np.abs(pivots))  # e^{-i arg(pivot)} without trig
        V *= phases[:, None]

        self._is_normalized = True

    # -------------------------------------------------------------- #
    # Manifold operations for (CP^{d-1})^n via unit‑sphere representatives (Hopf lift)
    # -------------------------------------------------------------- #
    def project(self, arr: Complex128Array | np.ndarray) -> Complex128Array:
        """
        Orthogonally project an ambient array onto the tangent space
        at this frame.

        For each row ``i`` the projection subtracts the
        inner product with the base vector so that the result satisfies

        <f_i, xi_i> = 0.

        This enforces full complex orthogonality, i.e., a genuine (CP^{d-1})^n tangent (not just sphere tangency Re <f_i, xi_i> = 0).

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

    def retract(self, tang: Complex128Array | np.ndarray) -> Frame:
        """
        Exponential-map retraction on (CP^{d-1})^n (implemented via the sphere lift).

        Given a base frame (self) and a CP-tangent perturbation (tang), this method
        returns a new Frame obtained by moving, for each row, along the great-circle
        in the unit sphere (the horizontal lift), which projects to the CP geodesic.

        Formula (per row):

            f_i' = cos(||xi_i||)*f_i + (sin(||xi_i||)/||xi_i||)*xi_i

        where xi_i is the i-th row of tang. For ||xi_i|| -> 0 the Taylor expansion
        reduces to the first-order update f_i + xi_i followed by renormalization.

        Parameters
        ----------
        tang :
            Complex array of shape self.shape representing a tangent vector field.
            It must satisfy <f_i, xi_i> = 0 (complex orthogonality) for every row;
            call project() if unsure.

        Returns
        -------
        Frame
            New frame with the same shape as self, normalized and gauge-fixed.

        Raises
        ------
        ValueError
            If tang.shape != self.shape.
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
        Riemannian logarithm in (CP^{d-1})^n (via horizontal lift).

        This returns the CP tangent xi at "self" such that exp_self(xi) reaches "other"
        in projective space, computed row-wise by phase-aligning and using the
        great-circle formula on the unit-sphere lift.

        For each row (f, g):
          1) s = <f, g> (complex)
          2) phase = s/|s| if |s| > eps else 1; define g_tilde = g / phase so that <f, g_tilde> is real and >= 0
          3) c = clip( Re <f, g_tilde>, 0, 1 ); theta = arccos(c)
          4) xi = (theta / sin(theta)) * ( g_tilde - c * f )      if theta > eps
                 0                                                otherwise

        The result satisfies <f, xi> = 0 (complex orthogonality), i.e., it is a bona fide
        CP tangent. When "other" equals "self" up to a global phase, the logarithm is zero.

        Parameters
        ----------
        other : Frame
            Target frame with the same (n, d) shape.

        Returns
        -------
        Complex128Array
            Tangent array of shape (n, d) at self (complex orthogonal row-wise).

        Raises
        ------
        ValueError
            If other.shape != self.shape.
        """
        if self.shape != other.shape:
            raise ValueError("Frame shapes mismatch")

        eps = 1e-12

        f = self._vectors  # (n, d)
        g = other._vectors  # (n, d)

        # Row-wise inner products s = <f, g> (complex)
        s = np.sum(f.conj() * g, axis=1, keepdims=True)  # (n,1)
        abs_s = np.abs(s)

        # Phase-align so that <f, g_tilde> is real and nonnegative
        phase = np.ones_like(s, dtype=np.complex128)
        nz = abs_s > eps
        phase[nz] = s[nz] / abs_s[nz]  # e^{i φ}
        g_tilde = g / phase

        # c = cos(theta) in [0, 1], numerically clipped
        c = np.real(np.sum(f.conj() * g_tilde, axis=1))  # (n,)
        c = np.clip(c, 0.0, 1.0)
        theta = np.arccos(c)  # (n,)

        # Avoid division by zero when theta ~ 0
        mask = theta > eps
        scale = np.zeros_like(theta)
        scale[mask] = theta[mask] / np.sin(theta[mask])

        diff = g_tilde - c[:, None] * f
        xi: Complex128Array = scale[:, None] * diff

        return xi

    def transport(
        self, other: Frame, tang: Complex128Array, eps: float = 1e-12
    ) -> Complex128Array:
        """
        Exact parallel transport on (CP^{d-1})^n via the sphere lift (row-wise), using phase alignment.
        For each row:
          - Let s = <x, y>. Set phase = s/|s| (or 1 if |s| ~ 0). Define y_tilde = y / phase so that <x, y_tilde> is real and >= 0.
          - Apply sphere PT with y_tilde:
              xi_tilde = U - ( <y_tilde, U> / (1 + <x, y_tilde>) ) * ( x + y_tilde ).
          - Rotate back: xi = phase * xi_tilde.

        This preserves tangency (<y, xi> = 0) and the tangent norm exactly in exact arithmetic.

        Parameters
        ----------
        other : Frame
            Target frame with the same (n, d) shape.
        tang : Complex128Array
            Tangent at self to be transported to "other". Must satisfy <self[i], tang[i]> = 0 per row.
        eps : float, optional
            Numerical threshold to detect near-zero denominators / antipodal cases (default 1e-12).

        Returns
        -------
        Complex128Array
            Transported tangent at "other", shape (n, d).

        Raises
        ------
        ValueError
            If shapes of "other" or "tang" do not match self.shape.
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
        The projection enforces <f_i, xi_i> = 0, i.e., a bona fide (CP^{d-1})^n tangent.

        Sampling: i.i.d. complex normal in ambient, projected to the tangent space.
        If unit=True, the result is normalized to Frobenius norm 1.

        Parameters
        ----------
        rng : numpy.random.Generator | None
            Optional generator for reproducibility. If None, uses a fresh default generator.
        unit : bool, keyword-only
            If True (default), normalize the tangent to Frobenius norm 1.

        Returns
        -------
        Complex128Array
            Tangent array of shape (n, d) at self.
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
        """Return a shallow copy of this Frame (vectors are copied; normalization flag preserved).

        Returns
        -------
        Frame
            New Frame with the same data and is_normalized flag.
        """
        f = Frame(self._vectors, normalize=False, copy=True)
        f._is_normalized = self._is_normalized
        return f

    def __iter__(self) -> Iterator[Complex128Array]:
        """Iterate over row vectors.

        Returns
        -------
        Iterator[Complex128Array]
            Iterator yielding each row (shape (d,)) as a complex vector.
        """
        return iter(self._vectors)

    def save_npy(self, path: str) -> None:
        """
        Save this frame's vectors to a NumPy .npy file.

        Parameters
        ----------
        path : str
            Path where the .npy file will be written.

        Returns
        -------
        None
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

        Returns
        -------
        None
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
