# src/evomof/core/frame.py
from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np

from evomof.core._types import Complex128Array


class Frame:
    """
    A frame represents an element of (CP^{d−1})ⁿ.

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
            If True, call :meth:`normalize` after storing the array.
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
        S^{2d−1} and then gauge‑fixed, i.e., a random element of (CP^{d−1})ⁿ
        represented by its unit‑sphere lifts.

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
        """Return the complex Gram matrix G = V Vᴴ (Hermitian transpose).

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

    # ------------------------------------------------------------------ #
    # Convenience & dunder methods                                       #
    # ------------------------------------------------------------------ #

    def copy(self) -> Frame:
        """Return a shallow copy of this :class:`Frame` (vectors are copied; normalization flag preserved).

        Returns
        -------
        :class:`Frame`
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

    # ------------------------------------------------------------------ #
    # Input/output                                                       #
    # ------------------------------------------------------------------ #

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
        :class:`Frame`
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
        :class:`Frame`
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
