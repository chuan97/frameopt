from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from evomof.core._types import Complex128Array, Float64Array
from evomof.core.frame import Frame


@dataclass(frozen=True, slots=True)
class ProductCP:
    """
    Policy object for geometry on (CP^{d−1})^n, operating on :class:`Frame` points.

    Parameters
    ----------
    eps : float
        Small numerical threshold used for phase alignment and antipodal guards.
    retraction_kind : {"exponential", "normalize"}
        Choice of retraction. "sphere" uses the great‑circle (lift) formula; "normalize"
        uses a first‑order retraction by normalizing ``f + ξ`` row‑wise.
    transport_kind : {"parallel", "projection"}
        Vector transport type. "parallel" uses the sphere‑lift parallel transport; "projection"
        projects the vector onto the target tangent at ``other``.
    """

    eps: float = 1e-12
    retraction_kind: Literal["exponential", "normalize"] = "exponential"
    transport_kind: Literal["parallel", "projection"] = "parallel"

    def project(
        self, frame: Frame, arr: Complex128Array | np.ndarray
    ) -> Complex128Array:
        """
        Orthogonally project an ambient array onto the tangent space at ``frame``.

        For each row ``i`` the projection subtracts the inner product with the base
        vector so that the result satisfies ⟨f_i, ξ_i⟩ = 0 (full complex orthogonality),
        i.e. a bona fide (CP^{d−1})ⁿ tangent.

        Parameters
        ----------
        frame : :class:`Frame`
            Base point whose tangent space is used for the projection.
        arr : Complex128Array | numpy.ndarray
            Complex array of shape ``frame.vectors.shape``. It does **not** need to
            be tangent already.

        Returns
        -------
        Complex128Array
            Tangent array with the same shape as ``frame``.
        """

        f = frame.vectors  # (n, d)
        inner = np.sum(f.conj() * arr, axis=1, keepdims=True)
        out: Complex128Array = arr - inner * f

        return out

    def retract(self, frame: Frame, tang: Complex128Array | np.ndarray) -> Frame:
        """
        Exponential‑map retraction on (CP^{d−1})ⁿ (implemented via the sphere lift).

        Given a base frame and a CP‑tangent perturbation, returns a new :class:`Frame`
        obtained row‑wise along the great‑circle in the unit‑sphere lift (projects to the
        CP geodesic):

            f_i′ = cos(‖ξ_i‖)·f_i + (sin(‖ξ_i‖)/‖ξ_i‖)·ξ_i.

        If ``retraction_kind == "normalize"``, uses a first‑order retraction: normalize ``f + ξ``
        row‑wise.

        Parameters
        ----------
        frame : :class:`Frame`
            Base point.
        tang : Complex128Array | numpy.ndarray
            Tangent array of shape ``frame.vectors.shape`` satisfying ⟨f_i, ξ_i⟩ = 0.

        Returns
        -------
        :class:`Frame`
            New frame with the same shape as ``frame`` (normalized and gauge‑fixed).

        Raises
        ------
        ValueError
            If ``tang.shape`` does not match ``frame.vectors.shape``.
        """

        f = frame.vectors
        if tang.shape != f.shape:
            raise ValueError("Tangent array shape mismatch.")

        if self.retraction_kind == "normalize":
            # First-order retraction: normalize(f + ξ) row-wise
            new_vecs = f + tang
            # Normalize rows to unit norm
            norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
            nz = norms > 0
            new_vecs[nz] = new_vecs[nz] / norms[nz]
            return Frame(new_vecs, normalize=False, copy=False)

        # Default: sphere-lift (great-circle) retraction
        norms = np.linalg.norm(tang, axis=1, keepdims=True)
        small = norms < 1e-12
        scale_sin = np.zeros_like(norms)
        scale_cos = np.zeros_like(norms)

        scale_sin[~small] = np.sin(norms[~small]) / norms[~small]
        scale_cos[~small] = np.cos(norms[~small])

        # First‑order Taylor for tiny norms: cos≈1, sin≈norm
        scale_sin[small] = 1.0
        scale_cos[small] = 1.0 - 0.5 * norms[small] ** 2

        new_vecs = scale_cos * f + scale_sin * tang

        return Frame(new_vecs, normalize=True, copy=False)

    def log_map(
        self, frame: Frame, other: Frame, eps: float | None = None
    ) -> Complex128Array:
        """
        Riemannian logarithm in (CP^{d−1})ⁿ (via the horizontal lift).

        Returns the CP tangent ξ at ``frame`` such that ``exp_frame(ξ) == other``
        in projective space, computed row‑wise by phase‑aligning and using the
        great‑circle formula on the unit‑sphere lift.

        Parameters
        ----------
        frame : :class:`Frame`
            Base point.
        other : :class:`Frame`
            Target frame with the same (n, d) shape.
        eps : float, optional
            Numerical threshold used when detecting near‑antipodal rows and for phase alignment.
            Defaults to ``self.eps``.

        Returns
        -------
        Complex128Array
            Tangent array of shape (n, d) at ``frame`` (complex orthogonal row‑wise).

        Raises
        ------
        ValueError
            If the shapes of ``frame`` and ``other`` do not match.
        """

        eps = self.eps if eps is None else eps
        if frame.vectors.shape != other.vectors.shape:
            raise ValueError("Frame shapes mismatch")

        f = frame.vectors  # (n, d)
        g = other.vectors  # (n, d)

        s = np.sum(f.conj() * g, axis=1, keepdims=True)  # (n,1) complex
        abs_s = np.abs(s)

        phase = np.ones_like(s, dtype=np.complex128)
        nz = abs_s > eps
        phase[nz] = s[nz] / abs_s[nz]  # e^{i φ}
        g_tilde = g / phase

        c = np.real(np.sum(f.conj() * g_tilde, axis=1))  # (n,)
        c = np.clip(c, 0.0, 1.0)
        theta = np.arccos(c)  # (n,)

        mask = theta > eps
        scale = np.zeros_like(theta)
        scale[mask] = theta[mask] / np.sin(theta[mask])

        diff = g_tilde - c[:, None] * f
        xi: Complex128Array = scale[:, None] * diff

        return xi

    def transport(
        self,
        frame: Frame,
        other: Frame,
        tang: Complex128Array,
        eps: float | None = None,
    ) -> Complex128Array:
        """
        Exact parallel transport on (CP^{d−1})ⁿ via the sphere lift (row‑wise), using phase alignment.

        If ``transport_kind == "projection"``, performs the simple projection transport:
        project the vector onto the target tangent at ``other`` row‑wise.

        Parameters
        ----------
        frame : :class:`Frame`
            Source base point.
        other : :class:`Frame`
            Target base point (same shape as ``frame``).
        tang : Complex128Array
            Tangent at ``frame`` to be transported to ``other``.
        eps : float, optional
            Numerical threshold for near‑antipodal detection. Defaults to ``self.eps``.

        Returns
        -------
        Complex128Array
            Transported tangent at ``other``, shape (n, d).

        Raises
        ------
        ValueError
            If shapes of ``frame``, ``other`` or ``tang`` do not match.
        """

        eps = self.eps if eps is None else eps
        if other.vectors.shape != frame.vectors.shape:
            raise ValueError("Frame shapes mismatch")
        if tang.shape != frame.vectors.shape:
            raise ValueError("Tangent array shape mismatch.")

        x = frame.vectors  # (n, d)
        y = other.vectors  # (n, d)

        if self.transport_kind == "projection":
            # Simple projection transport: project ξ onto T_y M row-wise
            inner = np.sum(y.conj() * tang, axis=1, keepdims=True)
            out: Complex128Array = tang - inner * y
            return out

        # Default: parallel transport via sphere-lift with phase alignment
        s = np.sum(x.conj() * y, axis=1, keepdims=True)  # (n,1) complex
        abs_s = np.abs(s)
        phase = np.ones_like(s, dtype=np.complex128)
        nz = abs_s > eps
        phase[nz] = s[nz] / abs_s[nz]  # e^{i φ}
        y_tilde = y / phase

        dot_yu = np.sum(y_tilde.conj() * tang, axis=1, keepdims=True)
        dot_xy = np.sum(x.conj() * y_tilde, axis=1, keepdims=True)
        dot_xy_real = np.real(dot_xy)

        denom = 1.0 + dot_xy_real
        bad = np.abs(denom) < eps  # antipodal guard
        denom_safe = denom.copy()
        denom_safe[bad] = 1.0

        transported_tilde = tang - (dot_yu / denom_safe) * (x + y_tilde)
        transported: Complex128Array = transported_tilde * phase

        # Fallback for (near) antipodal rows: projection transport row‑wise
        if np.any(bad):
            idx = bad.ravel()
            y_rows = y[idx, :]
            t_rows = tang[idx, :]
            inner_sub = np.sum(y_rows.conj() * t_rows, axis=1, keepdims=True)
            transported[idx, :] = t_rows - inner_sub * y_rows

        return transported

    def random_tangent(
        self,
        frame: Frame,
        rng: np.random.Generator | None = None,
        *,
        unit: bool = True,
    ) -> Complex128Array:
        """
        Draw a random tangent array at ``frame``.

        Sampling: i.i.d. complex normal in ambient, then projected to the tangent space
        at ``frame``. If ``unit=True``, the result is normalized to Frobenius norm 1.

        Parameters
        ----------
        frame : :class:`Frame`
            Base point.
        rng : numpy.random.Generator | None
            Optional generator for reproducibility. If ``None``, uses a fresh default generator.
        unit : bool, keyword-only
            If True (default), normalize the tangent to Frobenius norm 1.

        Returns
        -------
        Complex128Array
            Tangent array of shape ``frame.vectors.shape`` at ``frame``.
        """
        rng = rng or np.random.default_rng()
        shape = frame.vectors.shape
        Z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        U = self.project(frame, Z.astype(np.complex128, copy=False))

        if unit:
            nrm = np.linalg.norm(U)
            if nrm > 0:
                U = U / nrm

        return U


# Default policy instance for general use
PRODUCT_CP = ProductCP()


@dataclass(frozen=True, slots=True)
class Chart:
    """
    Orthonormal coordinate chart on (CP^{d−1})^n anchored at a base :class:`Frame`.

    Provides real coordinates of tangent vectors (encode/decode) and transports
    them between base points using the selected geometry policy.

    Notes
    -----
    * Real coordinate dimension is ``k = 2·n·(d−1)``.
    * Each row uses an orthonormal basis \(Q_i ∈ ℂ^{d×(d−1)}\) spanning the
      complex orthogonal complement of the frame row. Columns are phase‑stabilized
      so the largest‑magnitude entry per column is real and ≥ 0 (to reduce gauge flips).
    """

    frame: Frame
    Q_blocks: tuple[np.ndarray, ...]
    geom: ProductCP = PRODUCT_CP

    @classmethod
    def at(cls, frame: Frame, *, geom: ProductCP | None = None) -> Chart:
        """Build a chart anchored at ``frame`` with cached per‑row bases."""
        f = frame.vectors
        n, d = f.shape

        if d < 2:
            raise ValueError("Chart requires d ≥ 2 (tangent must be nontrivial).")

        Q_blocks = tuple(cls._orth_basis_row(f[i]) for i in range(n))

        g = PRODUCT_CP if geom is None else geom

        return cls(frame=frame, Q_blocks=tuple(Q_blocks), geom=g)

    def dim(self) -> int:
        """Return real tangent dimension ``k = 2·n·(d−1)``."""
        n, d = self.frame.vectors.shape
        dim: int = 2 * n * (d - 1)

        return dim

    def encode(self, U: Complex128Array) -> Float64Array:
        """
        Encode a CP‑tangent ``U`` at ``self.frame`` into real coordinates ``y``.

        Parameters
        ----------
        U : Complex128Array
            Tangent at ``self.frame`` with shape ``(n, d)``.

        Returns
        -------
        numpy.ndarray
            Real vector of length ``k = 2·n·(d−1)``.
        """
        f = self.frame.vectors

        if U.shape != f.shape:
            raise ValueError("Tangent array shape mismatch.")

        n, d = f.shape
        r = d - 1
        Y = np.empty((n, 2 * r), dtype=np.float64)
        for i in range(n):
            Qi = self.Q_blocks[i]
            c = Qi.conj().T @ U[i]
            Y[i, :r] = np.real(c)
            Y[i, r:] = np.imag(c)

        return Y.ravel()

    def decode(self, y: Float64Array) -> Complex128Array:
        """
        Decode real coordinates ``y`` into a CP‑tangent at ``self.frame``.

        Parameters
        ----------
        y : numpy.ndarray
            Real vector of length ``k = 2·n·(d−1)``.

        Returns
        -------
        Complex128Array
            Tangent at ``self.frame`` (shape ``(n, d)``).
        """
        f = self.frame.vectors
        n, d = f.shape
        r = d - 1
        expected = 2 * n * r
        if y.ndim != 1 or y.size != expected:
            raise ValueError(
                f"Coordinate length mismatch: got {y.size}, expected {expected}."
            )

        Y = y.reshape(n, 2 * r)
        U = np.empty_like(f)
        for i in range(n):
            Qi = self.Q_blocks[i]
            cre = Y[i, :r]
            cim = Y[i, r:]
            c = cre + 1j * cim
            U[i] = Qi @ c

        # Kill numerical drift along the base vector
        return self.geom.project(self.frame, U)

    def transport_coords(self, to: Chart, y: Float64Array) -> Float64Array:
        """Transport coordinates ``y`` from this chart to chart ``to``."""
        U = self.decode(y)
        V = self.geom.transport(self.frame, to.frame, U)

        return to.encode(V)

    def transport_basis(self, to: Chart, B: Float64Array) -> Float64Array:
        """
        Transport a set of coordinate directions (columns of ``B``) and
        re‑orthonormalize with a real QR.

        Parameters
        ----------
        to : Chart
            Target chart.
        B : numpy.ndarray
            Real matrix of shape ``(k, r)`` with coordinate directions.

        Returns
        -------
        numpy.ndarray
            Real matrix of shape ``(k, r)`` with orthonormal columns (Euclidean).
        """
        k, r = B.shape
        if B.ndim != 2 or k != self.dim():
            raise ValueError("Basis shape mismatch with chart dimension.")
        if r == 0:
            return B.copy()

        cols = [self.transport_coords(to, B[:, j]) for j in range(r)]
        M = np.column_stack(cols)
        Q, _ = np.linalg.qr(M, mode="reduced")

        return Q

    def transport_to(self, to: Frame, *, eps: float = 1e-12) -> Chart:
        """
        Transport this chart's per‑row complex bases to a new base point ``to``
        and return a stabilized chart at ``to``.

        This differs from :meth:`transport_basis`, which transports a **real**
        coordinate basis in ℝᵏ. Here we transport each row's complex basis
        \(Q_i ∈ ℂ^{d×(d−1)}\), re‑orthonormalize it, and align columns to reduce
        gauge flips across generations.

        Steps (per row):
        1. Transport each column of the old \(Q_i\) via :meth:`ProductCP.transport`.
        2. QR re‑orthonormalize the transported block.
        3. Procrustes‑align the QR basis to the transported block to preserve column identity.
        4. Phase‑stabilize columns at the **old pivot** index (largest‑magnitude entry of the old frame row).

        Parameters
        ----------
        to : :class:`Frame`
            Target frame (must have same shape as ``self.frame``).
        eps : float, keyword‑only
            Numerical threshold for phase stabilization at the pivot.

        Returns
        -------
        :class:`Chart`
            New chart anchored at ``to`` with stabilized per‑row bases.
        """
        if to.vectors.shape != self.frame.vectors.shape:
            raise ValueError("Frame shapes mismatch in chart transport.")

        X = self.frame
        n, d = to.vectors.shape
        r = d - 1
        geom = self.geom

        # Working array for per‑column transports
        work = np.zeros_like(X.vectors, dtype=np.complex128)
        Q_new: list[np.ndarray] = []

        for i in range(n):
            Qi_old = self.Q_blocks[i]  # d×r
            cols = np.empty((d, r), dtype=np.complex128)

            # Transport each column of the complex basis for this row
            for j in range(r):
                work.fill(0.0)
                work[i, :] = Qi_old[:, j]
                Vi = geom.transport(X, to, work)  # (n,d)
                cols[:, j] = Vi[i, :]

            # Orthonormalize transported columns
            Qtemp, _ = np.linalg.qr(cols, mode="reduced")  # d×r

            # Procrustes alignment to keep columns close to transported ones
            H = Qtemp.conj().T @ cols
            U, _, Vh = np.linalg.svd(H, full_matrices=False)
            R = Vh.conj().T @ U.conj().T
            Qrow = Qtemp @ R  # still orthonormal

            # Phase stabilization at the old pivot (lock pivot index)
            p = int(np.argmax(np.abs(X.vectors[i])))
            for j in range(r):
                mag = abs(Qrow[p, j])
                if mag > eps:
                    phase = Qrow[p, j] / mag
                    Qrow[:, j] = Qrow[:, j] / phase  # make pivot entry real ≥ 0

            Q_new.append(Qrow)

        return Chart(frame=to, Q_blocks=tuple(Q_new), geom=geom)

    @staticmethod
    def _orth_basis_row(f: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Orthonormal basis of the complex orthogonal complement of ``f``.

        Returns ``Q ∈ ℂ^{d×(d−1)}`` with columns spanning ``{u : ⟨f, u⟩ = 0}``.
        Columns are phase‑stabilized by aligning the entry at the frame pivot index `p`
        (largest entry of `f`) to be real and ≥ 0.
        """
        d = f.size

        # Choose pivot with largest magnitude for stability
        p = np.argmax(np.abs(f))

        # Build Ep = I without the pivot column
        Ep = np.eye(d, dtype=np.complex128)
        Ep = np.delete(Ep, p, axis=1)  # d×(d−1)

        # Project onto orthogonal complement of f
        inner = f.conj() @ Ep  # (d−1,)
        Qtilde = Ep - np.outer(f, inner)

        # Orthonormalize
        Q, _ = np.linalg.qr(Qtilde, mode="reduced")  # d×(d−1)

        # Column phase stabilization: align at the frame pivot `p` only.
        for j in range(Q.shape[1]):
            col = Q[:, j]
            mag_p = np.abs(col[p])
            if mag_p > eps:
                phase = col[p] / mag_p  # unit complex
                Q[:, j] = col / phase
            # else: leave column as is for simplicity

        return Q
