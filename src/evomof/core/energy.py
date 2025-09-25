"""
Energy and potential functions for frames (points in (CP^{d−1})ⁿ).

Public functions
----------------
- :func:`frame_potential`
- :func:`diff_coherence`
- :func:`coherence`
- :func:`grad_frame_potential`
- :func:`grad_diff_coherence`
"""

from __future__ import annotations

import numpy as np

from evomof.core._types import Complex128Array, Float64Array
from evomof.core.frame import Frame
from evomof.core.manifold import PRODUCT_CP

# -----------------------------------------------------------------------------#
# Helper utilities                                                             #
# -----------------------------------------------------------------------------#


def _absolute_inner(frame: Frame) -> Float64Array:
    """
    Return |⟨f_i, f_j⟩| with zeros on the diagonal.

    Parameters
    ----------
    frame : :class:`Frame`
        Input frame.

    Returns
    -------
    Float64Array
        Array of shape (n, n) with absolute inner products and zero diagonal.
    """
    g: Float64Array = np.abs(frame.gram).astype(np.float64, copy=False)
    np.fill_diagonal(g, 0.0)
    return g


# -----------------------------------------------------------------------------#
# Public energy functions                                                      #
# -----------------------------------------------------------------------------#


def frame_potential(frame: Frame, p: float = 4.0) -> float:
    """
    p‑frame potential: Φ_p(F) = ∑_{i≠j} |⟨f_i, f_j⟩|^{2p}.

    Parameters
    ----------
    frame : :class:`Frame`
        Input frame.
    p : float, optional
        Positive real exponent (default 4.0). Larger p penalizes large overlaps more.

    Raises
    ------
    ValueError
        If ``p`` is not positive.

    Returns
    -------
    float
        Value of the potential.
    """
    if p <= 0:
        raise ValueError("Exponent p must be positive.")
    aij = _absolute_inner(frame) ** (2 * p)
    return float(aij.sum())


def diff_coherence(frame: Frame, p: float = 16.0) -> float:
    """
    Differentiable surrogate for frame coherence.

    Computes L_p(F) = (Φ_p(F))^{1/(2p)} with Φ_p(F) = ∑_{i≠j} |⟨f_i, f_j⟩|^{2p},
    using an underflow‑stable evaluation by factoring out the largest overlap G⋆:

        L_p = G⋆ · ( ∑ (|g_{ij}|/G⋆)^{2p} )^{1/(2p)}

    Parameters
    ----------
    frame : :class:`Frame`
        Input frame.
    p : float, optional
        Positive exponent (default 16.0).

    Raises
    ------
    ValueError
        If ``p`` is not positive.

    Returns
    -------
    float
        Smooth proxy for coherence; approaches a multiple of the true coherence as p → ∞.
    """
    if p <= 0:
        raise ValueError("Exponent p must be positive.")

    g_abs = _absolute_inner(frame)  # (n,n) non-negative with zero diagonal
    g_max = float(g_abs.max())
    if g_max == 0.0:
        return 0.0

    q = 2 * p
    # Stable accumulation: ratios <= 1; underflow to zero is harmless.
    ratios = g_abs / g_max
    # Flatten upper triangle to save a bit (optional); full matrix fine too.
    r = ratios**q
    S = float(r.sum())
    # The p-norm surrogate
    return float(g_max * S ** (1.0 / q))


def coherence(frame: Frame) -> float:
    """
    True frame coherence μ(F) = max_{i<j} |⟨f_i, f_j⟩|.

    Parameters
    ----------
    frame : :class:`Frame`
        Input frame.

    Returns
    -------
    float
        Maximum absolute inner product between distinct rows.
    """
    return float(_absolute_inner(frame).max())


def grad_frame_potential(frame: Frame, p: float = 4.0) -> Complex128Array:
    """
    Analytic Riemannian gradient of the p‑frame potential on (CP^{d−1})ⁿ.

    Parameters
    ----------
    frame : :class:`Frame`
        Current frame (base point).
    p : float, optional
        Positive exponent in Φ_p(F) (default 4.0).

    Raises
    ------
    ValueError
        If ``p`` is not positive.

    Returns
    -------
    Complex128Array
        Tangent array of shape ``frame.shape`` satisfying ⟨f_i, ξ_i⟩ = 0 per row
        (complex orthogonality). Implemented as an ambient gradient projected onto the
        tangent using the geometry policy (:class:`ProductCP`).
    """
    if p <= 0:
        raise ValueError("Exponent p must be positive.")

    g = frame.gram  # (n, n) complex
    abs_g = np.abs(g)
    q = 2 * p  # effective power
    abs_pow = abs_g ** (q - 2)
    coeff = 2 * q * abs_pow * g  # factor 2 accounts for (i,j) and (j,i) contributions
    np.fill_diagonal(coeff, 0.0)
    grad = coeff @ frame.vectors
    return PRODUCT_CP.project_to_tangent(frame, grad)


def grad_diff_coherence(frame: Frame, p: float = 16.0) -> Complex128Array:
    """
    Gradient of the differentiable coherence surrogate L_p(F) on (CP^{d−1})ⁿ.

    Uses the same G⋆-factoring as :func:`diff_coherence` to avoid underflow.

    Parameters
    ----------
    frame : :class:`Frame`
        Current frame (base point).
    p : float, optional
        Positive exponent (default 16.0).

    Raises
    ------
    ValueError
        If ``p`` is not positive.

    Returns
    -------
    Complex128Array
        Tangent array of shape ``frame.shape`` with ⟨f_i, ξ_i⟩ = 0 per row,
        obtained by projecting the ambient gradient onto the tangent using the geometry
        policy (:class:`ProductCP`).
    """
    if p <= 0:
        raise ValueError("Exponent p must be positive.")

    g = frame.gram  # complex (n,n)
    abs_g = np.abs(g)
    np.fill_diagonal(abs_g, 0.0)

    g_max = float(abs_g.max())
    if g_max == 0.0:
        return np.zeros_like(frame.vectors)

    q = 2 * p
    ratios = abs_g / g_max
    r = ratios**q  # may underflow to 0 for small overlaps -> fine
    S = float(r.sum())
    L = g_max * S ** (1.0 / q)

    # Coefficient matrix for gradient: coeff_ij = (2L/S) * r_ij * g_ij / |g_ij|^2
    coeff = np.zeros_like(g)
    mask = abs_g > 0
    coeff[mask] = (2.0 * L / S) * r[mask] * g[mask] / (abs_g[mask] ** 2)
    np.fill_diagonal(coeff, 0.0)
    grad = coeff @ frame.vectors
    return PRODUCT_CP.project_to_tangent(frame, grad)
