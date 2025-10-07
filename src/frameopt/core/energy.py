"""
Energy and potential functions for frames (points in (CP^{d−1})ⁿ).
"""

from __future__ import annotations

import numpy as np

from frameopt.core._types import Complex128Array, Float64Array
from frameopt.core.frame import Frame
from frameopt.core.manifold import PRODUCT_CP

__all__ = [
    "frame_potential",
    "pnorm_coherence",
    "coherence",
    "grad_frame_potential",
    "grad_pnorm_coherence",
    "mellowmax_coherence",
    "grad_mellowmax_coherence",
]


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


def frame_potential(frame: Frame, p: float = 4.0) -> float:
    """
    p‑frame potential: Φ_p(F) = ∑_{i<j} |⟨f_i, f_j⟩|^{2p}.

    Uses the upper off‑diagonal only.

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
    n = aij.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(aij[iu].sum())


def pnorm_coherence(frame: Frame, p: float = 16.0) -> float:
    """
    p norm surrogate for frame coherence.

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
    g_max = g_abs.max()
    if g_max == 0.0:
        return 0.0

    q = 2 * p
    # Stable accumulation: ratios <= 1; underflow to zero is harmless.
    ratios = g_abs / g_max
    r = ratios**q
    S = r.sum()
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


def mellowmax_coherence(frame: Frame, omega: float = 10.0) -> float:
    """
    Mellowmax surrogate for coherence on (CP^{d−1})ⁿ.

    Computes
        mm_ω(x) = max(x) + (1/ω) [\log( (1/M) ∑_k e^{ω(x_k − max(x))} ) ],
    with \(x_k = |⟨f_i, f_j⟩|\) over unique pairs ``i<j`` (so ``M = n(n−1)/2``).

    This uses the standard log-sum-exp **max-shift** for numerical stability.
    For large ``ω``, ``mm_ω`` → ``max(x)`` from below; for ``ω → 0`` it approaches
    the arithmetic mean of the off-diagonal overlaps.

    Parameters
    ----------
    frame : :class:`Frame`
        Input frame.
    omega : float, optional
        Temperature parameter (default 10.0). Larger values concentrate the
        objective around the largest overlap. Must be non‑negative.

    Raises
    ------
    ValueError
        If ``omega`` is negative.

    Returns
    -------
    float
        Smooth, max‑like surrogate of the true coherence.
    """
    if omega < 0.0:
        raise ValueError("omega must be non-negative.")

    g_abs = _absolute_inner(frame)
    n = g_abs.shape[0]
    if n <= 1:
        return 0.0

    # Work on unique off-diagonal entries
    iu = np.triu_indices(n, k=1)
    x = g_abs[iu]

    m = x.max()
    if m == 0.0:
        return 0.0

    if omega == 0.0:
        # Continuous limit ω→0: mean of off-diagonal magnitudes
        return float(x.mean())

    # Max-shifted log-sum-exp: terms are exp(ω(x−m)) = exp(−ω·Δ), Δ≥0
    deltas = m - x
    y = np.exp(-omega * deltas)
    sum_y = y.sum()
    M = x.size

    return float(m + (np.log(sum_y) - np.log(M)) / omega)


def grad_frame_potential(frame: Frame, p: float = 4.0) -> Complex128Array:
    """
    Analytic Riemannian gradient of the p‑frame potential on (CP^{d−1})ⁿ.

    The potential is defined over unique pairs i<j, matching :func:`frame_potential`.

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
    coeff = q * abs_pow * g
    np.fill_diagonal(coeff, 0.0)
    grad = coeff @ frame.vectors
    return PRODUCT_CP.project_to_tangent(frame, grad)


def grad_pnorm_coherence(frame: Frame, p: float = 16.0) -> Complex128Array:
    """
    Gradient of the p-norm coherence surrogate L_p(F) on (CP^{d−1})ⁿ.

    Uses the same G⋆-factoring as :func:`pnorm_coherence` to avoid underflow.

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

    g_max = abs_g.max()
    if g_max == 0.0:
        return np.zeros_like(frame.vectors)

    q = 2 * p
    ratios = abs_g / g_max
    r = ratios**q  # may underflow to 0 for small overlaps -> fine
    S = r.sum()
    L = g_max * S ** (1.0 / q)

    # Coefficient matrix for gradient: coeff_ij = (2L/S) * r_ij * g_ij / |g_ij|^2
    coeff = np.zeros_like(g)
    mask = abs_g > 0
    coeff[mask] = (2.0 * L / S) * r[mask] * g[mask] / (abs_g[mask] ** 2)
    np.fill_diagonal(coeff, 0.0)
    grad = coeff @ frame.vectors
    return PRODUCT_CP.project_to_tangent(frame, grad)


def grad_mellowmax_coherence(frame: Frame, omega: float = 10.0) -> Complex128Array:
    """
    Gradient of the mellowmax coherence surrogate on (CP^{d−1})ⁿ.

    We define ``x_k = |⟨f_i, f_j⟩|`` for unique pairs ``i<j`` and
    ``mm_ω = max(x) + (1/ω) log((1/M) ∑_k exp(ω(x_k − max(x))))``.
    The gradient w.r.t. the Gram entries uses softmax weights
    ``w_k = exp(ω(x_k − max(x)))/∑_ℓ exp(ω(x_ℓ − max(x)))`` assigned to pairs,
    and the ambient gradient takes the form ``coeff_ij = w_{ij} · g_ij / |g_ij|``
    on off‑diagonal entries (symmetric for ``i≠j``). The division by ``M`` drops
    out of the gradient.

    Parameters
    ----------
    frame : :class:`Frame`
        Current frame.
    omega : float, optional
        Temperature parameter (default 10.0). Must be non‑negative.

    Returns
    -------
    Complex128Array
        Tangent gradient with the same shape as ``frame.vectors``.
    """
    if omega < 0.0:
        raise ValueError("omega must be non-negative.")

    g = frame.gram  # (n, n) complex Hermitian
    abs_g = np.abs(g).astype(np.float64, copy=False)
    np.fill_diagonal(abs_g, 0.0)

    n = abs_g.shape[0]
    if n <= 1:
        return np.zeros_like(frame.vectors)

    # Unique off-diagonal magnitudes and softmax weights
    iu = np.triu_indices(n, k=1)
    x = abs_g[iu]
    if x.size == 0:
        return np.zeros_like(frame.vectors)

    m = x.max()
    if m == 0.0:
        return np.zeros_like(frame.vectors)

    if omega == 0.0:
        # ω→0: weights become uniform over pairs
        w_pairs = np.full_like(x, 1.0 / x.size, dtype=np.float64)
    else:
        y = np.exp(omega * (x - m))  # max-shifted, safe
        s = y.sum()
        if s == 0.0:
            # All underflowed -> behave like max pair only
            w_pairs = np.zeros_like(x)
            w_pairs[np.argmax(x)] = 1.0
        else:
            w_pairs = y / s

    # Lift pair-weights to a full (n,n) symmetric weight matrix on off-diagonal
    W = np.zeros_like(abs_g)
    W[iu] = w_pairs
    W[(iu[1], iu[0])] = w_pairs  # symmetric

    # Ambient coefficient matrix: coeff_ij = W_ij * g_ij / |g_ij| on i≠j
    coeff = np.zeros_like(g)
    mask = abs_g > 0.0
    coeff[mask] = W[mask] * (g[mask] / abs_g[mask])
    np.fill_diagonal(coeff, 0.0)

    grad = coeff @ frame.vectors
    return PRODUCT_CP.project_to_tangent(frame, grad)
