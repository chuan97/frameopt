"""
Energy / potential functions for complex unit‑norm frames.

Public API
----------
frame_potential(frame, p)
riesz_energy(frame, s)
diff_coherence(frame, p)
coherence(frame)
"""

from __future__ import annotations

import typing

import numpy as np

from evomof.core._types import Complex128Array, Float64Array

from .frame import Frame

# -----------------------------------------------------------------------------#
# Helper utilities                                                             #
# -----------------------------------------------------------------------------#


def _absolute_inner(frame: Frame) -> Float64Array:
    """Return |⟨f_i, f_j⟩| with diagonal zeros (shape (n, n))."""
    g = np.abs(frame.gram)
    np.fill_diagonal(g, 0.0)
    return typing.cast(Float64Array, g.astype(np.float64, copy=False))


# -----------------------------------------------------------------------------#
# Public energy functions                                                      #
# -----------------------------------------------------------------------------#


def frame_potential(frame: Frame, p: float = 4.0) -> float:
    """
    p‑frame potential  Φ_p(F) = Σ_{i≠j} |⟨f_i,f_j⟩|^{2p}.

    Parameters
    ----------
    frame :
        Input frame.
    p :
        Positive real exponent.  Even integers appear in Welch/Riesz bounds;
        a larger *p* penalises large overlaps more aggressively.

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

    Underflow‑free evaluation of
        (Φ_p(F))^{1/(2p)}  with  Φ_p(F) = Σ_{i≠j} |⟨f_i,f_j⟩|^{2p}.

    We factor out the largest overlap G* to avoid |g|^{2p} underflow:
        L = G* * ( Σ (|g_ij|/G*)^{2p} )^{1/(2p)}

    As *p*→∞ this approaches a multiple of the true coherence µ(F).

    Raises
    ------
    ValueError
        If ``p`` is not positive.
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
    True frame coherence µ(F) = max_{i<j} |⟨f_i,f_j⟩|.

    Non‑differentiable but useful for reporting final results.
    """
    return float(_absolute_inner(frame).max())


def riesz_energy(frame: Frame, s: float = 2.0, eps: float = 1e-12) -> float:
    r"""
    Riesz *s*‑energy (p‑design surrogate).

    Parameters
    ----------
    frame :
        Input frame.
    s :
        Positive exponent.  Common choices: s = 1, 2, 4 …
    eps :
        Minimum chordal distance used to clamp nearly‑parallel pairs.  Values
        smaller than *eps* are replaced by *eps* to avoid numerical overflow
        in ``dist**(-s)`` when two vectors align.

    Raises
    ------
    ValueError
        If ``s`` is not positive.

    Returns
    -------
    float
        Riesz energy.
    """
    if s <= 0:
        raise ValueError("Exponent s must be positive.")

    # Pair‑wise chordal distances (n, n), zero on diagonal
    dist = frame.chordal_distances()

    # Extract upper‑triangle (k=1) and leverage symmetry by doubling.
    i, j = np.triu_indices_from(dist, k=1)
    dist_sub = dist[i, j]

    # Clamp to avoid division by zero / overflow for nearly‑parallel vectors
    dist_sub = np.maximum(dist_sub, eps)

    energy_half = np.sum(dist_sub ** (-s))
    return float(2.0 * energy_half)


def grad_frame_potential(frame: Frame, p: float = 4.0) -> Complex128Array:
    """
    Analytic Riemannian gradient of the *p*-frame potential.

    Parameters
    ----------
    frame :
        The current frame :math:`F \\in (S^{2d-1})^{n}`.
    p :
        Positive exponent in the potential
        :math:`\\Phi_p(F) = \\sum_{i\\neq j} |\\langle f_i,f_j\\rangle|^{2p}`.

    Raises
    ------
    ValueError
        If ``p`` is not positive.

    Returns
    -------
    Complex128Array
        Tangent array of shape ``frame.shape`` satisfying
        :math:`\\operatorname{Re}\\langle f_i,\\xi_i\\rangle = 0` for every row.
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
    return frame.project(grad)


def grad_diff_coherence(frame: Frame, p: float = 16.0) -> Complex128Array:
    """
    Underflow‑free gradient of the differentiable coherence surrogate
        L_p(F) = (Φ_p(F))^{1/(2p)}  with  Φ_p = Σ_{i≠j} |⟨f_i,f_j⟩|^{2p}.

    Using q = 2p and G* = max |⟨f_i,f_j⟩|, define r_ij = (|g_ij|/G*)^q and
    S = Σ r_ij. Then

        L_p = G* S^{1/q},
        ∇L_p = (2 L_p / S) Σ_{j≠i} r_ij * g_ij / |g_ij|^2 f_j,

    implemented without forming |g_ij|^{q} explicitly.
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
    return frame.project(grad)


# TODO: Fix gradient for Riesz energy (fails finite-difference test)
# See issue #1
def grad_riesz_energy(
    frame: Frame, s: float = 2.0, eps: float = 1e-12
) -> Complex128Array:
    """
    Analytic gradient of the Riesz *s*-energy on the product sphere.

    For each unordered pair ``(i,j)`` the contribution is

    .. math::

        -\\frac{s}{\\|f_i - f_j\\|^{s+2}}\\,(f_i - f_j),

    projected onto the tangent space.

    Parameters
    ----------
    frame :
        Input frame.
    s :
        Positive Riesz exponent.
    eps :
        Clamp for very small chordal distances to avoid overflow
        (must match the value used in :func:`riesz_energy`).

    Raises
    ------
    ValueError
        If ``s`` is not positive.

    Returns
    -------
    Complex128Array
        Tangent gradient array.
    """
    # Compute all chordal distances and index upper triangle
    dist = frame.chordal_distances()
    i, j = np.triu_indices_from(dist, k=1)
    # Clamp distances for safety
    dist_sub = np.maximum(dist[i, j], eps)

    # Precompute inner products
    g = frame.gram  # complex array of shape (n,n)
    # The derivative factor: 4 * s * dist^{-s-2}
    coeff = 4 * s * dist_sub ** (-(s + 2))

    # Accumulate Euclidean gradient
    grad = np.zeros_like(frame.vectors)
    for idx_i, idx_j, c, w in zip(i, j, g[i, j], coeff, strict=False):
        # Contribution from |<f_i, f_j>|^2 dependence
        grad[idx_i] += w * np.conj(c) * frame.vectors[idx_j]
        grad[idx_j] += w * c * frame.vectors[idx_i]

    # Project onto tangent space
    return frame.project(grad)
