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

from evomof.core._types import Float64Array

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
    p‑frame potential  Φ_p(F) = Σ_{i≠j} |⟨f_i,f_j⟩|^p.

    Parameters
    ----------
    frame :
        Input frame.
    p :
        Positive real exponent.  Even integers appear in Welch/Riesz bounds;
        a larger *p* penalises large overlaps more aggressively.

    Returns
    -------
    float
        Value of the potential.
    """
    if p <= 0:
        raise ValueError("Exponent p must be positive.")
    aij = _absolute_inner(frame) ** p
    return float(aij.sum())


def diff_coherence(frame: Frame, p: float = 16.0) -> float:
    """
    Differentiable surrogate for frame coherence.

    Defined as  (Φ_p(F))^{1/p}.  As *p*→∞ this approaches a multiple of
    the true coherence µ(F) = max_{i<j} |⟨f_i,f_j⟩|.
    """
    phi_p = frame_potential(frame, p)
    # Guard against numerical underflow when phi_p ≈ 0
    return float(phi_p ** (1.0 / p)) if phi_p != 0.0 else 0.0


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
