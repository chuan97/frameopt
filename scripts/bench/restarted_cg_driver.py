#!/usr/bin/env python3
"""
simple_cg_benchmark.py  –  Hard‑coded CG-only test

Runs 20 random starts of conjugate‑gradient (CG) polishing on 30×4
frames with max 300 iterations each, then prints the best diff‑coherence
energy found.

Usage
-----
$ python simple_cg_benchmark.py
"""

from __future__ import annotations

import numpy as np

from evomof.core.energy import coherence, diff_coherence, grad_diff_coherence
from evomof.core.frame import Frame

# --- Hard‑coded parameters ---------------------------------------------------
N_STARTS = 1
N = 30
D = 4
CG_ITERS = 300
P_EXP = 40  # exponent for diff‑coherence, matches other benchmarks

# ---------------------------------------------------------------------------

best_E = float("inf")
best_frame: Frame | None = None

for seed in range(N_STARTS):
    rng = np.random.default_rng()
    frame0 = Frame.random(N, D, rng=rng)
    polished = cg_minimize(
        frame0,
        energy_fn=lambda F: diff_coherence(F, p=P_EXP),
        grad_fn=lambda F: grad_diff_coherence(F, p=P_EXP),
        maxiter=CG_ITERS,
    )
    E = diff_coherence(polished, p=P_EXP)
    if E < best_E:
        best_E = E
        best_frame = polished

best_coh = coherence(best_frame) if best_frame is not None else float("nan")
print(
    f"Best energy over {N_STARTS} starts: {best_E:.6e} " f"(coherence {best_coh:.6e})"
)
