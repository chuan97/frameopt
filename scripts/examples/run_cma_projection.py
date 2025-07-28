"""
Example / smoke‑benchmark for projection‑based CMA‑ES.

Runs two short optimisations on (n = 36, d = 6):

1. Differentiable coherence with p = 16  (default metric)
2. Riesz s‑energy with s = 2             (custom energy_fn)

Both use identical CMA parameters so you can compare convergence speed.
"""

from evomof.core.energy import coherence, riesz_energy
from evomof.optim.cma import ProjectionCMA

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N, D = 121, 11
SIGMA0 = 0.3
POPSIZE = 40
MAX_GEN = 400
LOG_EVERY = 20

# ---------------------------------------------------------------------------
# 1. Diff‑coherence baseline
# ---------------------------------------------------------------------------
print("=== Diff‑coherence (p = 16) ===")
cma_diff = ProjectionCMA(
    N,
    D,
    sigma0=SIGMA0,
    popsize=POPSIZE,
    energy_kwargs={"p": 4 * D},  # forwarded to diff_coherence
)
best_diff = cma_diff.run(max_gen=MAX_GEN, log_every=LOG_EVERY)
print("Final coherence:", coherence(best_diff))

# ---------------------------------------------------------------------------
# 2. Riesz‑2 energy
# ---------------------------------------------------------------------------
print("\n=== Riesz energy (s = 2) ===")
cma_riesz = ProjectionCMA(
    N,
    D,
    sigma0=SIGMA0,
    popsize=POPSIZE,
    energy_fn=riesz_energy,
    energy_kwargs={"s": 2 * D},
)
best_riesz = cma_riesz.run(max_gen=MAX_GEN, log_every=LOG_EVERY)
print("Final coherence:", coherence(best_riesz))
