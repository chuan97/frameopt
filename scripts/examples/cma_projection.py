"""
Example for projection‑based CMA‑ES.
"""

from frameopt.core.energy import coherence, diff_coherence
from frameopt.optim.cma import ProjectionCMA

N, D = 16, 4
SIGMA0 = 0.3
POPSIZE = 50
MAX_GEN = 1000
LOG_EVERY = 20

cma_diff = ProjectionCMA(
    N,
    D,
    sigma0=SIGMA0,
    popsize=POPSIZE,
    energy_fn=diff_coherence,
    energy_kwargs={"p": 2 * D},  # forwarded to diff_coherence
)
best_diff = cma_diff.run(max_gen=MAX_GEN, log_every=LOG_EVERY)

print(
    f"Final coherence: {coherence(best_diff):.8f}",
)
