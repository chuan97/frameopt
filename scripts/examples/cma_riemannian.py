"""
Example for projection‑based CMA‑ES.
"""

from evomof.core.energy import coherence
from evomof.optim.cma import RiemannianCMA

N, D = 16, 4
SIGMA0 = 0.3
POPSIZE = 50
MAX_GEN = 1000
LOG_EVERY = 20

cma_diff = RiemannianCMA(
    N,
    D,
    sigma0=SIGMA0,
    popsize=POPSIZE,
    energy_kwargs={"p": 2 * D},  # forwarded to diff_coherence
)
best_diff = cma_diff.run(max_gen=MAX_GEN, log_every=LOG_EVERY)

print(
    f"Final coherence: {coherence(best_diff):.8f}",
)
