"""
Example for projection‑based CMA‑ES.
"""

from functools import partial

from frameopt.core.energy import coherence, pnormmax_coherence
from frameopt.optim.cma import RiemannianCMA

N, D = 16, 4
SIGMA0 = 0.3
POPSIZE = 50
MAX_GEN = 1000
LOG_EVERY = 20

cma = RiemannianCMA(
    N,
    D,
    sigma0=SIGMA0,
    popsize=POPSIZE,
    energy_fn=partial(pnormmax_coherence, p=2 * D),
)
best = cma.run(max_gen=MAX_GEN, log_every=LOG_EVERY)

print(
    f"Final coherence: {coherence(best):.8f}",
)
