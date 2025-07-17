import time

import numpy as np

from evomof.core.energy import riesz_energy
from evomof.core.frame import Frame

d, n = 6, 36
rng = np.random.default_rng(0)
frame = Frame.random(n, d, rng=rng)

t0 = time.perf_counter_ns()
E = riesz_energy(frame, s=2)
t1 = time.perf_counter_ns()

print(f"Riesz-2 energy: {E:.6f}")
print(f"Elapsed: {(t1-t0)/1e6:.3f} ms for n={n}, d={d}")
