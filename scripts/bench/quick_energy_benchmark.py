import time

import numpy as np

from frameopt.core.energy import (
    frame_potential,
    grad_frame_potential,
    grad_pnorm_coherence,
    pnorm_coherence,
)
from frameopt.core.frame import Frame


def timed(fn, *args, repeat=10):
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    t1 = time.perf_counter()
    return 1e3 * (t1 - t0) / repeat  # ms per call


def main():
    n, d = 16, 4
    rng = np.random.default_rng(42)
    F = Frame.random(n, d, rng=rng)

    results = [
        ("Frame potential", timed(frame_potential, F)),
        ("Grad FP", timed(grad_frame_potential, F)),
        ("p norm coherence", timed(pnorm_coherence, F)),
        ("Grad DC", timed(grad_pnorm_coherence, F)),
    ]

    print("=== Quick Benchmark (ms per call) ===")
    for name, t in results:
        print(f"{name:<20}: {t:6.2f} ms")


if __name__ == "__main__":
    main()
