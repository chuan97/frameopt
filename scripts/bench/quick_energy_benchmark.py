import time

import numpy as np

from evomof.core.energy import (
    diff_coherence,
    frame_potential,
    grad_diff_coherence,
    grad_frame_potential,
)
from evomof.core.frame import Frame


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
        ("Diff. coherence", timed(diff_coherence, F)),
        ("Grad DC", timed(grad_diff_coherence, F)),
    ]

    print("=== Quick Benchmark (ms per call) ===")
    for name, t in results:
        print(f"{name:<20}: {t:6.2f} ms")


if __name__ == "__main__":
    main()
