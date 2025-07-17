#!/usr/bin/env python
"""
Run Projection-CMA on a frame problem and then polish the best frame
with the Conjugate-Gradient (CG) local optimiser.

Example
-------
$ python examples/run_cma_cg.py               # default n=16, d=4
$ python examples/run_cma_cg.py -n 48 -d 8 \
      --sigma0 0.4 --popsize 80 --gen 200

"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence, grad_diff_coherence
from evomof.optim.cma.projection import ProjectionCMA
from evomof.optim.conjgrad import polish_with_conjgrad

here = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CMA → CG pipeline benchmark")
    p.add_argument("-n", type=int, default=16, help="Number of frame vectors")
    p.add_argument("-d", type=int, default=4, help="Ambient dimension")
    p.add_argument("--p", type=float, default=10, help="p exponent (diff-coherence)")
    p.add_argument("--sigma0", type=float, default=0.3, help="Initial CMA sigma")
    p.add_argument("--popsize", type=int, default=40, help="CMA population λ")
    p.add_argument("--gen", type=int, default=100, help="CMA generations")
    p.add_argument("--cg-iters", type=int, default=30, help="CG polish iterations")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(0)

    # ----------------------------- CMA stage -----------------------------
    print(f"Running Projection-CMA: n={args.n}, d={args.d}, gens={args.gen}")
    t0 = time.perf_counter()
    cma = ProjectionCMA(
        n=args.n,
        d=args.d,
        sigma0=args.sigma0,
        popsize=args.popsize,
        energy_fn=diff_coherence,
        energy_kwargs={"p": args.p},
        rng=rng,
    )
    best_frame = cma.run(max_gen=args.gen, log_every=max(args.gen // 10, 1))
    e_cma = diff_coherence(best_frame, p=args.p)
    coh_cma = coherence(best_frame)
    t_cma = time.perf_counter() - t0
    print(
        f"CMA done in {t_cma:.2f} s, diff-coh = {e_cma:.6f}, coherence = {coh_cma:.6f}"
    )

    # ----------------------------- CG polish -----------------------------
    print(f"Polishing with CG ({args.cg_iters} iterations)…")
    t1 = time.perf_counter()
    best_polished = polish_with_conjgrad(
        best_frame,
        energy_fn=lambda F: diff_coherence(F, p=args.p),
        grad_fn=lambda F: grad_diff_coherence(F, p=args.p),
        maxiter=args.cg_iters,
    )
    t_cg = time.perf_counter() - t1
    e_cg = diff_coherence(best_polished, p=args.p)
    coh = coherence(best_polished)

    # ----------------------------- Report -------------------------------
    print(
        f"CG done in {t_cg:.2f} s  →  diff-coh {e_cg:.6f} "
        f"(Δ = {e_cma - e_cg:+.3e}),  coherence = {coh:.6f}"
    )


if __name__ == "__main__":
    main()
