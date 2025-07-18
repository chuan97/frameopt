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
from evomof.core.frame import Frame
from evomof.optim.cma.projection import ProjectionCMA
from evomof.optim.local import polish_with_cg

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
    p.add_argument(
        "--cg-run-every",
        type=int,
        default=None,
        help="Run CG polish on best individual every N CMA generations (None to skip)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (None for random)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ----------------------------- CMA stage with interleaved CG -----------------------------
    print(
        f"Running Projection-CMA with {args.gen} generations (interleave CG every {args.cg_run_every})"
    )
    t0 = time.perf_counter()
    cma = ProjectionCMA(
        n=args.n,
        d=args.d,
        sigma0=args.sigma0,
        popsize=args.popsize,
        energy_fn=diff_coherence,
        energy_kwargs={"p": args.p},
        seed=args.seed,
    )
    # Initialize best_frame and best_energy (handles cases like gen=0)
    best_frame = Frame.random(args.n, args.d, rng=rng)
    best_energy = diff_coherence(best_frame, p=args.p)
    # Main CMA loop
    for gen in range(1, args.gen + 1):
        # Sample population and evaluate
        population = cma.ask()
        energies = [diff_coherence(f, p=args.p) for f in population]
        # Update global best
        min_idx = int(np.argmin(energies))
        if energies[min_idx] < best_energy:
            best_energy = energies[min_idx]
            best_frame = population[min_idx]
        # Interleaved CG Polish
        if args.cg_run_every and gen % args.cg_run_every == 0:
            # Polish current best individual
            polished = polish_with_cg(
                best_frame,
                energy_fn=lambda F: diff_coherence(F, p=args.p),
                grad_fn=lambda F: grad_diff_coherence(F, p=args.p),
                maxiter=args.cg_iters,
            )
            # Recompute energy and replace in population
            energies[min_idx] = diff_coherence(polished, p=args.p)
            population[min_idx] = polished
            # Update global best if improved
            if energies[min_idx] < best_energy:
                best_energy = energies[min_idx]
                best_frame = polished
        # Reinjection
        cma.tell(population, energies)
        # Optional logging every 10% of run
        if gen % max(args.gen // 10, 1) == 0:
            print(f"  Gen {gen}: best diff-coh = {best_energy:.6f}")
    t_cma = time.perf_counter() - t0
    coh_cma = coherence(best_frame)
    print(
        f"CMA stage complete in {t_cma:.2f}s; diff-coh = {best_energy:.6f}, coherence = {coh_cma:.6f}"
    )

    # ----------------------------- CG polish -----------------------------
    print(f"Polishing with CG ({args.cg_iters} iterations)…")
    t1 = time.perf_counter()
    best_polished = polish_with_cg(
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
        f"(Δ = {best_energy - e_cg:+.3e}),  coherence = {coh:.6f}"
    )


if __name__ == "__main__":
    main()
