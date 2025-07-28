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
import csv
import pathlib
import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence, grad_diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma.projection import ProjectionCMA
from evomof.optim.cma.utils import frame_to_realvec

here = pathlib.Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CMA → CG pipeline benchmark")
    p.add_argument("-n", type=int, default=16, help="Number of frame vectors")
    p.add_argument("-d", type=int, default=4, help="Ambient dimension")
    p.add_argument("-p", type=float, default=10, help="p exponent (diff-coherence)")
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
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to output CSV file logging per-generation metrics",
    )
    p.add_argument(
        "--export-npy",
        type=str,
        default=None,
        help="Output filename to save final frame as .npy",
    )
    p.add_argument(
        "--export-txt",
        type=str,
        default=None,
        help="Output filename to save final frame as flat text submission format",
    )
    p.add_argument(
        "--mean-mix-coeff",
        type=float,
        default=0.5,
        help="Blend coefficient (0–1) for mixing CMA mean toward the CG‑polished individual; 0 disables mean mixing",
    )
    p.add_argument(
        "--sigma-boost",
        type=float,
        default=2.0,
        help="Multiplicative factor (>1) to increase CMA sigma after mean mixing; 1 disables boosting",
    )
    p.add_argument(
        "--no-inject-polished",
        action="store_false",
        dest="inject_polished",
        help="Skip inserting the CG‑polished individual into the CMA population; rely only on mean mixing/sigma boosting",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Prepare metrics logging
    metrics: list[dict] = []

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
        # Generation start timestamp
        gen_start = time.perf_counter()

        # Sample population and evaluate
        population = cma.ask()
        energies = [diff_coherence(f, p=args.p) for f in population]

        # Update global best
        min_idx = int(np.argmin(energies))
        if energies[min_idx] < best_energy:
            best_energy = energies[min_idx]
            best_frame = population[min_idx]

        # Interleaved CG Polish
        cg_time = 0.0
        cg_gain = 0.0
        polished_vec: np.ndarray | None = None
        if args.cg_run_every and gen % args.cg_run_every == 0:
            # Polish current best individual
            cg_start = time.perf_counter()
            polished = cg_minimize(
                best_frame,
                energy_fn=lambda F: diff_coherence(F, p=args.p),
                grad_fn=lambda F: grad_diff_coherence(F, p=args.p),
                maxiter=args.cg_iters,
            )
            polished_vec = frame_to_realvec(polished)
            cg_time = time.perf_counter() - cg_start

            # Recompute energy and replace in population
            polished_energy = diff_coherence(polished, p=args.p)
            cg_gain = best_energy - polished_energy
            if args.inject_polished:
                energies[min_idx] = polished_energy
                population[min_idx] = polished

            # Update global best if improved
            if energies[min_idx] < best_energy:
                best_energy = energies[min_idx]
                best_frame = polished

        # Reinjection
        cma.tell(population, energies)

        # --- Mean mixing and sigma boosting (post‑tell) -----------------
        if polished_vec is not None:
            if args.mean_mix_coeff > 0.0:
                # --- Direction‑only blend ----------------------------------
                old_norm = np.linalg.norm(cma.mean)
                if old_norm > 0.0:  # guard against numerical edge case
                    mean_dir = cma.mean / old_norm
                    pol_dir = polished_vec / np.linalg.norm(polished_vec)
                    mixed_dir = (
                        1.0 - args.mean_mix_coeff
                    ) * mean_dir + args.mean_mix_coeff * pol_dir
                    mixed_dir /= np.linalg.norm(mixed_dir)  # renormalise to unit length
                    # re‑apply original radius so radius statistics stay unchanged
                    cma.mean = mixed_dir * old_norm
            if args.sigma_boost > 1.0:
                cma.sigma *= args.sigma_boost

        # Record metrics for this generation (after mix/boost)
        metrics.append(
            {
                "gen": gen,
                "elapsed_time": time.perf_counter() - t0,
                "best_diff_coh": best_energy,
                "best_coh": float(coherence(best_frame)),
                "cg_run": bool(args.cg_run_every and gen % args.cg_run_every == 0),
                "cg_time": cg_time,
                "cg_gain": cg_gain,
                "sigma": cma.sigma,
                "mean_norm": float(np.linalg.norm(cma.mean)),
            }
        )

        # Optional logging every 10% of run
        if gen % max(args.gen // 10, 1) == 0:
            print(f"  Gen {gen}: best diff-coh = {best_energy:.6f}")

    t_cma = time.perf_counter() - t0
    coh_cma = coherence(best_frame)
    print(
        f"CMA stage complete in {t_cma:.2f}s; diff-coh = {best_energy:.6f}, coherence = {coh_cma:.6f}"
    )

    # Write metrics to CSV if requested
    if args.log_file:
        with open(args.log_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)

    # Export final frame if requested
    if args.export_npy:
        best_frame.save_npy(args.export_npy)
        print(f"Saved final frame to .npy → {args.export_npy}")
    if args.export_txt:
        best_frame.export_txt(args.export_txt)
        print(f"Saved final frame to .txt → {args.export_txt}")


if __name__ == "__main__":
    main()
