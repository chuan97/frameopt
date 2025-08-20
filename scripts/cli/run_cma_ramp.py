#!/usr/bin/env python3
"""
run_cma_ramp.py  –  CMA-ES with p-exponent ramp (projection or riemannian).

Run a single CMA-ES instance (no restarts) while progressively increasing the
exponent *p* used in the differentiable coherence objective.

Two schedulers are available:
  • fixed:      periodic p increase, p <- min(p * p_mult, p_max) every `switch_every`.
  • adaptive:   event-based constant-window scheduler (budgeted): ramp when no
                new global best for `window` steps AND at least `window` since last ramp;
                per-ramp multiplier is chosen so that, if the interval repeats,
                p reaches p_max by generation `--gen`.

Optional CSV logging records per-generation metrics.

Example
-------
# Fixed scheduler (periodic)
$ python run_cma_ramp.py -n 30 -d 4 --gen 1000 --scheduler fixed \
      --p0 2 --p-mult 1.5 --switch-every 200 --p-max 80

# Riemannian CMA (periodic)
$ python run_cma_ramp.py -n 30 -d 4 --gen 1000 --algo riemannian --scheduler fixed \
      --p0 2 --p-mult 1.5 --switch-every 200 --p-max 80

# Adaptive scheduler (budgeted, constant window)
$ python run_cma_ramp.py -n 30 -d 4 --gen 1000 --scheduler adaptive \
      --p0 2 --p-max 1e6 --window 100
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma import ProjectionCMA, RiemannianCMA
from evomof.optim.utils.p_scheduler import (
    AdaptivePScheduler,
    FixedPScheduler,
    Scheduler,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pure Projection-CMA with progressive p ramp (no restarts).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", type=int, default=16, help="Number of frame vectors")
    p.add_argument("-d", type=int, default=4, help="Ambient dimension")
    p.add_argument("--gen", type=int, default=500, help="Total CMA generations")
    p.add_argument("--sigma0", type=float, default=0.5, help="Initial CMA sigma")
    p.add_argument(
        "--algo",
        type=str,
        choices=("projection", "riemannian"),
        default="riemannian",
        help="Which CMA variant to run: projection or riemannian",
    )
    p.add_argument(
        "--popsize",
        type=int,
        default=None,
        help="CMA population size λ (if omitted or 0, use 4 + floor(3*ln(2*n*d)))",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    # p-ramp parameters
    p.add_argument("--p0", type=float, default=2.0, help="Initial p exponent")
    p.add_argument(
        "--scheduler",
        type=str,
        choices=("fixed", "adaptive"),
        default="adaptive",
        help="p-ramp scheduler: fixed (periodic) or adaptive (budgeted, constant window)",
    )
    p.add_argument(
        "--p-mult",
        type=float,
        default=1.5,
        help="Multiplier applied to p at each switch step",
    )
    p.add_argument(
        "--p-max", type=float, default=1e9, help="Maximum p exponent (cap after ramp)"
    )
    p.add_argument(
        "--switch-every",
        type=int,
        default=200,
        help="Increase p every this many generations (0 disables ramp)",
    )
    p.add_argument(
        "--window",
        type=int,
        default=100,
        help="(adaptive) constant patience window in generations",
    )
    # Output / logging
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="CSV file to log per-generation metrics",
    )
    p.add_argument(
        "--export-npy", type=str, default=None, help="Save final best frame as .npy"
    )
    p.add_argument(
        "--export-txt",
        type=str,
        default=None,
        help="Save final best frame as .txt submission",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Resolve population size (script-level default)
    if args.popsize in (None, 0):
        dim = 2 * args.n * args.d
        popsize = 4 + int(3 * np.log(dim))
        print(f"[popsize] Using default λ={popsize} for dim={dim}")
    else:
        popsize = args.popsize

    # Build p-scheduler (Fixed or Adaptive)
    if args.scheduler == "fixed":
        switch_every = (
            args.switch_every if args.switch_every and args.switch_every > 0 else None
        )
        sched: Scheduler = FixedPScheduler(
            p0=args.p0,
            p_mult=args.p_mult,
            p_max=args.p_max,
            switch_every=switch_every,
        )
        print(
            f"[scheduler] fixed: p0={args.p0}, p_mult={args.p_mult}, switch_every={switch_every}, p_max={args.p_max}"
        )
    else:  # adaptive (budgeted, constant window)
        sched = AdaptivePScheduler(
            p0=args.p0,
            p_max=args.p_max,
            total_steps=args.gen,
            window=args.window,
        )
        print(
            f"[scheduler] adaptive: p0={args.p0}, window={args.window}, p_max={args.p_max}, total_steps={args.gen}"
        )
    p_exp = sched.current_p()

    if args.algo == "projection":
        cma = ProjectionCMA(
            n=args.n,
            d=args.d,
            sigma0=args.sigma0,
            popsize=popsize,
            energy_fn=diff_coherence,
            energy_kwargs={"p": p_exp},
            seed=args.seed,
        )
        print("[algo] projection CMA")
    else:  # riemannian
        cma = RiemannianCMA(
            n=args.n,
            d=args.d,
            sigma0=args.sigma0,
            popsize=popsize,
            energy_fn=diff_coherence,
            energy_kwargs={"p": p_exp},
            seed=args.seed,
        )
        print("[algo] riemannian CMA")

    # Best frame/energy tracked under *current* p
    best_frame = Frame.random(args.n, args.d, rng=rng)
    best_energy = diff_coherence(best_frame, p=p_exp)

    metrics: list[dict] = []

    t0 = time.perf_counter()
    global_best_coh = coherence(best_frame)
    global_best_frame = best_frame  # track frame achieving minimal coherence

    # Main CMA loop
    for gen in range(1, args.gen + 1):
        population = cma.ask()
        # Evaluate with current p exponent
        energies = [diff_coherence(f, p=p_exp) for f in population]

        # Generation best
        idx = int(np.argmin(energies))
        gen_best_energy = energies[idx]
        gen_best_frame = population[idx]
        gen_best_coh = coherence(gen_best_frame)

        # Update energy-based best (under current p)
        if gen_best_energy < best_energy:
            best_energy = gen_best_energy
            best_frame = gen_best_frame

        # Update coherence-based global best (monotonic)
        if gen_best_coh < global_best_coh:
            global_best_coh = gen_best_coh
            global_best_frame = gen_best_frame

        cma.tell(population, energies)

        # Log metrics
        elapsed = time.perf_counter() - t0

        sigma_val = getattr(cma, "sigma", None)
        mean_vec = getattr(cma, "mean", None)
        sigma_f = float(sigma_val) if sigma_val is not None else float("nan")
        mean_norm = (
            float(np.linalg.norm(mean_vec)) if mean_vec is not None else float("nan")
        )

        metrics.append(
            {
                "gen": gen,
                "elapsed_time": elapsed,
                "p": p_exp,
                "gen_best_diff_coh": gen_best_energy,
                "gen_best_coh": gen_best_coh,
                "best_diff_coh": best_energy,
                "best_coh": global_best_coh,
                "sigma": sigma_f,
                "mean_norm": mean_norm,
                "algo": args.algo,
            }
        )

        # Decide p ramp for the *next* generation using the scheduler
        p_next, switched = sched.update(
            step=gen,
            global_best_coh=global_best_coh,
        )
        if switched:
            print(f"[p-ramp] Generation {gen}: p {p_exp:g} -> {p_next:g}")
            # Re-evaluate best_energy under new p for consistent next-gen logging
            best_energy = diff_coherence(best_frame, p=p_next)

        p_exp = p_next

        # Light console feedback (10% intervals)
        if gen % max(args.gen // 10, 1) == 0:
            print(
                f"Gen {gen:5d} | p={p_exp:g} | best diff-coh={best_energy:.8f} | coherence={global_best_coh:.8f}"
            )

    runtime = time.perf_counter() - t0
    print(
        f"Finished {args.gen} generations in {runtime:.2f}s | final p={p_exp:g} | "
        f"best diff-coh (current p) {best_energy:.6e} | global best coherence {global_best_coh:.10f}"
    )

    # Write CSV if requested
    if args.log_file and metrics:
        with open(args.log_file, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
        print(f"Saved metrics to {args.log_file}")

    # Export final frame
    if args.export_npy:
        global_best_frame.save_npy(args.export_npy)
        print(f"Saved final frame to .npy → {args.export_npy}")
    if args.export_txt:
        global_best_frame.export_txt(args.export_txt)
        print(f"Saved final frame to .txt → {args.export_txt}")


if __name__ == "__main__":  # pragma: no cover
    main()
