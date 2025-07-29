#!/usr/bin/env python3
"""
run_cma_ramp.py  –  Pure Projection-CMA with a p-exponent ramp.

Run a single CMA-ES instance (no restarts) while progressively increasing the
exponent *p* used in the differentiable coherence objective. The ramp follows:
    p <- min(p * p_mult, p_max)
whenever the generation counter hits a multiple of `switch_every`.

Optional live plotting shows generation-level best coherence, global best
coherence, and sigma over time.

Example
-------
$ python run_cma_ramp.py -n 30 -d 4 --gen 1000 --p0 2 --p-mult 1.5 \
      --switch-every 200 --p-max 80 --plot
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma import ProjectionCMA
from evomof.optim.utils.p_scheduler import PScheduler

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
        "--popsize",
        type=int,
        default=None,
        help="CMA population size λ (if omitted or 0, use 4 + floor(3*ln(2*n*d)))",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    # p-ramp parameters
    p.add_argument("--p0", type=float, default=2.0, help="Initial p exponent")
    p.add_argument(
        "--p-mult",
        type=float,
        default=1.5,
        help="Multiplier applied to p at each switch step",
    )
    p.add_argument(
        "--p-max", type=float, default=1e12, help="Maximum p exponent (cap after ramp)"
    )
    p.add_argument(
        "--switch-every",
        type=int,
        default=200,
        help="Increase p every this many generations (0 disables ramp)",
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
    p.add_argument(
        "--plot",
        action="store_true",
        help="Show dynamic plot of coherence/sigma across generations",
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

    # CMA initialisation (energy kwargs will be updated as p changes)
    sched = PScheduler(
        mode="adaptive",
        p0=args.p0,
        p_mult=args.p_mult,
        p_max=args.p_max,
        switch_every=(args.switch_every if args.switch_every > 0 else None),
    )
    p_exp = sched.current_p()
    cma = ProjectionCMA(
        n=args.n,
        d=args.d,
        sigma0=args.sigma0,
        popsize=popsize,
        energy_fn=diff_coherence,
        energy_kwargs={"p": p_exp},  # used only if user calls convenience wrappers
        seed=args.seed,
    )

    # Best frame/energy tracked under *current* p
    best_frame = Frame.random(args.n, args.d, rng=rng)
    best_energy = diff_coherence(best_frame, p=p_exp)

    metrics: list[dict] = []

    # ----- Optional dynamic plotting -------------------------------------
    coh_history: list[float] = []  # generation best
    global_best_history: list[float] = []  # global best coherence
    sigma_history: list[float] = []
    p_history: list[float] = []

    plotting = False
    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.ion()
            plotting = True
            fig, (ax_coh, ax_sigma) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
            (line_cur,) = ax_coh.plot([], [], lw=2, label="Generation best coherence")
            (line_best,) = ax_coh.plot([], [], lw=2, label="Global best coherence")
            (line_sigma,) = ax_sigma.plot([], [], lw=2, label="Sigma")
            ax_coh.set_ylabel("Coherence")
            ax_coh.set_title("CMA with p-ramp")
            ax_coh.legend()
            ax_sigma.set_ylabel("Sigma")
            ax_sigma.set_xlabel("Generation")
            ax_sigma.axhline(0.0, color="black", linewidth=2.0, alpha=0.6)
            ax_sigma.legend()
        except Exception as e:  # pragma: no cover
            print(f"Plotting disabled (matplotlib import failed: {e})")
            plotting = False
            fig = ax_coh = ax_sigma = None  # type: ignore
            line_cur = line_best = line_sigma = None  # type: ignore

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
        metrics.append(
            {
                "gen": gen,
                "elapsed_time": elapsed,
                "p": p_exp,
                "gen_best_diff_coh": gen_best_energy,
                "gen_best_coh": gen_best_coh,
                "best_diff_coh": best_energy,
                "best_coh": global_best_coh,
                "sigma": cma.sigma,
                "mean_norm": float(np.linalg.norm(cma.mean)),
            }
        )

        # Update dynamic plot
        if plotting:
            coh_history.append(gen_best_coh)
            global_best_history.append(global_best_coh)
            sigma_history.append(cma.sigma)
            p_history.append(p_exp)
            line_cur.set_data(range(1, gen + 1), coh_history)
            line_best.set_data(range(1, gen + 1), global_best_history)
            line_sigma.set_data(range(1, gen + 1), sigma_history)
            ax_coh.relim()
            ax_coh.autoscale_view()
            ax_sigma.relim()
            ax_sigma.autoscale_view()
            ax_sigma.set_xlim(left=1, right=gen + 1)
            fig.canvas.draw_idle()  # type: ignore
            import matplotlib.pyplot as plt  # type: ignore

            plt.pause(0.001)

        # Decide p ramp for the *next* generation using the scheduler
        p_next, switched = sched.update(
            step=gen,
            global_best_coh=global_best_coh,
        )
        if switched:
            print(f"[p-ramp] Generation {gen}: p {p_exp:g} -> {p_next:g}")
            # Re-evaluate best_energy under new p for consistent next-gen logging
            best_energy = diff_coherence(best_frame, p=p_next)
            if plotting:
                ax_coh.axvline(gen, color="gray", linestyle="--", alpha=0.5)
                ax_sigma.axvline(gen, color="gray", linestyle="--", alpha=0.5)
        p_exp = p_next

        # Light console feedback (10% intervals)
        if gen % max(args.gen // 10, 1) == 0:
            print(
                f"Gen {gen:5d} | p={p_exp:g} | best diff-coh={best_energy:.8f} | coherence={global_best_coh:.8f}"
            )

    runtime = time.perf_counter() - t0
    print(
        f"Finished {args.gen} generations in {runtime:.2f}s | final p={p_exp:g} | "
        f"best diff-coh (current p) {best_energy:.6e} | global best coherence {global_best_coh:.8f}"
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

    if plotting:
        import matplotlib.pyplot as plt  # type: ignore

        plt.ioff()
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
