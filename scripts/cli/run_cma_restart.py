#!/usr/bin/env python3
"""
run_cma_restart_p.py  –  Pure Projection-CMA with progressive-p restarts.

Each restart multiplies the exponent p used in the differentiable coherence
(diff_coherence) and re-launches CMA from the previous best frame. Stopping
conditions:
  * coherence increases (worse), OR
  * improvement smaller than absolute tolerance, OR
  * maximum number of loops reached.

Example
-------
$ python run_cma_restart.py -n 30 -d 4 --gen 200 --loops 5 --tol 1e-4
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma.projection import ProjectionCMA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pure CMA with progressive-p restarts")
    p.add_argument("-n", type=int, default=16, help="Number of frame vectors")
    p.add_argument("-d", type=int, default=4, help="Ambient dimension")
    p.add_argument(
        "--sigma0", type=float, default=0.3, help="Initial CMA sigma each loop"
    )
    p.add_argument("--popsize", type=int, default=40, help="CMA population λ")
    p.add_argument("--gen", type=int, default=100, help="Generations per loop")
    p.add_argument("--seed", type=int, default=None, help="Base random seed")
    p.add_argument("--loops", type=int, default=5, help="Maximum restart loops")
    p.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Absolute coherence improvement tolerance for early stopping",
    )
    p.add_argument("--p0", type=float, default=6.0, help="Initial p exponent")
    p.add_argument(
        "--p-mult",
        type=float,
        default=2.0,
        help="Multiplier applied to p after each loop",
    )
    p.add_argument(
        "--export-npy",
        type=str,
        default=None,
        help="Optional filename to save final best frame (.npy)",
    )
    p.add_argument(
        "--export-txt",
        type=str,
        default=None,
        help="Optional filename to save final best frame (.txt submission)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Show dynamic plot of coherence across generations (marks loop boundaries).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Initial frame and stats
    best_frame = Frame.random(args.n, args.d, rng=rng)
    best_coh = coherence(best_frame)
    best_energy = diff_coherence(best_frame, p=args.p0)

    p_exp = args.p0
    start_time = time.perf_counter()
    cur_gen = args.gen
    cur_sigma = args.sigma0
    cur_popsize = args.popsize

    coh_history: list[float] = []  # per-generation generation-best coherence
    global_best_history: list[float] = []  # per-generation global best coherence
    sigma_history: list[float] = []  # per-generation sigma after tell()
    global_gen = 0
    plotting = False
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            plt.ion()
            plotting = True
            fig, (ax_coh, ax_sigma) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
            (line_cur,) = ax_coh.plot([], [], lw=2, label="Generation best coherence")
            (line_best,) = ax_coh.plot([], [], lw=2, label="Global best coherence")
            (line_sigma,) = ax_sigma.plot([], [], lw=2, label="Sigma")
            ax_coh.legend()
            ax_coh.set_ylabel("Coherence")
            ax_coh.set_title("Coherence across CMA generations with p restarts")
            ax_sigma.set_ylabel("Sigma")
            ax_sigma.set_xlabel("Generation")
            ax_sigma.legend()
        except Exception as e:
            print(f"Plotting disabled (matplotlib import failed: {e})")
            plotting = False

    print(
        f"Starting restart protocol: max_loops={args.loops}, p0={args.p0}, "
        f"p_mult={args.p_mult}, tol={args.tol}"
    )

    for loop in range(1, args.loops + 1):
        print(
            f"\nLoop {loop}: p = {p_exp:g}, gen={cur_gen}, sigma0={cur_sigma}, popsize={cur_popsize}"
        )
        # Loop boundary marker (skip first)
        if plotting and global_gen > 0:
            ax_coh.axvline(global_gen, color="gray", linestyle="--", alpha=0.5)
            ax_sigma.axvline(global_gen, color="gray", linestyle="--", alpha=0.5)
        prev_best_coh = best_coh  # store global best before this loop

        loop_seed = None if args.seed is None else args.seed + loop

        # --- Initialise CMA for this loop ---
        cma = ProjectionCMA(
            n=args.n,
            d=args.d,
            sigma0=cur_sigma,
            start_frame=best_frame,
            popsize=cur_popsize,
            energy_fn=diff_coherence,
            energy_kwargs={"p": p_exp},
            seed=loop_seed,
        )

        loop_best_frame = best_frame.copy()
        loop_best_E = diff_coherence(loop_best_frame, p=p_exp)

        for _ in range(cur_gen):
            pop = cma.ask()
            energies = [diff_coherence(f, p=p_exp) for f in pop]
            idx = int(np.argmin(energies))
            if energies[idx] < loop_best_E:
                loop_best_E = energies[idx]
                loop_best_frame = pop[idx]
            cma.tell(pop, energies)

            # Track generation-level best (current population's best)
            gen_best_coh = coherence(pop[idx])
            coh_history.append(gen_best_coh)
            sigma_history.append(cma.sigma)

            # Update global best across all generations
            if gen_best_coh < best_coh:
                best_coh = gen_best_coh
            global_best_history.append(best_coh)

            global_gen += 1
            if plotting:
                line_cur.set_data(range(1, global_gen + 1), coh_history)
                line_best.set_data(range(1, global_gen + 1), global_best_history)
                line_sigma.set_data(range(1, global_gen + 1), sigma_history)
                ax_coh.relim()
                ax_coh.autoscale_view()
                ax_sigma.relim()
                ax_sigma.autoscale_view()
                ax_sigma.set_xlim(left=1, right=global_gen + 1)
                fig.canvas.draw_idle()
                import matplotlib.pyplot as plt

                plt.pause(0.001)

        loop_coh = coherence(loop_best_frame)
        improvement = np.abs(prev_best_coh - loop_coh)
        print(
            f"  Previous coherence {prev_best_coh:.8f} → New {loop_coh:.8f} "
            f"(improvement {improvement:.3e})"
        )

        # Update best frame/energy if loop improved global best
        if loop_coh <= prev_best_coh:
            best_frame = loop_best_frame
            best_energy = diff_coherence(best_frame, p=p_exp)

        # Stopping criteria
        if improvement < args.tol:
            print(f"  Stopping: improvement {improvement:.3e} < tol {args.tol}.")
            break

        # Prepare parameters for next loop
        p_exp = min(int(1.3 * p_exp), 1350)
        cur_gen = int(1 * cur_gen)
        cur_sigma = cma.sigma
        cur_popsize *= 1

    runtime = time.perf_counter() - start_time
    print("\n=== Summary ===")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Final p exponent: {p_exp / args.p_mult:g}")  # last used
    print(f"Best diff-coherence energy: {best_energy:.6e}")
    print(f"Best coherence: {best_coh:.8f}")

    if plotting:
        import matplotlib.pyplot as plt

        plt.ioff()
        # Ensure sigma history marker lines are drawn for final loop
        ax_coh.axvline(global_gen, color="gray", linestyle="--", alpha=0.3)
        ax_sigma.axvline(global_gen, color="gray", linestyle="--", alpha=0.3)
        plt.show()

    if args.export_npy:
        best_frame.save_npy(args.export_npy)
        print(f"Saved frame → {args.export_npy}")
    if args.export_txt:
        best_frame.export_txt(args.export_txt)
        print(f"Saved frame → {args.export_txt}")


if __name__ == "__main__":
    main()
