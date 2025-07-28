#!/usr/bin/env python3
"""
run_cma_cg_restart.py  –  Loop: CMA → CG → restart CMA

For each *loop*:
    1. Run Projection‑CMA for `--cma-gens` generations.
    2. Polish the current best frame with Conjugate‑Gradient (CG).
    3. Restart CMA with the polished frame as its mean.

A *CG run counts as one generation* in the live plot / log.

Features
--------
• Live plot of best diff‑coherence and coherence via `--live-plot`
• Optional CSV logging (`--log-file`)
• Optional input frame (`--input`) and export (`--export-npy / --export-txt`)
• Basic loop termination on `--loops` or when CG yields < tol improvement

Example
-------
$ python run_cma_cg_restart.py -n 30 -d 4 --loops 5 \
      --cma-max-gens 25 --cg-iters 300 --live-plot

"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence, grad_diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma.projection import ProjectionCMA


# --------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMA → CG → restart CMA loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # problem size
    p.add_argument("-n", type=int, default=16, help="Number of frame vectors")
    p.add_argument("-d", type=int, default=4, help="Ambient dimension")
    p.add_argument("-p", type=float, default=40, help="p exponent (diff-coherence)")

    # CMA params
    p.add_argument("--cma-sigma0", type=float, default=0.3, help="Initial CMA sigma")
    p.add_argument("--cma-popsize", type=int, default=40, help="CMA population λ")
    p.add_argument(
        "--cma-max-gens",
        type=int,
        default=25,
        help="Maximum CMA generations per loop before CG polish",
    )
    p.add_argument(
        "--cma-tol",
        type=float,
        default=0.0,
        help="Stop current CMA loop early when generation-best energy improves "
        "by less than this tolerance.",
    )
    p.add_argument(
        "--cma-sigma-restart-factor",
        type=float,
        default=0.5,
        help="sigma_restart = sigma_factor × sigma_end_of_loop (min 0.2*sigma0)",
    )

    # CG params
    p.add_argument("--cg-iters", type=int, default=300, help="Max CG iterations")
    p.add_argument(
        "--cg-tol",
        type=float,
        default=1e-8,
        help="Energy tolerance for early CG stop (passed to cg_minimize)",
    )

    # loop control
    p.add_argument("--loops", type=int, default=5, help="Max CMA→CG cycles")
    p.add_argument(
        "--no-stop-on-plateau",
        action="store_true",
        help="Disable automatic stop when CG improvement < 1e-6",
    )

    # I/O
    p.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input .npy frame (otherwise random)",
    )
    p.add_argument("--export-npy", type=str, default=None, help="Save best frame .npy")
    p.add_argument("--export-txt", type=str, default=None, help="Save best frame .txt")
    p.add_argument("--log-file", type=str, default=None, help="CSV log path")

    # misc
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument(
        "--live-plot",
        action="store_true",
        help="Show interactive plot of best energy & coherence",
    )
    return p.parse_args()


# --------------------------------------------------------------------- plotting helper
def setup_plot() -> tuple:
    """Return (fig, ax_e, ax_c, line_e, line_c, txt_coh) or (None,)*6 if matplotlib not available."""
    try:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, (ax_e, ax_c) = plt.subplots(
            nrows=2, sharex=True, figsize=(6, 6), constrained_layout=True
        )

        (line_e,) = ax_e.plot([], [], label="diff‑coh")
        (line_c,) = ax_c.plot([], [], label="coherence", color="tab:orange")

        ax_e.set_ylabel("diff‑coh")
        ax_e.set_yscale("log")  # log‑scale energy
        ax_c.set_ylabel("coherence")
        ax_c.set_xlabel("pseudo‑generation")

        # dynamic coherence label (top-left of coherence axis)
        txt_coh = ax_c.text(
            0.02,
            0.07,
            "",
            transform=ax_c.transAxes,
            va="top",
            ha="left",
            fontsize="small",
        )
        ax_e.legend(loc="upper right")
        ax_c.legend(loc="upper right")
        return fig, ax_e, ax_c, line_e, line_c, txt_coh
    except Exception:
        return (None, None, None, None, None, None)


def update_plot(fig, ax_e, ax_c, line_e, line_c, txt_coh, xs, ys_e, ys_c, cg_x, cg_y):
    if fig is None:
        return
    line_e.set_data(xs, ys_e)
    line_c.set_data(xs, ys_c)

    # clear and redraw CG markers on **energy** axis
    if cg_x:
        for coll in list(ax_e.collections):
            coll.remove()
        ax_e.scatter(cg_x, cg_y, marker="x", color="red", label="CG")

    if txt_coh is not None and ys_c:
        txt_coh.set_text(f"coh={ys_c[-1]:.8f}")

    for ax in (ax_e, ax_c):
        ax.relim()
        ax.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()


# --------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # initial frame / mean
    if args.input:
        best_frame = Frame.load_npy(args.input)
    else:
        best_frame = Frame.random(args.n, args.d, rng=rng)
    best_energy = diff_coherence(best_frame, p=args.p)
    best_coh = coherence(best_frame)

    # live plot setup
    fig, ax_e, ax_c, line_e, line_c, txt_coh = (None, None, None, None, None, None)
    if args.live_plot:
        fig, ax_e, ax_c, line_e, line_c, txt_coh = setup_plot()

    # logging containers
    metrics: list[dict] = []
    plot_x, plot_e, plot_c = [], [], []
    plot_cg_x, plot_cg_y = [], []

    start_time = time.perf_counter()
    pseudo_gen = 0  # counts CMA gens + 1 per CG
    sigma0 = args.cma_sigma0
    sigma_restart = sigma0
    popsize = args.cma_popsize

    for loop_idx in range(args.loops):
        # --- CMA phase --------------------------------------------------
        cma = ProjectionCMA(
            n=args.n,
            d=args.d,
            sigma0=sigma_restart,
            popsize=popsize,
            seed=rng.integers(0, 2**32),
            energy_fn=diff_coherence,
            energy_kwargs={"p": args.p},
            start_frame=best_frame,  # start from previous CG result
        )

        for gen in range(args.cma_max_gens):
            population = cma.ask()
            energies = [diff_coherence(f, p=args.p) for f in population]

            min_idx = int(np.argmin(energies))
            if energies[min_idx] < best_energy:
                best_energy = energies[min_idx]
                best_frame = population[min_idx]
                best_coh = coherence(best_frame)

            # generation-level convergence on CMA tol
            if args.cma_tol > 0:
                gen_best_E = energies[min_idx]
                if (
                    "prev_gen_best_E" in locals()
                    and abs(prev_gen_best_E - gen_best_E) < args.cma_tol
                ):
                    break  # early exit CMA loop
                prev_gen_best_E = gen_best_E

            cma.tell(population, energies)

            # per‑generation log
            elapsed = time.perf_counter() - start_time
            metrics.append(
                {
                    "pseudo_gen": pseudo_gen,
                    "loop": loop_idx,
                    "stage": "CMA",
                    "elapsed": elapsed,
                    "diff_coh": best_energy,
                    "coh": best_coh,
                }
            )
            plot_x.append(pseudo_gen)
            plot_e.append(best_energy)
            plot_c.append(best_coh)
            if args.live_plot and gen % 5 == 0:
                update_plot(
                    fig,
                    ax_e,
                    ax_c,
                    line_e,
                    line_c,
                    txt_coh,
                    plot_x,
                    plot_e,
                    plot_c,
                    plot_cg_x,
                    plot_cg_y,
                )
            pseudo_gen += 1

        # --- CG polish ---------------------------------------------------
        energy_before_cg = best_energy
        cg_start = time.perf_counter()
        polished = cg_minimize(
            best_frame,
            energy_fn=lambda F: diff_coherence(F, p=args.p),
            grad_fn=lambda F: grad_diff_coherence(F, p=args.p),
            maxiter=args.cg_iters,
            verbosity=1,
        )
        cg_time = time.perf_counter() - cg_start
        best_frame = polished
        best_energy = diff_coherence(polished, p=args.p)
        best_coh = coherence(polished)

        plot_cg_x.append(pseudo_gen)
        plot_cg_y.append(best_energy)

        metrics.append(
            {
                "pseudo_gen": pseudo_gen,
                "loop": loop_idx,
                "stage": "CG",
                "elapsed": time.perf_counter() - start_time,
                "diff_coh": best_energy,
                "coh": best_coh,
            }
        )
        plot_x.append(pseudo_gen)
        plot_e.append(best_energy)
        plot_c.append(best_coh)
        if args.live_plot:
            update_plot(
                fig,
                ax_e,
                ax_c,
                line_e,
                line_c,
                txt_coh,
                plot_x,
                plot_e,
                plot_c,
                plot_cg_x,
                plot_cg_y,
            )
        pseudo_gen += 1

        # termination check
        if not args.no_stop_on_plateau and abs(energy_before_cg - best_energy) < 1e-6:
            print(f"CG improvement <1e-6 after loop {loop_idx}; stopping.")
            break

        # prepare restart parameters
        sigma_restart = max(args.cma_sigma_restart_factor * cma.sigma, 0.2 * sigma0)

    total_time = time.perf_counter() - start_time

    # summary
    print(
        f"Finished {pseudo_gen} pseudo‑generations in {total_time:.2f}s → "
        f"diff‑coh {best_energy:.6e} | coherence {best_coh:.6e}"
    )

    # export
    if args.export_npy:
        best_frame.save_npy(args.export_npy)
    if args.export_txt:
        best_frame.export_txt(args.export_txt)

    # CSV log
    if args.log_file:
        with open(args.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)

    # keep interactive figure open
    if args.live_plot and fig is not None:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
