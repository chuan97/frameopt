#!/usr/bin/env python3
"""
run_cg_ramp.py  –  Pure CG with a p-exponent ramp (no CMA, no restarts).

Runs Conjugate-Gradient in consecutive chunks. After each chunk of `switch_every`
iterations, increases the exponent p via `p <- min(p * p_mult, p_max)` and continues
from the current frame.

The script logs per-chunk metrics (coherence and diff_coherence under current p),
supports live plotting, and can start from a random frame or from an input .npy file.

Examples
--------
$ python scripts/cli/run_cg_ramp.py -n 30 -d 4 --iters 2000 --p0 2 --p-mult 1.5 \
      --switch-every 200 --p-max 2048 --plot

$ python scripts/cli/run_cg_ramp.py -n 32 -d 6 --init-npy seed_frame.npy --iters 3000 \
      --p0 2 --p-mult 1.25 --switch-every 300 --plot --log-file cg_ramp.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from evomof.core.energy import coherence, diff_coherence, grad_diff_coherence
from evomof.core.frame import Frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pure CG with progressive p ramp (single run, no restarts).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", type=int, default=16, help="Number of frame vectors")
    p.add_argument("-d", type=int, default=4, help="Ambient dimension")
    p.add_argument("--iters", type=int, default=2000, help="Total CG iterations budget")
    p.add_argument(
        "--switch-every",
        type=int,
        default=200,
        help="Increase p every this many CG iterations (0 disables ramp)",
    )
    p.add_argument("--p0", type=float, default=2.0, help="Initial p exponent")
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
        "--seed", type=int, default=None, help="Random seed (for random initial frame)"
    )
    p.add_argument(
        "--init-npy",
        type=str,
        default=None,
        help="Optional path to .npy file with an initial frame (shape (n,d) or (d,n))",
    )
    p.add_argument(
        "--log-file", type=str, default=None, help="CSV file for per-chunk metrics"
    )
    p.add_argument(
        "--export-npy", type=str, default=None, help="Save final best frame as .npy"
    )
    p.add_argument(
        "--export-txt",
        type=str,
        default=None,
        help="Save final best frame as submission .txt",
    )
    p.add_argument(
        "--plot", action="store_true", help="Live plot: coherence and global best"
    )
    return p.parse_args()


def load_init_frame(path: Path, n: int, d: int) -> Frame:
    arr = np.load(path, allow_pickle=False)
    if arr.shape == (n, d):
        pass
    elif arr.shape == (d, n):
        arr = arr.T
        print("[init] Loaded (d,n) array; transposed to (n,d).")
    else:
        raise ValueError(
            f"Loaded array has shape {arr.shape}, expected (n,d)=({n},{d}) or (d,n)=({d},{n})."
        )
    # Ensure complex dtype (project uses complex arithmetic)
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128)
    return Frame.from_array(arr, copy=False)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Initial frame
    if args.init_npy:
        init_path = Path(args.init_npy)
        if not init_path.exists():
            raise FileNotFoundError(f"init-npy not found: {init_path}")
        frame = load_init_frame(init_path, args.n, args.d)
        print(f"[init] Using provided frame from {init_path}")
    else:
        frame = Frame.random(args.n, args.d, rng=rng)
        print(f"[init] Using random frame (seed={args.seed})")

    # Ensure shapes match args
    if frame.vectors.shape != (args.n, args.d):
        raise ValueError(
            f"Frame shape {frame.vectors.shape} != (n,d)=({args.n},{args.d})"
        )

    # p ramp state
    p_exp = float(args.p0)

    # Live plotting state
    plotting = False
    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.ion()
            plotting = True
            fig, ax_coh = plt.subplots(1, 1, figsize=(8, 4))
            (line_cur,) = ax_coh.plot([], [], lw=2, label="Chunk-end coherence")
            (line_best,) = ax_coh.plot([], [], lw=2, label="Global best coherence")
            ax_coh.set_xlabel("CG iterations (cumulative)")
            ax_coh.set_ylabel("Coherence")
            ax_coh.set_title("CG with p-ramp")
            ax_coh.legend()
        except Exception as e:  # pragma: no cover
            print(f"Plotting disabled (matplotlib import failed: {e})")
            plotting = False
            fig = ax_coh = line_cur = line_best = None  # type: ignore

    # Histories
    coh_history: list[float] = []
    best_history: list[float] = []
    iter_history: list[int] = []
    p_marks: list[int] = []

    # Global best (by actual coherence)
    best_frame = frame
    best_coh = float(coherence(frame))

    # Metrics store
    metrics: list[dict] = []
    t0 = time.perf_counter()
    it_cum = 0
    print(
        f"Running CG for {args.iters} iterations (switch p every {args.switch_every})"
    )

    while it_cum < args.iters:
        # Chunk size: either switch_every or the remaining iterations (if 0: use all remaining)
        if args.switch_every and args.switch_every > 0:
            chunk = min(args.switch_every, args.iters - it_cum)
        else:
            chunk = args.iters - it_cum

        # Define energy and gradient under current p
        E = lambda F: diff_coherence(F, p=p_exp)
        G = lambda F: grad_diff_coherence(F, p=p_exp)

        # Run CG for this chunk
        t_chunk0 = time.perf_counter()
        frame = cg_minimize(
            frame,
            energy_fn=E,
            grad_fn=G,
            maxiter=chunk,
        )
        t_chunk = time.perf_counter() - t_chunk0
        it_cum += chunk

        # Evaluate metrics at chunk end
        cur_coh = float(coherence(frame))
        cur_diff = float(diff_coherence(frame, p=p_exp))

        # Update global best
        if cur_coh < best_coh:
            best_coh = cur_coh
            best_frame = frame

        # Record metrics
        elapsed = time.perf_counter() - t0
        metrics.append(
            {
                "iters_cumulative": it_cum,
                "elapsed_time": elapsed,
                "p": p_exp,
                "coherence": cur_coh,
                "diff_coherence": cur_diff,
            }
        )

        # Update histories / live plot
        iter_history.append(it_cum)
        coh_history.append(cur_coh)
        best_history.append(best_coh)

        if plotting:
            line_cur.set_data(iter_history, coh_history)
            line_best.set_data(iter_history, best_history)
            ax_coh.relim()
            ax_coh.autoscale_view()
            fig.canvas.draw_idle()  # type: ignore
            import matplotlib.pyplot as plt  # type: ignore

            plt.pause(0.001)

        # p-ramp: increase p for next chunk
        if (
            args.switch_every
            and args.switch_every > 0
            and it_cum < args.iters
            and p_exp < args.p_max
        ):
            old_p = p_exp
            p_exp = min(p_exp * args.p_mult, args.p_max)
            print(f"[p-ramp] after {it_cum} iters: p {old_p:g} -> {p_exp:g}")
            if plotting:
                # vertical marker at this chunk boundary
                ax_coh.axvline(it_cum, color="gray", linestyle="--", alpha=0.5)
                fig.canvas.draw_idle()  # type: ignore

        # Console progress (every ~10% of budget)
        if len(iter_history) == 1 or it_cum >= (
            args.iters * (len(iter_history) / max(len(iter_history), 10))
        ):
            print(
                f"  iters={it_cum:6d} | p={p_exp:g} | coh={cur_coh:.8f} | best={best_coh:.8f} | chunk_time={t_chunk:.2f}s"
            )

    runtime = time.perf_counter() - t0
    print(
        f"CG complete in {runtime:.2f}s | final p={p_exp:g} | best coherence={best_coh:.8f}"
    )

    # CSV export
    if args.log_file and metrics:
        with open(args.log_file, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
        print(f"Saved metrics to {args.log_file}")

    # Frame export (best)
    if args.export_npy:
        best_frame.save_npy(args.export_npy)
        print(f"Saved best frame to .npy → {args.export_npy}")
    if args.export_txt:
        best_frame.export_txt(args.export_txt)
        print(f"Saved best frame to .txt → {args.export_txt}")

    if plotting:
        import matplotlib.pyplot as plt  # type: ignore

        plt.ioff()
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
