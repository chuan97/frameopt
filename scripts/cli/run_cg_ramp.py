#!/usr/bin/env python3
"""
run_cg_ramp.py  –  Pure CG with a p-exponent ramp (no CMA, no restarts).

Runs Conjugate-Gradient in consecutive **chunks** of iterations. After each
chunk completes, the scheduler may increase the exponent p according to one of
two policies:

Schedulers
---------
• fixed      – periodic: ramp p at every chunk boundary (equivalently, every
               `--switch-every` iterations). Effect: p <- min(p * p_mult, p_max).
• adaptive   – budgeted, constant-window (event-based): treat each chunk end as
               one "step"; ramp when there has been **no new global best coherence**
               for the last `--window` chunks AND at least `--window` chunks since
               the previous ramp. The per-ramp multiplier is chosen so that, if the
               observed stall interval repeats, p reaches p_max by the final chunk.

The script logs per-chunk metrics (coherence and diff_coherence under current p),
and can start from a random frame or from an input .npy file.

Examples
--------
# Fixed scheduler: ramp every 200 iterations (one chunk)
$ python scripts/cli/run_cg_ramp.py -n 30 -d 4 --iters 2000 --scheduler fixed \
      --p0 2 --p-mult 1.5 --switch-every 200 --p-max 2048

# Adaptive scheduler: chunk size = 300 iterations, window over chunks = 3
$ python scripts/cli/run_cg_ramp.py -n 32 -d 6 --init-npy seed.npy --iters 3000 \
      --scheduler adaptive --p0 2 --p-max 1e6 --switch-every 300 --window 3
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from evomof.core.energy import coherence, diff_coherence, grad_diff_coherence
from evomof.core.frame import Frame
from evomof.optim.local import cg_minimize
from evomof.optim.utils.p_scheduler import (
    AdaptivePScheduler,
    FixedPScheduler,
    Scheduler,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pure CG with progressive p ramp (single run, no restarts).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", type=int, default=16, help="Number of frame vectors")
    p.add_argument("-d", type=int, default=4, help="Ambient dimension")
    p.add_argument("--iters", type=int, default=2000, help="Total CG iterations budget")
    p.add_argument(
        "--scheduler",
        type=str,
        choices=("fixed", "adaptive"),
        default="fixed",
        help=(
            "p-ramp scheduler: 'fixed' (default) ramps every CHUNK (i.e., at each "
            "--switch-every boundary). 'adaptive' is event-based with a constant "
            "window over CHUNKS: it ramps only at chunk boundaries when there has been "
            "no new global best for --window chunks and at least --window chunks since "
            "the last ramp. For CG, adaptive is useful mainly when chunks are small "
            "(e.g., 50–150 iterations) so it can detect stalls across chunks."
        ),
    )
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
        "--window",
        type=int,
        default=3,
        help="(adaptive) constant patience window in CHUNKS (not iterations)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Initial frame
    if args.init_npy:
        init_path = Path(args.init_npy)
        if not init_path.exists():
            raise FileNotFoundError(f"init-npy not found: {init_path}")
        frame = Frame.load_npy(init_path)
        print(f"[init] Using provided frame from {init_path}")
    else:
        frame = Frame.random(args.n, args.d, rng=rng)
        print(f"[init] Using random frame (seed={args.seed})")

    # Ensure shapes match args
    if frame.vectors.shape != (args.n, args.d):
        raise ValueError(
            f"Frame shape {frame.vectors.shape} != (n,d)=({args.n},{args.d})"
        )

    # Chunking: how many CG iterations per chunk (evaluation cadence)
    if args.switch_every and args.switch_every > 0:
        chunk_iters = int(args.switch_every)
    else:
        # No chunking: single chunk with all iterations
        chunk_iters = args.iters

    total_chunks = max(1, math.ceil(args.iters / chunk_iters))

    # Build p-scheduler. For fixed scheduler we ramp at each chunk boundary,
    # i.e., switch_every=1 in scheduler-steps (chunks). If only one chunk, disable ramp.
    if args.scheduler == "fixed":
        sched: Scheduler = FixedPScheduler(
            p0=args.p0,
            p_mult=args.p_mult,
            p_max=args.p_max,
            switch_every=(1 if total_chunks > 1 else None),
        )
        print(
            f"[scheduler] fixed | p0={args.p0}, p_mult={args.p_mult}, p_max={args.p_max}, "
            f"chunk_iters={chunk_iters}, total_chunks={total_chunks}"
        )
    else:
        if chunk_iters <= 0:
            raise ValueError(
                "adaptive scheduler requires --switch-every > 0 (chunk size)"
            )
        sched = AdaptivePScheduler(
            p0=args.p0,
            p_max=args.p_max,
            total_steps=total_chunks,
            window=args.window,
        )
        print(
            f"[scheduler] adaptive | p0={args.p0}, p_max={args.p_max}, window={args.window}, "
            f"chunk_iters={chunk_iters}, total_chunks={total_chunks}"
        )

    # Histories removed (unused)

    # Global best (by actual coherence)
    best_frame = frame
    best_coh = float(coherence(frame))

    t0 = time.perf_counter()
    it_cum = 0
    metrics: list[dict] = []

    print(
        f"Running CG for {args.iters} iterations in {total_chunks} chunks of {chunk_iters} (last may be smaller)"
    )

    for chunk_idx in range(1, total_chunks + 1):
        # Determine iterations for this chunk
        remaining = args.iters - it_cum
        if remaining <= 0:
            break
        this_chunk = min(chunk_iters, remaining)

        # Current p from scheduler
        p_exp = sched.current_p()

        # Define energy and gradient under current p
        E = lambda F: diff_coherence(F, p=p_exp)
        G = lambda F: grad_diff_coherence(F, p=p_exp)

        # Run CG for this chunk
        t_chunk0 = time.perf_counter()
        frame = cg_minimize(frame, energy_fn=E, grad_fn=G, maxiter=this_chunk)
        t_chunk = time.perf_counter() - t_chunk0
        it_cum += this_chunk

        # Evaluate metrics at chunk end
        cur_coh = float(coherence(frame))
        cur_diff = float(diff_coherence(frame, p=p_exp))

        # Update global best
        if cur_coh < best_coh:
            best_coh = cur_coh
            best_frame = frame

        # Record metrics (per CHUNK)
        elapsed = time.perf_counter() - t0
        metrics.append(
            {
                "chunk": chunk_idx,
                "iters_cumulative": it_cum,
                "elapsed_time": elapsed,
                "p": p_exp,
                "coherence": cur_coh,
                "diff_coherence": cur_diff,
            }
        )

        # Ask scheduler to possibly ramp for the **next** chunk
        p_before = p_exp
        p_after, switched = sched.update(step=chunk_idx, global_best_coh=best_coh)
        if switched:
            print(f"[p-ramp] chunk {chunk_idx}: p {p_before:g} -> {p_after:g}")

        # Console progress
        print(
            f"  chunk={chunk_idx:3d}/{total_chunks} | iters={it_cum:6d} | p={p_after:g} | "
            f"coh={cur_coh:.8f} | best={best_coh:.8f} | chunk_time={t_chunk:.2f}s"
        )

    runtime = time.perf_counter() - t0
    print(
        f"CG complete in {runtime:.2f}s | final p={p_after:g} | best coherence={best_coh:.8f}"
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


if __name__ == "__main__":  # pragma: no cover
    main()
