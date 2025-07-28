#!/usr/bin/env python3
"""
run_cg.py

Conjugate‐gradient for frames

Usage examples:
  # Polish an existing frame:
  python run_cg.py --input results/frame.npy --cg-iters 50 --export-npy results/frame_polished.npy

  # Generate a random 4×40 frame and polish it:
  python run_cg.py -n 40 -d 4 --seed 123 --cg-iters 30 --export-txt 4x40_abcd.txt
"""

import argparse
import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence, grad_diff_coherence
from evomof.core.frame import Frame


def parse_args():
    p = argparse.ArgumentParser(description="CG for frames")
    p.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of vectors (n) (required if --input is omitted)",
    )
    p.add_argument(
        "-d",
        type=int,
        default=None,
        help="Ambient dimension (d) (required if --input is omitted)",
    )
    p.add_argument(
        "-p", type=int, default=40, help="Exponent p for differentiable coherence"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for initial frame when --input is omitted",
    )
    p.add_argument(
        "--cg-iters", type=int, default=30, help="Maximum CG polish iterations"
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print CG iteration count and stopping criterion",
    )
    p.add_argument(
        "--export-npy",
        type=str,
        default=None,
        help="If set, save the polished Frame to this .npy file",
    )
    p.add_argument(
        "--export-txt",
        type=str,
        default=None,
        help="If set, export the polished Frame to flat .txt (submission format)",
    )
    p.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input Frame .npy file; if omitted, a random frame is created",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load or generate initial Frame
    if args.input:
        frame = Frame.load_npy(args.input)
    else:
        if args.n is None or args.d is None:
            raise ValueError("Must specify --n and --d when no --input is provided")
        rng = np.random.default_rng(args.seed)
        frame = Frame.random(n=args.n, d=args.d, rng=rng)

    # 3) Run CG polish
    t0 = time.perf_counter()
    polished = cg_minimize(
        frame,
        energy_fn=lambda F: diff_coherence(F, p=args.p),
        grad_fn=lambda F: grad_diff_coherence(F, p=args.p),
        maxiter=args.cg_iters,
        verbosity=1 if args.verbose else 0,
    )
    t_polish = time.perf_counter() - t0

    # 4) Report final quality
    final_diff = diff_coherence(polished, p=args.p)
    final_coh = coherence(polished)
    print(
        f"[Polished] time={t_polish:.2f}s  |  "
        f"diff-coh = {final_diff:.8f}  |  coherence = {final_coh:.8f}"
    )

    # 5) Export if requested
    if args.export_npy:
        polished.save_npy(args.export_npy)
        print(f"Saved frame as .npy → {args.export_npy}")
    if args.export_txt:
        polished.export_txt(args.export_txt)
        print(f"Saved frame as .txt → {args.export_txt}")


if __name__ == "__main__":
    main()
