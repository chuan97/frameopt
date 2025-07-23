#!/usr/bin/env python3
"""
run_cma.py

Projection-CMA for frames

Example:
    python run_cma.py -n 48 -d 8 --sigma0 0.4 --popsize 80 --max-gen 200
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from evomof.core.energy import coherence, diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma.projection import ProjectionCMA
from evomof.optim.cma.utils import frame_to_realvec


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Projection-CMA for frames")
    p.add_argument("-n", type=int, default=16, help="Number of frame vectors")
    p.add_argument("-d", type=int, default=4, help="Ambient dimension")
    p.add_argument("-p", type=float, default=40, help="Exponent p for diff-coherence")
    p.add_argument("--sigma0", type=float, default=0.3, help="Initial CMA sigma")
    p.add_argument("--popsize", type=int, default=40, help="Population λ")
    p.add_argument("--max-gen", type=int, default=100, help="Maximum generations")
    p.add_argument(
        "--tol",
        type=float,
        default=1e-20,
        help="Absolute tolerance; stop when best‑energy improvement < tol",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument(
        "--export-npy",
        type=str,
        default=None,
        help="If set, save the best Frame to this .npy file",
    )
    p.add_argument(
        "--export-txt",
        type=str,
        default=None,
        help="If set, export the best Frame to flat .txt (submission format)",
    )
    p.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input Frame .npy file; if omitted, a random frame is created",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

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

    if args.input:
        best_frame = Frame.load_npy(args.input)
        # use supplied frame as CMA mean
        cma.mean = frame_to_realvec(best_frame)
    else:
        best_frame = Frame.random(args.n, args.d, rng=rng)

    best_frame = cma.run(
        max_gen=args.max_gen,
        tol=args.tol,
        log_every=max(args.max_gen // 10, 1),
    )
    best_energy = diff_coherence(best_frame, p=args.p)

    runtime = time.perf_counter() - t0
    print(
        f"CMA finished in {runtime:.2f}s  |  "
        f"diff-coh {best_energy:.8f}  |  coherence {coherence(best_frame):.8f}"
    )

    # Optional export of best frame
    if args.export_npy:
        best_frame.save_npy(args.export_npy)
        print(f"Saved frame as .npy → {args.export_npy}")
    if args.export_txt:
        best_frame.export_txt(args.export_txt)
        print(f"Saved frame as .txt → {args.export_txt}")


if __name__ == "__main__":
    main()
