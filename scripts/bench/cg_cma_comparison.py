#!/usr/bin/env python3
"""
cg_cma_comparison.py  –  Quick benchmark: pure CMA vs. multi‑start CG


The script calls the existing runners:
    scripts/cli/run_cma.py      (pure CMA)
    scripts/cli/run_cg.py       (pure CG)

For each (n,d) it prints:
    n  d   diff‑coh_CMA   diff‑coh_CG   Δenergy   Δcoherence
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from evomof.core.energy import coherence, diff_coherence
from evomof.core.frame import Frame

# ----------------------------------------------------------------------
# Generate all (n,d) pairs with  d ∈ [2..6]  and  n ∈ [7..50]
# ----------------------------------------------------------------------
PAIRS: list[tuple[int, int]] = [
    (n, d) for d in range(2, 6 + 1) for n in range(7, 50 + 1)
]

ROOT = Path(__file__).resolve().parents[2]  # repo root

RUN_CMA = ROOT / "scripts" / "cli" / "run_cma.py"

RUN_CG = ROOT / "scripts" / "cli" / "run_cg.py"

P_EXP = 40

COMMON_CMA = [
    "--sigma0",
    "0.3",
    "--popsize",
    "40",
    "-p",
    str(P_EXP),
    "--max-gen",
    "10000",
    "--tol",
    "1e-20",
    "--seed",
    "42",
]

CG_N_STARTS = 20

COMMON_CG = ["--cg-iters", "1000", "-p", str(P_EXP), "--verbose"]


def run_and_load(cmd: list[str], out_path: Path, p: int) -> tuple[float, float]:
    """Run external runner that saves a frame to *out_path*; return (energy, coherence)."""
    # Ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove any stale file
    out_path.unlink(missing_ok=True)
    # Append export flag to command
    cmd_ext = cmd + ["--export-npy", str(out_path)]
    subprocess.run(cmd_ext, check=True, text=True)
    frame = Frame.load_npy(out_path)
    return diff_coherence(frame, p=p), coherence(frame)


def main() -> None:
    results = []
    for n, d in PAIRS:
        # CMA
        cma_tmp = Path(tempfile.gettempdir()) / f"cma_{n}x{d}.npy"
        cmd_cma = [
            sys.executable,
            str(RUN_CMA),
            "-n",
            str(n),
            "-d",
            str(d),
            *COMMON_CMA,
        ]
        t0 = time.perf_counter()
        e_cma, c_cma = run_and_load(cmd_cma, cma_tmp, P_EXP)
        t_cma = time.perf_counter() - t0

        # CG multistart
        best_e_cg = float("inf")
        best_c_cg = float("nan")
        t_cg_total = 0.0
        for seed in range(CG_N_STARTS):
            cg_tmp = Path(tempfile.gettempdir()) / f"cg_{n}x{d}_s{seed}.npy"
            cmd_cg = [
                sys.executable,
                str(RUN_CG),
                "-n",
                str(n),
                "-d",
                str(d),
                "--seed",
                str(seed),
                *COMMON_CG,
            ]
            t0 = time.perf_counter()
            e_tmp, c_tmp = run_and_load(cmd_cg, cg_tmp, P_EXP)
            t_cg_total += time.perf_counter() - t0
            if e_tmp < best_e_cg:
                best_e_cg, best_c_cg = e_tmp, c_tmp
        results.append((n, d, e_cma, best_e_cg, c_cma, best_c_cg, t_cma, t_cg_total))

    # Print summary once
    print(" n   d   ΔE          CMA_coh      CG_coh      Δcoh        tCMA    tCG    Δt")
    print("-" * 80)
    for n, d, e_cma, e_cg, c_cma, c_cg, t_cma, t_cg in results:
        print(
            f"{n:2d}  {d:2d}  {e_cma - e_cg: .2e}  "
            f"{c_cma:.8f}  {c_cg:.8f}  {c_cma - c_cg: .2e}  "
            f"{t_cma:6.2f}  {t_cg:6.2f}  {t_cma - t_cg: .2e}"
        )


if __name__ == "__main__":
    main()
