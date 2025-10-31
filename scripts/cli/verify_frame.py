#!/usr/bin/env python3
"""
Verify a frame's coherence from a .npy file.

This tool loads a real/complex frame array of shape (n, d), normalizes rows to
unit ℓ2 norm, computes the maximum off‑diagonal absolute inner product (the
coherence) in float64 and, optionally, in high precision via mpmath. It prints a
human‑readable summary (including top‑K pairs and simple statistics) and can
optionally write a JSON certificate next to the input file.

Notes:
  • Only .npy inputs are supported (submission .txt is intentionally not supported here).
  • High‑precision verification is optional and used when --mp-dps > 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mpmath as mp
except Exception:
    mp = None  # optional; only used if --mp-dps > 0


FNAME_RE = re.compile(r"(\d+)x(\d+)_\w{1,4}\.txt$")


def load_txt_submission(path: Path, d: int, n: int) -> np.ndarray:
    """Load a submission-format .txt file into a complex (n, d) array.

    The file must contain exactly 2*d*n floats: first all real parts (vector-by-vector,
    component-by-component), then all imaginary parts, one number per line.
    """
    data = np.loadtxt(path, dtype=float)
    expected = 2 * d * n
    if data.size != expected:
        raise ValueError(
            f"Expected {expected} entries for d={d}, n={n}; got {data.size}."
        )
    # First d*n are reals, next d*n are imags; reshape as (n, d)
    reals = data[: d * n].reshape(n, d)
    imags = data[d * n :].reshape(n, d)
    return (reals + 1j * imags).astype(np.complex128, copy=False)


def sha256_file(path: Path) -> str:
    """Return the SHA‑256 hex digest of the file contents at `path`."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------- loading & normalization -------------------------


def load_frame(path: Path) -> np.ndarray:
    """
    Load a frame from a .npy or .txt file.

    Parameters
    ----------
    path : Path
        Path to a NumPy `.npy` file containing a 2‑D array of shape (n, d), real or complex,
        or a submission-format `.txt` file containing 2*d*n floats as reals then imaginaries.

    Returns
    -------
    np.ndarray
        Array with dtype complex128 and shape (n, d). Values are not modified beyond dtype casting.
    """
    suffix = path.suffix.lower()
    if suffix == ".txt":
        m = FNAME_RE.search(path.name)
        if not m:
            raise ValueError(
                "Txt filename must be like 'dxn_xxx.txt' (e.g. '4x6_init.txt'), with a 3–4 char suffix."
            )
        d, n = map(int, m.groups())
        return load_txt_submission(path, d=d, n=n)
    if suffix != ".npy":
        raise ValueError(
            f"Unsupported file extension: {suffix}. Only .npy is supported."
        )
    F = np.load(path)
    if F.ndim != 2:
        raise ValueError(f".npy must be 2D (n x d); got shape {F.shape}")
    return F.astype(np.complex128, copy=False)


def normalize_rows(F: np.ndarray) -> np.ndarray:
    """
    Normalize each row of F to unit ℓ2 norm.

    Rows with zero norm are left unchanged (protected by replacing 0 with 1 in the divisor).
    Returns a new array; does not modify the input in place.
    """
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return F / norms


# ---------------------------- coherence & stats ----------------------------


def coherence_float64(
    F: np.ndarray,
) -> tuple[float, tuple[int, int], float, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Compute coherence in float64 and supporting data.

    The input is row‑normalized internally, then the Gram matrix G = F F* is formed
    and the absolute upper‑triangular entries are scanned.

    Returns
    -------
    (mu, argmax_pair, mu2, off, iu)
        mu : float
            Maximum |G_ij| over i<j (coherence).
        argmax_pair : tuple[int, int]
            Indices (i, j) attaining the maximum.
        mu2 : float
            Second largest off‑diagonal magnitude (masking out one occurrence of the max).
        off : np.ndarray
            Vectorized upper‑triangular absolute values, length n*(n−1)/2.
        iu : tuple[np.ndarray, np.ndarray]
            The (i, j) index arrays used to map back from `off` to matrix pairs.
    """
    F = normalize_rows(F)
    G = F @ F.conj().T
    A = np.abs(G)
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    off = A[iu]
    # max and argmax
    k = int(np.argmax(off))
    mu = float(off[k])
    # second largest (mask-out one max occurrence)
    off2 = off.copy()
    off2[k] = -np.inf
    mu2 = float(off2.max())
    # map back to (i,j)
    i, j = iu[0][k], iu[1][k]
    return mu, (int(i), int(j)), mu2, off, iu


def coherence_mpmath(
    F: np.ndarray, dps: int = 120, top_pairs: list[tuple[int, int]] | None = None
) -> float:
    """
    Compute coherence with arbitrary precision using mpmath.

    Parameters
    ----------
    F : np.ndarray
        Frame array (n, d). Rows are normalized internally (in high precision).
    dps : int, default 120
        Decimal digits of precision to use in mpmath.
    top_pairs : list[tuple[int, int]] | None
        If provided, only these index pairs (i, j) are evaluated; otherwise a full
        O(n^2 d) search is performed.

    Returns
    -------
    float
        Maximum |<f_i, f_j>| over the evaluated pairs.
    """
    if mp is None:
        raise RuntimeError("mpmath not installed; pip install mpmath")

    mp.mp.dps = dps
    # normalize rows
    Fm = [[mp.mpc(z.real, z.imag) for z in row] for row in F]
    for r, _ in enumerate(Fm):
        nr = mp.sqrt(sum((mp.re(z) ** 2 + mp.im(z) ** 2) for z in Fm[r]))
        if nr != 0:
            Fm[r] = [z / nr for z in Fm[r]]

    if top_pairs is not None and len(top_pairs) > 0:
        mu = mp.mpf("0")
        for i, j in top_pairs:
            s = mp.mpc("0")
            for k in range(len(Fm[i])):
                s += mp.conj(Fm[i][k]) * Fm[j][k]
            val = mp.sqrt((mp.re(s)) ** 2 + (mp.im(s)) ** 2)
            if val > mu:
                mu = val
        return float(mu)

    # full search
    n = len(Fm)
    mu = mp.mpf("0")
    for i in range(n):
        for j in range(i + 1, n):
            s = mp.mpc("0")
            for k in range(len(Fm[i])):
                s += mp.conj(Fm[i][k]) * Fm[j][k]
            val = mp.sqrt((mp.re(s)) ** 2 + (mp.im(s)) ** 2)
            if val > mu:
                mu = val
    return float(mu)


def summarize_offdiag(off: np.ndarray, mu: float) -> dict[str, Any]:
    """
    Summarize the distribution of off‑diagonal absolute Gram entries.

    Returns a dict with count, mean, std, min/median/quantiles, max, and a simple
    `degeneracy_at_max` count using atol=1e‑12.
    """
    q = {p: float(np.quantile(off, p)) for p in (0.5, 0.9, 0.95, 0.99)}
    mu_rd = np.round(mu, 8)
    off_rd = np.round(off, 8)
    degeneracy = int(np.count_nonzero(off_rd == mu_rd))
    distinct_overlaps = int(np.unique(off_rd).size)
    return {
        "count": int(off.size),
        "mean": float(off.mean()),
        "std": float(off.std()),
        "min": float(off.min()),
        "median": q[0.5],
        "q90": q[0.9],
        "q95": q[0.95],
        "q99": q[0.99],
        "max": float(off.max()),
        "degeneracy_at_max_8dp": degeneracy,
        "distinct_overlaps_8dp": distinct_overlaps,
    }


def topk_pairs(
    off: np.ndarray, iu: tuple[np.ndarray, np.ndarray], k: int
) -> list[tuple[float, tuple[int, int]]]:
    """
    Return the top‑k pairs (value, (i, j)) from the vectorized upper‑tri off‑diagonal
    entries `off`, sorted in descending order. `iu` are the index arrays from
    `np.triu_indices` used to map indices back to (i, j) pairs.
    """
    if k <= 0:
        return []
    k = min(k, off.size)
    idx = np.argpartition(off, -k)[-k:]
    # sort descending
    idx = idx[np.argsort(off[idx])[::-1]]
    out: list[tuple[float, tuple[int, int]]] = [
        (float(off[t]), (int(iu[0][t]), int(iu[1][t]))) for t in idx
    ]
    return out


# ---------------------------------- CLI ------------------------------------


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments for the verifier."""
    ap = argparse.ArgumentParser(
        description="Verify frame coherence from .npy with optional high-precision check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=str, help="Path to frame (.npy or .txt)")
    ap.add_argument(
        "--mp-dps", type=int, default=80, help="mpmath precision (0 to skip)"
    )
    ap.add_argument(
        "--mp-topk",
        type=int,
        default=10,
        help="Verify only top-K float64 pairs in mpmath (0 = full)",
    )
    ap.add_argument(
        "--report-k", type=int, default=5, help="Report top-K pairs by magnitude"
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default=None,
        help=(
            "Explicit JSON report path. If this is a directory, the file "
            "<stem>_certificate.json will be created inside it."
        ),
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help=(
            "Write a JSON report next to the input frame as <stem>_certificate.json "
            "unless --json-out is provided."
        ),
    )
    return ap.parse_args()


def main() -> None:
    """CLI entry point: load frame, compute/report coherences, optionally write JSON."""
    args = parse_args()
    path = Path(args.input).resolve()
    F = load_frame(path)

    mu64, argmax_pair, mu2, off, iu = coherence_float64(F)
    stats = summarize_offdiag(off, mu64)
    topk = topk_pairs(off, iu, args.report_k)

    mu_mp = None
    if args.mp_dps > 0:
        if args.mp_topk > 0:
            # Verify only the top-K pairs from float64
            pairs = [p for (_, p) in topk_pairs(off, iu, args.mp_topk)]
            mu_mp = coherence_mpmath(F, dps=args.mp_dps, top_pairs=pairs)
        else:
            # Full high-precision search (O(n^2 d) mp operations)
            mu_mp = coherence_mpmath(F, dps=args.mp_dps, top_pairs=None)

    # ---- Print human report ----
    print(f"file: {path}")
    print(f"sha256: {sha256_file(path)}")
    print(f"shape (d, n): ({F.shape[1]}, {F.shape[0]})")
    print(f"coherence64     : {mu64:.17g} @ pair {argmax_pair}")
    print(f"coherence64_8dp : {mu64:.8f}")
    if mu_mp is not None:
        mp_str = mp.nstr(mu_mp, args.mp_dps)
        print(f"coherence_mp    : {mp_str}  (dps={args.mp_dps})")
        print(f"coherence_mp_8dp: {mu_mp:.8f}")
        delta = abs(mu_mp - mu64)
        print(f"delta(mp-64): {delta:.3e}")
    print("offdiag stats:")
    for k, v in stats.items():
        print(f"    {k:>21s}: {v}")
    if args.report_k > 0:
        print(f"top-{args.report_k} pairs:")
        for val, (i, j) in topk:
            print(f"    ({i:>3d},{j:>3d}) -> {val:.17g}")

    # ---- Optional JSON ----
    json_path: Path | None = None
    if args.json_out is not None:
        candidate = Path(args.json_out)
        if candidate.exists() and candidate.is_dir():
            json_path = candidate / f"{path.stem}_certificate.json"
        else:
            json_path = candidate
    elif args.json:
        json_path = path.with_name(f"{path.stem}_certificate.json")

    if json_path is not None:
        out = {
            "file": str(path),
            "sha256": sha256_file(path),
            "n": int(F.shape[0]),
            "d": int(F.shape[1]),
            "coherence_float64": mu64,
            "coherence_float64_8dp": f"{mu64:.8f}",
            "coherence_mpmath": mu_mp,
            "coherence_mpmath_8dp": (f"{mu_mp:.8f}" if mu_mp is not None else None),
            "second_largest": mu2,
            "argmax_pair": argmax_pair,
            "stats": stats,
            "top_pairs": [{"value": v, "i": i, "j": j} for (v, (i, j)) in topk],
            "mp_dps": args.mp_dps,
            "mp_topk": args.mp_topk,
        }
        json_path.write_text(json.dumps(out, indent=2))
        print(f"[json] wrote {json_path}")


if __name__ == "__main__":
    main()
