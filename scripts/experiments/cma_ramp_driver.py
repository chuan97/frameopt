#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evomof.core.energy import coherence, diff_coherence
from evomof.core.frame import Frame
from evomof.optim.cma import ProjectionCMA, RiemannianCMA
from evomof.optim.utils.p_scheduler import (
    AdaptivePScheduler,
    FixedPScheduler,
    Scheduler,
)

# ------------------------------- helpers ---------------------------------- #


def _timestamp_utc() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception as e:
        raise RuntimeError(
            f"Unable to read git commit hash in repo root {repo_root}.\n"
            f"Are you running inside a git checkout? Original error: {e}"
        ) from e


def _assert_git_clean(repo_root: Path) -> None:
    """
    Raise if there are uncommitted *tracked* changes (staged or unstaged).

    We ignore untracked files to avoid false positives from scratch artifacts.
    """
    try:
        # Quick sanity: ensure we're in a repo
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"], cwd=str(repo_root), text=True
        )
        # list tracked changes only (ignore untracked). Non-empty means dirty.
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root),
            text=True,
        )
        if status.strip():
            raise RuntimeError(
                "Repository has uncommitted tracked changes. "
                "For certifiable experiments, commit or stash changes and retry."
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Git check failed. Are you inside a git repo at {repo_root}? {e}"
        ) from e


# ------------------------------ config model ------------------------------ #


@dataclass
class ExperimentConfig:
    experiment: str
    problem: dict[str, Any]  # {"n": int, "d": int}
    cma: dict[
        str, Any
    ]  # {"gen": int, "sigma0": float, "popsize": int, "algo": "projection"|"riemannian"}
    scheduler: dict[str, Any]  # {"mode": "fixed"|"adaptive", ... params ...}
    seeds: dict[str, Any]  # {"list":[...]} OR {"count":int, "start":int}
    logging: dict[str, Any] | None = None  # {"save_metrics": bool}
    exports: dict[str, Any] | None = None  # {"save_npy": bool, "save_txt": bool}
    certify: dict[str, Any] | None = (
        None  # {"enabled": bool, "mp_dps": int, "mp_topk": int}
    )

    @staticmethod
    def load(path: Path) -> ExperimentConfig:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} did not parse to a mapping.")
        for k in ("experiment", "problem", "cma", "scheduler", "seeds"):
            if k not in data:
                raise ValueError(f"Missing required key '{k}' in {path}")
        return ExperimentConfig(**data)


# ------------------------------ driver core -------------------------------- #


def _resolve_verify_cli(this_file: Path) -> Path:
    # this file: repo/scripts/bench/cma_ramp_driver.py
    # verifier  : repo/scripts/cli/verify_frame.py
    repo_root = this_file.parents[2]
    vf = repo_root / "scripts" / "cli" / "verify_frame.py"
    if not vf.exists():
        raise FileNotFoundError(f"Could not find verify_frame.py at {vf}")
    return vf


def _discover_config(this_file: Path) -> Path:
    """
    Look for a YAML config next to the driver:
        <stem>_driver.py  ->  <stem>_config.yaml

    Examples:
      cma_ramp_driver.py -> cma_ramp_config.yaml
      foo_driver.py      -> foo_config.yaml
    """
    base = this_file.stem  # e.g., "cma_ramp_driver"
    if base.endswith("_driver"):
        base = base[: -len("_driver")]
    cfg_name = f"{base}_config.yaml"
    cfg_path = this_file.with_name(cfg_name)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path}\n"
            "Place a YAML config alongside the driver with the same base name, "
            "ending in _config.yaml."
        )
    return cfg_path


def _seeds_from_cfg(seeds_cfg: dict[str, Any]) -> list[int]:
    if "list" in seeds_cfg and isinstance(seeds_cfg["list"], list):
        return [int(s) for s in seeds_cfg["list"]]
    count = int(seeds_cfg.get("count", 1))
    start = int(seeds_cfg.get("start", 0))
    return [start + i for i in range(count)]


# --------------------------- in-process CMA runner --------------------------- #


def run_cma_ramp(
    *,
    n: int,
    d: int,
    gen: int,
    sigma0: float,
    popsize: int | None,
    algo: str,
    scheduler_cfg: dict[str, Any],
    seed: int | None,
) -> tuple[Frame, list[dict], float]:
    rng = np.random.default_rng(seed)

    # Resolve population size default
    if popsize in (None, 0):
        dim = 2 * n * d
        popsize_eff = 4 + int(3 * np.log(dim))
        print(f"[popsize] Using default λ={popsize_eff} for dim={dim}")
    else:
        popsize_eff = int(popsize)

    # Build p-scheduler (Fixed or Adaptive)
    mode = str(scheduler_cfg.get("mode", "adaptive"))
    p0 = float(scheduler_cfg.get("p0", 2.0))
    p_max = float(scheduler_cfg.get("p_max", 1e9))
    if mode == "fixed":
        switch_every = scheduler_cfg.get("switch_every", 200)
        switch_every = int(switch_every) if switch_every and switch_every > 0 else None
        p_mult = float(scheduler_cfg.get("p_mult", 1.5))
        sched: Scheduler = FixedPScheduler(
            p0=p0, p_mult=p_mult, p_max=p_max, switch_every=switch_every
        )
        print(
            f"[scheduler] fixed: p0={p0}, p_mult={p_mult}, switch_every={switch_every}, p_max={p_max}"
        )
    elif mode == "adaptive":
        window = int(scheduler_cfg.get("window", 100))
        sched = AdaptivePScheduler(p0=p0, p_max=p_max, total_steps=gen, window=window)
        print(
            f"[scheduler] adaptive: p0={p0}, window={window}, p_max={p_max}, total_steps={gen}"
        )
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")

    p_exp = sched.current_p()

    # Instantiate CMA variant
    if algo == "projection":
        cma = ProjectionCMA(
            n=n,
            d=d,
            sigma0=sigma0,
            popsize=popsize_eff,
            energy_fn=diff_coherence,
            energy_kwargs={"p": p_exp},
            seed=seed,
        )
        print("[algo] projection CMA")
    elif algo == "riemannian":
        cma = RiemannianCMA(
            n=n,
            d=d,
            sigma0=sigma0,
            popsize=popsize_eff,
            energy_fn=diff_coherence,
            energy_kwargs={"p": p_exp},
            seed=seed,
        )
        print("[algo] riemannian CMA")
    else:
        raise ValueError(
            f"Unknown algo '{algo}'. Expected 'projection' or 'riemannian'."
        )

    # Initialize bests
    best_frame = Frame.random(n, d, rng=rng)
    best_energy = diff_coherence(best_frame, p=p_exp)
    global_best_frame = best_frame
    global_best_coh = float(coherence(best_frame))

    metrics: list[dict] = []
    t0 = time.perf_counter()

    for g in range(1, gen + 1):
        population = cma.ask()
        energies = [diff_coherence(F, p=p_exp) for F in population]

        idx = int(np.argmin(energies))
        gen_best_energy = float(energies[idx])
        gen_best_frame = population[idx]
        gen_best_coh = float(coherence(gen_best_frame))

        if gen_best_energy < best_energy:
            best_energy = gen_best_energy
            best_frame = gen_best_frame

        if gen_best_coh < global_best_coh:
            global_best_coh = gen_best_coh
            global_best_frame = gen_best_frame

        cma.tell(population, energies)

        elapsed = time.perf_counter() - t0
        sigma_val = getattr(cma, "sigma", None)
        mean_vec = getattr(cma, "mean", None)
        sigma_f = float(sigma_val) if sigma_val is not None else float("nan")
        mean_norm = (
            float(np.linalg.norm(mean_vec)) if mean_vec is not None else float("nan")
        )

        metrics.append(
            {
                "gen": g,
                "elapsed_time": elapsed,
                "p": p_exp,
                "gen_best_diff_coh": gen_best_energy,
                "gen_best_coh": gen_best_coh,
                "best_diff_coh": best_energy,
                "best_coh": global_best_coh,
                "sigma": sigma_f,
                "mean_norm": mean_norm,
                "algo": algo,
            }
        )

        p_next, switched = sched.update(step=g, global_best_coh=global_best_coh)
        if switched:
            print(f"[p-ramp] Generation {g}: p {p_exp:g} -> {p_next:g}")
            best_energy = float(diff_coherence(best_frame, p=p_next))
        p_exp = p_next

        if g % max(gen // 10, 1) == 0:
            print(
                f"Gen {g:5d} | p={p_exp:g} | best diff-coh={best_energy:.8f} | coherence={global_best_coh:.8f}"
            )

    runtime = time.perf_counter() - t0
    print(
        f"Finished {gen} generations in {runtime:.2f}s | final p={p_exp:g} | "
        f"best diff-coh (current p) {best_energy:.6e} | global best coherence {global_best_coh:.10f}"
    )

    return global_best_frame, metrics, runtime


def _run_verifier(
    *,
    npy_path: Path,
    verify_cli: Path,
    repo_root: Path,
    log_path: Path,
    seed: int,
    mp_dps: int | None,
    mp_topk: int | None,
) -> tuple[Path | None, float | None, str]:
    """
    Run verify_frame.py on `npy_path`, write JSON next to it, and return:
      (certificate_path_or_None, numeric_best_coh_or_None, best_coh_8dp_string_or_empty)

    Prints concise [ver]/[cmd-ver] messages to console and appends detailed logs to `run.log`.
    Falls back to float64 verification if mpmath is unavailable or the high-precision run fails.
    """
    if not npy_path.exists():
        print(f"[ver] seed={seed} | best_frame.npy not found; skipping verification")
        return None, None, ""
    print(f"[ver] seed={seed} → verify {npy_path.name}")

    # Build verifier command
    vcmd = [sys.executable, str(verify_cli), str(npy_path), "--json"]

    # Determine whether to use high precision
    want_mp = isinstance(mp_dps, int) and mp_dps > 0
    have_mp = False
    if want_mp:
        try:
            importlib.import_module("mpmath")
            have_mp = True
        except Exception:
            have_mp = False
            print(f"[ver] seed={seed} | mpmath not available; will verify in float64")
    if have_mp:
        vcmd += ["--mp-dps", str(int(mp_dps))]  # type: ignore[arg-type]
        if isinstance(mp_topk, int) and mp_topk >= 0:
            vcmd += ["--mp-topk", str(int(mp_topk))]

    # Echo and run
    print(f"[cmd-ver] {' '.join(vcmd)}")
    with log_path.open("a") as logf:
        logf.write("\n[verify] running verify_frame.py\n")
        logf.write(f"[verify] cmd: {' '.join(vcmd)}\n")
        rc_v = subprocess.call(
            vcmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(repo_root)
        )
        logf.write(f"[verify] return_code={rc_v}\n")

    # If high-precision attempt failed, retry once without mp flags
    if rc_v != 0 and have_mp and "--mp-dps" in vcmd:
        vcmd_fallback = [sys.executable, str(verify_cli), str(npy_path), "--json"]
        with log_path.open("a") as logf:
            logf.write("[verify] retry without mpmath flags\n")
            logf.write(f"[verify] cmd: {' '.join(vcmd_fallback)}\n")
            rc_v2 = subprocess.call(
                vcmd_fallback, stdout=logf, stderr=subprocess.STDOUT, cwd=str(repo_root)
            )
            logf.write(f"[verify] return_code={rc_v2}\n")
        if rc_v2 == 0:
            print(f"[ver] seed={seed} | fallback float64 verification succeeded")
            rc_v = 0
        else:
            print(f"[ver] seed={seed} | fallback verifier also failed (rc={rc_v2})")

    # Parse certificate JSON
    cert_path = npy_path.with_name(f"{npy_path.stem}_certificate.json")
    if not cert_path.exists():
        print(f"[ver] seed={seed} | certificate not found (expected {cert_path.name})")
        return None, None, ""
    try:
        cert = json.loads(cert_path.read_text())
    except Exception as e:
        print(f"[ver] seed={seed} | failed to parse certificate JSON: {e}")
        return cert_path, None, ""
    # prefer high-precision coherence if available
    coh_num = cert.get("coherence_mpmath", None)
    if coh_num is None:
        coh_num = cert.get("coherence_float64", None)
    coh8 = cert.get("coherence_mpmath_8dp", None) or cert.get(
        "coherence_float64_8dp", ""
    )
    if coh8:
        print(f"[ver] seed={seed} | coherence_8dp={coh8}")
    return (
        cert_path,
        (float(coh_num) if isinstance(coh_num, int | float) else None),
        (coh8 if isinstance(coh8, str) else ""),
    )


def main() -> None:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    verify_cli = _resolve_verify_cli(this_file)
    cfg_path = _discover_config(this_file)

    # Enforce clean repo for certifiable experiments
    _assert_git_clean(repo_root)
    git_sha = _git_sha(repo_root)

    cfg = ExperimentConfig.load(cfg_path)

    # Output root: results/<experiment>_<timestamp>/
    ts = _timestamp_utc()
    out_root = repo_root / "results" / f"{cfg.experiment}_{ts}"
    _ensure_dir(out_root)

    # Save augmented config in output root
    base_cfg = yaml.safe_load(cfg_path.read_text())
    if not isinstance(base_cfg, dict):
        raise ValueError(f"Config file {cfg_path} did not parse to a mapping.")
    augmented = {
        **base_cfg,
        "_meta": {
            "git_sha": git_sha,
            "timestamp_utc": ts,
            "driver": str(this_file.relative_to(repo_root)),
        },
    }
    (out_root / "config_used.yaml").write_text(
        yaml.safe_dump(augmented, sort_keys=False)
    )

    # Seeds
    seeds = _seeds_from_cfg(cfg.seeds)
    best_seed: int | None = None
    best_coh_num: float | None = None
    best_coh_8dp: str | None = None
    print(f"[exp] {cfg.experiment} | seeds={seeds}")

    # Summary CSV
    summary_rows: list[dict[str, Any]] = []

    for seed in seeds:
        run_dir = out_root / f"seed_{seed:04d}"
        _ensure_dir(run_dir)
        print(f"[run] seed={seed} → {run_dir}")

        # Resolve CMA params
        n = int(cfg.problem["n"])
        d = int(cfg.problem["d"])
        gen = int(cfg.cma.get("gen", 0))
        if gen <= 0:
            raise ValueError("Config.cma must include positive 'gen' (generations).")
        sigma0 = float(cfg.cma.get("sigma0", 0.5))
        popsize = cfg.cma.get("popsize", None)
        algo = str(cfg.cma.get("algo", "projection"))

        # Determine logging/export flags
        save_metrics = bool((cfg.logging or {}).get("save_metrics", True))
        save_npy = bool((cfg.exports or {}).get("save_npy", True))
        save_txt = bool((cfg.exports or {}).get("save_txt", False))

        # Log file
        log_path = run_dir / "run.log"
        t0 = time.time()
        # Run CMA-ES in-process
        best_frame, metrics, _runtime = run_cma_ramp(
            n=n,
            d=d,
            gen=gen,
            sigma0=sigma0,
            popsize=(
                int(popsize)
                if popsize not in (None, 0) and popsize is not None
                else None
            ),
            algo=algo,
            scheduler_cfg=cfg.scheduler,
            seed=seed,
        )
        dt = time.time() - t0

        # Write metrics CSV if requested
        if save_metrics and metrics:
            with (run_dir / "cma_metrics.csv").open("w", newline="") as fh:
                w = csv.dictWriter(fh, fieldnames=list(metrics[0].keys()))
                w.writeheader()
                w.writerows(metrics)
        # Write best_frame.npy if requested
        if save_npy:
            best_frame.save_npy(run_dir / "best_frame.npy")
        # Write best_frame.txt if requested
        if save_txt:
            best_frame.export_txt(run_dir / f"{d}x{n}_jrr.txt")
        # Write run.log (summary)
        with log_path.open("w") as logf:
            logf.write(f"Seed: {seed}\n")
            logf.write(f"Runtime: {dt:.3f} s\n")
            logf.write(
                f"n={n}, d={d}, gen={gen}, sigma0={sigma0}, popsize={popsize}, algo={algo}\n"
            )
            if metrics:
                logf.write(f"Final best coherence: {metrics[-1]['best_coh']:.10f}\n")
        # Per-seed meta
        (run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "runtime_sec": round(dt, 3),
                    "gen": gen,
                    "algo": algo,
                    "sigma0": sigma0,
                    "popsize": (
                        int(popsize)
                        if popsize not in (None, 0) and popsize is not None
                        else None
                    ),
                },
                indent=2,
            )
        )

        # Optional certification step
        cert_json_rel = ""
        coh8 = ""
        if cfg.certify and bool(cfg.certify.get("enabled", False)):
            mp_dps = cfg.certify.get("mp_dps")
            mp_topk = cfg.certify.get("mp_topk")
            cert_path, coh_num, coh8 = _run_verifier(
                npy_path=run_dir / "best_frame.npy",
                verify_cli=verify_cli,
                repo_root=repo_root,
                log_path=log_path,
                seed=seed,
                mp_dps=(int(mp_dps) if isinstance(mp_dps, int) else None),
                mp_topk=(int(mp_topk) if isinstance(mp_topk, int) else None),
            )
            if cert_path:
                cert_json_rel = str(cert_path.relative_to(out_root))
            if isinstance(coh_num, int | float):
                if best_coh_num is None or float(coh_num) < best_coh_num:
                    best_coh_num = float(coh_num)
                    best_seed = seed
                    best_coh_8dp = coh8 or best_coh_8dp

        summary_rows.append(
            {
                "seed": seed,
                "return_code": 0,
                "runtime_sec": round(dt, 3),
                "certificate_json": cert_json_rel,
                "best_coherence_8dp": coh8,
                "metrics_csv": (
                    str((run_dir / "cma_metrics.csv").relative_to(out_root))
                    if (run_dir / "cma_metrics.csv").exists()
                    else ""
                ),
                "best_frame_npy": (
                    str((run_dir / "best_frame.npy").relative_to(out_root))
                    if (run_dir / "best_frame.npy").exists()
                    else ""
                ),
                "best_frame_txt": (
                    str(
                        (
                            run_dir / f"{cfg.problem['d']}x{cfg.problem['n']}_jrr.txt"
                        ).relative_to(out_root)
                    )
                    if (
                        run_dir / f"{cfg.problem['d']}x{cfg.problem['n']}_jrr.txt"
                    ).exists()
                    else ""
                ),
                "log_file": str((run_dir / "run.log").relative_to(out_root)),
            }
        )
        print(f"[done] seed={seed} | OK | {dt:.2f}s")

    # Write summary
    if summary_rows:
        with (out_root / "summary.csv").open("w", newline="") as fh:
            w = csv.dictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[summary] {out_root / 'summary.csv'}")

    if best_seed is not None and best_coh_8dp:
        print(f"[best] seed={best_seed:04d}  coherence_8dp={best_coh_8dp}")

    print(f"[out] {out_root}")


if __name__ == "__main__":
    main()
