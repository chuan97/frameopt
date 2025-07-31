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
from typing import Any, Dict, List

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
        )


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
        # List tracked changes only (ignore untracked). Non-empty means dirty.
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
        )


# ------------------------------ config model ------------------------------ #


@dataclass
class ExperimentConfig:
    experiment: str
    problem: Dict[str, Any]  # {"n": int, "d": int}
    cma: Dict[str, Any]  # {"gen": int, "sigma0": float, "popsize": int, ...}
    scheduler: Dict[str, Any]  # {"mode": "fixed"|"adaptive", ... params ...}
    seeds: Dict[str, Any]  # {"list":[...]} OR {"count":int, "start":int}
    logging: Dict[str, Any] | None = None  # {"save_metrics": bool}
    exports: Dict[str, Any] | None = None  # {"save_npy": bool, "save_txt": bool}
    certify: Dict[str, Any] | None = None  # {"enabled": bool, "mp_dps": int, "mp_topk": int}

    @staticmethod
    def load(path: Path) -> "ExperimentConfig":
        data = json.loads(path.read_text())
        for k in ("experiment", "problem", "cma", "scheduler", "seeds"):
            if k not in data:
                raise ValueError(f"Missing required key '{k}' in {path}")
        return ExperimentConfig(**data)


# ------------------------------ driver core -------------------------------- #


def _resolve_cli_path(this_file: Path) -> Path:
    # this file: repo/scripts/experiments/cma_ramp_driver.py
    # cli file : repo/scripts/cli/run_cma_ramp.py
    repo_root = this_file.parents[2]
    cli = repo_root / "scripts" / "cli" / "run_cma_ramp.py"
    if not cli.exists():
        raise FileNotFoundError(f"Could not find run_cma_ramp.py at {cli}")
    return cli


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
    Look for a JSON config next to the driver:
        <stem>_driver.py  ->  <stem>_config.json

    Examples:
      cma_ramp_driver.py -> cma_ramp_config.json
      foo_driver.py      -> foo_config.json
    """
    base = this_file.stem  # e.g., "cma_ramp_driver"
    if base.endswith("_driver"):
        base = base[: -len("_driver")]
    cfg_name = f"{base}_config.json"
    cfg_path = this_file.with_name(cfg_name)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path}\n"
            "Place a JSON config alongside the driver with the same base name, "
            "ending in _config.json."
        )
    return cfg_path


def _seeds_from_cfg(seeds_cfg: Dict[str, Any]) -> List[int]:
    if "list" in seeds_cfg and isinstance(seeds_cfg["list"], list):
        return [int(s) for s in seeds_cfg["list"]]
    count = int(seeds_cfg.get("count", 1))
    start = int(seeds_cfg.get("start", 0))
    return [start + i for i in range(count)]


def _build_cmd(
    py: str,
    cli_path: Path,
    cfg: ExperimentConfig,
    seed: int,
    out_dir: Path,
) -> list[str]:
    n = int(cfg.problem["n"])
    d = int(cfg.problem["d"])

    gen = int(cfg.cma.get("gen", 0))
    sigma0 = float(cfg.cma.get("sigma0", 0.3))
    popsize = int(cfg.cma.get("popsize", 40))

    sched = cfg.scheduler
    mode = str(sched.get("mode", "adaptive"))
    p0 = float(sched.get("p0", 2.0))
    p_max = float(sched.get("p_max", 1e9))

    cmd = [
        py,
        str(cli_path),
        "-n",
        str(n),
        "-d",
        str(d),
        "--gen",
        str(gen),
        "--sigma0",
        str(sigma0),
        "--popsize",
        str(popsize),
        "--scheduler",
        mode,
        "--p0",
        str(p0),
        "--p-max",
        str(p_max),
        "--seed",
        str(seed),
    ]

    # scheduler-specific fields
    if mode == "fixed":
        if "p_mult" in sched:
            cmd += ["--p-mult", str(float(sched["p_mult"]))]
        if "switch_every" in sched:
            cmd += ["--switch-every", str(int(sched["switch_every"]))]
    elif mode == "adaptive":
        if "window" not in sched:
            raise ValueError(
                "scheduler.mode='adaptive' requires 'window' in scheduler config"
            )
        cmd += ["--window", str(int(sched["window"]))]
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")

    # outputs from CLI
    save_metrics = bool((cfg.logging or {}).get("save_metrics", True))
    save_npy = bool((cfg.exports or {}).get("save_npy", True))
    save_txt = bool((cfg.exports or {}).get("save_txt", False))

    if save_metrics:
        cmd += ["--log-file", str(out_dir / "cma_metrics.csv")]
    if save_npy:
        cmd += ["--export-npy", str(out_dir / "best_frame.npy")]
    if save_txt:
        cmd += ["--export-txt", str(out_dir / "best_frame.txt")]

    return cmd


def main() -> None:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]
    cli_path = _resolve_cli_path(this_file)
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
    augmented = {
        **json.loads(cfg_path.read_text()),
        "_meta": {
            "git_sha": git_sha,
            "timestamp_utc": ts,
            "driver": str(this_file.relative_to(repo_root)),
            "cli": str(cli_path.relative_to(repo_root)),
        },
    }
    (out_root / "config_used.json").write_text(json.dumps(augmented, indent=2))

    # Seeds
    seeds = _seeds_from_cfg(cfg.seeds)
    best_seed: int | None = None
    best_coh_num: float | None = None
    best_coh_8dp: str | None = None
    print(f"[exp] {cfg.experiment} | seeds={seeds}")

    # Summary CSV
    summary_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        run_dir = out_root / f"seed_{seed:04d}"
        _ensure_dir(run_dir)

        cmd = _build_cmd(sys.executable, cli_path, cfg, seed=seed, out_dir=run_dir)
        print(f"[run] seed={seed} → {run_dir}")
        print(f"[cmd] {' '.join(cmd)}")

        # Log file (stdout+stderr)
        log_path = run_dir / "run.log"
        t0 = time.time()
        with log_path.open("w") as logf:
            rc = subprocess.call(
                cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(repo_root)
            )
        dt = time.time() - t0

        # Per-seed meta
        (run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "return_code": rc,
                    "runtime_sec": round(dt, 3),
                    "cmd": cmd,
                },
                indent=2,
            )
        )

        # Optional certification step: run verify_frame.py to produce JSON certificate next to the frame
        cert_json_rel = ""
        coh8 = ""
        if cfg.certify and bool(cfg.certify.get("enabled", False)):
            npy_path = run_dir / "best_frame.npy"
            if npy_path.exists():
                # Console notice for verifier
                print(f"[ver] seed={seed} → verify {npy_path.name}")
                # Build verifier command
                vcmd = [sys.executable, str(verify_cli), str(npy_path), "--json"]
                mp_dps = cfg.certify.get("mp_dps")
                mp_topk = cfg.certify.get("mp_topk")
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
                    vcmd += ["--mp-dps", str(int(mp_dps))]
                    if isinstance(mp_topk, int) and mp_topk >= 0:
                        vcmd += ["--mp-topk", str(int(mp_topk))]

                # Echo verifier command to console for visibility
                print(f"[cmd-ver] {' '.join(vcmd)}")
                # Append verifier output to the same log and run
                with log_path.open("a") as logf:
                    logf.write("\n[verify] running verify_frame.py\n")
                    logf.write(f"[verify] cmd: {' '.join(vcmd)}\n")
                    rc_v = subprocess.call(vcmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(repo_root))
                    logf.write(f"[verify] return_code={rc_v}\n")
                if rc_v != 0:
                    print(f"[ver] seed={seed} | verifier exited with rc={rc_v}")
                    # If we asked for mpmath, try once more without high-precision flags
                    if have_mp and "--mp-dps" in vcmd:
                        vcmd_fallback = [sys.executable, str(verify_cli), str(npy_path), "--json"]
                        with log_path.open("a") as logf:
                            logf.write("[verify] retry without mpmath flags\n")
                            logf.write(f"[verify] cmd: {' '.join(vcmd_fallback)}\n")
                            rc_v2 = subprocess.call(vcmd_fallback, stdout=logf, stderr=subprocess.STDOUT, cwd=str(repo_root))
                            logf.write(f"[verify] return_code={rc_v2}\n")
                        if rc_v2 == 0:
                            print(f"[ver] seed={seed} | fallback float64 verification succeeded")
                            rc_v = 0
                        else:
                            print(f"[ver] seed={seed} | fallback verifier also failed (rc={rc_v2})")

                # Read certificate JSON if created
                cert_path = run_dir / f"{npy_path.stem}_certificate.json"
                if cert_path.exists():
                    cert_json_rel = str(cert_path.relative_to(out_root))
                    try:
                        cert = json.loads(cert_path.read_text())
                        # prefer high-precision coherence if available
                        coh_num = cert.get("coherence_mpmath", None)
                        if coh_num is None:
                            coh_num = cert.get("coherence_float64", None)
                        coh8 = cert.get("coherence_mpmath_8dp", None) or cert.get("coherence_float64_8dp", "")
                        # track best across seeds
                        if isinstance(coh_num, (int, float)):
                            if best_coh_num is None or float(coh_num) < best_coh_num:
                                best_coh_num = float(coh_num)
                                best_seed = seed
                                best_coh_8dp = coh8 if isinstance(coh8, str) else None
                        if coh8:
                            print(f"[ver] seed={seed} | coherence_8dp={coh8}")
                    except Exception as e:
                        print(f"[ver] seed={seed} | failed to parse certificate JSON: {e}")
                else:
                    print(f"[ver] seed={seed} | certificate not found (expected {cert_path.name})")
            else:
                print(f"[ver] seed={seed} | best_frame.npy not found; skipping verification")

        summary_rows.append(
            {
                "seed": seed,
                "return_code": rc,
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
                    str((run_dir / "best_frame.txt").relative_to(out_root))
                    if (run_dir / "best_frame.txt").exists()
                    else ""
                ),
                "log_file": str((run_dir / "run.log").relative_to(out_root)),
            }
        )
        status = "OK" if rc == 0 else f"rc={rc}"
        print(f"[done] seed={seed} | {status} | {dt:.2f}s")

    # Write summary
    if summary_rows:
        with (out_root / "summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[summary] {out_root / 'summary.csv'}")

    if best_seed is not None and best_coh_8dp:
        print(f"[best] seed={best_seed:04d}  coherence_8dp={best_coh_8dp}")

    print(f"[out] {out_root}")


if __name__ == "__main__":
    main()
