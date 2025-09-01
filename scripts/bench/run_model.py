#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from scripts._utils import (
    ensure_dir,
    git_sha,
    rel_name_from_config,
    repo_root_from_here,
    timestamp_utc,
)

# ---------------------------
# CLI
# ---------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-node parallel model runner (multiprocessing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--inputs", required=True, type=Path, help="Inputs YAML with problems & seeds"
    )
    p.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Model YAML: import & init (+ optional scheduler)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for outputs",
    )
    p.add_argument(
        "--workers", type=int, default=None, help="Number of worker processes"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite finished case dirs"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Only enumerate cases, do not run"
    )
    return p.parse_args(argv)


# ---------------------------
# YAML helpers
# ---------------------------


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Top-level YAML at {path} must be a mapping")
    return data


def _parse_seeds(seeds: Any) -> list[int]:
    # Accept: int | list[int] | {count,start,step?}
    if isinstance(seeds, int):
        return [seeds]
    if isinstance(seeds, list | tuple):
        return [int(s) for s in seeds]
    if isinstance(seeds, Mapping):
        count = seeds.get("count", 0)
        start = seeds.get("start", 0)
        step = seeds.get("step", 1)
        return [start + i * step for i in range(count)]
    raise ValueError("Unsupported seeds format")


def _range_or_list(obj: Any) -> list[int]:
    if isinstance(obj, Mapping) and "range" in obj:
        r = obj["range"]
        start, stop = r["start"], r["stop"]
        step = r.get("step", 1)
        stop_incl = stop + (1 if step > 0 else -1)
        return list(range(start, stop_incl, step))
    if isinstance(obj, list | tuple):
        return [int(x) for x in obj]
    raise ValueError("Expected list or {range:{start,stop,step}}")


def enumerate_cases(inputs_yaml: Path) -> list[dict[str, Any]]:
    cfg = _read_yaml(inputs_yaml)
    seeds = _parse_seeds(cfg["seeds"])  # seeds always in inputs

    problems: list[tuple[int, int]] = []
    if "problems" in cfg:
        problems = [(p["d"], p["n"]) for p in cfg["problems"]]
    elif "sweep" in cfg:
        d_list = _range_or_list(cfg["sweep"]["d"])
        n_list = _range_or_list(cfg["sweep"]["n"])
        problems = list(itertools.product(d_list, n_list))
    else:
        raise ValueError("Inputs YAML must define either 'problems' or 'sweep'")

    return [{"d": d, "n": n, "seed": s} for (d, n) in problems for s in seeds]


# ---------------------------
# Dynamic import & model/scheduler
# ---------------------------


def _import(spec: str) -> Any:
    mod_name, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod_name), attr)


def build_model(model_yaml: Path) -> tuple[type, dict[str, Any], str]:
    m = _read_yaml(model_yaml)
    Model = _import(m["import"])  # "pkg.mod:Class"
    kwargs = dict(m.get("init", {}) or {})

    # Optional nested scheduler (supports 'scheduler' or 'p_scheduler')
    sched_cfg = m.get("scheduler") or m.get("p_scheduler")
    if isinstance(sched_cfg, Mapping) and "import" in sched_cfg:
        Sched = _import(sched_cfg["import"])  # "pkg.mod:Class"
        kwargs_name = "scheduler"
        try:
            # Match ctor arg name if it exists
            import inspect

            names = {
                p.name for p in inspect.signature(Model.__init__).parameters.values()
            }
            if "p_scheduler" in names and "scheduler" not in names:
                kwargs_name = "p_scheduler"
        except Exception:
            pass
        kwargs[kwargs_name] = Sched(**(sched_cfg.get("init", {}) or {}))

    label = str(m["import"]).split(":")[-1]
    return Model, kwargs, label


# ---------------------------
# Worker
# ---------------------------


def run_one_case(case: Mapping[str, Any]) -> Mapping[str, Any]:
    d, n, seed = case["d"], case["n"], case["seed"]
    out_dir = Path(case["out_dir"])
    ensure_dir(out_dir)

    # Provenance copies (best-effort)
    try:
        out_dir.joinpath("model_used.yaml").write_text(
            Path(case["model_yaml"]).read_text()
        )
        out_dir.joinpath("inputs_used.yaml").write_text(
            Path(case["inputs_yaml"]).read_text()
        )
    except Exception:
        pass

    t0 = time.perf_counter()
    status = "ok"
    message = ""
    score: float | None = None

    try:
        Model, kwargs, _ = build_model(Path(case["model_yaml"]))
        model = Model(**kwargs)
        if hasattr(model, "run"):
            try:
                res = model.run(d=d, n=n, seed=seed)
            except TypeError:
                res = model.run(d=d, n=n)
        else:
            try:
                res = model(d=d, n=n, seed=seed)
            except TypeError:
                res = model(d=d, n=n)
        if isinstance(res, Mapping):
            for k in ("score", "energy_final", "energy", "coherence", "loss"):
                v = res.get(k)
                if isinstance(v, int | float):
                    score = float(v)
                    break
        else:
            # if model returns a scalar
            if isinstance(res, int | float):
                score = float(res)
    except Exception as e:
        status = "error"
        message = f"{type(e).__name__}: {e}"
        out_dir.joinpath("ERROR.txt").write_text(traceback.format_exc())

    elapsed = time.perf_counter() - t0

    row = {
        "stamp": case["stamp"],
        "model": case["model_label"],
        "inputs_yaml": Path(case["inputs_yaml"]).name,
        "model_yaml": Path(case["model_yaml"]).name,
        "d": d,
        "n": n,
        "seed": seed,
        "status": status,
        "elapsed_s": f"{elapsed:.6f}",
        "score": "" if score is None else f"{score:.16g}",
        "git_sha": git_sha(repo_root_from_here()),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "out_dir": str(out_dir),
        "message": message,
    }
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    if status == "ok":
        out_dir.joinpath("DONE.ok").write_text("ok\n")
    return row


# ---------------------------
# Collation
# ---------------------------


def collate_results(exp_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for p in exp_root.rglob("metrics.csv"):
        with p.open("r", newline="") as f:
            r = list(csv.DictReader(f))
            if r:
                rows.append(r[0])
    if not rows:
        return exp_root / "results.csv"

    rows.sort(key=lambda r: (int(r["d"]), int(r["n"]), int(r["seed"])))
    fieldnames = list(rows[0].keys())
    out_csv = exp_root / "results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return out_csv


# ---------------------------
# Worker count & BLAS clamp
# ---------------------------


def _detect_workers(user_workers: int | None, ncases: int) -> int:
    if user_workers and user_workers > 0:
        return min(user_workers, ncases)
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(k)
        if v:
            try:
                return max(1, min(int(v), ncases))
            except ValueError:
                pass
    return max(1, min(os.cpu_count() or 1, ncases))


def _clamp_blas() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ---------------------------
# Main
# ---------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    _clamp_blas()

    cases = enumerate_cases(args.inputs)
    model_info = _read_yaml(args.model)
    label = str(model_info.get("import", rel_name_from_config(args.model))).split(":")[
        -1
    ]

    stamp = timestamp_utc()
    exp_root = (args.output_root / stamp / label).resolve()
    ensure_dir(exp_root)

    payloads: list[dict[str, Any]] = []
    for c in cases:
        out_dir = exp_root / f"d{c['d']}_n{c['n']}" / f"seed_{c['seed']}"
        ensure_dir(out_dir)
        payloads.append(
            {
                **c,
                "model_yaml": str(args.model),
                "inputs_yaml": str(args.inputs),
                "out_dir": str(out_dir),
                "model_label": label,
                "stamp": stamp,
            }
        )

    if args.dry_run:
        print(f"[dry-run] {len(payloads)} cases")
        return 0

    to_run = [
        p
        for p in payloads
        if args.overwrite or not Path(p["out_dir"]).joinpath("DONE.ok").exists()
    ]
    if not to_run:
        out_csv = collate_results(exp_root)
        print(f"Nothing to do. Results at: {out_csv}")
        return 0

    workers = _detect_workers(args.workers, len(to_run))
    print(f"[runner] {len(to_run)} / {len(payloads)} cases; workers={workers}")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one_case, p): p for p in to_run}
        try:
            for fut in as_completed(futs):
                fut.result()  # exceptions will surface here and still continue
        except KeyboardInterrupt:
            print("\nInterrupted. Finishing running tasks...")

    out_csv = collate_results(exp_root)
    print(f"[runner] Done → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
