#!/usr/bin/env python3
"""
run_model.py — Load an inputs suite and a model preset, run all cases, and log results.

Inputs:
  - --inputs configs/inputs/<name>.yaml
  - --model-config configs/models/<family>/<preset>.yaml

Design:
  - Inputs own geometry & seeds (list or range).
  - Model config owns algorithm & hyperparameters (incl. p or schedules).
  - Model implements its own training objective; the runner only measures wall-time and logs results.

Output:
  results/runs/<inputs_name>/<model_name>/<timestamp>/
    - results.csv
    - inputs_used.yaml
    - model_config_used.yaml
"""

from __future__ import annotations

import argparse
import csv
import importlib
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from models.api import Problem

# --------------------------- helpers ---------------------------


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def timestamp_utc() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rel_name_from_config(path: Path, anchor: str) -> str:
    """
    Derive a stable name from a config file path relative to a known anchor directory.
    Example:
      path = /repo/configs/models/projection/quick.yaml, anchor="configs"
      -> "models/projection/quick"
    """
    try:
        idx = path.parts.index(anchor)
    except ValueError:
        # Fallback: use stem without extension
        return path.with_suffix("").name
    rel = Path(*path.parts[idx + 1 :]).with_suffix("")
    # Normalize to POSIX-like with forward slashes
    return str(rel).replace("\\", "/")


# --------------------------- loaders ---------------------------


def load_inputs(
    inputs_path: Path,
) -> tuple[str, list[tuple[int, int]], list[int], dict[str, Any]]:
    """
    Returns:
      inputs_name: derived name from path
      pairs: list of (n, d)
      seeds: list of seeds
      raw_dict: the loaded YAML for provenance
    """
    data = yaml.safe_load(inputs_path.read_text())
    inputs_name = rel_name_from_config(inputs_path, anchor="configs")
    if inputs_name.startswith("inputs/"):
        inputs_name = inputs_name.split("/", 1)[1]

    # Seeds: int, dict(count/start), or dict(list)
    seeds_cfg = data.get("seeds", 0)
    seeds: list[int]
    if isinstance(seeds_cfg, int):
        seeds = [seeds_cfg]
    elif isinstance(seeds_cfg, dict):
        if "list" in seeds_cfg:
            seeds = seeds_cfg["list"]
        else:
            count = seeds_cfg["count"]
            start = seeds_cfg.get("start", 0)
            seeds = [start + i for i in range(count)]
    else:
        raise ValueError("Invalid 'seeds' format in inputs YAML")

    # Problems: for this first pass, support explicit list only
    problems = data.get("problems")
    if not isinstance(problems, list):
        raise ValueError(
            "This runner expects an explicit 'problems:' list in inputs YAML."
        )
    pairs: list[tuple[int, int]] = []
    for item in problems:
        try:
            d = item["d"]
            n = item["n"]
        except Exception as e:
            raise ValueError(
                f"Invalid problem entry (expected mapping with 'n' and 'd'): {item}"
            ) from e
        pairs.append((n, d))

    return inputs_name, pairs, seeds, data


def load_model_class_and_kwargs(
    model_cfg_path: Path,
) -> tuple[str, type, dict[str, Any], dict[str, Any]]:
    """
    Returns:
      model_name: derived from path (configs/models/...)
      model_cls: class object to instantiate
      base_kwargs: kwargs dict from YAML 'init'
      raw_dict: loaded YAML for provenance
    """
    data = yaml.safe_load(model_cfg_path.read_text())
    model_name = rel_name_from_config(model_cfg_path, anchor="configs")
    if model_name.startswith("models/"):
        model_name = model_name.split("/", 1)[1]

    import_path = data.get("import")
    if not isinstance(import_path, str) or ":" not in import_path:
        raise ValueError("Model config must include 'import: module.path:ClassName'")

    mod_name, _, cls_name = import_path.partition(":")
    mod = importlib.import_module(mod_name)
    model_cls = getattr(mod, cls_name)
    base_kwargs: dict[str, Any] = data.get("init", {})

    # If 'p' is present and is a string, mark for evaluation later
    if "p" in base_kwargs and isinstance(base_kwargs["p"], str):
        base_kwargs["p"] = {"__expr__": base_kwargs["p"]}

    return model_name, model_cls, base_kwargs, data


# --------------------------- main ---------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a model preset over an inputs suite.")
    ap.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="Path to inputs YAML (configs/inputs/*.yaml)",
    )
    ap.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to model preset YAML (configs/models/*/*.yaml)",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("results/runs"), help="Output root directory"
    )
    args = ap.parse_args()

    repo_root = repo_root_from_here()
    sha = git_sha(repo_root)
    ts = timestamp_utc()

    inputs_name, pairs, seeds, inputs_dict = load_inputs(args.inputs)
    model_name, model_cls, base_kwargs, model_cfg_dict = load_model_class_and_kwargs(
        args.model_config
    )

    # Output directory: results/runs/<inputs_name>/<model_name>/<timestamp>/
    out_dir = args.out / inputs_name / model_name / ts
    ensure_dir(out_dir)

    # Save the original configs
    (out_dir / "inputs_used.yaml").write_text(args.inputs.read_text())
    (out_dir / "model_config_used.yaml").write_text(args.model_config.read_text())

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "git_sha",
                "inputs",
                "model",
                "d",
                "n",
                "n_seeds",
                "coh_min",
                "coh_median",
                "coh_mean",
                "coh_max",
                "wall_time_s_mean",
            ],
        )
        writer.writeheader()

        for n, d in pairs:
            init_kwargs_base = dict(base_kwargs)
            if "p" in init_kwargs_base and isinstance(init_kwargs_base["p"], dict):
                expr = init_kwargs_base["p"].get("__expr__")
                if expr is not None:
                    init_kwargs_base["p"] = eval(expr, {}, {"n": n, "d": d})

            cohs: list[float] = []
            times: list[float] = []

            for seed in seeds:
                init_kwargs = init_kwargs_base | {"seed": seed}
                model = model_cls(**init_kwargs)

                result = model.run(Problem(n=n, d=d))

                cohs.append(result.best_coherence)
                times.append(result.wall_time_s)

            n_seeds = len(cohs)
            coh_min = min(cohs)
            coh_median = statistics.median(cohs)
            coh_mean = statistics.fmean(cohs)
            coh_max = max(cohs)
            wall_time_s_mean = statistics.fmean(times)

            writer.writerow(
                {
                    "timestamp": ts,
                    "git_sha": sha,
                    "inputs": inputs_name,
                    "model": model_name,
                    "d": d,
                    "n": n,
                    "n_seeds": n_seeds,
                    "coh_min": coh_min,
                    "coh_median": coh_median,
                    "coh_mean": coh_mean,
                    "coh_max": coh_max,
                    "wall_time_s_mean": wall_time_s_mean,
                }
            )
            f.flush()

    print(f"✅ wrote {csv_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
