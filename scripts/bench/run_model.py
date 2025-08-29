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
from pathlib import Path
from typing import Any, cast

import yaml

from models.api import Problem
from scripts._utils import (
    ensure_dir,
    git_sha,
    rel_name_from_config,
    repo_root_from_here,
    timestamp_utc,
)

# --------------------------- helpers ---------------------------


def _eval_expr(x: Any, env: dict[str, int]) -> int:
    """Evaluate an int or simple expression string like '2*d' with variables from env."""
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        return int(eval(x, {}, env))
    raise TypeError(f"Unsupported expression type: {type(x)!r} (value={x!r})")


def _expand_axis(axis_cfg: dict[str, Any], env: dict[str, int]) -> list[int]:
    """
    Expand an axis configuration into a list of ints.

    Supported forms:
      - {'values': [2, 3, '2*d', ...]}
      - {'range': {'start': 2, 'stop': 7, 'step': 1}}  # 'start'/'stop'/'step' may be expressions.
    The 'stop' is treated as inclusive.
    """
    if "values" in axis_cfg:
        vals = [_eval_expr(v, env) for v in axis_cfg["values"]]
        return vals
    if "range" in axis_cfg:
        r = axis_cfg["range"]
        start = _eval_expr(r["start"], env)
        stop = _eval_expr(r["stop"], env)
        step = _eval_expr(r.get("step", 1), env)
        if step == 0:
            raise ValueError("range.step must be nonzero")
        # inclusive stop
        stop_inclusive = stop + (1 if step > 0 else -1)
        return list(range(start, stop_inclusive, step))
    raise ValueError("Axis config must contain either 'values' or 'range'")


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

    pairs: list[tuple[int, int]] = []

    if "problems" in data:
        problems = data["problems"]
        if not isinstance(problems, list):
            raise ValueError("Invalid 'problems' format: expected a list.")
        for item in problems:
            try:
                d = item["d"]
                n = item["n"]
            except Exception as e:
                raise ValueError(
                    f"Invalid problem entry (expected mapping with 'n' and 'd'): {item}"
                ) from e
            pairs.append((n, d))
    elif "sweep" in data:
        sweep = data["sweep"]
        if not isinstance(sweep, dict):
            raise ValueError("Invalid 'sweep' format: expected a mapping.")
        if "d" not in sweep or "n" not in sweep:
            raise ValueError("Sweep must define both 'd' and 'n' axes.")
        d_cfg = cast(dict[str, Any], sweep["d"])
        n_cfg = cast(dict[str, Any], sweep["n"])
        d_values = _expand_axis(d_cfg, {})
        pairs.extend((n, d) for d in d_values for n in _expand_axis(n_cfg, {"d": d}))
    else:
        raise ValueError("Inputs YAML must contain either 'problems' or 'sweep'.")

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
        "--out",
        type=Path,
        default=Path("results/model-runs"),
        help="Output root directory",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Enable verbose/debug logging"
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
            if args.verbose:
                print(
                    f"Running model {model_name} on inputs {inputs_name} with n={n}, d={d}"
                )
            init_kwargs_base = dict(base_kwargs)
            if "p" in init_kwargs_base and isinstance(init_kwargs_base["p"], dict):
                expr = init_kwargs_base["p"].get("__expr__")
                if expr is not None:
                    init_kwargs_base["p"] = eval(expr, {}, {"n": n, "d": d})

            cohs: list[float] = []
            times: list[float] = []

            for seed in seeds:
                if args.verbose:
                    print(f"  Seed {seed}...")
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
