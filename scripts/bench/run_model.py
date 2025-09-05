#!/usr/bin/env python3
"""
run_model.py — Load an inputs suite and a model preset, run all cases, and log results.

Inputs:
  - --inputs configs/inputs/<name>.yaml
  - --model-config configs/models/<family>/<preset>.yaml

Output:
  results/model-runs/<inputs_name>/<model_name>/<timestamp>/
    - results.csv
    - inputs_used.yaml
    - model_config_used.yaml
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
from typing import Any

import yaml

from models.api import Problem
from scripts._utils import (
    ensure_dir,
    git_sha,
    rel_name_from_config,
    repo_root_from_here,
    timestamp_utc,
)

# --------------------------- loaders ---------------------------


def load_inputs(
    inputs_cfg_path: Path,
) -> tuple[str, list[Problem]]:
    """
    Returns:
      inputs_name: derived name from path
      problems: list of Problem(n, d)
    """
    data = yaml.safe_load(inputs_cfg_path.read_text())
    inputs_name = rel_name_from_config(inputs_cfg_path, anchor="configs")
    if inputs_name.startswith("inputs/"):
        inputs_name = inputs_name.split("/", 1)[1]

    problems: list[Problem] = []

    if "problems" in data:
        problems = [Problem(item["n"], item["d"]) for item in data["problems"]]

    elif "sweep" in data:
        sweep = data["sweep"]
        d_range = sweep["d"]["range"]
        n_range = sweep["n"]["range"]
        d_values = range(d_range["start"], d_range["stop"], d_range.get("step", 1))
        n_values = range(n_range["start"], n_range["stop"], n_range.get("step", 1))
        problems = [Problem(d=d, n=n) for d in d_values for n in n_values]
    else:
        raise ValueError("Inputs YAML must contain either 'problems' or 'sweep'.")

    return inputs_name, problems


def load_model(
    model_cfg_path: Path,
) -> tuple[str, Any]:
    """
    Returns:
      model_name: derived from path (configs/models/...)
      model instance
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

    instance = model_cls.from_config(model_cfg_path)
    return model_name, instance


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
    args = ap.parse_args()

    repo_root = repo_root_from_here()
    sha = git_sha(repo_root)
    ts = timestamp_utc()

    inputs_name, problems = load_inputs(args.inputs)
    model_name, model = load_model(args.model_config)

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
                "coh_min",
                "wall_time_s_mean",
            ],
        )
        writer.writeheader()

        for problem in problems:
            print(f"Running {model_name} on inputs d={problem.d}, n={problem.n}...")
            result = model.run(problem)

            writer.writerow(
                {
                    "timestamp": ts,
                    "git_sha": sha,
                    "inputs": inputs_name,
                    "model": model_name,
                    "d": problem.d,
                    "n": problem.n,
                    "coh_min": result.best_coherence,
                    "wall_time_s_mean": result.wall_time_s,
                }
            )
            f.flush()

    print(f"✅ wrote {csv_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
