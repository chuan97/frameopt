#!/usr/bin/env python3
"""
interleaved_cg_bench_driver.py

Driver to compare pure CMA vs. CMA interleaved with CG.

Usage:
  python scripts/cli/interleaved_cg_bench_driver.py \
      --config interleaved_cg_bench_config.json [--output-dir RESULTS_BASE]
"""
import datetime
import json
import subprocess
import sys
from pathlib import Path


def get_git_commit_hash(short: bool = True) -> str:
    """
    Retrieve the current Git commit hash. If short=True, returns the abbreviated hash; otherwise the full hash.
    """
    try:
        if short:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
        else:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
        return out.decode().strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main() -> None:
    script_path = Path(__file__)
    script_dir = script_path.parent
    script_stem = script_path.stem
    # Expect script_stem to end with '_bench_driver'
    if not script_stem.endswith("_bench_driver"):
        raise RuntimeError(
            f"Script filename must end with '_bench_driver.py', got {script_stem}"
        )
    prefix = script_stem[: -len("_bench_driver")]

    matching_configs = list(script_dir.glob(f"*{prefix}_bench_config.json"))
    matching_configs = [
        p for p in matching_configs if p.name.endswith("_bench_config.json")
    ]

    if len(matching_configs) == 0:
        print(
            f"No config file ending in '{prefix}_bench_config.json' found",
            file=sys.stderr,
        )
        sys.exit(1)
    elif len(matching_configs) > 1:
        print(
            f"Multiple matching config files found: {[str(p) for p in matching_configs]}",
            file=sys.stderr,
        )
        sys.exit(1)

    config_path = matching_configs[0]

    # Load JSON config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Determine experiment name and timestamp
    exp_name = config.get(
        "experiment_name", config_path.stem.replace("_bench_config", "")
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare results directory
    base_dir = Path(config.get("output_dir", script_dir / "results"))
    exp_dir = base_dir / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Copy and annotate config for provenance
    commit = get_git_commit_hash()
    config["git_commit"] = commit
    config["timestamp"] = timestamp
    # Save annotated config
    with open(exp_dir / config_path.name, "w") as outf:
        json.dump(config, outf, indent=2)

    # Extract parameters
    dims = config["dims"]  # list of [n, d]
    cg_every_list = config["cg_run_every"]  # list of int or None
    seeds = config["seeds"]  # list of int or None
    gen = config["gen"]
    popsize = config["popsize"]
    sigma0 = config["sigma0"]
    p_exp = config["p_exp"]
    cg_iters = config.get("cg_iters")

    # Loop through combinations
    for n, d in dims:
        for cg_every in cg_every_list:
            for seed in seeds:
                # Unique tag per run
                tag = f"{n}x{d}"
                if seed is not None:
                    tag += f"_seed{seed}"
                if cg_every is not None:
                    tag += f"_cg{cg_every}"
                # Output log path
                log_file = exp_dir / f"{tag}.csv"

                # Build command
                cmd = [
                    sys.executable,
                    "scripts/cli/run_cma_cg.py",
                    "-n",
                    str(n),
                    "-d",
                    str(d),
                    "--gen",
                    str(gen),
                    "--popsize",
                    str(popsize),
                    "--sigma0",
                    str(sigma0),
                    "--p",
                    str(p_exp),
                    "--log-file",
                    str(log_file),
                ]
                if cg_every is not None:
                    cmd += ["--cg-run-every", str(cg_every)]
                if seed is not None:
                    cmd += ["--seed", str(seed)]
                if cg_iters is not None:
                    cmd += ["--cg-iters", str(cg_iters)]

                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

    print(f"All experiments complete. Results in {exp_dir}")


main()
