"""Train and test every YAML config: python run_batch.py --num_seed 3 --threads 2.

Append tab-separated experiment, total_seconds, train_seconds, test_seconds,
bn_seconds, explanations_seconds, ad_seconds to results/times after each
experiment. Skipped training phases are recorded as null. At most --threads
subprocesses run at once, with training preceding testing for each experiment.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from threading import Lock
import time

import numpy as np
import torch
import yaml


# Seed inside each child process as well: subprocesses do not inherit RNG state.
SEEDED_RUNNER = """
import random
import runpy
import sys
import numpy as np
import torch

seed = int(sys.argv.pop(1))
sys.argv.pop(0)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run_script(project_dir, script, arguments, seed):
    # Hash ordering must be configured before Python starts, independently of
    # the experiment seed used for model randomness.
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    subprocess.run(
        [sys.executable, "-c", SEEDED_RUNNER, str(seed), script, *arguments],
        cwd=project_dir,
        env=env,
        check=True,
    )


def run_experiment(project_dir, run_config, name, seed, times_path, times_lock):
    results_dir = times_path.parent
    print(f"Running {name}", flush=True)
    started = time.perf_counter()
    run_script(
        project_dir, "train.py",
        ["--config", str(run_config), "--name", name, "--seed", str(seed)],
        seed,
    )
    train_elapsed = time.perf_counter() - started
    timing_path = results_dir / f"train_times_{name}.json"
    with timing_path.open(encoding="utf-8") as timing_file:
        train_timings = json.load(timing_file)
    if train_timings["status"] != "completed":
        raise RuntimeError(f"Training did not complete for {name}")
    phase_times = "\t".join(
        "null" if train_timings[key] is None else f"{train_timings[key]:.6f}"
        for key in ("bn_seconds", "explanations_seconds", "ad_seconds")
    )
    test_started = time.perf_counter()
    run_script(
        project_dir, "test.py",
        ["--exp_path", str(results_dir / name)], seed,
    )
    finished = time.perf_counter()
    test_elapsed = finished - test_started
    elapsed = finished - started
    # Serialize appends and close after each experiment to save its row promptly.
    with times_lock:
        with times_path.open("a", encoding="utf-8") as times_file:
            times_file.write(
                f"{name}\t{elapsed:.6f}\t{train_elapsed:.6f}\t{test_elapsed:.6f}"
                f"\t{phase_times}\n"
            )
    # Retain the subprocess report until the combined row is saved.
    timing_path.unlink()
    print(
        f"Finished {name} in {elapsed:.6f} seconds "
        f"(train: {train_elapsed:.6f}, test: {test_elapsed:.6f})",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train and test all configs in configs/.")
    parser.add_argument("--num_seed", type=int, default=1, help="Runs per config (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed (default: 42)")
    parser.add_argument("--threads", type=int, default=1, help="Maximum concurrent subprocesses (default: 1)")
    args = parser.parse_args()
    if args.num_seed < 1:
        parser.error("--num_seed must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")

    project_dir = Path(__file__).resolve().parent
    configs_dir = project_dir / "configs"
    if not configs_dir.is_dir():
        parser.error(f"Config directory does not exist: {configs_dir}")
    configs = sorted(
        path for path in configs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    if not configs:
        parser.error(f"No YAML configs found in {configs_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Sampling without replacement guarantees distinct, reproducible run seeds.
    run_count = len(configs) * args.num_seed
    if run_count > 2**32:
        parser.error("The number of runs exceeds the available unique seeds")
    seeds = iter(random.sample(range(2**32), run_count))

    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    times_path = results_dir / "times"
    header = (
        "experiment\ttotal_seconds\ttrain_seconds\ttest_seconds\t"
        "bn_seconds\texplanations_seconds\tad_seconds\n"
    )
    existing_times = times_path.read_text(encoding="utf-8") if times_path.exists() else ""
    if not existing_times.startswith(header):
        times_path.write_text(header + existing_times, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="nesy_nids_batch_") as temporary_dir:
        experiments = []
        for config_path in configs:
            with config_path.open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
            if not isinstance(config, dict):
                parser.error(f"Config must contain a YAML mapping: {config_path}")

            for _ in range(args.num_seed):
                seed = next(seeds)
                name = f"{config_path.stem}_{seed}"
                # train.py prioritizes the YAML seed over its CLI seed. Save the
                # effective seed in its config without changing the source file.
                # Separate directories prevent parallel runs from overwriting
                # configs while preserving the basename used in timing reports.
                run_dir = Path(temporary_dir) / name
                run_dir.mkdir()
                run_config = run_dir / config_path.name
                with run_config.open("w", encoding="utf-8") as config_file:
                    yaml.safe_dump({**config, "seed": seed}, config_file)
                experiments.append((run_config, name, seed))

        times_lock = Lock()
        # Each worker waits for one child at a time, so the worker count also
        # bounds the total number of simultaneous training/testing subprocesses.
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [
                executor.submit(
                    run_experiment, project_dir, run_config, name, seed,
                    times_path, times_lock,
                )
                for run_config, name, seed in experiments
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                # Let active experiments save their results; skip queued work.
                for future in futures:
                    future.cancel()
                raise


if __name__ == "__main__":
    main()
