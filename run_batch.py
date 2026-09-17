"""Train and test every YAML config: python run_batch.py --num_seed 3 --threads 2.

Append tab-separated experiment, total_seconds, train_seconds, test_seconds,
bn_seconds, explanations_seconds, ad_seconds to results/times after each
experiment. Skipped training phases are recorded as null. At most --threads
subprocesses run at once, with training preceding testing for each experiment.
Each subprocess has a --timeout deadline (default: 86400 seconds). On failure
or interruption, active process groups are terminated and queued work cancelled.
--skip N skips the first N experiments in sorted config order, preserving seeds.
"""

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
from threading import Event, Lock
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


class ProcessRunner:
    """Own subprocess groups and stop them together when the batch fails."""

    def __init__(self, timeout, grace_period=5.0):
        self.timeout = timeout
        self.grace_period = grace_period
        self.stopped = Event()
        self.failure = None
        self._lock = Lock()
        self._active = set()

    @staticmethod
    def _signal_group(process, signum):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return False
        return True

    def stop(self, failure=None):
        # Serialize stopping with process creation: no child may start after
        # shutdown takes its snapshot of active processes.
        with self._lock:
            if not self.stopped.is_set():
                self.failure = failure
                self.stopped.set()
            for process in self._active:
                self._signal_group(process, signal.SIGTERM)

    def _cleanup(self, process):
        try:
            # A successful direct child can still leave worker processes behind.
            if self._signal_group(process, signal.SIGTERM):
                deadline = time.monotonic() + self.grace_period
                while self._signal_group(process, 0):
                    process.poll()  # Reap the direct child if it has exited.
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._signal_group(process, signal.SIGKILL)
                        break
                    time.sleep(min(0.05, remaining))
        finally:
            process.wait()

    def run_script(self, project_dir, script, arguments, seed):
        # Hash ordering must be configured before Python starts, independently
        # of the experiment seed used for model randomness.
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0" # should be set seed for ensure different BNs
        command = [sys.executable, "-c", SEEDED_RUNNER, str(seed), script, *arguments]
        process = None
        try:
            with self._lock:
                if self.stopped.is_set():
                    raise CancelledError("Batch stopped before subprocess launch")
                process = subprocess.Popen(
                    command, cwd=project_dir, env=env, start_new_session=True,
                )
                self._active.add(process)

            deadline = time.monotonic() + self.timeout
            while True:
                if self.stopped.is_set():
                    raise CancelledError(f"Batch stopped while running {script}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, self.timeout)
                try:
                    returncode = process.wait(timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            if self.stopped.is_set():
                raise CancelledError(f"Batch stopped while running {script}")
        except BaseException as error:
            self.stop(error)
            raise
        finally:
            if process is not None:
                try:
                    self._cleanup(process)
                finally:
                    with self._lock:
                        self._active.discard(process)


def run_experiment(project_dir, run_config, name, seed, times_path, times_lock, processes):
    results_dir = times_path.parent
    print(f"Running {name}", flush=True)
    started = time.perf_counter()
    processes.run_script(
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
    processes.run_script(
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


def run_experiments(project_dir, experiments, times_path, threads, timeout):
    processes = ProcessRunner(timeout)
    times_lock = Lock()
    futures = []
    executor = None
    previous_handlers = {}

    def handle_signal(signum, frame):
        # Further signals must not interrupt termination and child reaping.
        for shutdown_signal in (signal.SIGINT, signal.SIGTERM):
            signal.signal(shutdown_signal, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, handle_signal)
        # Each worker waits for one child at a time, bounding active subprocesses.
        executor = ThreadPoolExecutor(max_workers=threads)
        for run_config, name, seed in experiments:
            if processes.stopped.is_set():
                break
            futures.append(executor.submit(
                run_experiment, project_dir, run_config, name, seed,
                times_path, times_lock, processes,
            ))
        for future in as_completed(futures):
            future.result()
    except BaseException as error:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        processes.stop(error)
        for future in futures:
            future.cancel()
        # Report the original failure instead of another worker's cancellation.
        if isinstance(error, Exception) and processes.failure is not None:
            raise processes.failure
        raise
    finally:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description="Train and test all configs in configs/.")
    parser.add_argument("--num_seed", type=int, default=1, help="Runs per config (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed (default: 42)")
    parser.add_argument("--threads", type=int, default=1, help="Maximum concurrent subprocesses (default: 1)")
    parser.add_argument(
        "--skip", type=int,
        help="Skip the first N experiments (1 through num_seed * number of configs)",
    )
    parser.add_argument(
        "--timeout", type=float, default=86400,
        help="Maximum seconds per train/test subprocess (default: 86400 = 24 hours)",
    )
    args = parser.parse_args()

    if args.num_seed < 1:
        parser.error("--num_seed must be positive")
    if args.num_seed > 20:
        parser.error("--num_seed exceeds the recommended maximum of 20")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.threads > 4:
        parser.error("--threads exceeds the recommended maximum of 4")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive, finite number of seconds")
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

    run_count = len(configs) * args.num_seed
    if run_count > 2**32:
        parser.error("The number of runs exceeds the available unique seeds")
    if args.skip is not None and not 1 <= args.skip <= run_count:
        parser.error(f"--skip must be between 1 and {run_count}")
    skip_count = args.skip or 0
    if skip_count:
        print(f"Skipping the first {skip_count} of {run_count} experiments", flush=True)
    if skip_count == run_count:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Generate the full sequence so skipping does not change subsequent seeds.
    # Sampling without replacement guarantees distinct, reproducible run seeds.
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
        for config_index, config_path in enumerate(configs):
            with config_path.open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
            if not isinstance(config, dict):
                parser.error(f"Config must contain a YAML mapping: {config_path}")

            for repetition in range(args.num_seed):
                seed = next(seeds)
                if config_index * args.num_seed + repetition < skip_count:
                    continue
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

        run_experiments(project_dir, experiments, times_path, args.threads, args.timeout)


if __name__ == "__main__":
    main()
