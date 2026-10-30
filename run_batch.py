"""Train and test every YAML config: python run_batch.py --num_seed 3 --threads 2.

Explicit config seeds are used in order; otherwise --num_seed seeds are sampled.
EFC runs once per config, using the first seed when explicitly configured.
Append tab-separated experiment, total_seconds, train_seconds, test_seconds,
bn_seconds, explanations_seconds, ad_seconds, train_bn_seconds,
bn_to_pc_seconds, test_bn_seconds to results/times after each
experiment. All algorithms retain train/test totals in each run's timings.json;
baseline phase columns, like skipped NeSy phases, are null. At most --threads
subprocesses run at once, with training preceding testing for each experiment.
Each subprocess has a --timeout deadline (default: 86400 seconds). Failed runs
are reported after the remaining runs finish. On interruption, active process
groups are terminated and queued work cancelled.
--skip N skips the first N experiments in sorted config order, preserving seeds.
"""

import argparse
import csv
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

from config_validator import validate_config


BN_PHASE_KEYS = ("train_bn_seconds", "bn_to_pc_seconds", "test_bn_seconds")
LEGACY_TIMING_COLUMNS = (
    "experiment", "total_seconds", "train_seconds", "test_seconds",
    "bn_seconds", "explanations_seconds", "ad_seconds",
)
TIMING_COLUMNS = LEGACY_TIMING_COLUMNS + BN_PHASE_KEYS


def prepare_timing_file(path):
    """Upgrade the old TSV before appending rows, preserving historical values."""
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            columns = tuple(next(reader, []))
            if columns == TIMING_COLUMNS:
                return
            rows = list(reader)
        if columns != LEGACY_TIMING_COLUMNS:
            raise ValueError(f"{path}: unrecognized timing header; file left unchanged")
        if any(len(row) != len(columns) for row in rows):
            raise ValueError(f"{path}: malformed timing row; file left unchanged")
        rows = [list(TIMING_COLUMNS)] + [row + ["null"] * 3 for row in rows]
    else:
        rows = [list(TIMING_COLUMNS)]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# Seed inside each child process as well: subprocesses do not inherit RNG state.
SEEDED_RUNNER = """
import random
import runpy
import sys
from pathlib import Path
import numpy as np
import torch

seed = int(sys.argv.pop(1))
sys.argv.pop(0)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
runpy.run_path(sys.argv[0], run_name="__main__")
"""


class ProcessRunner:
    """Clean up each subprocess group; stop all groups on batch interruption."""

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
        env["PYTHONHASHSEED"] = str(seed) # should be set seed for ensure different BNs, "0" for same BNs
        env["TQDM_DISABLE"] = "1" # disable tqdm progress bars in subprocesses to avoid cluttering output
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
        finally:
            if process is not None:
                try:
                    self._cleanup(process)
                finally:
                    with self._lock:
                        self._active.discard(process)


def run_experiment(project_dir, run_config, name, seed, times_path, times_lock, processes, test=False, classes=None):
    results_dir = times_path.parent
    with run_config.open(encoding="utf-8") as config_file:
        config = validate_config(yaml.safe_load(config_file))
    algorithm = config["algorithm"]["type"]
    baseline = algorithm in {"EFC", "OCN", "CLOSR", "Siamese", "CROSR", "OpenMax", "DOC"}
    experiment_dir = results_dir / name
    timing_path = experiment_dir / "timings.json"
    if baseline:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with (experiment_dir / "config.yaml").open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(config, config_file)
        data_dir = Path(config["dataset_path"])
        if not data_dir.is_absolute():
            data_dir = project_dir / data_dir
        arguments = ["--data-dir", str(data_dir), "--results-dir", str(experiment_dir)]
        if classes is not None:
            arguments.extend(["--classes", *classes])
        train_script = f"baseline_wrappers/{algorithm.lower()}/train_{algorithm.lower()}.py"
        test_script = f"baseline_wrappers/{algorithm.lower()}/test_{algorithm.lower()}.py"
        train_arguments = arguments + (["--seed", str(seed)] if algorithm in {"OCN", "CLOSR", "Siamese", "CROSR", "OpenMax", "DOC"} else [])
        test_arguments = arguments
    else:
        train_script, test_script = "train.py", "test.py"
        # train arguments
        train_arguments = ["--config", str(run_config), "--name", name, "--seed", str(seed)]
        test_arguments = ["--exp_path", str(experiment_dir)]
        if test:
            train_arguments.extend(["--exp_path", str(experiment_dir)])
            test_arguments.extend(["--no_expl"])
            
    print(f"Running {name}", flush=True)
    started = time.perf_counter()
    processes.run_script(project_dir, train_script, train_arguments, seed)
    with timing_path.open(encoding="utf-8") as timing_file:
        train_timings = json.load(timing_file)
    if train_timings["status"] != "trained":
        raise RuntimeError(f"Training did not complete for {name}")
    phase_values = [None] * 6
    if not baseline:
        bn_phases = [train_timings.get(key) for key in BN_PHASE_KEYS]
        if any(key in train_timings for key in BN_PHASE_KEYS):
            measured = [value for value in bn_phases if value is not None]
            bn_total = sum(measured) if measured else None
        else:
            bn_total = train_timings["bn_seconds"]
        phase_values = [bn_total, train_timings["explanations_seconds"],
                        train_timings["ad_seconds"], *bn_phases]
    phase_times = "\t".join(
        "null" if value is None else f"{value:.6f}" for value in phase_values
    )
    processes.run_script(project_dir, test_script, test_arguments, seed)
    finished = time.perf_counter()
    elapsed = finished - started
    with timing_path.open(encoding="utf-8") as timing_file:
        timings = json.load(timing_file)
    if timings["status"] != "completed":
        raise RuntimeError(f"Testing did not complete for {name}")
    train_elapsed = timings["train_seconds"]
    test_elapsed = timings["test_seconds"]
    # Serialize appends and close after each experiment to save its row promptly.
    with times_lock:
        prepare_timing_file(times_path)
        with times_path.open("a", encoding="utf-8") as times_file:
            times_file.write(
                f"{name}\t{elapsed:.6f}\t{train_elapsed:.6f}\t{test_elapsed:.6f}"
                f"\t{phase_times}\n"
            )
    print(
        f"Finished {name} in {elapsed:.6f} seconds "
        f"(train: {train_elapsed:.6f}, test: {test_elapsed:.6f})",
        flush=True,
    )


def run_experiments(project_dir, experiments, times_path, threads, timeout, test = False, classes=None):
    processes = ProcessRunner(timeout)
    times_lock = Lock()
    futures = {}
    failures = []
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
            future = executor.submit(
                run_experiment, project_dir, run_config, name, seed,
                times_path, times_lock, processes, test, classes
            )
            futures[future] = name
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                name = futures[future]
                failures.append(error)
                print(f"Failed {name}: {error}", file=sys.stderr, flush=True)
                # A killed child may leave stale running or completed timings.
                timing_path = times_path.parent / name / "timings.json"
                try:
                    timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
                    if not isinstance(timings, dict):
                        timings = {}
                    timings["status"] = "failed"
                    timing_path.parent.mkdir(parents=True, exist_ok=True)
                    timing_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
                except (OSError, ValueError) as timing_error:
                    print(f"Warning: could not mark {name} failed: {timing_error}", file=sys.stderr)
        if failures:
            print(f"{len(failures)} of {len(futures)} experiments failed", file=sys.stderr)
            raise failures[0]
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
    parser.add_argument("--num_seed", type=int, default=1, help="Runs per config without explicit seeds; EFC always runs once (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed (default: 42)")
    parser.add_argument("--threads", type=int, default=1, help="Maximum concurrent subprocesses (default: 1)")
    parser.add_argument("--test", action="store_true", help="Assume training and run only testing")
    parser.add_argument("--classes", nargs="+", help="Subset of partitions passed to baseline train/test scripts; default: all (NeSy-NIDS unaffected)")
    parser.add_argument(
        "--skip", type=int,
        help="Skip the first N experiments; count each EFC config once",
    )
    parser.add_argument(
        "--timeout", type=float, default=172800,
        help="Maximum seconds per train/test subprocess (default: 172800 = 48 hours)",
    )
    args = parser.parse_args()

    if args.num_seed < 1:
        parser.error("--num_seed must be positive")
    if args.num_seed > 20:
        parser.error("--num_seed exceeds the recommended maximum of 20")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.threads > os.cpu_count():
        parser.error("--threads exceeds the recommended maximum of os.cpu_count()")
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

    planned_configs = []
    for config_path in configs:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        has_seed = isinstance(config, dict) and "seed" in config
        try:
            config = validate_config(config)
        except ValueError as error:
            parser.error(f"{config_path}: {error}")
        config_seeds = config["seed"] if has_seed else None
        if has_seed and not isinstance(config_seeds, list):
            config_seeds = [config_seeds]
        repetitions = len(config_seeds) if has_seed else args.num_seed
        if config["algorithm"]["type"] == "EFC":
            repetitions = 1
        if has_seed and len(set(config_seeds[:repetitions])) != repetitions:
            parser.error(f"{config_path}: seed list must contain unique seeds")
        planned_configs.append((config_path, config, repetitions, config_seeds))

    run_count = sum(repetitions for _, _, repetitions, _ in planned_configs)
    random_count = sum(repetitions for _, _, repetitions, seeds in planned_configs if seeds is None)
    if random_count > 2**32:
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
    seeds = iter(random.sample(range(2**32), random_count))

    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    times_path = results_dir / "times"
    prepare_timing_file(times_path)

    with tempfile.TemporaryDirectory(prefix="nesy_nids_batch_") as temporary_dir:
        experiments = []
        run_index = 0
        for config_path, config, repetitions, config_seeds in planned_configs:
            for index in range(repetitions):
                seed = config_seeds[index] if config_seeds is not None else next(seeds)
                run_index += 1
                if run_index <= skip_count:
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

        run_experiments(project_dir, experiments, times_path, args.threads, args.timeout, args.test, args.classes)


if __name__ == "__main__":
    main()
