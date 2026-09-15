import argparse
from contextlib import contextmanager
import json
import yaml
import os
import numpy as np
import torch
import logging
import traceback
import time
import random

from bn.train_bn import train_bn
from bn.expl import create_expl
from ad.train_ad import train_ad
from config_validator import validate_config


def save_timings(path, timings):
    with open(path, "w", encoding="utf-8") as timing_file:
        json.dump(timings, timing_file, indent=2)

@contextmanager
def time_phase(path, timings, phase):
    started = time.perf_counter()
    timings["phase_status"][phase] = "failed"
    try:
        yield
        timings["phase_status"][phase] = "completed"
    finally:
        timings[f"{phase}_seconds"] = time.perf_counter() - started
        save_timings(path, timings)

if __name__ == "__main__":

    timing_path = None
    timings = None
    try:
        # Parse config arguments
        parser = argparse.ArgumentParser(description="Train Neurosymbolic Intrusion Detection System")
        parser.add_argument('--config', type=str, required=False, help='Path to the config file', default="config.yml")
        parser.add_argument('--data_path', type=str, required=False, help='Path to the dataset', default="data/dataset/ton-iot_net")
        parser.add_argument('--name', type=str, required=False, help='Name of the experiment', default=None)
        parser.add_argument('--exp_path', type=str, required=False, help='Path to the experiment')
        parser.add_argument('--seed', type=int, required=False, help='Random seed for reproducibility', default=42)
        parser.add_argument('--no_ad', action='store_true', help='Flag to skip training the anomaly detection model')
        args = parser.parse_args()

        # Create results and logs directories if they don't exist
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Experiment name
        if args.exp_path:
            exp_name = os.path.basename(args.exp_path)
        else:
            if args.name:
                exp_name = args.name
            else:
                exp_name = "exp_" + str(int(time.time()))

        logging.basicConfig(filename=f"logs/{exp_name}.log", level=logging.INFO, encoding='utf-8', force=True)

        # Load yaml config file and convert to json
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        seed = config.get("seed", args.seed)
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        timing_path = os.path.join("results", f"train_times_{config_name}_{seed}.json")
        timings = {
            "status": "running",
            "bn_seconds": None,
            "explanations_seconds": None,
            "ad_seconds": None,
            "phase_status": {
                "bn": "skipped" if args.exp_path else "pending",
                "explanations": "skipped" if args.exp_path else "pending",
                "ad": "skipped" if args.no_ad else "pending",
            },
        }
        save_timings(timing_path, timings)

        os.makedirs(f"results/{exp_name}", exist_ok=True)

        # Save config yaml to base path
        with open(os.path.join(f"results/{exp_name}", "config.yaml"), 'w') as f:
            yaml.dump(config, f)
        
        # Validate config
        validate_config(config)

        # Defining seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        if not args.exp_path:
            # Train Bayesian Network
            with time_phase(timing_path, timings, "bn"):
                train_bn(exp_name, config.get("bayesian_network"), args.data_path)

            # Create explanation vectors for Train 2 partition
            with time_phase(timing_path, timings, "explanations"):
                create_expl(exp_name, config["anomaly_detection"]["explanations"], args.data_path)

        if not args.no_ad:
            # Train anomaly detection model
            with time_phase(timing_path, timings, "ad"):
                train_ad(exp_name, config.get("anomaly_detection"), seed)

        timings["status"] = "completed"
        save_timings(timing_path, timings)

    except Exception as e:
        logging.error(traceback.format_exc())
        if timings is not None:
            timings["status"] = "failed"
            save_timings(timing_path, timings)
        raise
