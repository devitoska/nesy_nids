import argparse
import yaml
import os
import numpy as np
import torch
import logging
import traceback
import time

from bn.train_bn import train_bn
from bn.expl import create_expl
from ad.train_ad import train_ad
from config_validator import validate_config

if __name__ == "__main__":

    try:
        # Parse config arguments
        parser = argparse.ArgumentParser(description="Train Neurosymbolic Intrusion Detection System")
        parser.add_argument('--config', type=str, required=False, help='Path to the config file', default="config.yml")
        parser.add_argument('--data_path', type=str, required=False, help='Path to the dataset', default="data/dataset/ton-iot_net")
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
            exp_name = "exp_" + str(int(time.time()))

        logging.basicConfig(filename=f"logs/{exp_name}.log", level=logging.INFO, encoding='utf-8', force=True)

        # Load yaml config file and convert to json
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        os.makedirs(f"results/{exp_name}")

        # Save config yaml to base path
        with open(os.path.join(f"results/{exp_name}", "config.yaml"), 'w') as f:
            yaml.dump(config, f)
        
        # Validate config
        validate_config(config)

        # Defining seeds for reproducibility
        np.random.seed(config.get("seed", args.seed))
        torch.manual_seed(config.get("seed", args.seed))

        if not args.exp_path:
            # Train Bayesian Network
            train_bn(exp_name, config.get("bayesian_network"), args.data_path)

            # Create explanation vectors for Train 2 partition
            create_expl(exp_name, args.data_path)

        if not args.no_ad:
            # Train anomaly detection model
            train_ad(exp_name, config.get("anomaly_detection"))

    except Exception as e:
        logging.error(traceback.format_exc())