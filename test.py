import argparse
import os
import traceback
import logging
import yaml

from bn.expl import create_expl
from ad.test_ad import test_ad
from utils import create_metrics_table

if __name__ == "__main__":

    try:
        # Parse config arguments
        parser = argparse.ArgumentParser(description="Test Neurosymbolic Intrusion Detection System")
        parser.add_argument('--data_path', type=str, required=False, help='Path to the dataset', default="data/dataset/ton-iot_net")
        parser.add_argument('--exp_path', type=str, required=True, help='Path to the experiment')
        args = parser.parse_args()

        path_to_exp = args.exp_path
        exp_name = os.path.basename(path_to_exp)

        with open(os.path.join(f"results/{exp_name}", "config.yaml"), 'r') as f:
            config = yaml.safe_load(f)
    
        # Create explanation vectors for Test partition
        create_expl(exp_name, args.data_path, mode="test")
        
        # Test anomaly detection model
        test_ad(exp_name, config.get("anomaly_detection"))

        # Create metrics table
        create_metrics_table(path_to_exp)

    except Exception as e:
        logging.error(traceback.format_exc())
