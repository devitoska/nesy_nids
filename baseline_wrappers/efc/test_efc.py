import pandas as pd
import os
import numpy as np
import pickle
import json
import argparse
from pathlib import Path

import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import compute_metrics
from baseline_wrappers.utils_baselines import DEFAULT_DATA_DIR, create_metrics_table, discover_classes, record_stage_time

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test EFC for each no_<class> data directory.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results/efc")
    parser.add_argument("--classes", nargs="+", help="Optional subset of discovered classes; default: all")
    args = parser.parse_args()
    try:
        classes = discover_classes(args.data_dir, args.classes)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    with record_stage_time(args.results_dir, "test"):
        print("Testing EFCs...")

        for unknown_cls in classes:
            test_data = pd.read_csv(os.path.join(
                args.data_dir, f"no_{unknown_cls}", "test_data.csv",
            ))

            # load the model
            model = pickle.load(open(args.results_dir / "efc" / f"efc_no_{unknown_cls}.pkl", "rb"))

            # prepare test data
            X = test_data.drop(columns=["class"]).values.astype(float)  # ensure numeric

            y_gt_mul = test_data["class"].values
            y_gt_bin = (y_gt_mul == unknown_cls).astype(int)

            # get predictions
            y_pred_mul, energies = model.predict(X, unknown_class=True, return_energies=True)

            # replace "unknown" with the unknown class label for multiclass evaluation
            y_pred_mul = np.where(y_pred_mul == "unknown", unknown_cls, y_pred_mul)
            y_pred_bin = (y_pred_mul == unknown_cls).astype(int)

            ad_scores = energies  # assuming energies are used as anomaly detection scores
            metrics = compute_metrics(y_gt_bin, y_pred_bin, y_gt_mul, y_pred_mul, ad_scores)

            os.makedirs(args.results_dir / "metrics" / f"no_{unknown_cls}", exist_ok=True)
            with open(args.results_dir / "metrics" / f"no_{unknown_cls}" / "results.json", "w") as f:
                json.dump(metrics, f, indent=4)

        create_metrics_table(args.results_dir)
