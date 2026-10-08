import pandas as pd
import os
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc, recall_score
import json
import argparse
from pathlib import Path

if __package__:
    from .utils_baselines import DEFAULT_DATA_DIR, create_metrics_table, discover_classes, record_stage_time
else:
    from utils_baselines import DEFAULT_DATA_DIR, create_metrics_table, discover_classes, record_stage_time

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test EFC for each no_<class> data directory.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results/efc")
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

            pr, rc, _ = precision_recall_curve(y_gt_bin, energies)
            # save results
            metrics = {
                "roc_auc_score": roc_auc_score(y_gt_bin, energies),
                "auc_score": auc(rc, pr),
                "fpr" : 1 - recall_score(y_gt_bin, y_pred_bin, pos_label=0),
                "classification_report_binary": classification_report(y_gt_bin, y_pred_bin, output_dict=True),
                "confusion_matrix_multiclass": confusion_matrix(y_gt_mul, y_pred_mul).tolist(),
                "classification_report_multiclass": classification_report(y_gt_mul, y_pred_mul, output_dict=True)
            }

            os.makedirs(args.results_dir / "metrics" / f"no_{unknown_cls}", exist_ok=True)
            with open(args.results_dir / "metrics" / f"no_{unknown_cls}" / "results.json", "w") as f:
                json.dump(metrics, f, indent=4)

        create_metrics_table(args.results_dir)
