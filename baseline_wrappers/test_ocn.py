"""Evaluate OCN checkpoints using the EFC wrapper's metrics and output schema.

Run: python baseline_wrappers/test_ocn.py
Inputs: no_<attack>/test_raw_data.csv and results/ocn/ocn/ocn_no_<attack>.pt.
Outputs: results/ocn/metrics/no_<attack>/results.json and metrics/table.csv.
Unknown means the held-out attack, including for the binary metrics.
Only version 2 checkpoints with the unpadded 1D backbone are supported.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    auc, classification_report, confusion_matrix, precision_recall_curve,
    recall_score, roc_auc_score,
)
import torch

if __package__:
    from . import utils_ocn as utils
else:
    import utils_ocn as utils


def evaluate_partition(ocn, unknown_class, args):
    frame = utils.read_raw(args.data_dir / f"no_{unknown_class}" / "test_raw_data.csv")
    path = args.results_dir / "ocn" / f"ocn_no_{unknown_class}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != utils.CHECKPOINT_VERSION:
        raise ValueError(
            f"Incompatible OCN checkpoint: {path}. The 1D model requires retraining "
            "with train_ocn.py; old 2D checkpoints cannot be reused."
        )
    if checkpoint["unknown_class"] != unknown_class:
        raise ValueError(f"Checkpoint held-out class does not match {unknown_class!r}: {path}")
    classes = checkpoint["classes"]
    y_gt_mul = frame["class"].to_numpy()
    unexpected = set(y_gt_mul) - set(classes) - {unknown_class}
    if unexpected:
        raise ValueError(f"Unexpected test classes for {unknown_class}: {sorted(unexpected)}")
    y_gt_bin = (y_gt_mul == unknown_class).astype(int)
    if np.unique(y_gt_bin).size != 2:
        raise ValueError("AUROC evaluation requires both known and held-out examples.")

    features = utils.transform_features(frame, checkpoint["preprocessing"])
    device = utils.resolve_device(args.device)
    architecture = checkpoint["architecture"]
    if architecture["num_features"] != features.shape[-1]:
        raise ValueError("Checkpoint architecture and preprocessing feature counts disagree.")
    model = utils.build_model(architecture, len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    centroids, thresholds = checkpoint["centroids"], checkpoint["thresholds"]
    utils.check_finite(model, centroids, thresholds)
    predicted, distances = utils.predict(
        ocn, model, features, centroids, thresholds, args.batch_size, device,
    )
    if not np.isfinite(distances).all():
        raise RuntimeError("The original OCN produced nonfinite anomaly scores.")
    # As in EFC, use the held-out name only for evaluation of unknown predictions.
    y_pred_mul = np.asarray(classes + [unknown_class])[predicted]
    y_pred_bin = (y_pred_mul == unknown_class).astype(int)
    precision, recall, _ = precision_recall_curve(y_gt_bin, distances)
    metrics = {
        "roc_auc_score": roc_auc_score(y_gt_bin, distances),
        "auc_score": auc(recall, precision),
        "fpr": 1 - recall_score(y_gt_bin, y_pred_bin, pos_label=0),
        "classification_report_binary": classification_report(
            y_gt_bin, y_pred_bin, output_dict=True, zero_division=0,
        ),
        "confusion_matrix_multiclass": confusion_matrix(y_gt_mul, y_pred_mul).tolist(),
        "classification_report_multiclass": classification_report(
            y_gt_mul, y_pred_mul, output_dict=True, zero_division=0,
        ),
    }
    output = args.results_dir / "metrics" / f"no_{unknown_class}" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(metrics, handle, indent=4, allow_nan=False)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=utils.DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=utils.DEFAULT_RESULTS_DIR)
    parser.add_argument("--classes", nargs="+", help="Optional subset of discovered classes; default: all")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda:0")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be >=1")
    try:
        args.classes = utils.discover_classes(args.data_dir, args.classes)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    with utils.record_stage_time(args.results_dir, "test"):
        ocn = utils.load_ocn()
        print("Testing OCNs...")
        for unknown_class in args.classes:
            evaluate_partition(ocn, unknown_class, args)
        utils.create_metrics_table(args.results_dir)


if __name__ == "__main__":
    main()
