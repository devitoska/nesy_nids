"""Evaluate CLOSR with unknown-positive scores and the shared baseline metrics.

The rejection threshold is a quantile of known training scores, not a threshold
selected using held-out test labels. Architecture and preprocessing come from
the checkpoint. Test CSV rows are evaluated exactly once.
"""

import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch as T
import torch.nn.functional as F

from baseline_wrappers.closr.utils_closr import load_partition, get_features
from baseline_wrappers.closr.model.model import CLOSRMLP
from baseline_wrappers.closr.util.distance import get_centroids, calc_class_sims_chunked
from baseline_wrappers.utils_baselines import (
    DEFAULT_DATA_DIR, create_metrics_table, discover_classes, record_stage_time,
)
from utils import compute_metrics


def parse_option():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/closr")
    parser.add_argument("--classes", nargs="+", help="Optional subset of partition names")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--rejection-rate", type=float, default=0.05,
                        help="Fraction of known training scores above the rejection threshold")
    opt = parser.parse_args()
    if opt.chunk_size < 1:
        parser.error("chunk_size must be positive")
    if not 0 < opt.rejection_rate < 1:
        parser.error("rejection-rate must be in (0, 1)")
    if opt.device == "auto":
        opt.device = "cuda" if T.cuda.is_available() else "cpu"
    try:
        opt.classes = discover_classes(opt.data_dir, opt.classes)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return opt


def anomaly_scores(similarities):
    # The original CLOSR score measures knownness; larger anomaly scores must
    # mean unknown for the common AUROC/AUPR contract used by EFC.
    return -(similarities.square() * T.softmax(similarities, dim=-1)).sum(dim=-1)


@T.no_grad()
def closr_eval(model, x_train, y_train, x_test, y_test, x_zd, y_zd, opt):
    num_known = model.n_classes
    x_test = T.cat((x_test, x_zd), dim=0)
    y_test = T.cat((y_test, y_zd), dim=0)
    train_features, train_labels = get_features(
        model, x_train, y_train, chunk_size=opt.chunk_size, move_to_cpu=True,
    )
    centroids = F.normalize(get_centroids(train_features, train_labels), dim=-1)
    train_sims = calc_class_sims_chunked(centroids, train_features, opt.chunk_size)
    train_scores = anomaly_scores(train_sims).numpy()
    threshold = float(np.quantile(train_scores, 1 - opt.rejection_rate))
    del train_features, train_sims

    test_features, test_labels = get_features(
        model, x_test, y_test, chunk_size=opt.chunk_size, move_to_cpu=True,
    )
    test_sims = calc_class_sims_chunked(centroids, test_features, opt.chunk_size)
    scores = anomaly_scores(test_sims).numpy()
    if not np.isfinite(train_scores).all() or not np.isfinite(scores).all():
        raise RuntimeError("CLOSR produced nonfinite anomaly scores")
    predictions = test_sims.argmax(dim=1).numpy()
    rejected = scores > threshold
    predictions[rejected] = num_known
    labels = test_labels.numpy()
    names = np.asarray(opt.class_names)
    metrics = compute_metrics(
        (labels == num_known).astype(int), rejected.astype(int),
        names[labels], names[predictions], scores,
    )
    return metrics


def main():
    opt = parse_option()
    with record_stage_time(opt.results_dir, "test"):
        print("Testing CLOSR...")
        for unknown_cls in opt.classes:
            checkpoint_path = opt.results_dir / "closr" / f"closr_no_{unknown_cls}.pt.tar"
            checkpoint = T.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("format_version") != 1:
                raise ValueError(f"Retrain CLOSR to create a checkpoint with model metadata: {checkpoint_path}")
            arrays = load_partition(
                opt.data_dir / f"no_{unknown_cls}", preprocessing=checkpoint["preprocessing"],
            )
            opt.class_names = checkpoint["preprocessing"]["classes"] + [unknown_cls]
            model = CLOSRMLP(**checkpoint["model_config"]).to(opt.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            # Keep input arrays on CPU and transfer only each inference chunk.
            data = [T.as_tensor(array) for array in arrays[:6]]
            metrics = closr_eval(model, *data, opt)
            output = opt.results_dir / "metrics" / f"no_{unknown_cls}" / "results.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metrics, indent=4, allow_nan=False), encoding="utf-8")
        create_metrics_table(opt.results_dir)


if __name__ == "__main__":
    main()
