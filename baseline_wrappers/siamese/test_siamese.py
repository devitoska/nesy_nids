"""Evaluate Siamese class-KNN rejection with saved references and calibration."""

import argparse
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch as T

from baseline_wrappers.siamese.utils_siamese import (
    build_model, class_distances, embed, resolve_device, transform,
)
from baseline_wrappers.utils_baselines import (
    DEFAULT_DATA_DIR, create_metrics_table, discover_classes, record_stage_time,
)
from utils import compute_metrics


def parse_option():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/siamese")
    parser.add_argument("--classes", nargs="+", help="Subset of partition names")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]; KNN runs on CPU")
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=256)
    parser.add_argument("--reference-chunk-size", type=int, default=4096)
    opt = parser.parse_args()
    try:
        if min(opt.chunk_size, opt.reference_chunk_size) < 1:
            raise ValueError("Chunk sizes must be positive")
        opt.device = resolve_device(opt.device)
        opt.classes = discover_classes(opt.data_dir, opt.classes)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    return opt


@T.no_grad()
def eval(opt, checkpoint, unknown_cls):
    preprocessing = checkpoint["preprocessing"]
    if preprocessing["unknown_class"] != unknown_cls:
        raise ValueError("Checkpoint belongs to a different held-out class")
    frame = pd.read_csv(opt.data_dir / f"no_{unknown_cls}" / "test_raw_data.csv")
    x, y = transform(frame, preprocessing)
    num_known = len(preprocessing["classes"])
    unknown = y.numpy() == num_known
    if not unknown.any() or unknown.all():
        raise ValueError("Evaluation requires both known and unknown test samples")
    model = build_model(checkpoint["model_config"]).to(opt.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    features = embed(model, x, opt.chunk_size)
    labels = checkpoint["reference_labels"]
    if not T.equal(T.unique(labels, sorted=True), T.arange(num_known)):
        raise ValueError("Checkpoint must contain references for every known class")
    distances = class_distances(
        checkpoint["reference_embeddings"], labels, features, k=checkpoint["k"],
        metric=checkpoint["distance"], chunk_size=opt.chunk_size,
        reference_chunk_size=opt.reference_chunk_size,
    )
    scores, predicted = distances.min(dim=1)
    scores, predicted = scores.numpy(), predicted.numpy()
    rejected = scores > checkpoint["threshold"]
    predicted[rejected] = num_known
    names = np.asarray(preprocessing["classes"] + [unknown_cls])
    return compute_metrics(
        unknown.astype(int), rejected.astype(int),
        names[y.numpy()], names[predicted], scores,
    )


def main():
    opt = parse_option()
    with record_stage_time(opt.results_dir, "test"):
        print("Testing Siamese network...")
        for unknown_cls in opt.classes:
            path = opt.results_dir / f"siamese" / f"siamese_no_{unknown_cls}.pt.tar"
            checkpoint = T.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("format_version") != 1 or checkpoint.get("algorithm") != "siamese":
                raise ValueError(f"Unsupported Siamese checkpoint: {path}")
            metrics = eval(opt, checkpoint, unknown_cls)
            output = opt.results_dir / "metrics" / f"no_{unknown_cls}" / "results.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metrics, indent=4, allow_nan=False), encoding="utf-8")
            print(f"Saved {output}")
        create_metrics_table(opt.results_dir)


if __name__ == "__main__":
    main()
