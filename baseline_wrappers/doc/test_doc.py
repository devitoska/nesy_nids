"""Evaluate DOC using saved Gaussian thresholds and unknown-positive scores."""

import argparse
import importlib
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch as T

from baseline_wrappers.utils_baselines import (
    DEFAULT_DATA_DIR, create_metrics_table, discover_classes, record_stage_time,
)
from utils import compute_metrics


def parse_option():
    parser = argparse.ArgumentParser(description=__doc__)

    # data config
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to dataset")
    parser.add_argument("--results-dir", type=Path,
                        help="Exact run directory; default: results/doc_<seed>")
    parser.add_argument("--classes", nargs="+", help="Optional subset of discovered partitions")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=1024)

    opt = parser.parse_args()
    if not 0 <= opt.seed < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    if opt.device == "auto":
        opt.device = "cuda" if T.cuda.is_available() else "cpu"
    try:
        opt.classes = discover_classes(opt.data_dir, opt.classes)
        device = T.device(opt.device)
        if device.type == "cuda" and not T.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        opt.device = str(device)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))

    # config validation and transformation
    if opt.chunk_size < 1:
        parser.error("chunk-size must be positive")
    if opt.results_dir is None:
        opt.results_dir = Path(__file__).resolve().parents[2] / "results" / f"doc_{opt.seed}"
    return opt


@T.no_grad()
def eval(opt, checkpoint, unknown_cls):
    if checkpoint.get("format_version") != 1 or checkpoint.get("algorithm") != "doc":
        raise ValueError("Unsupported doc checkpoint")
    if checkpoint["seed"] != opt.seed:
        raise ValueError("Checkpoint seed differs from --seed")
    preprocessing = checkpoint["preprocessing"]
    if preprocessing["unknown_class"] != unknown_cls:
        raise ValueError("Checkpoint belongs to a different held-out class")
    frame = pd.read_csv(opt.data_dir / f"no_{unknown_cls}" / "test_raw_data.csv")
    features = preprocessing["features"]
    if set(frame.columns) != set(features + ["class"]):
        raise ValueError("CSV feature schema differs from training")
    values = frame[features].to_numpy(dtype=np.float64)
    if not len(values) or not features or not np.isfinite(values).all():
        raise ValueError("Features must be nonempty, finite numeric values")
    values = ((values - np.asarray(preprocessing["mean"]))
              / np.asarray(preprocessing["std"])).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Preprocessing produced nonfinite features")
    names = preprocessing["classes"] + [unknown_cls]
    labels = frame["class"].map({name: i for i, name in enumerate(names)})
    if labels.isna().any():
        raise ValueError("CSV contains missing or unexpected class labels")
    x = T.from_numpy(values)
    y = T.from_numpy(labels.to_numpy(dtype=np.int64))
    unknown = y == len(preprocessing["classes"])
    if not unknown.any() or unknown.all():
        raise ValueError("Evaluation requires both known and unknown test samples")

    # Resolve the original implementation's import paths without editing it.
    for old, new in {
        "model": "baseline_wrappers.closr.model",
        "model.model": "baseline_wrappers.closr.model.model",
        "baseline_implementations": "baseline_wrappers",
        "baseline_implementations.common": "baseline_wrappers.common",
    }.items():
        sys.modules.setdefault(old, importlib.import_module(new))
    api = importlib.import_module("baseline_wrappers.doc.doc")
    model = api.DOC(**checkpoint["model_config"]).to(opt.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # Calibration attributes are not registered buffers. Inference below passes
    # CPU probabilities to the original prediction and scoring APIs.
    model.stds = checkpoint["stds"].cpu()
    model.thresholds = checkpoint["thresholds"].cpu()
    predictions, scores = [], []
    for chunk in x.split(opt.chunk_size):
        probabilities = model(chunk.to(opt.device)).cpu()
        if not T.isfinite(probabilities).all():
            raise RuntimeError("Classifier produced nonfinite outputs")
        predictions.append(model.predict(probabilities))
        # forward_osr_eval reverses the sign. This is the API used by doc_evaluation.
        scores.append(model.get_osr_scores(probabilities))
    predictions, scores = T.cat(predictions).numpy(), T.cat(scores).numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError("DOC produced nonfinite novelty scores")
    num_known = len(preprocessing["classes"])
    names = np.asarray(preprocessing["classes"] + [unknown_cls])
    return compute_metrics(
        (y.numpy() == num_known).astype(int), (predictions == num_known).astype(int),
        names[y.numpy()], names[predictions], scores,
    )


def main():
    opt = parse_option()
    with record_stage_time(opt.results_dir, "test"):
        print("Testing DOC...")
        for unknown_cls in opt.classes:
            checkpoint_path = opt.results_dir / "doc" / f"doc_no_{unknown_cls}.pt.tar"
            checkpoint = T.load(checkpoint_path, map_location="cpu", weights_only=True)
            metrics = eval(opt, checkpoint, unknown_cls)
            output = opt.results_dir / "metrics" / f"no_{unknown_cls}" / "results.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metrics, indent=4, allow_nan=False), encoding="utf-8")
            print(f"Saved {output}")
        create_metrics_table(opt.results_dir)


if __name__ == "__main__":
    main()
