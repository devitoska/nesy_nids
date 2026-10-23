"""Evaluate CROSR using saved known-only calibration and shared baseline metrics."""

import argparse
import json
import sys
import libmr
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch as T

from baseline_wrappers.crosr.utils_crosr import build_model, crosr_api, resolve_device, transform
from baseline_wrappers.utils_baselines import (
    DEFAULT_DATA_DIR, create_metrics_table, discover_classes, record_stage_time,
)
from utils import compute_metrics


def parse_option():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/crosr")
    parser.add_argument("--classes", nargs="+", help="Subset of partition names")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=1024)
    opt = parser.parse_args()
    try:
        if opt.chunk_size < 1:
            raise ValueError("chunk-size must be positive")
        opt.device = resolve_device(opt.device)
        opt.classes = discover_classes(opt.data_dir, opt.classes)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    return opt


@T.no_grad()
def eval(opt, checkpoint, unknown_cls):
    crosr_api()

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
    model.centroids = checkpoint["centroids"].cpu()
    model.weibulls = []
    for params in checkpoint["weibull_params"]:
        mr = libmr.MR()
        mr.set_params(*params)
        model.weibulls.append(mr)

    predictions, scores = [], []
    for chunk in x.split(opt.chunk_size):
        z, latent, _ = model(chunk.to(opt.device))
        z, latent = z.cpu(), latent.cpu()
        # The API returns probabilities with UNKNOWN LAST. Do not use
        # predict_label (assumes unknown first) or forward_openmax (latent only).
        probs = model.update_probs(z=z, mavs=T.cat((z, latent), dim=-1),
                                   alpha=checkpoint["alpha"])
        if not T.isfinite(probs).all():
            raise RuntimeError("CROSR produced nonfinite OpenMax probabilities")
        confidence, predicted = probs.max(dim=1)
        predicted[confidence < checkpoint["threshold"]] = num_known
        predictions.append(predicted)
        scores.append(probs[:, -1])
    predicted, scores = T.cat(predictions).numpy(), T.cat(scores).numpy()
    names = np.asarray(preprocessing["classes"] + [unknown_cls])
    return compute_metrics(
        unknown.astype(int), (predicted == num_known).astype(int),
        names[y.numpy()], names[predicted], scores,
    )


def main():
    opt = parse_option()
    with record_stage_time(opt.results_dir, "test"):
        print("Testing CROSR...")
        for unknown_cls in opt.classes:
            path = opt.results_dir / f"crosr" / f"crosr_no_{unknown_cls}.pt.tar"
            checkpoint = T.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("format_version") != 1 or checkpoint.get("algorithm") != "crosr":
                raise ValueError(f"Unsupported CROSR checkpoint: {path}")
            metrics = eval(opt, checkpoint, unknown_cls)
            output = opt.results_dir / "metrics" / f"no_{unknown_cls}" / "results.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metrics, indent=4, allow_nan=False), encoding="utf-8")
            print(f"Saved {output}")
        create_metrics_table(opt.results_dir)


if __name__ == "__main__":
    main()
