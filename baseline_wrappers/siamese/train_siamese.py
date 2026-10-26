"""Train the supplied binary contrastive Siamese loss for open-set recognition.

The encoder uses ContrastiveMLP. Original known-class labels are retained for
class-specific KNN inference. Rejection is calibrated on held-out known rows;
no test rows are read during training. Checkpoints include reference embeddings.
"""

import argparse
import random
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch as T
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from baseline_wrappers.siamese.siamese_network import OnlineContrastiveLoss
from baseline_wrappers.siamese.utils_siamese import (
    build_model, class_distances, embed, load_training, reference_indices, resolve_device,
)
from baseline_wrappers.utils_baselines import DEFAULT_DATA_DIR, discover_classes, record_stage_time


def parse_option():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/siamese")
    parser.add_argument("--classes", nargs="+", help="Subset of no_<class> partitions")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]; KNN runs on CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benign-class", default="normal", help="Class mapped to label zero in the binary loss")
    parser.add_argument("--neurons", default="256,256,256", help="Comma-separated hidden widths")
    parser.add_argument("--d-out", "--d_out", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=256,
                        help="Contrastive loss computes all pairs within each batch")
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=256)
    parser.add_argument("--reference-chunk-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True,
                        help="Inverse-frequency sampling using original known-class labels")
    parser.add_argument("--distance", choices=("cosine", "euclidean", "manhattan"), default="cosine")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--squared", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--k", type=int, default=5, help="Neighbors averaged within each known class")
    parser.add_argument("--calibration-fraction", type=float, default=0.2,
                        help="Fraction held out per known class before model/scaler fitting")
    parser.add_argument("--rejection-rate", type=float, default=0.05,
                        help="Upper-tail fraction of held-out known scores used for the threshold")
    parser.add_argument("--max-reference-per-class", type=int, default=0,
                        help="Optional random reference cap for KNN cost; 0 keeps all fitting rows")
    opt = parser.parse_args()
    try:
        opt.neurons = [int(n) for n in opt.neurons.split(",")]
        if min(opt.neurons) < 1:
            raise ValueError("neurons must be positive integers")
        if min(opt.d_out, opt.chunk_size, opt.reference_chunk_size, opt.epochs, opt.k) < 1:
            raise ValueError("Dimensions, chunk sizes, epochs and k must be positive")
        if opt.batch_size < 2 or opt.max_reference_per_class < 0:
            raise ValueError("batch-size must be >=2; max-reference-per-class must be >=0")
        if not 0 <= opt.seed < 2**32:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        if not 0 <= opt.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < opt.calibration_fraction < 1 or not 0 < opt.rejection_rate < 1:
            raise ValueError("calibration-fraction and rejection-rate must be in (0, 1)")
        if not all(np.isfinite(v) for v in (opt.lr, opt.weight_decay, opt.margin)):
            raise ValueError("Optimizer and loss parameters must be finite")
        if opt.lr <= 0 or opt.weight_decay < 0 or opt.margin <= 0:
            raise ValueError("lr and margin must be positive; weight-decay must be nonnegative")
        opt.device = resolve_device(opt.device)
        opt.classes = discover_classes(opt.data_dir, opt.classes)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    return opt


def train(opt, model, loader, optimizer, criterion):
    """One epoch; the original loss maps all attack labels to the positive class."""
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(opt.device), y.to(opt.device)
        loss = criterion(model(x), y)
        if not T.isfinite(loss):
            raise RuntimeError("Siamese network produced a nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


def main():
    opt = parse_option()
    with record_stage_time(opt.results_dir, "train"):
        print("Training Siamese network...")
        directory = opt.results_dir / f"siamese"
        directory.mkdir(parents=True, exist_ok=True)
        for cls in opt.classes:
            random.seed(opt.seed)
            np.random.seed(opt.seed)
            T.manual_seed(opt.seed)
            x, y, x_cal, y_cal, preprocessing = load_training(
                opt.data_dir / f"no_{cls}", benign_class=opt.benign_class,
                calibration_fraction=opt.calibration_fraction, seed=opt.seed,
            )
            config = {
                "d_in": x.shape[1], "neurons": opt.neurons, "d_out": opt.d_out,
                "dropout": opt.dropout, "project_to_sphere": True,
            }
            model = build_model(config).to(opt.device)
            sampler = None
            if opt.balanced:
                weights = T.bincount(y).double().reciprocal()[y]
                sampler = WeightedRandomSampler(weights, len(y), replacement=True)
            # Avoid a singleton final batch, which has no contrastive pairs.
            batch_size = min(opt.batch_size, len(y))
            loader = DataLoader(TensorDataset(x, y), batch_size=batch_size,
                                sampler=sampler, shuffle=sampler is None, num_workers=0,
                                drop_last=len(y) % batch_size == 1)
            criterion = OnlineContrastiveLoss(m=opt.margin, distance_metric=opt.distance,
                                              squared=opt.squared)
            optimizer = T.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            for epoch in range(1, opt.epochs + 1):
                loss = train(opt, model, loader, optimizer, criterion)
                print(f"{cls}: epoch {epoch}/{opt.epochs}, loss={loss:.6f}", flush=True)

            indices = reference_indices(y, opt.max_reference_per_class, opt.seed)
            reference = embed(model, x[indices], opt.chunk_size)
            labels = y[indices]
            calibration = embed(model, x_cal, opt.chunk_size)
            distances = class_distances(
                reference, labels, calibration, k=opt.k, metric=opt.distance,
                chunk_size=opt.chunk_size, reference_chunk_size=opt.reference_chunk_size,
            )
            scores = distances.min(dim=1).values.numpy()
            threshold = float(np.quantile(scores, 1 - opt.rejection_rate))
            checkpoint = {
                "format_version": 1, "algorithm": "siamese",
                "model_config": config, "preprocessing": preprocessing,
                "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "reference_embeddings": reference, "reference_labels": labels,
                "distance": opt.distance, "k": opt.k, "threshold": threshold,
                "calibration": {"samples": len(y_cal), "fraction": opt.calibration_fraction,
                                "rejection_rate": opt.rejection_rate},
                "training": {key: getattr(opt, key) for key in
                             ("seed", "epochs", "lr", "weight_decay", "batch_size", "balanced",
                              "margin", "squared", "benign_class", "max_reference_per_class")},
            }
            path = directory / f"siamese_no_{cls}.pt.tar"
            T.save(checkpoint, path)
            print(f"Saved {path} (references={len(labels)}, threshold={threshold:.6g})")


if __name__ == "__main__":
    main()
