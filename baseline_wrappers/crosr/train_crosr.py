"""Train CROSR on known classes and save a calibrated checkpoint per partition.

Requires libmr (with NumPy/Cython installed: pip install --no-build-isolation libmr).
The final epoch is used; no unknown/test samples participate in model selection.
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

from baseline_wrappers.crosr.utils_crosr import (
    build_model, crosr_api, fit_calibration, load_training, resolve_device,
)
from baseline_wrappers.utils_baselines import DEFAULT_DATA_DIR, discover_classes, record_stage_time


def parse_option():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/crosr")
    parser.add_argument("--classes", nargs="+", help="Subset of no_<class> partitions")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neurons", default="256,256,256", help="Comma-separated hidden widths")
    parser.add_argument("--bottlesize", type=int, default=16, help="Bottleneck width per layer")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=512)
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True,
                        help="Inverse-frequency sampling of known training classes")
    parser.add_argument("--distance", choices=("euclidean", "cosine", "eucos"), default="euclidean")
    parser.add_argument("--eucos-weight", type=float, default=1.0)
    parser.add_argument("--tail-size", type=int, default=50)
    parser.add_argument("--alpha", type=int, default=3,
                        help="Top classes recalibrated by OpenMax; capped at known-class count")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Reject when maximum OpenMax probability is below this value")
    opt = parser.parse_args()
    try:
        opt.neurons = [int(n) for n in opt.neurons.split(",")]
        if len(opt.neurons) < 2 or min(opt.neurons) < 1:
            raise ValueError("neurons must contain at least two positive widths (DHRN ladder)")
        if min(opt.bottlesize, opt.batch_size, opt.chunk_size, opt.epochs, opt.alpha) < 1:
            raise ValueError("bottlesize, batch-size, chunk-size, epochs and alpha must be positive")
        if opt.tail_size < 2:
            raise ValueError("tail-size must be at least two")
        if not 0 <= opt.seed < 2**32:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        if not 0 <= opt.dropout < 1 or not 0 <= opt.threshold <= 1:
            raise ValueError("dropout must be in [0, 1); threshold must be in [0, 1]")
        if not all(np.isfinite(v) for v in (opt.lr, opt.weight_decay, opt.eucos_weight)):
            raise ValueError("optimizer and distance parameters must be finite")
        if opt.lr <= 0 or opt.weight_decay < 0 or opt.eucos_weight < 0:
            raise ValueError("lr must be positive; weight-decay and eucos-weight nonnegative")
        opt.device = resolve_device(opt.device)
        opt.classes = discover_classes(opt.data_dir, opt.classes)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    return opt


def train(opt, model, loader, optimizer, criterion):
    """One epoch using the CROSR API's classification-plus-reconstruction loss."""
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(opt.device), y.to(opt.device)
        z, _, reconstruction = model(x)
        loss = criterion(x, reconstruction, z, y)
        if not T.isfinite(loss):
            raise RuntimeError("CROSR produced a nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


def main():
    opt = parse_option()
    api = crosr_api()
    with record_stage_time(opt.results_dir, "train"):
        print("Training CROSR...")
        directory = opt.results_dir / f"crosr"
        directory.mkdir(parents=True, exist_ok=True)
        for cls in opt.classes:
            random.seed(opt.seed)
            np.random.seed(opt.seed)
            T.manual_seed(opt.seed)
            x, y, preprocessing = load_training(opt.data_dir / f"no_{cls}")
            config = {
                "d_in": x.shape[1], "n_classes": len(preprocessing["classes"]),
                "neurons": opt.neurons, "bottlesize": opt.bottlesize,
                "dropout": opt.dropout, "distance_metric": opt.distance,
                "eucos_weight": opt.eucos_weight,
            }
            model = build_model(config).to(opt.device)
            sampler = None
            if opt.balanced:
                weights = T.bincount(y).double().reciprocal()[y]
                sampler = WeightedRandomSampler(weights, len(y), replacement=True)
            loader = DataLoader(TensorDataset(x, y), batch_size=opt.batch_size,
                                sampler=sampler, shuffle=sampler is None, num_workers=0)
            optimizer = T.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            criterion = api.CROSRLoss_()
            for epoch in range(1, opt.epochs + 1):
                loss = train(opt, model, loader, optimizer, criterion)
                print(f"{cls}: epoch {epoch}/{opt.epochs}, loss={loss:.6f}", flush=True)

            fit_calibration(model, x, y, chunk_size=opt.chunk_size,
                            tail_size=opt.tail_size, classes=preprocessing["classes"])
            # Serialize plain parameters, not C-extension MR objects or modules.
            checkpoint = {
                "format_version": 1, "algorithm": "crosr",
                "model_config": config, "preprocessing": preprocessing,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "centroids": model.centroids.cpu(),
                "weibull_params": [list(mr.get_params()) for mr in model.weibulls],
                "alpha": min(opt.alpha, config["n_classes"]), "threshold": opt.threshold,
                "training": {key: getattr(opt, key) for key in
                             ("seed", "epochs", "lr", "weight_decay", "batch_size", "balanced", "tail_size")},
            }
            path = directory / f"crosr_no_{cls}.pt.tar"
            T.save(checkpoint, path)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
