"""Train DOC and fit Gaussian rejection thresholds using known training rows only."""

import argparse
import importlib
import pickle
import random
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch as T
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from baseline_wrappers.utils_baselines import DEFAULT_DATA_DIR, discover_classes, record_stage_time


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
    # model and training config
    parser.add_argument("--neurons", default="256,256,256", help="Comma-separated hidden widths")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-4)
    parser.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True,
                        help="Inverse-frequency sampling of known classes during training")

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
    try:
        opt.neurons = [int(n) for n in opt.neurons.split(",")]
        if min(opt.neurons) < 1 or min(opt.epochs, opt.batch_size) < 1:
            raise ValueError("Hidden widths, epochs and batch-size must be positive")
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= opt.dropout < 1:
        parser.error("dropout must be in [0, 1)")
    if not np.isfinite(opt.lr) or opt.lr <= 0:
        parser.error("lr must be positive and finite")
    if not np.isfinite(opt.weight_decay) or opt.weight_decay < 0:
        parser.error("weight-decay must be nonnegative and finite")
    if opt.results_dir is None:
        opt.results_dir = Path(__file__).resolve().parents[2] / "results" / f"doc_{opt.seed}"
    return opt


def train(opt, model, loader, optimizer, criterion):
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(opt.device), y.to(opt.device)
        loss = criterion(model(x), y)
        if not T.isfinite(loss):
            raise RuntimeError("Classifier produced a nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


def main():
    opt = parse_option()
    with record_stage_time(opt.results_dir, "train"):
        print("Training DOC...")
        # Resolve the original implementation's import paths without editing it.
        for old, new in {
            "model": "baseline_wrappers.closr.model",
            "model.model": "baseline_wrappers.closr.model.model",
            "baseline_implementations": "baseline_wrappers",
            "baseline_implementations.common": "baseline_wrappers.common",
        }.items():
            sys.modules.setdefault(old, importlib.import_module(new))
        api = importlib.import_module("baseline_wrappers.doc.doc")
        opt.results_dir.mkdir(parents=True, exist_ok=True)
        for cls in opt.classes:
            random.seed(opt.seed)
            np.random.seed(opt.seed)
            T.manual_seed(opt.seed)
            partition_path = opt.data_dir / f"no_{cls}"
            # Read known training splits only; test data never influences preprocessing.
            frames = [pd.read_csv(partition_path / f"train_{i}_raw_data.csv") for i in (1, 2)]
            if set(frames[0].columns) != set(frames[1].columns) or "class" not in frames[0]:
                raise ValueError("Training CSVs require matching schemas and a class column")
            frame = pd.concat(frames, ignore_index=True)
            if frame["class"].isna().any():
                raise ValueError("Training class labels must not be missing")
            classes = sorted(frame["class"].unique().tolist())
            if cls in classes:
                raise ValueError(f"Held-out attack {cls!r} appears in training data")
            if len(classes) < 2:
                raise ValueError("Training requires at least two known classes")
            if "normal" in classes:
                classes.remove("normal")
                classes.insert(0, "normal")
            features = frame.drop(columns="class").columns.tolist()
            values = frame[features].to_numpy(dtype=np.float64)
            if not len(values) or not features or not np.isfinite(values).all():
                raise ValueError("Features must be nonempty, finite numeric values")
            mean, std = values.mean(axis=0), values.std(axis=0)
            std[std == 0] = 1.0
            # Preserve encoded categorical columns when project metadata is available.
            metadata = partition_path.parent / "categorical_columns.pkl"
            if metadata.exists():
                with metadata.open("rb") as handle:
                    categorical = pickle.load(handle)
                mask = np.isin(features, list(categorical))
                mean[mask], std[mask] = 0.0, 1.0
            preprocessing = {
                "features": features, "classes": classes, "unknown_class": cls,
                "mean": mean.tolist(), "std": std.tolist(),
            }
            values = ((values - mean) / std).astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Preprocessing produced nonfinite features")
            labels = frame["class"].map({name: i for i, name in enumerate(classes)})
            x = T.from_numpy(values)
            y = T.from_numpy(labels.to_numpy(dtype=np.int64))
            config = {
                "d_in": x.shape[1], "d_out": len(preprocessing["classes"]),
                "neurons": opt.neurons, "dropout": opt.dropout,
            }
            model = api.DOC(**config).to(opt.device)
            sampler = None
            if opt.balanced:
                weights = T.bincount(y).double().reciprocal()[y]
                sampler = WeightedRandomSampler(weights, len(y), replacement=True)
            loader = DataLoader(TensorDataset(x, y), batch_size=opt.batch_size,
                                sampler=sampler, shuffle=sampler is None, num_workers=0)
            criterion = api.DOCLoss_()
            optimizer = T.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            for epoch in range(1, opt.epochs + 1):
                loss = train(opt, model, loader, optimizer, criterion)
                print(f"{cls}: epoch {epoch}/{opt.epochs}, loss={loss:.6f}", flush=True)
            model.eval()
            outputs = []
            with T.no_grad():
                for chunk in x.split(opt.chunk_size):
                    output = model(chunk.to(opt.device)).cpu()
                    if not T.isfinite(output).all():
                        raise RuntimeError("Classifier produced nonfinite outputs")
                    outputs.append(output)
            probabilities = T.cat(outputs)
            # Use the array API: the supplied DataLoader fitting branch is broken.
            model.fit_gaussian(x=probabilities, y=y, transform=False)
            if not T.isfinite(model.stds).all() or not T.isfinite(model.thresholds).all():
                raise RuntimeError("DOC produced nonfinite Gaussian calibration")
            directory = opt.results_dir / "doc"
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "format_version": 1, "algorithm": "doc", "seed": opt.seed,
                "model_config": config, "preprocessing": preprocessing,
                "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "stds": model.stds.cpu(), "thresholds": model.thresholds.cpu(),
                "training": {key: getattr(opt, key) for key in
                             ("epochs", "lr", "weight_decay", "batch_size", "balanced")},
            }
            path = directory / f"doc_no_{cls}.pt.tar"
            T.save(checkpoint, path)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
