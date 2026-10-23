"""Data and import adapters for the unchanged CROSR/OpenMax implementations."""

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as T


def crosr_api():
    """Resolve the upstream package names without editing the vendored sources."""
    try:
        import libmr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CROSR requires libmr. Install NumPy and Cython, then run "
            "python -m pip install --no-build-isolation libmr"
        ) from exc
    aliases = {
        "model": "baseline_wrappers.closr.model",
        "model.model": "baseline_wrappers.closr.model.model",
        "baseline_implementations": "baseline_wrappers",
        "baseline_implementations.common": "baseline_wrappers.common",
    }
    for old, new in aliases.items():
        sys.modules.setdefault(old, importlib.import_module(new))
    # CROSR imports OSRModel from the package, but the class lives in its file.
    package = importlib.import_module("baseline_wrappers.openmax")
    package.OSRModel = importlib.import_module(
        "baseline_wrappers.openmax.openmax"
    ).OSRModel
    return importlib.import_module("baseline_wrappers.crosr.crosr")


def build_model(config):
    model = crosr_api().DHRN(**config)
    metric = model.distance_metric

    def distance(a, b, chunk_size=None):
        # fit_weibull supplies chunk_size even for the Euclidean/eucos APIs,
        # which only accept two tensors. These are point-to-centroid distances.
        return metric(a, b)

    model.distance_metric = distance
    return model


def _features(frame, names):
    if set(frame.columns) != set(names + ["class"]):
        raise ValueError("CSV feature schema differs from the training schema")
    values = frame[names].to_numpy(dtype=np.float64)
    if not names or len(frame) == 0 or not np.isfinite(values).all():
        raise ValueError("Features must be nonempty, finite numeric values")
    return values


def transform(frame, preprocessing):
    values = _features(frame, preprocessing["features"])
    values = ((values - np.asarray(preprocessing["mean"]))
              / np.asarray(preprocessing["std"])).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Preprocessing produced nonfinite features")
    names = preprocessing["classes"] + [preprocessing["unknown_class"]]
    labels = frame["class"].map({name: i for i, name in enumerate(names)})
    if labels.isna().any():
        raise ValueError("CSV contains missing or unexpected class labels")
    return T.from_numpy(values), T.from_numpy(labels.to_numpy(dtype=np.int64))


def load_training(partition):
    """Pool the two known-only splits; fit scaling without reading test data.

    Numeric inputs are standardized. If the dataset provides the project's
    categorical_columns.pkl metadata, those encoded columns are left unscaled.
    """
    partition = Path(partition)
    unknown = partition.name.removeprefix("no_")
    first, second = [pd.read_csv(partition / f"train_{i}_raw_data.csv") for i in (1, 2)]
    if set(first.columns) != set(second.columns) or "class" not in first:
        raise ValueError("Training CSVs require matching schemas and a class column")
    frame = pd.concat((first, second), ignore_index=True)
    if frame["class"].isna().any():
        raise ValueError("Training class labels must not be missing")
    classes = sorted(frame["class"].unique().tolist())
    if unknown in classes:
        raise ValueError(f"Held-out attack {unknown!r} appears in training data")
    if len(classes) < 2:
        raise ValueError("CROSR requires at least two known classes")
    if "normal" in classes:
        classes.remove("normal")
        classes.insert(0, "normal")
    features = frame.drop(columns="class").columns.tolist()
    values = _features(frame, features)
    mean, std = values.mean(axis=0), values.std(axis=0)
    std[std == 0] = 1.0
    metadata = partition.parent / "categorical_columns.pkl"
    if metadata.exists():
        with metadata.open("rb") as handle:
            categorical = pickle.load(handle)
        mask = np.isin(features, list(categorical))
        mean[mask], std[mask] = 0.0, 1.0
    preprocessing = {
        "features": features, "classes": classes, "unknown_class": unknown,
        "mean": mean.tolist(), "std": std.tolist(),
    }
    return (*transform(frame, preprocessing), preprocessing)


@T.no_grad()
def extract_features(model, x, chunk_size):
    """Keep the dataset on CPU and transfer one inference batch at a time."""
    device = next(model.parameters()).device
    logits, latents = [], []
    model.eval()
    for chunk in x.split(chunk_size):
        z, latent, _ = model(chunk.to(device))
        logits.append(z.cpu())
        latents.append(latent.cpu())
    z, latent = T.cat(logits), T.cat(latents)
    if not T.isfinite(z).all() or not T.isfinite(latent).all():
        raise RuntimeError("CROSR produced nonfinite features")
    return z, T.cat((z, latent), dim=-1)


def fit_calibration(model, x, y, *, chunk_size, tail_size, classes):
    z, joint = extract_features(model, x, chunk_size)
    correct_counts = T.bincount(y[z.argmax(dim=1) == y], minlength=len(classes))
    missing = [name for name, count in zip(classes, correct_counts) if count < 2]
    if missing:
        raise ValueError(
            "CROSR needs at least two correctly classified training examples per "
            f"class for Weibull fitting; insufficient for {missing}. Train longer "
            "or adjust the model using known data only."
        )
    # CPU tensors are required by libmr and update_probs. Avoid the broken
    # DataLoader branch and use the joint representation from crosr_engine.
    model.fit_weibull(x=z, y=y, latents=joint, nu_val=tail_size)
    if any(mr is None or not mr.is_valid for mr in model.weibulls):
        raise RuntimeError("CROSR Weibull fitting failed; inspect known-class feature tails")


def resolve_device(value):
    if value == "auto":
        value = "cuda" if T.cuda.is_available() else "cpu"
    device = T.device(value)
    if device.type == "cuda" and not T.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return str(device)
