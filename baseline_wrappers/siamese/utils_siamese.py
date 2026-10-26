"""Partition, embedding and bounded-memory KNN helpers for the Siamese API."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch as T

from baseline_wrappers.closr.model.model import ContrastiveMLP
from baseline_wrappers.siamese.siamese_network import distance_matrix


def resolve_device(value):
    if value == "auto":
        value = "cuda" if T.cuda.is_available() else "cpu"
    device = T.device(value)
    if device.type == "cuda" and not T.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return str(device)


def build_model(config):
    return ContrastiveMLP(**config)


def transform(frame, preprocessing):
    features = preprocessing["features"]
    if set(frame.columns) != set(features + ["class"]):
        raise ValueError("CSV feature schema differs from training")
    x = frame[features].to_numpy(dtype=np.float64)
    if not len(x) or not features or not np.isfinite(x).all():
        raise ValueError("Features must be nonempty, finite numeric values")
    x = ((x - np.asarray(preprocessing["mean"]))
         / np.asarray(preprocessing["std"])).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Scaling produced nonfinite features")
    names = preprocessing["classes"] + [preprocessing["unknown_class"]]
    y = frame["class"].map({name: i for i, name in enumerate(names)})
    if y.isna().any():
        raise ValueError("CSV contains missing or unexpected class labels")
    return T.from_numpy(x), T.from_numpy(y.to_numpy(dtype=np.int64))


def load_training(partition, *, benign_class, calibration_fraction, seed):
    """Split known rows before fitting scaling; never read the test CSV.

    Each class contributes at least one fitting and one calibration row. Encoded
    categorical columns are left unscaled when project metadata is available.
    """
    partition = Path(partition)
    unknown = partition.name.removeprefix("no_")
    frames = [pd.read_csv(partition / f"train_{i}_raw_data.csv") for i in (1, 2)]
    if set(frames[0].columns) != set(frames[1].columns) or "class" not in frames[0]:
        raise ValueError("Training CSVs require matching schemas and a class column")
    frame = pd.concat(frames, ignore_index=True)
    if frame["class"].isna().any():
        raise ValueError("Missing training labels")
    classes = sorted(frame["class"].unique().tolist())
    if unknown in classes:
        raise ValueError(f"Held-out attack {unknown!r} appears in training data")
    if benign_class not in classes or len(classes) < 2:
        raise ValueError(f"Training requires benign class {benign_class!r} and known attacks")
    classes.remove(benign_class)
    classes.insert(0, benign_class)  # OnlineContrastiveLoss collapses all y > 0.
    rng = np.random.default_rng(seed)
    fit_indices, calibration_indices = [], []
    for name in classes:
        indices = rng.permutation(np.flatnonzero(frame["class"].to_numpy() == name))
        if len(indices) < 2:
            raise ValueError(f"Class {name!r} needs at least two rows for calibration split")
        n_cal = min(len(indices) - 1, max(1, int(np.ceil(len(indices) * calibration_fraction))))
        calibration_indices.extend(indices[:n_cal])
        fit_indices.extend(indices[n_cal:])
    fit = frame.iloc[fit_indices]
    calibration = frame.iloc[calibration_indices]
    features = frame.drop(columns="class").columns.tolist()
    values = fit[features].to_numpy(dtype=np.float64)
    if not features or not np.isfinite(values).all():
        raise ValueError("Training requires finite numeric features")
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
    return (*transform(fit, preprocessing), *transform(calibration, preprocessing), preprocessing)


@T.no_grad()
def embed(model, x, chunk_size):
    model.eval()
    device = next(model.parameters()).device
    features = T.cat([model(chunk.to(device)).cpu() for chunk in x.split(chunk_size)])
    if not T.isfinite(features).all():
        raise RuntimeError("Siamese network produced nonfinite embeddings")
    return features


@T.no_grad()
def class_distances(reference, labels, query, *, k, metric, chunk_size, reference_chunk_size):
    """Exact class_knn_distances semantics with both tensor axes chunked on CPU.

    Use the supplied distance_matrix API and merge nearest neighbors across
    reference blocks. Means use min(k, class count), as in the original API.
    Columns follow sorted labels, which the wrappers keep contiguous from zero.
    """
    scores = []
    for cls in T.unique(labels, sorted=True):
        class_reference = reference[labels == cls]
        n_neighbors = min(k, len(class_reference))
        class_scores = []
        for query_chunk in query.split(chunk_size):
            nearest = T.empty((len(query_chunk), 0), dtype=query.dtype)
            for reference_chunk in class_reference.split(reference_chunk_size):
                distances = distance_matrix(query_chunk, reference_chunk, metric)
                candidates = T.cat((nearest, distances), dim=1)
                nearest = candidates.topk(min(n_neighbors, candidates.shape[1]),
                                          largest=False, sorted=False).values
            class_scores.append(nearest.mean(dim=1))
        scores.append(T.cat(class_scores))
    result = T.stack(scores, dim=1)
    if not T.isfinite(result).all():
        raise RuntimeError("Siamese KNN produced nonfinite distances")
    return result


def reference_indices(labels, max_per_class, seed):
    """Optionally cap references per class without using calibration/test rows."""
    if max_per_class == 0:
        return T.arange(len(labels))
    generator = T.Generator().manual_seed(seed)
    indices = []
    for cls in T.unique(labels, sorted=True):
        candidates = T.where(labels == cls)[0]
        indices.append(candidates[T.randperm(len(candidates), generator=generator)[:max_per_class]])
    return T.cat(indices)
