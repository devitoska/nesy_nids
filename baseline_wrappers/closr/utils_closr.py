"""Adapters for project partitions and device handling; CLOSR core stays intact."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from baseline_wrappers.closr.data.utils import ZNormaliser
from baseline_wrappers.closr.util.features import get_features as _get_features


def load_partition(data_path, *, include_test=True, preprocessing=None):
    """Load existing train/test splits, with contiguous known labels and one unknown."""
    data_path = Path(data_path)
    if not data_path.name.startswith("no_"):
        raise ValueError(f"Expected a no_<class> partition: {data_path}")
    unknown_class = data_path.name[3:]
    frames = [pd.read_csv(data_path / f"train_{i}_raw_data.csv") for i in (1, 2)]
    if set(frames[0].columns) != set(frames[1].columns):
        raise ValueError(f"Training feature schemas differ in {data_path}")
    frame = pd.concat(frames, ignore_index=True)
    if "class" not in frame or len(frame) < 2:
        raise ValueError("Training requires a class column and at least two rows")
    classes = sorted(frame["class"].unique().tolist())
    if unknown_class in classes:
        raise ValueError(f"Held-out class {unknown_class!r} appears in training data")
    if "normal" in classes:
        classes.remove("normal")
        classes.insert(0, "normal")
    if len(classes) < 2:
        raise ValueError("CLOSR requires at least two known training classes")
    features = frame.drop(columns="class").columns.tolist()
    if preprocessing is not None:
        if (preprocessing["classes"] != classes
                or preprocessing["unknown_class"] != unknown_class
                or set(preprocessing["features"]) != set(features)):
            raise ValueError("Partition does not match checkpoint classes or features")
        features = preprocessing["features"]
    x_train = frame[features].to_numpy(dtype=np.float64)
    if not features or not np.isfinite(x_train).all():
        raise ValueError("Training features must be nonempty, finite numeric values")
    norm = ZNormaliser()
    if preprocessing is None:
        norm.fit(x_train)
        # Constant columns in project CSVs must not cause division by zero.
        norm.std = np.where(norm.std == 0, 1.0, norm.std)
        with (data_path.parent / "categorical_columns.pkl").open("rb") as f:
            categorical_columns = pickle.load(f)
        categorical_mask = np.isin(features, list(categorical_columns))
        norm.mean[categorical_mask] = 0.0
        norm.std[categorical_mask] = 1.0
        preprocessing = {
            "classes": classes, "unknown_class": unknown_class, "features": features,
            "mean": norm.mean.tolist(), "std": norm.std.tolist(),
        }
    else:
        norm.mean = np.asarray(preprocessing["mean"])
        norm.std = np.asarray(preprocessing["std"])
    mapping = {name: index for index, name in enumerate(classes + [unknown_class])}
    x_train = norm.normalise(x_train).astype(np.float32)
    y_train = frame["class"].map(mapping).to_numpy(dtype=np.int64)
    if not include_test:
        return x_train, y_train, None, None, None, None, preprocessing

    test = pd.read_csv(data_path / "test_raw_data.csv")
    if set(test.columns) != set(features + ["class"]):
        raise ValueError("Test feature schema differs from training")
    unexpected = set(test["class"]) - set(mapping)
    if unexpected:
        raise ValueError(f"Unexpected test classes: {sorted(unexpected)}")
    x_test = test[features].to_numpy(dtype=np.float64)
    if not np.isfinite(x_test).all():
        raise ValueError("Test features must be finite numeric values")
    x_test = norm.normalise(x_test).astype(np.float32)
    y_test = test["class"].map(mapping).to_numpy(dtype=np.int64)
    unknown = y_test == len(classes)
    if not unknown.any() or unknown.all():
        raise ValueError("Evaluation requires both known and held-out test examples")
    # Disjoint subsets: closr_eval concatenates these exactly once.
    return (x_train, y_train, x_test[~unknown], y_test[~unknown],
            x_test[unknown], y_test[unknown], preprocessing)


def get_features(model, x_data, y_data, **kwargs):
    """Transfer each input chunk without changing CLOSR's feature extractor."""
    device = next(model.parameters(), x_data).device
    return _get_features(
        lambda chunk: model(chunk.to(device)), x_data, y_data, **kwargs,
    )
