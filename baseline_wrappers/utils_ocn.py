"""Integration helpers for the unmodified scalable-NIDS implementation.

Raw CSVs must contain a ``class`` column and the same feature columns in each
split. Numeric features are not discretized. String features are ordinal-encoded
using the training vocabulary (unseen values become -1). Training-only min/max
scaling is followed by zero-padding to the original CNN's 256 inputs.

The authors' loss, centroid updates, skipped final training batch and last-batch
thresholds are intentionally preserved. Compatibility adapters only provide the
old import/iterator interfaces and convert tensors for sklearn's reporting.
"""

import collections
import collections.abc
from contextlib import contextmanager
from functools import lru_cache
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/dataset/ton-iot_net"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results/ocn"
CLASSES = (
    "backdoor", "ddos", "dos", "injection", "mitm", "password",
    "ransomware", "scanning", "xss",
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _LegacyIterator:
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    next = __next__


class _LegacyLoader:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return _LegacyIterator(iter(self.loader))


def _cpu_classification_report(*args, **kwargs):
    args = tuple(x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in args)
    return classification_report(*args, **kwargs)


@lru_cache(maxsize=1)
def load_ocn():
    """Import the authors' code without editing it or retaining import aliases."""
    source = PROJECT_ROOT / "scalable-NIDS"
    names = ("cnn", "mmd", "cal_metrics")
    saved = {name: sys.modules.get(name) for name in names}
    had_iterable = hasattr(collections, "Iterable")
    try:
        if not had_iterable:
            collections.Iterable = collections.abc.Iterable
        for name in names:
            sys.modules[name] = _load_module(name, source / f"{name}.py")
        ocn = _load_module("_baseline_ocn", source / "main_ocn.py")
    finally:
        if not had_iterable:
            del collections.Iterable
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    original_loader = ocn.get_source_loader

    def compatible_loader(data, labels):
        return _LegacyLoader(original_loader(data, labels))

    ocn.get_source_loader = compatible_loader
    ocn.classification_report = _cpu_classification_report
    return ocn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available; use --device cpu.")
    return device


def read_raw(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw OCN input: {path}")
    frame = pd.read_csv(path)
    if "class" not in frame.columns or frame.empty:
        raise ValueError(f"{path} must have a 'class' column and at least one row.")
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing values; prepare complete raw features first.")
    frame["class"] = frame["class"].astype(str)
    return frame


def _numeric_features(frame, preprocessing):
    names = preprocessing["feature_names"]
    actual = set(frame.columns) - {"class"}
    if actual != set(names):
        raise ValueError(
            f"Feature schema mismatch: missing={sorted(set(names) - actual)}, "
            f"extra={sorted(actual - set(names))}"
        )
    columns = []
    for name in names:
        values = frame[name]
        if values.isna().any():
            raise ValueError(f"Missing values in feature {name!r}.")
        if name in preprocessing["categories"]:
            values = values.astype(str).map(preprocessing["categories"][name]).fillna(-1)
        try:
            columns.append(values.to_numpy(dtype=np.float64))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Feature {name!r} was numeric in training but is not numeric now.") from exc
    values = np.column_stack(columns)
    if not np.isfinite(values).all():
        raise ValueError("OCN input contains nonfinite feature values.")
    return values


def fit_preprocessing(frame):
    names = [name for name in frame.columns if name != "class"]
    if not 1 <= len(names) <= 256:
        raise ValueError(f"The original OCN accepts 1 to 256 features with padding; got {len(names)}.")
    preprocessing = {"feature_names": names, "categories": {}}
    for name in names:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            vocabulary = sorted(frame[name].astype(str).unique())
            preprocessing["categories"][name] = {value: i for i, value in enumerate(vocabulary)}
    scaler = MinMaxScaler().fit(_numeric_features(frame, preprocessing))
    # Plain metadata allows torch.load(weights_only=True), without pickled sklearn objects.
    preprocessing["scale"] = scaler.scale_.tolist()
    preprocessing["offset"] = scaler.min_.tolist()
    return preprocessing


def transform_features(frame, preprocessing):
    values = _numeric_features(frame, preprocessing)
    values = values * np.asarray(preprocessing["scale"]) + np.asarray(preprocessing["offset"])
    padded = np.zeros((len(values), 256), dtype=np.float32)
    padded[:, :values.shape[1]] = values
    if not np.isfinite(padded).all():
        raise ValueError("Features overflowed after scaling to the CNN's float32 input.")
    return padded.reshape(-1, 1, 16, 16)


@contextmanager
def working_directory(path):
    """Isolate the original training function's hard-coded centroid filename."""
    import os

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def check_finite(model, *tensors):
    values = list(model.state_dict().values()) + list(tensors)
    if any(not torch.isfinite(value).all().item() for value in values):
        raise RuntimeError(
            "The original OCN produced nonfinite model values, centroids or thresholds. "
            "No numerical corrections are applied by these wrappers."
        )


def predict(ocn, model, features, centroids, thresholds, batch_size, device):
    """Reproduce the authors' argmax + nearest-centroid rejection rule.

    Standalone inference uses eval mode for all rows, without test labels or
    updating BatchNorm. Scores are the squared distance to the nearest centroid,
    matching the score export in the authors' evaluation routines.
    """
    model.eval()
    predictions, scores = [], []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            inputs = torch.from_numpy(features[start:start + batch_size]).to(device)
            activations, outputs, _, _ = model(inputs, inputs)
            distances, nearest = ocn.cal_min_dis_to_centroid(activations.cpu(), centroids)
            labels = outputs.argmax(dim=1).cpu()
            labels[distances > thresholds[nearest]] = len(centroids)
            predictions.append(labels.numpy())
            scores.append(distances.numpy())
    return np.concatenate(predictions), np.concatenate(scores)


def create_metrics_table(results_dir):
    # Load the project helper by path so both script and `python -m` use work.
    helper = _load_module("_ocn_project_utils", PROJECT_ROOT / "utils.py")
    helper.create_metrics_table(str(results_dir))
