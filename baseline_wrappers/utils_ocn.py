"""Integration helpers for the unmodified scalable-NIDS implementation.

Raw CSVs must contain a ``class`` column and the same feature columns in each
split. Numeric features are not discretized. String features are ordinal-encoded
using the training vocabulary (unseen values become -1). Training-only min/max
scaling produces (N, 1, F) inputs with no added feature values.

The wrapper implements the KDD 1D backbone from Section 4.2 of Zhang et al.
(2021): 16/32 filters, convolution kernel 3/stride 1, pooling size 4/stride 2.
The paper does not specify boundary handling. We use no convolution padding and
ceil-mode pooling to retain the final partial windows without zero-padding.

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

if __package__:
    from .utils_baselines import DEFAULT_DATA_DIR, create_metrics_table, discover_classes, record_stage_time
else:
    from utils_baselines import DEFAULT_DATA_DIR, create_metrics_table, discover_classes, record_stage_time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results/ocn"
CHECKPOINT_VERSION = 2


def model_config(num_features):
    """Serializable architecture specification; version 1 checkpoints were 2D."""
    if type(num_features) is not int or num_features < 1:
        raise ValueError("OCN requires a positive integer feature count.")
    return {
        "name": "ocn_kdd_1d",
        "num_features": num_features,
        "channels": [16, 32],
        "conv_kernel_size": 3,
        "conv_stride": 1,
        "conv_padding": 0,
        "pool_kernel_size": 4,
        "pool_stride": 2,
        "pool_padding": 0,
        "pool_ceil_mode": True,
        "head": "batchnorm_linear_sigmoid",
    }


class SharedCNN1D(torch.nn.Module):
    """Shared 1D backbone with the four-output interface used by original OCN.

    BatchNorm and sigmoid preserve the released code's head behavior. The extra
    singleton axis accepted here only accommodates its 4D noise-generation code;
    convolution always runs on (batch, 1, features).
    """

    def __init__(self, config, num_classes):
        super().__init__()
        self.num_features = config["num_features"]
        blocks = []
        in_channels = 1
        for out_channels in config["channels"]:
            blocks.extend([
                torch.nn.Conv1d(
                    in_channels, out_channels, config["conv_kernel_size"],
                    stride=config["conv_stride"], padding=config["conv_padding"],
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool1d(
                    config["pool_kernel_size"], stride=config["pool_stride"],
                    padding=config["pool_padding"], ceil_mode=config["pool_ceil_mode"],
                ),
            ])
            in_channels = out_channels
        self.features = torch.nn.Sequential(*blocks)
        try:
            with torch.no_grad():
                output = self.features(torch.zeros(1, 1, self.num_features))
        except RuntimeError as exc:
            raise ValueError(
                f"{self.num_features} features are too few for the unpadded 1D OCN "
                "convolution/pooling stack (minimum 13)."
            ) from exc
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(output.numel()),
            torch.nn.Linear(output.numel(), num_classes),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def _forward_one(self, inputs):
        if inputs.ndim == 4 and inputs.shape[2] == 1:
            inputs = inputs.squeeze(2)
        if inputs.ndim != 3 or inputs.shape[1:] != (1, self.num_features):
            raise ValueError(
                f"Expected (N, 1, {self.num_features}) or "
                f"(N, 1, 1, {self.num_features}), got {tuple(inputs.shape)}."
            )
        activations = self.fc(self.features(inputs))
        return activations, self.sigmoid(activations)

    def forward(self, indata, outdata):
        in_activations, in_predictions = self._forward_one(indata)
        out_activations, out_predictions = self._forward_one(outdata)
        return in_activations, in_predictions, out_activations, out_predictions


def build_model(config, num_classes):
    if config != model_config(config.get("num_features")):
        raise ValueError("Unsupported OCN architecture metadata; retrain with train_ocn.py.")
    return SharedCNN1D(config, num_classes)


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
    if len(names) < 13:
        raise ValueError(f"The unpadded 1D OCN requires at least 13 features; got {len(names)}.")
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
    values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Features overflowed after scaling to the CNN's float32 input.")
    return values[:, np.newaxis, :]


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
