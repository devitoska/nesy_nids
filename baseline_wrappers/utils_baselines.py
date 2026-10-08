"""Shared partition discovery and result summaries for baseline wrappers."""

import json
from contextlib import contextmanager
from pathlib import Path
import time

import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data/dataset/ton-iot_net"


@contextmanager
def record_stage_time(results_dir, stage):
    """Persist experiment train/test totals, including incomplete/failed runs.

    A new training run resets both totals. Testing preserves its training total.
    Batch execution reads this JSON before appending to results/times.
    """
    path = Path(results_dir) / "timings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if stage == "test" and path.exists():
        timings = json.loads(path.read_text(encoding="utf-8"))
    else:
        timings = {"train_seconds": None, "test_seconds": None}
    timings[f"{stage}_seconds"] = None
    timings["status"] = "running"

    def save():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(timings, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    save()
    started = time.perf_counter()
    try:
        yield timings
    except BaseException:
        timings["status"] = "failed"
        raise
    else:
        timings["status"] = "trained" if stage == "train" else "completed"
    finally:
        timings[f"{stage}_seconds"] = time.perf_counter() - started
        save()


def discover_classes(data_dir, selected=None):
    """Return class suffixes of immediate no_<class> directories, sorted by name."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist or is not a directory: {data_dir}")
    classes = sorted(
        path.name[3:] for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith("no_") and len(path.name) > 3
    )
    if not classes:
        raise ValueError(f"No no_<class> directories found in {data_dir}")
    if selected is not None:
        missing = sorted(set(selected) - set(classes))
        if missing:
            raise ValueError(f"No partition directories for requested classes: {', '.join(missing)}")
        classes = [name for name in classes if name in selected]
    return classes


def create_metrics_table(exp_path):
    """Match the project summary columns while preserving complete class names."""
    metrics_dir = Path(exp_path) / "metrics"
    results = {}
    for partition in sorted(metrics_dir.iterdir()):
        if not partition.is_dir() or not partition.name.startswith("no_") or len(partition.name) <= 3:
            continue
        with (partition / "results.json").open() as handle:
            data = json.load(handle)
        binary = data["classification_report_binary"]
        multi = data["classification_report_multiclass"]
        scores = {
            "auroc": data["roc_auc_score"],
            "aupr": data["auc_score"],
            "recall_pos": binary["1"]["recall"],
            "prec_pos": binary["1"]["precision"],
            "fpr": data["fpr"],
            "f1_pos": binary["1"]["f1-score"],
            "macro_f1_bin": binary["macro avg"]["f1-score"],
            "avg_f1_bin": binary["weighted avg"]["f1-score"],
            "macro_f1_multi": multi["macro avg"]["f1-score"],
            "avg_f1_multi": multi["weighted avg"]["f1-score"],
        }
        results[partition.name[3:]] = {name: round(value, 3) for name, value in scores.items()}
    pd.DataFrame.from_dict(results, orient="index").sort_index().to_csv(metrics_dir / "table.csv")
