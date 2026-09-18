"""Summarize runs named <config>_<seed>: python report_batch.py results.

The independent observations are seeds, not held-out attack classes. Standard
deviations use ddof=1; 95% confidence intervals use Student's t distribution.
For the known [0, 1] metrics, intervals are intersected with [0, 1]. This
restriction preserves coverage of a bounded mean but does not fix the t
approximation with few or non-normal seed results. Count intervals are unbounded.
For one seed, uncertainty is undefined and is written as empty CSV cells.

Also writes recall_precision_f1_unknown.png (unknown recall/precision/F1),
bin_multi_f1.png (weighted binary/multiclass F1), and
unknown_mis.png (unknown misclassification counts). Figure cells show
mean and ± sample SD. Metric plots use a superscript ? for SD > 0.1 in
original units (10 percentage points). Count means are rounded to integers
for display and never have superscripts. Yellow bold marks row maxima for
metrics and column minima for counts, including ties before display rounding.
If an efc/ subfolder exists, include its fixed baseline in every comparison
and write metrics_efc.csv and unknown_mis_efc.csv with values only (no SD/CI).
Use --no-plots for CSV only.
Comparison labels use the saved config's explanation settings: Type <number>,
with optional u (unobserved) and m (misclassified) suffixes, in that order.
Append <score>@<rejection_rate>, defaulting to mse@0.01 as in training.
"""

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import t
import yaml


METRIC_PATHS = {
    "auroc": ("roc_auc_score",),
    "aupr": ("auc_score",),
    "recall_pos": ("classification_report_binary", "1", "recall"),
    "prec_pos": ("classification_report_binary", "1", "precision"),
    "fpr": ("fpr",),
    "f1_pos": ("classification_report_binary", "1", "f1-score"),
    "macro_f1_bin": ("classification_report_binary", "macro avg", "f1-score"),
    "avg_f1_bin": ("classification_report_binary", "weighted avg", "f1-score"),
    "macro_f1_multi": ("classification_report_multiclass", "macro avg", "f1-score"),
    "avg_f1_multi": ("classification_report_multiclass", "weighted avg", "f1-score"),
}
STATISTICS = ("mean", "std", "ci95_lower", "ci95_upper")
PLOT_METRICS = (
    ("recall_pos", "Unknown-class recall (%)"),
    ("prec_pos", "Unknown-class precision (%)"),
    ("f1_pos", "Unknown-class F1 (%)"),
    ("avg_f1_bin", "Weighted binary F1 (%)"),
    ("avg_f1_multi", "Weighted multiclass F1 (%)"),
)


def summarize(values):
    """Return statistics along the first (seed) axis, without rounding/clipping."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    if len(values) == 1:
        std = np.full_like(mean, np.nan)
        margin = std
    else:
        std = values.std(axis=0, ddof=1)
        margin = t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))
    return np.stack([mean, std, mean - margin, mean + margin])


def read_results(experiment):
    results = {}
    for path in sorted((experiment / "metrics").glob("no_*/results.json")):
        with path.open(encoding="utf-8") as stream:
            results[path.parent.name.removeprefix("no_")] = json.load(stream)
    if not results:
        raise ValueError(f"{experiment}: no metrics/no_*/results.json files")
    return results


def read_metrics(experiment, results):
    # Support both the requested metrics.csv name and this project's table.csv.
    candidates = (
        experiment / "metrics.csv",
        experiment / "metrics" / "metrics.csv",
        experiment / "metrics" / "table.csv",
    )
    path = next((path for path in candidates if path.is_file()), None)
    if path is not None:
        table = pd.read_csv(path, index_col=0)
    else:
        rows = {}
        for unknown, result in results.items():
            row = {}
            for metric, keys in METRIC_PATHS.items():
                value = result
                for key in keys:
                    value = value[key]
                row[metric] = value
            rows[unknown] = row
        table = pd.DataFrame.from_dict(rows, orient="index")
    if table.index.has_duplicates or table.empty or not len(table.columns):
        raise ValueError(f"{experiment}: empty metric table or duplicate attack rows")
    if set(table.index) != set(results):
        raise ValueError(f"{experiment}: metric rows do not match held-out results")
    table = table.astype(float).sort_index()
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError(f"{experiment}: metrics must all be finite numbers")
    bounded = table.loc[:, table.columns.isin(METRIC_PATHS)]
    if ((bounded < 0) | (bounded > 1)).to_numpy().any():
        raise ValueError(f"{experiment}: known metrics must be between 0 and 1")
    return table


def unknown_counts(experiment, results):
    """Sum off-diagonal unknown-class rows across this seed's held-out runs."""
    counts = {}
    for unknown, result in results.items():
        report = result["classification_report_multiclass"]
        # sklearn's default confusion_matrix and classification_report use the
        # same sorted labels. Report summary entries do not have precision.
        labels = sorted(
            label for label, scores in report.items()
            if isinstance(scores, dict) and "precision" in scores
            and label not in {"macro avg", "weighted avg", "micro avg", "samples avg"}
        )
        matrix = np.asarray(result["confusion_matrix_multiclass"], dtype=float)
        if (matrix.shape != (len(labels), len(labels))
                or unknown not in labels or "normal" not in labels
                or unknown == "normal" or "attack" in labels or "total" in labels):
            raise ValueError(f"{experiment}/metrics/no_{unknown}: invalid confusion-matrix labels")
        if (not np.isfinite(matrix).all() or (matrix < 0).any()
                or (matrix != np.floor(matrix)).any()):
            raise ValueError(f"{experiment}/metrics/no_{unknown}: invalid confusion-matrix counts")
        row = matrix[labels.index(unknown)]
        if row.sum() != report[unknown]["support"]:
            raise ValueError(f"{experiment}/metrics/no_{unknown}: matrix and report support disagree")
        for label, count in zip(labels, row):
            counts.setdefault(label, 0.0)
            if label != unknown:
                counts[label] += count
    attacks = sorted(set(counts) - {"normal"})
    attack_total = sum(counts[label] for label in attacks)
    return pd.Series({
        "normal": counts["normal"],
        "attack": attack_total,
        **{label: counts[label] for label in attacks},
        "total": counts["normal"] + attack_total,
    })


def discover_experiments(results_dir):
    groups = {}
    for experiment in sorted(results_dir.iterdir()):
        if not experiment.is_dir() or experiment.name == "efc":
            continue
        config_path = next((experiment / name for name in ("config.yaml", "config.yml")
                            if (experiment / name).is_file()), None)
        if config_path is None:
            continue
        match = re.fullmatch(r"(.+)_(\d+)", experiment.name)
        if match is None:
            print(f"Warning: skipping {experiment}: expected <config>_<seed> name", file=sys.stderr)
            continue
        if not (experiment / "metrics").is_dir():
            print(f"Warning: skipping {experiment}: no evaluation metrics yet", file=sys.stderr)
            continue
        config_name, seed_text = match.groups()
        seed = int(seed_text)
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError(f"{config_path}: expected a YAML mapping")
        saved_seed = config.pop("seed", seed)
        if saved_seed != seed:
            raise ValueError(f"{experiment}: directory seed differs from saved config seed")
        group = groups.setdefault(config_name, {"config": config, "seeds": set(), "runs": []})
        if config != group["config"]:
            raise ValueError(f"{experiment}: settings differ within config {config_name} (excluding seed)")
        if seed in group["seeds"]:
            raise ValueError(f"{experiment}: duplicate seed {seed} for config {config_name}")
        group["seeds"].add(seed)
        group["runs"].append(experiment)
    efc = results_dir / "efc"
    if efc.is_dir():
        if "efc" in groups:
            raise ValueError("Cannot combine seeded efc runs with the fixed efc/ baseline")
        groups["efc"] = {"config": {}, "runs": [efc], "baseline": True}
    if not groups:
        raise ValueError(f"{results_dir}: no evaluated <config>_<seed> experiments or efc baseline found")
    return groups


def build_reports(runs):
    metric_tables, count_tables = [], []
    for experiment in runs:
        results = read_results(experiment)
        metrics = read_metrics(experiment, results)
        counts = unknown_counts(experiment, results)
        if metric_tables:
            reference = metric_tables[0]
            if (not metrics.index.equals(reference.index)
                    or set(metrics.columns) != set(reference.columns)
                    or set(counts.index) != set(count_tables[0].index)):
                raise ValueError(f"{experiment}: seeds have different held-out classes or metrics")
            metrics = metrics.reindex(columns=reference.columns)
            counts = counts.reindex(count_tables[0].index)
        metric_tables.append(metrics)
        count_tables.append(counts)

    metric_stats = summarize(np.stack([table.to_numpy() for table in metric_tables]))
    # Intersect only CI endpoints with the known parameter domain. Keep means,
    # standard deviations, custom metric columns and count statistics intact.
    bounded = metric_tables[0].columns.isin(METRIC_PATHS)
    metric_stats[2:, :, bounded] = np.clip(metric_stats[2:, :, bounded], 0, 1)
    index = pd.MultiIndex.from_product(
        [metric_tables[0].index, STATISTICS], names=["unknown_class", "statistic"],
    )
    metrics_report = pd.DataFrame(
        metric_stats.transpose(1, 0, 2).reshape(len(index), -1),
        index=index, columns=metric_tables[0].columns,
    )
    counts_report = pd.DataFrame(
        summarize(np.stack([table.to_numpy() for table in count_tables])),
        index=pd.Index(STATISTICS, name="statistic"), columns=count_tables[0].index,
    )
    return metrics_report, counts_report


def build_baseline_reports(experiment):
    """Keep the single EFC evaluation as values, with no uncertainty estimates."""
    results = read_results(experiment)
    metrics = read_metrics(experiment, results)
    metrics.index = pd.MultiIndex.from_arrays(
        [metrics.index, ["mean"] * len(metrics)], names=["unknown_class", "statistic"],
    )
    counts = unknown_counts(experiment, results).to_frame().T
    counts.index = pd.Index(["mean"], name="statistic")
    return metrics, counts


def build_comparison(groups, reports):
    """Align configs and compute Average rows within seeds before summarizing."""
    comparison = []
    required = [metric for metric, _ in PLOT_METRICS]
    reference_classes = None
    reference_counts = None
    for name, group in groups.items():
        metrics, counts = reports[name]
        baseline = group.get("baseline", False)
        means = metrics.xs("mean", level="statistic")
        stds = (pd.DataFrame(np.nan, index=means.index, columns=means.columns)
                if baseline else metrics.xs("std", level="statistic"))
        missing = set(required) - set(means.columns)
        if missing:
            raise ValueError(f"{name}: missing figure metrics {sorted(missing)}; use --no-plots for CSV only")
        if reference_classes is None:
            reference_classes = means.index
            reference_counts = counts.columns
        if (set(means.index) != set(reference_classes)
                or set(counts.columns) != set(reference_counts)):
            raise ValueError("Figures require the same held-out and predicted classes across configurations")
        means = means.reindex(reference_classes)[required].copy()
        stds = stds.reindex(reference_classes)[required].copy()
        if baseline:
            means.loc["Average"] = means.mean(axis=0)
            stds.loc["Average"] = np.nan
        else:
            per_seed_averages = [
                read_metrics(run, read_results(run))[required].mean(axis=0).to_numpy()
                for run in group["runs"]
            ]
            average_stats = summarize(per_seed_averages)
            means.loc["Average"] = average_stats[0]
            stds.loc["Average"] = average_stats[1]
        anomaly_detection = group["config"].get("anomaly_detection", {})
        explanations = anomaly_detection.get("explanations", {})
        label = "EFC" if baseline else f"Type {explanations.get('type', 1)}"
        if not baseline:
            if explanations.get("unobserved", False):
                label += " u"
            if explanations.get("misclassified", False):
                label += " m"
            score = anomaly_detection.get("score", "mse")
            rejection_rate = anomaly_detection.get("rejection_rate", 0.01)
            label += f" {score}@{rejection_rate}"
        comparison.append({
            "name": name, "label": label, "n": len(group["runs"]),
            "baseline": baseline,
            "means": means, "stds": stds,
            "counts": counts.reindex(index=["mean", "std"], columns=reference_counts),
        })
    return comparison


def draw_heatmap(ax, means, stds, row_labels, column_labels, *, percentages, show_std=None):
    """Draw annotated mean cells; SD and its flag always use original units."""
    import matplotlib.patheffects as path_effects

    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    show_std = np.broadcast_to(True if show_std is None else show_std, means.shape)
    vmax = 1 if percentages else max(float(means.max()), 1)
    heatmap = ax.imshow(means, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(column_labels)), column_labels,
                  rotation=0 if percentages else 32,
                  ha="center" if percentages else "right", fontsize=11)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=11)
    ax.tick_params(axis="both", length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    best = means.max(axis=1, keepdims=True) if percentages else means.min(axis=0, keepdims=True)
    highlights = np.isclose(means, best, rtol=0, atol=1e-12)
    for row, column in np.ndindex(means.shape):
        mean, std = means[row, column], stds[row, column]
        scale = 100 if percentages else 1
        mean_text = f"{mean * scale:.2f}" if percentages else f"{mean:,.0f}"
        std_text = (f"{std * scale:.2f}" if percentages else f"{std:,.0f}") if np.isfinite(std) else "n/a"
        # Luminance gives legible text on both the light and dark blue cells.
        red, green, blue, _ = heatmap.cmap(heatmap.norm(mean))
        color = "#172b40" if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.55 else "white"
        highlight = bool(highlights[row, column])
        effects = [path_effects.withStroke(linewidth=1.8, foreground="#24384d")] if highlight else []
        if highlight:
            color = "#ffe600"
        mean_y = row - 0.13 if show_std[row, column] else row
        text = ax.text(column, mean_y, mean_text, ha="center", va="center",
                       fontsize=12, fontweight="bold" if highlight else "normal",
                       color=color, path_effects=effects)
        if percentages and show_std[row, column] and np.isfinite(std) and std > 0.1:
            # Place the superscript after the rendered number, preserving the
            # number's centering, weight and comma formatting.
            from matplotlib.offsetbox import AnnotationBbox, TextArea
            marker = TextArea("?", textprops={"fontsize": 8, "color": color,
                                            "fontweight": text.get_fontweight(),
                                            "path_effects": effects})
            width = ax.figure.canvas.get_renderer().get_text_width_height_descent(
                mean_text, text.get_fontproperties(), ismath=False,
            )[0]
            marker_offset = width / ax.figure.dpi * 72 / 2 + 1
            ax.add_artist(AnnotationBbox(marker, (column, mean_y),
                                        xybox=(marker_offset, 4.5), boxcoords="offset points",
                                        box_alignment=(0, 0.5), frameon=False))
        if show_std[row, column]:
            ax.text(column, row + 0.23, f"± {std_text}", ha="center", va="center",
                    fontsize=8.5, color=color, path_effects=effects)
    if percentages:
        ax.axhline(len(row_labels) - 1.5, color="#536273", linewidth=1.2)
        ax.get_yticklabels()[-1].set_fontweight("bold")


def comparison_figures(comparison):
    """Create publication figures with a headless canvas; no GUI is required."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    labels = [item["label"] if item["baseline"] else f"{item['label']}\n(n={item['n']})"
              for item in comparison]
    show_std = np.array([not item["baseline"] for item in comparison])
    baseline_note = " EFC: single evaluation, no SD/CI." if not show_std.all() else ""
    rows = comparison[0]["means"].index.tolist()
    figures = []
    for filename, panels in (("recall_precision_f1_unknown.png", PLOT_METRICS[:3]),
                             ("bin_multi_f1.png", PLOT_METRICS[3:])):
        width = len(panels) * max(7, len(comparison) * 1.4 + 1.8)
        figure = Figure(figsize=(width, max(8, len(rows) * 0.66 + 2)), dpi=160)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, len(panels), squeeze=False)[0]
        figure.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.17, wspace=0.36)
        title = "Explanation representations" + (" and EFC" if not show_std.all() else "")
        figure.suptitle(f"{title}: {len(rows) - 1} held-out attacks",
                       fontsize=21, fontweight="bold", y=0.98)
        for ax, (metric, title) in zip(axes, panels):
            means = np.column_stack([item["means"][metric] for item in comparison])
            stds = np.column_stack([item["stds"][metric] for item in comparison])
            draw_heatmap(ax, means, stds, rows, labels, percentages=True, show_std=show_std)
            ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        figure.text(0.5, 0.04,
                    "Cells: mean (%) and ± SD (percentage points). Superscript ?: SD > 10 percentage points."
                    + baseline_note + "\n"
                    "Average: equal-weight mean across held-out attacks within each seed. Yellow bold: row maximum, including ties, before rounding.",
                    ha="center", fontsize=10, color="#555555", linespacing=1.6)
        figures.append((filename, figure))

    columns = comparison[0]["counts"].columns.tolist()
    figure = Figure(figsize=(max(18, len(columns) * 1.6), max(5, len(comparison) * 0.9 + 2.8)), dpi=160)
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    figure.subplots_adjust(left=0.08, right=0.99, top=0.83, bottom=0.34)
    figure.suptitle("Unknown attacks misclassified as known classes", fontsize=21,
                   fontweight="bold", y=0.98)
    means = np.stack([item["counts"].loc["mean"].to_numpy() for item in comparison])
    stds = np.stack([item["counts"].loc["std"].to_numpy() for item in comparison])
    draw_heatmap(ax, means, stds, labels, columns, percentages=False, show_std=show_std[:, None])
    ax.set_xlabel("Predicted class", fontsize=12, labelpad=14)
    figure.text(0.5, 0.035,
                "Cells: mean count (rounded to integer) and ± SD across seeds, after summing held-out experiments."
                + baseline_note + "\n"
                "attack = sum of attack-class columns; total = normal + attack. Correct unknown detections excluded. Yellow bold: column minimum, including ties.",
                ha="center", fontsize=10, color="#555555", linespacing=1.6)
    figures.append(("unknown_mis.png", figure))
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_folder", type=Path, help="Folder containing <config>_<seed> runs")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV summaries only")
    args = parser.parse_args(argv)
    if not args.results_folder.is_dir():
        parser.error(f"Not a directory: {args.results_folder}")
    try:
        groups = discover_experiments(args.results_folder)
        # Validate all groups before replacing any existing report.
        reports = {
            name: (build_baseline_reports(group["runs"][0]) if group.get("baseline")
                   else build_reports(group["runs"]))
            for name, group in groups.items()
        }
        comparison = build_comparison(groups, reports) if not args.no_plots else None
        for name, (metrics, counts) in reports.items():
            metrics_path = args.results_folder / f"metrics_{name}.csv"
            counts_path = args.results_folder / f"unknown_mis_{name}.csv"
            metrics.to_csv(metrics_path, na_rep="", float_format="%.10g")
            counts.to_csv(counts_path, na_rep="", float_format="%.10g")
            sample = "fixed baseline" if groups[name].get("baseline") else f"{len(groups[name]['runs'])} seeds"
            print(f"{name}: {sample} -> {metrics_path}, {counts_path}")
        if comparison is not None:
            for filename, figure in comparison_figures(comparison):
                path = args.results_folder / filename
                figure.savefig(path, facecolor="white")
                figure.clear()
                print(f"Figure -> {path}")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
