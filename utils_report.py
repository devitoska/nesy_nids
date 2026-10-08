"""Statistics and timing figures shared by batch reports."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import t


TRAINING_PHASES = (
    ("bn_seconds", "Bayesian network", "#4477AA"),
    ("explanations_seconds", "Explanations", "#228833"),
    ("ad_seconds", "Anomaly detection", "#EEAA33"),
    ("other_seconds", "Other / overhead", "#BBBBBB"),
)
ALGORITHM_COLORS = {"NeSy-NIDS": "#4477AA", "EFC": "#AA3377", "OCN": "#CC6677"}


def summarize(values):
    """Return mean, sample SD and t-based CI along the first (seed) axis."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    if len(values) == 1:
        std = np.full_like(mean, np.nan)
        margin = std
    else:
        std = values.std(axis=0, ddof=1)
        margin = t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))
    return np.stack([mean, std, mean - margin, mean + margin])


def read_timing_summary(path, groups):
    """Read the batch TSV, keeping one observation per config/seed.

    Repeated experiment rows represent reruns: the last row wins. EFC uses
    only its last recorded evaluation, with no averaging or uncertainty.
    Null NeSy phase durations mean skipped phases (zero elapsed time).
    Only evaluated experiment groups supplied by the caller are included.
    """
    path = Path(path)
    table = pd.read_csv(path, sep="\t", dtype={"experiment": str})
    phases = [key for key, _, _ in TRAINING_PHASES[:-1]]
    required = ["experiment", "train_seconds", "test_seconds", *phases]
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing timing columns {sorted(missing)}")
    if table.empty or table["experiment"].isna().any():
        raise ValueError(f"{path}: empty timings or missing experiment names")
    table = table.drop_duplicates("experiment", keep="last")
    summaries = []
    for name, group in groups.items():
        algorithm = group["config"].get("algorithm", {})
        kind = "EFC" if group.get("baseline") else algorithm.get("type", "NeSy-NIDS")
        kind = {"nesy-nids": "NeSy-NIDS", "efc": "EFC", "ocn": "OCN"}.get(str(kind).casefold(), kind)
        if kind not in ALGORITHM_COLORS:
            raise ValueError(f"{name}: unknown timing algorithm {kind!r}")
        run_names = {Path(run).name for run in group["runs"]}
        rows = table.loc[table["experiment"].isin(run_names)].copy()
        if rows.empty:
            # The historical fixed efc/ metrics baseline predates batch timing.
            if group.get("baseline"):
                continue
            raise ValueError(f"{path}: no timings for config {name}")
        if kind == "EFC":
            rows = rows.tail(1)
        else:
            absent = run_names - set(rows["experiment"])
            if absent:
                raise ValueError(f"{path}: missing timings for {sorted(absent)}")
            seeds = [int(re.fullmatch(r"(.+)_(\d+)", value).group(2)) for value in rows["experiment"]]
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{path}: duplicate seeds for config {name}")
        columns = ["train_seconds", "test_seconds"] + (phases if kind == "NeSy-NIDS" else [])
        values = rows[columns].apply(pd.to_numeric, errors="raise")
        if kind == "NeSy-NIDS":
            values[phases] = values[phases].fillna(0)
        if not np.isfinite(values.to_numpy()).all() or (values < 0).to_numpy().any():
            raise ValueError(f"{path}: {name} durations must be finite, nonnegative seconds")
        if kind == "NeSy-NIDS":
            remainder = values["train_seconds"] - values[phases].sum(axis=1)
            # TSV durations are rounded independently to six decimal places.
            tolerance = 2e-6 + 1e-12 * values["train_seconds"]
            if (remainder < -tolerance).any():
                raise ValueError(f"{path}: {name} training phases exceed the training total")
            values["other_seconds"] = remainder.clip(lower=0)
        if kind == "EFC":
            means = values.iloc[0]
            stds = pd.Series(np.nan, index=values.columns)
        else:
            stats = summarize(values.to_numpy())
            means = pd.Series(stats[0], index=values.columns)
            stds = pd.Series(stats[1], index=values.columns)
        summaries.append({"name": name, "algorithm": kind, "n": len(values),
                          "means": means, "stds": stds})
    if not summaries:
        raise ValueError(f"{path}: no timings matching evaluated configurations")
    return summaries


def format_seconds(mean, std, *, deterministic=False):
    """Integer seconds with sample SD as a mathematical subscript."""
    value = f"{mean:.0f}"
    if deterministic:
        return value
    deviation = f"{std:.0f}" if np.isfinite(std) else r"\mathrm{n/a}"
    return rf"${value}_{{\pm {deviation}}}$"


def timing_figures(summary):
    """One bar per config: stacked training totals, and testing totals."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import textwrap

    labels = []
    for item in summary:
        sampling = "single run" if item["algorithm"] == "EFC" else f"n = {item['n']}"
        labels.append(textwrap.fill(item["name"], width=24) + f"\n{item['algorithm']} ({sampling})")
    width = max(9, 2.4 * len(summary) + 2)
    figures = []
    for stage in ("train", "test"):
        training = stage == "train"
        has_phases = training and any(item["algorithm"] == "NeSy-NIDS" for item in summary)
        figure = Figure(figsize=(width, 9 if has_phases else 6.5), dpi=160)
        FigureCanvasAgg(figure)
        if has_phases:
            grid = figure.add_gridspec(2, 1, height_ratios=[3.5, 1.3], hspace=0.55)
            ax = figure.add_subplot(grid[0])
            detail_ax = figure.add_subplot(grid[1])
            detail_ax.axis("off")
        else:
            ax = figure.subplots()
        figure.subplots_adjust(left=0.14, right=0.97, top=0.85, bottom=0.17 if not has_phases else 0.13)
        totals = []
        legend_labels = set()
        for index, item in enumerate(summary):
            kind = item["algorithm"]
            mean = item["means"][f"{stage}_seconds"]
            std = item["stds"][f"{stage}_seconds"]
            totals.append(mean)
            if training and kind == "NeSy-NIDS":
                bottom = 0
                for key, label, color in TRAINING_PHASES:
                    height = item["means"][key]
                    ax.bar(index, height, bottom=bottom, width=0.62, color=color,
                           label=label if label not in legend_labels else None)
                    legend_labels.add(label)
                    bottom += height
            else:
                ax.bar(index, mean, width=0.62, color=ALGORITHM_COLORS[kind],
                       label=kind if kind not in legend_labels else None)
                legend_labels.add(kind)
            ax.annotate(format_seconds(mean, std, deterministic=kind == "EFC"),
                        (index, mean), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=13)
        ax.set_ylim(0, max(max(totals) * 1.2, 1))
        ax.set_xlim(-0.6, len(summary) - 0.4)
        ax.set_xticks(range(len(summary)), labels, fontsize=10)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        title = "Training time" if training else "Testing time"
        figure.suptitle(title + " by configuration", fontsize=20, fontweight="bold", y=0.97)
        if has_phases:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=9)
            cells = []
            for key, _, _ in TRAINING_PHASES:
                cells.append([
                    format_seconds(item["means"][key], item["stds"][key])
                    if item["algorithm"] == "NeSy-NIDS" else "—"
                    for item in summary
                ])
            # The table keeps short phase measurements legible even when their
            # stack segment is too small to contain an annotation.
            table = detail_ax.table(cellText=cells,
                                    rowLabels=[label for _, label, _ in TRAINING_PHASES],
                                    colLabels=[textwrap.fill(item["name"], width=24) for item in summary],
                                    cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            for (row, column), cell in table.get_celld().items():
                cell.set_edgecolor("#DDDDDD")
                if row == 0:
                    cell.set_facecolor("#F2F4F6")
                elif column == -1:
                    cell.set_facecolor(TRAINING_PHASES[row - 1][2])
        figure.text(0.5, 0.035,
                    "Seconds rounded to integers; subscript: ± sample SD across seeds (n/a for one seed).\n"
                    "EFC: one recorded run, without averaging or SD."
                    + (" Stacks sum to total training time." if has_phases else ""),
                    ha="center", fontsize=10, color="#555555")
        figures.append((f"{stage}_times.png", figure))
    return figures
