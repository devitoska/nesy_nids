import os
import warnings

import optbinning as optb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

random_state = 42
train_1_split = 0.4
train_2_split = 0.4
test_split = 0.2
num_bins = 50
hierarchical_coarse_bins = 32
hierarchical_within_bins = 32


def quantile_splits(values, requested_bins):
    """Choose quantile-like splits only between distinct observed values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    unique_values, counts = np.unique(values, return_counts=True)
    n_bins = min(requested_bins, len(unique_values))

    if n_bins <= 1:
        return np.array([], dtype=float)

    cumulative_counts = np.cumsum(counts)[:-1]
    split_positions = []
    previous_position = -1

    # Select distinct gaps closest to the desired cumulative quantiles.
    # Restricting the search range leaves enough gaps for later splits.
    for split_number in range(1, n_bins):
        remaining_splits = n_bins - split_number - 1
        first_position = previous_position + 1
        last_position = len(unique_values) - 2 - remaining_splits
        positions = np.arange(first_position, last_position + 1)
        target_count = split_number * counts.sum() / n_bins
        position = positions[
            np.argmin(np.abs(cumulative_counts[positions] - target_count))
        ]
        split_positions.append(position)
        previous_position = position

    split_positions = np.asarray(split_positions, dtype=int)
    left_values = unique_values[split_positions]
    right_values = unique_values[split_positions + 1]
    splits = left_values + (right_values - left_values) / 2

    # Preserve a strict boundary even for adjacent floating-point values.
    invalid_midpoints = (splits <= left_values) | (splits >= right_values)
    splits[invalid_midpoints] = np.nextafter(
        left_values[invalid_midpoints], right_values[invalid_midpoints]
    )
    return splits


def hierarchical_planning(
    train_values,
    coarse_bins=hierarchical_coarse_bins,
    within_bins=hierarchical_within_bins,
):
    """Learn fine quantile edges and the radix used for two-level encoding."""
    if coarse_bins < 1 or within_bins < 1:
        raise ValueError("coarse_bins and within_bins must be positive integers")

    leaf_splits = quantile_splits(train_values, coarse_bins * within_bins)
    leaf_edges = np.concatenate(([-np.inf], leaf_splits, [np.inf]))
    return leaf_edges, within_bins


def hierarchical_binning(values, leaf_edges, within_bins):
    """Encode fine interval IDs as ``(coarse, within)`` integer pairs."""
    leaf_ids = pd.cut(
        values, bins=leaf_edges, labels=False, include_lowest=True,
    )
    coarse_ids = (leaf_ids // within_bins).astype("Int64")
    within_ids = (leaf_ids % within_bins).astype("Int64")
    return coarse_ids, within_ids


def get_bins(data, col, col_data, mode="base"):
    """Return bin edges learned exclusively from the known training data."""
    bin_edges = None

    if mode == "base" or mode == "hierarchical":
        raw_values = pd.to_numeric(data[col], errors="coerce").to_numpy(
            dtype=float
        )
        normalized_values = np.asarray(col_data, dtype=float)
        valid = np.isfinite(raw_values) & np.isfinite(normalized_values)
        raw_valid = raw_values[valid]
        normalized_valid = normalized_values[valid]

        if not len(raw_valid):
            return np.array([-np.inf, np.inf])

        zero_fraction = np.count_nonzero(raw_values == 0) / len(raw_values)
        if zero_fraction > 0.25:
            # Reserve one bin for zero. For count-valued fields, also reserve
            # one bin for every observed small integer from 1 through 10.
            is_count_valued = np.all(
                np.isclose(raw_valid, np.round(raw_valid), rtol=0, atol=1e-12)
            )
            if is_count_valued:
                exact_values = np.unique(
                    raw_valid[
                        (raw_valid >= 0)
                        & (raw_valid <= 10)
                    ]
                )
            else:
                exact_values = np.array([0.0])

            exact_values = exact_values[exact_values == np.round(exact_values)]
            exact_values = exact_values[:num_bins]
            tail_start = exact_values[-1]

            value_pairs = pd.DataFrame({
                "raw": raw_valid,
                "normalized": normalized_valid,
            }).drop_duplicates("raw").sort_values("raw")
            distinct_raw = value_pairs["raw"].to_numpy()
            distinct_normalized = value_pairs["normalized"].to_numpy()

            exact_splits = []
            for value in exact_values:
                position = np.searchsorted(distinct_raw, value)
                if position < len(distinct_raw) - 1:
                    left = distinct_normalized[position]
                    right = distinct_normalized[position + 1]
                    split = left + (right - left) / 2
                    if not left < split < right:
                        split = np.nextafter(left, right)
                    exact_splits.append(split)

            tail_values = normalized_valid[raw_valid > tail_start]
            tail_bin_budget = max(num_bins - len(exact_values), 0)
            tail_splits = quantile_splits(tail_values, tail_bin_budget)
            internal_splits = np.unique(np.concatenate((
                np.asarray(exact_splits), tail_splits,
            )))
        else:
            internal_splits = quantile_splits(normalized_valid, num_bins)

        bin_edges = np.concatenate(([-np.inf], internal_splits, [np.inf]))
    elif mode == "optb":
        # First require exactly num_bins final bins. Quantile prebinning
        # supplies extra candidate splits that the optimizer can merge.
        binning_params = dict(
            name=col,
            prebinning_method="quantile",
            min_n_bins=num_bins,
            max_n_bins=num_bins,
            max_n_prebins=2 * num_bins,
            min_prebin_size=1 / (2 * num_bins),
            monotonic_trend=None,
            max_pvalue=None,
        )
        categorical_target = (
            data["class"] != "normal"
        ).astype(np.int8).to_numpy()
        model = optb.OptimalBinning(**binning_params)
        model.fit(col_data, categorical_target)

        # Some fields have fewer than num_bins usable prebins because
        # repeated values collapse quantile boundaries. In that case,
        # retain the upper bound and let the optimizer use fewer bins.
        exact_binning_succeeded = (
            model.status in {"OPTIMAL", "FEASIBLE"}
            and len(model.splits) + 1 == num_bins
        )
        if not exact_binning_succeeded:
            binning_params["min_n_bins"] = None
            model = optb.OptimalBinning(**binning_params)
            model.fit(col_data, categorical_target)
            if model.status not in {"OPTIMAL", "FEASIBLE"}:
                raise RuntimeError(
                    f"Optimal binning failed for {col!r}: {model.status}"
                )
            warnings.warn(
                f"{col!r}: {num_bins} bins are infeasible; using "
                f"{len(model.splits) + 1} bins instead.",
                stacklevel=2,
            )

        bin_edges = np.concatenate((
            [-np.inf],
            model.splits,
            [np.inf],
        ))
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return bin_edges


def preprocess(data, output_dir="data/dataset/ton-iot_net", mode = "base"):
    data = data.copy()
    data.columns = [col.lower() for col in data.columns]
    data = data.rename(columns={"type": "class"})
    data = data.drop(columns=["label", "src_ip", "dst_ip", "dns_query"])

    # Keep the same raw rows in each split across all held-out attacks.
    raw_train, raw_test = train_test_split(
        data, test_size=test_split, stratify=data["class"],
        random_state=random_state,
    )
    raw_train1, raw_train2 = train_test_split(
        raw_train, test_size=train_2_split / (train_1_split + train_2_split),
        stratify=raw_train["class"], random_state=random_state,
    )

    categorical_columns = set(
        data.select_dtypes(include=["object", "category", "string"]).columns
    ) | {"dns_qclass", "dns_qtype", "dns_rcode", "http_status_code"}
    unknown_classes = sorted(set(raw_train["class"]) - {"normal"})

    for unknown_cls in unknown_classes:
        train1 = raw_train1.loc[raw_train1["class"] != unknown_cls].copy()
        train2 = raw_train2.loc[raw_train2["class"] != unknown_cls].copy()
        test = raw_test.copy()
        known_train = pd.concat([train1.copy(), train2.copy()], ignore_index=True)
        partitions = [train1, train2, test]
        drop_columns = []

        for col in known_train.columns:
            if col == "class":
                continue
            if known_train[col].nunique() <= 1:
                drop_columns.append(col)
                continue

            if col in categorical_columns:
                # Vocabulary comes exclusively from known training examples.
                categories = pd.Index(known_train[col].dropna().unique())
                for i in range(len(partitions)):
                    partitions[i][col] = pd.Categorical(
                        partitions[i][col], categories=categories,
                    ).codes  # Unseen or missing categories receive -1.
            else:
                train_min = known_train[col].min()
                train_range = known_train[col].max() - train_min
                # Normalize the training data
                normalized_train = (known_train[col] - train_min) / train_range
                
                n_unique = known_train[col].nunique()
                n_samples = len(known_train[col])
                
                minimum_unique = max(
                    4 * num_bins,             # at least 200 distinct values
                    int(0.005 * n_samples),   # at least 0.5% sample cardinality
                )

                if mode == "hierarchical" and minimum_unique < n_unique:
                    leaf_edges, within_bins = hierarchical_planning(
                        normalized_train
                    )
                    for i in range(len(partitions)):
                        normalized_partition = (
                            partitions[i][col] - train_min
                        ) / train_range
                        coarse_ids, within_ids = hierarchical_binning(
                            normalized_partition, leaf_edges, within_bins
                        )
                        partitions[i][f"{col}_1"] = coarse_ids
                        partitions[i][f"{col}_2"] = within_ids
                    drop_columns.append(col)
                else:
                    bin_edges = get_bins(
                        known_train, col, normalized_train, mode=mode
                    )

                    for i in range(len(partitions)):
                        partitions[i][col] = pd.cut(
                            (partitions[i][col] - train_min) / train_range,
                            bins=bin_edges, labels=False, include_lowest=True,
                        )

                    if pd.concat([train1[col], train2[col]]).nunique() <= 1:
                        drop_columns.append(col)

        experiment_dir = os.path.join(output_dir, f"no_{unknown_cls}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        for i, name in enumerate(("train_1", "train_2", "test")):
            partitions[i] = partitions[i].drop(columns=drop_columns)
            # order fields alphabetically, with class last
            partitions[i] = partitions[i][
                sorted(c for c in partitions[i].columns if c != "class")
                + ["class"]
            ]
            # Save the partition to a CSV file in the experiment directory.
            partitions[i].to_csv(
                os.path.join(experiment_dir, f"{name}_data.csv"), index=False,
            )
            print(f"no_{unknown_cls}/{name}: {len(partitions[i])} samples")

        all = pd.concat(partitions, ignore_index=True)
        all.to_csv(
            os.path.join(experiment_dir, "all" \
            "_data.csv"), index=False,
        )

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset", "ton-iot_net")
    data = pd.read_csv(os.path.join(dataset_dir, "train_test_data.csv"))
    preprocess(data, output_dir=dataset_dir, mode = "hierarchical")
