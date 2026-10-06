import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import hierarchical_binning, hierarchical_planning, get_bins

random_state = 42
train_1_split = 0.4
train_2_split = 0.4
test_split = 0.2
num_bins = 50

def preprocess(data, output_dir="data/dataset/CIC-IDS-2017", mode = "base"):
    data = data.copy()
    # remove leading spaces in column names and convert to lowercase
    data.columns = [col.lower().strip() for col in data.columns]
    data = data.rename(columns={"label": "class"})

    # Delete rows with class 'Infiltration' and 'Heartbleed' as they are too few to be useful.
    data = data[~data["class"].isin(["Infiltration", "Heartbleed"])]

    # Rename all class labels starting with "Web Attack" to "Web Attack" for aggregation purposes.
    data["class"] = data["class"].replace(
        to_replace=r"^Web Attack.*", value="Web Attack", regex=True
    )

    # Rename "BENIGN" label to "normal" for consistency with other datasets.
    data["class"] = data["class"].replace(to_replace="BENIGN", value="normal")

    # keep only the 20% of normal rows to balance the dataset
    normal_rows = data[data["class"] == "normal"]
    normal_rows = normal_rows.sample(frac=0.2, random_state=random_state)
    data = pd.concat([data[data["class"] != "normal"], normal_rows], ignore_index=True)

    # Keep the same raw rows in each split across all held-out attacks.
    raw_train, raw_test = train_test_split(
        data, test_size=test_split, stratify=data["class"],
        random_state=random_state,
    )
    raw_train1, raw_train2 = train_test_split(
        raw_train, test_size=train_2_split / (train_1_split + train_2_split),
        stratify=raw_train["class"], random_state=random_state,
    )

    # N.B. no categorical columns in CIC-IDS-2017
    categorical_columns = set(
        data.select_dtypes(include=["object", "category", "string"]).columns
    )
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
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset", "CIC-IDS-2017")
    # concatenate csv files in dataset_dir into a single dataframe
    data = pd.concat(
        [pd.read_csv(os.path.join(dataset_dir, f)) for f in os.listdir(dataset_dir) if f.endswith(".csv")],
        ignore_index=True,
    )
    preprocess(data, output_dir=dataset_dir, mode = "hierarchical")