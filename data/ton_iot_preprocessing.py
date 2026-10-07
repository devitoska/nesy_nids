import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import hierarchical_binning, hierarchical_planning, get_bins

random_state = 42
train_1_split = 0.4
train_2_split = 0.4
test_split = 0.2
num_bins = 50

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

        # save raw partitions to CSV files in the experiment directory
        experiment_dir = os.path.join(output_dir, f"no_{unknown_cls}")
        os.makedirs(experiment_dir, exist_ok=True)

        for i, name in enumerate(("train_1", "train_2", "test")):
            partitions[i].to_csv(
                os.path.join(experiment_dir, f"{name}_raw_data.csv"), index=False,
            )

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
