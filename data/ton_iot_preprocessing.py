import os

import pandas as pd
from sklearn.model_selection import train_test_split

random_state = 42
train_1_split = 0.4
train_2_split = 0.4
test_split = 0.2
num_bins = 50

def preprocess(data, output_dir="data/dataset/ton-iot_net"):
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
        known_train = pd.concat([train1, train2], ignore_index=True)
        partitions = (train1, train2, test)
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
                for partition in partitions:
                    partition[col] = pd.Categorical(
                        partition[col], categories=categories,
                    ).codes  # Unseen or missing categories receive -1.
            else:
                train_min = known_train[col].min()
                train_range = known_train[col].max() - train_min
                normalized_train = (known_train[col] - train_min) / train_range
                _, bin_edges = pd.qcut(
                    normalized_train, q=num_bins, labels=False,
                    duplicates="drop", retbins=True,
                )
                bin_edges[0] = float("-inf")
                bin_edges[-1] = float("inf")
                for partition in partitions:
                    partition[col] = pd.cut(
                        (partition[col] - train_min) / train_range,
                        bins=bin_edges, labels=False, include_lowest=True,
                    )
                if pd.concat([train1[col], train2[col]]).nunique() <= 1:
                    drop_columns.append(col)

        experiment_dir = os.path.join(output_dir, f"no_{unknown_cls}")
        os.makedirs(experiment_dir, exist_ok=True)
        for name, partition in zip(("train_1", "train_2", "test"), partitions):
            partition = partition.drop(columns=drop_columns)
            partition.to_csv(
                os.path.join(experiment_dir, f"{name}_data.csv"), index=False,
            )
            print(f"no_{unknown_cls}/{name}: {len(partition)} samples")


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset", "ton-iot_net")
    data = pd.read_csv(os.path.join(dataset_dir, "train_test_data.csv"))
    preprocess(data, output_dir=dataset_dir)
