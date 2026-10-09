import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import preprocess_loop

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

    preprocess_loop(
        output_dir=output_dir,
        raw_train1=raw_train1,
        raw_train2=raw_train2,
        raw_test=raw_test,
        unknown_classes=unknown_classes,
        categorical_columns=categorical_columns,
        mode=mode,
    )

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset", "CIC-IDS-2017")
    # concatenate csv files in dataset_dir into a single dataframe
    data = pd.concat(
        [pd.read_csv(os.path.join(dataset_dir, f)) for f in os.listdir(dataset_dir) if f.endswith(".csv")],
        ignore_index=True,
    )
    preprocess(data, output_dir=dataset_dir, mode = "hierarchical")