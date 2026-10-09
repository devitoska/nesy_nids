import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import preprocess_loop

random_state = 42
train_1_split = 0.4
train_2_split = 0.4
test_split = 0.2
num_bins = 50

def preprocess(data, output_dir="data/dataset/CIC-UNSW-NB15", mode = "base"):
    data = data.copy()
    # remove leading spaces in column names and convert to lowercase
    data.columns = [col.lower().strip() for col in data.columns]
    data = data.rename(columns={"label": "class"})

    # Rename "Benign" label to "normal" for consistency with other datasets.
    data["class"] = data["class"].replace(to_replace="Benign", value="normal")

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

    # N.B. no categorical columns in CIC-UNSW-NB15
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
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset", "CIC-UNSW-NB15")
    # create a file named LabeledData.csv in the dataset directory
    # that contains the vertical concatenation of Data.csv and Label.csv
    numeric_to_label = {
        0 : "Benign",
        1 : "Analysis",
        2 : "Backdoor",
        3 : "DoS",
        4 : "Exploits",
        5 : "Fuzzers",
        6 : "Generic",
        7 : "Reconnaissance",
        8 : "Shellcode",
        9 : "Worms"
    }
    data_csv = os.path.join(dataset_dir, "Data.csv")
    label_csv = os.path.join(dataset_dir, "Label.csv")
    data = pd.read_csv(data_csv)
    labels = pd.read_csv(label_csv)
    data["label"] = labels["Label"].map(numeric_to_label)
    data.to_csv(os.path.join(dataset_dir, "LabeledData.csv"), index=False)
    preprocess(data, output_dir=dataset_dir, mode = "hierarchical")