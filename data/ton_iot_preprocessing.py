import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import preprocess_loop

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
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset", "ton-iot_net")
    data = pd.read_csv(os.path.join(dataset_dir, "train_test_data.csv"))
    preprocess(data, output_dir=dataset_dir, mode = "hierarchical")
