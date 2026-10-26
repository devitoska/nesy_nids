import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import preprocess_loop

random_state = 42
num_bins = 50

def preprocess(train_data, test_data, output_dir="data/dataset/CICIOMT-2024", mode = "base"):
    
    # remove leading spaces in column names and convert to lowercase
    train_data.columns = [col.lower().strip() for col in train_data.columns]
    test_data.columns = [col.lower().strip() for col in test_data.columns]

    # assert train_data and test_data have the same columns
    assert set(train_data.columns) == set(test_data.columns), "Train and test data have different columns"

    # Keep the same raw rows in each split across all held-out attacks.
    raw_train = train_data.copy()
    raw_test = test_data.copy()

    raw_train1, raw_train2 = train_test_split(
        raw_train, test_size=0.5,
        stratify=raw_train["class"], random_state=random_state,
    )

    categorical_columns = set(
        raw_train.select_dtypes(include=["object", "category", "string"]).columns
    )
    unknown_classes = sorted(set(raw_train["class"]) - {"normal"})

    preprocess_loop(
        output_dir=output_dir,
        raw_train1=raw_train1,
        raw_train2=raw_train2,
        raw_test=raw_test,
        unknown_classes=unknown_classes,
        categorical_columns=categorical_columns,
        num_bins=num_bins,
        mode=mode,
    )

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset", "CICIOMT-2024")
    # In CICIOMT-2024, train and test are already split
    # read all csv files under dataset_dir / "train" folder with a for loop
    train_dir = os.path.join(dataset_dir, "train")
    class_to_train_df = {}

    for f in os.listdir(train_dir):
        if f.endswith(".csv"):
            # get file name and remove "_train.pcap.csv" from the end of the file name
            class_name = f[:-len("_train.pcap.csv")]
            # remove prefixes "Recon-", "MQTT-", "TCP_IP-" if present in the class name
            class_name = class_name.replace("Recon-", "").replace("MQTT-", "").replace("TCP_IP-", "")
            # remove trailing digits from the class name
            class_name = class_name[:-1] if class_name[-1].isdigit() else class_name

            df = pd.read_csv(os.path.join(train_dir, f))
            # if class_name is "Benign", retain up to 100_000 rows,
            # otherwise retain up to 20_000 rows for other classes
            if class_name == "Benign":
                df = df.sample(n=min(200_000, len(df)), random_state=random_state)
                class_name = "normal"
            else:
                df = df.sample(n=min(20_000, len(df)), random_state=random_state)
            
            # add class column
            df["class"] = class_name
            
            if class_name not in class_to_train_df:
                class_to_train_df[class_name] = df
            else:
                class_to_train_df[class_name] = pd.concat( [class_to_train_df[class_name], df],
                    ignore_index=True,
                )

    test_dir = os.path.join(dataset_dir, "test")
    class_to_test_df = {}

    for f in os.listdir(test_dir):
        if f.endswith(".csv"):
            # get file name and remove "_test.pcap.csv" from the end of the file name
            class_name = f[:-len("_test.pcap.csv")]
            # remove prefixes "Recon-", "MQTT-", "TCP_IP-" if present in the class name
            class_name = class_name.replace("Recon-", "").replace("MQTT-", "").replace("TCP_IP-", "")
            # remove trailing digits from the class name
            class_name = class_name[:-1] if class_name[-1].isdigit() else class_name

            df = pd.read_csv(os.path.join(test_dir, f))
            # if class_name is "Benign", retain up to 100_000 rows,
            # otherwise retain up to 10_000 rows for other classes
            if class_name == "Benign":
                df = df.sample(n=min(100_000, len(df)), random_state=random_state)
                class_name = "normal"
            else:
                df = df.sample(n=min(10_000, len(df)), random_state=random_state)
            
            # add class column
            df["class"] = class_name
            
            if class_name not in class_to_test_df:
                class_to_test_df[class_name] = df
            else:
                class_to_test_df[class_name] = pd.concat( [class_to_test_df[class_name], df],
                    ignore_index=True,
                )

    # for class_name different from "normal", retain only 20_000 (train) or 10_000 (test) rows
    # redo because some attacks are split in multiple csv files
    for class_name, df in class_to_train_df.items():
        if class_name != "normal":
            class_to_train_df[class_name] = df.sample(n=min(20_000, len(df)), random_state=random_state)

    for class_name, df in class_to_test_df.items():
            if class_name != "normal":
                class_to_test_df[class_name] = df.sample(n=min(20_000, len(df)), random_state=random_state)

    # aggregate all dataframes into a single dataframe for train and test
    train_data = pd.concat(class_to_train_df.values(), ignore_index=True)
    test_data = pd.concat(class_to_test_df.values(), ignore_index=True)

    preprocess(train_data, test_data, output_dir=dataset_dir, mode = "base")