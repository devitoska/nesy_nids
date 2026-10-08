from efc import EnergyBasedFlowClassifier
import pickle
import os
import pandas as pd
import time
import argparse
from pathlib import Path

if __package__:
    from .utils_baselines import DEFAULT_DATA_DIR, discover_classes, record_stage_time
else:
    from utils_baselines import DEFAULT_DATA_DIR, discover_classes, record_stage_time

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train EFC for each no_<class> data directory.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results/efc")
    parser.add_argument("--classes", nargs="+", help="Optional subset of discovered classes; default: all")
    args = parser.parse_args()
    try:
        classes = discover_classes(args.data_dir, args.classes)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    with record_stage_time(args.results_dir, "train"):
        print("Training EFCs...")
        times = {cls : 0 for cls in classes}

        # Create "results" directory if it doesn't exist
        args.results_dir.mkdir(parents=True, exist_ok=True)

        for cls in classes:

            partition_path = args.data_dir / f"no_{cls}"
            df_train_1 = pd.read_csv(os.path.join(partition_path, "train_1_data.csv"))
            df_train_2 = pd.read_csv(os.path.join(partition_path, "train_2_data.csv"))

            df_train = pd.concat([df_train_1, df_train_2], ignore_index=True)
            y = df_train["class"].values

            X_train = df_train.drop(columns=["class"]).values.astype(float)  # ensure numeric

            categorical_indexes = [i for i in range(X_train.shape[1])]
            num_cols = X_train.shape[1]

            t0 = time.time()
            model = EnergyBasedFlowClassifier()
            model.fit(X_train, y, categorical_columns = categorical_indexes)
            t1 = time.time()
            times[cls] = round(t1 - t0, 2)

            os.makedirs(args.results_dir / "efc", exist_ok=True)
            with open(args.results_dir / "efc" / f"efc_no_{cls}.pkl", "wb") as f:
                pickle.dump(model, f)

        print("Time taken for each class:")
        for cls, t in times.items():
            print(f"{cls} {t} s")
