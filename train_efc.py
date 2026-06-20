import numpy as np
from efc import EnergyBasedFlowClassifier
import pickle
import os
import pandas as pd
import time

if __name__ == "__main__":

    print("Training EFCs...")

    classes = ["backdoor", "ddos", "dos", "injection", "mitm", "password", "ransomware", "scanning", "xss"]
    times = {cls : 0 for cls in classes}

    train_1_data = pd.read_csv("data/dataset/ton-iot_net/train_1_data.csv")
    train_2_data = pd.read_csv("data/dataset/ton-iot_net/train_2_data.csv")

    # Create "results" directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    seeds = [42] # EFC is deterministic, seed not needed, just keeping it for consistency with other models

    for seed in seeds:
        np.random.seed(seed)
        
        for cls in classes:

            df_train_1 = train_1_data[train_1_data["class"] != cls]
            df_train_2 = train_2_data[train_2_data["class"] != cls]

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

            os.makedirs(f"results/efc_{seed}", exist_ok=True)
            with open(f"results/efc_{seed}/efc_no_{cls}.pkl", "wb") as f:
                pickle.dump(model, f)
                
        print(f"Finished training EFC with seed {seed}.")
    
    print("Time taken for each class:")
    for cls, t in times.items():
        print(f"{cls} {t} s")