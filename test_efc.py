import pandas as pd
import os
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc, recall_score
import json

if __name__ == "__main__":

    print("Testing EFCs...")

    classes = ["backdoor", "ddos", "dos", "injection", "mitm", "password", "ransomware", "scanning", "xss"]
    test_data = pd.read_csv("data/dataset/ton-iot_net/test_data.csv")
    
    seeds = [42] # EFC is deterministic, seed not needed, just keeping it for consistency with other models

    for seed in seeds:
        for unknown_cls in classes:

            # load the model
            model = pickle.load(open(f"results/efc_{seed}/efc_no_{unknown_cls}.pkl", "rb"))

            # prepare test data
            X = test_data.drop(columns=["class"]).values.astype(float)  # ensure numeric

            y_gt_mul = test_data["class"].values
            y_gt_bin = (y_gt_mul == unknown_cls).astype(int)
            
            # get predictions
            y_pred_mul = model.predict(X, unknown_class=True)

            # replace "unknown" with the unknown class label for multiclass evaluation
            y_pred_mul = np.where(y_pred_mul == "unknown", unknown_cls, y_pred_mul)
            y_pred_bin = (y_pred_mul == unknown_cls).astype(int)
            
            pr, rc, _ = precision_recall_curve(y_gt_bin, y_pred_bin)
            # save results
            metrics = {
                "roc_auc_score": roc_auc_score(y_gt_bin, y_pred_bin),
                "auc_score": auc(pr, rc),
                "fpr" : 1 - recall_score(y_gt_bin, y_pred_bin, pos_label=0),
                "classification_report_binary": classification_report(y_gt_bin, y_pred_bin, output_dict=True),
                "confusion_matrix_multiclass": confusion_matrix(y_gt_mul, y_pred_mul).tolist(),
                "classification_report_multiclass": classification_report(y_gt_mul, y_pred_mul, output_dict=True)
            }
        
            with open(f"results/efc_{seed}/results_no_{unknown_cls}.json", "w") as f:
                json.dump(metrics, f, indent=4)
        
        print(f"Finished testing seed {seed}")