import os
import pandas as pd
import json

def create_metrics_table(exp_path):

    # list subfolders in exp_path
    subfolders = [f.name for f in os.scandir(os.path.join(exp_path, "metrics")) if f.is_dir()]
    res = {}
   
    for dir in subfolders:
        cls = dir.split("_")[1]
        # open json file
        with open(os.path.join(exp_path, f"metrics/{dir}/results.json"), "r") as f:
            data = json.load(f)
            scores = {}
            scores[f"auroc"] = round(data["roc_auc_score"], 3)
            scores[f"aupr"] = round(data["auc_score"], 3)
            scores[f"recall_pos"] = round(data["classification_report_binary"]["1"]["recall"], 3)
            scores[f"prec_pos"] = round(data["classification_report_binary"]["1"]["precision"], 3)
            scores[f"fpr"] = round(data["fpr"], 3)
            scores[f"f1_pos"] = round(data["classification_report_binary"]["1"]["f1-score"], 3)
            scores[f"macro_f1_bin"] = round(data["classification_report_binary"]["macro avg"]["f1-score"], 3)
            scores[f"avg_f1_bin"] = round(data["classification_report_binary"]["weighted avg"]["f1-score"], 3)
            scores[f"macro_f1_multi"] = round(data["classification_report_multiclass"]["macro avg"]["f1-score"], 3)
            scores[f"avg_f1_multi"] = round(data["classification_report_multiclass"]["weighted avg"]["f1-score"], 3)
            res[cls] = scores

    # create dataframe
    df = pd.DataFrame(res).T

    # reorder rows alphabetically
    df = df.sort_index()

    # save dataframe to csv
    df.to_csv(os.path.join(exp_path, "metrics/table.csv"))