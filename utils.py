import os
import pandas as pd
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc, recall_score


def compute_metrics(y_gt_bin, y_pred_bin, y_gt_mul, y_pred_mul, ad_scores):
    pr, rc, _ = precision_recall_curve(y_gt_bin, ad_scores)
    fpr, tpr, _ = roc_curve(y_gt_bin, ad_scores)

    # save results
    metrics = {
        "roc_auc_score": roc_auc_score(y_gt_bin, ad_scores),
        "auc_score": auc(rc, pr),
        "tpr" : recall_score(y_gt_bin, y_pred_bin, pos_label=1),
        "fpr" : 1 - recall_score(y_gt_bin, y_pred_bin, pos_label=0),
        "fpr_at_95_tpr":  float(fpr[np.flatnonzero(tpr >= 0.95)[0]]),
        "classification_report_binary": classification_report(y_gt_bin, y_pred_bin, output_dict=True),
        "confusion_matrix_multiclass": confusion_matrix(y_gt_mul, y_pred_mul).tolist(),
        "classification_report_multiclass": classification_report(y_gt_mul, y_pred_mul, output_dict=True)
    }
    return metrics

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