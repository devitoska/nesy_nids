import os
import torch
import pickle
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc, recall_score
import json
import logging
import numpy as np

from bn.utils import load_bn
from ad._if import IF
from ad._ae import AE
from ad.utils import transform_explanations
 
def test_ad(exp_name, config):

    # get all subdirectories
    bn_paths = sorted(
        d for d in os.listdir(f"results/{exp_name}/bn")
        if os.path.isdir(os.path.join(f"results/{exp_name}/bn", d))
    )
    
    if config["method"] == "IF":
        model_cls = IF
    elif config["method"] == "AE":
        model_cls = AE

    loss = config.get("loss", "l2")
    rejection_rate = config.get("rejection_rate", 0.01)
    EVT_rejection_rate = config.get("EVT_rejection_rate", None)
    score = config.get("score", "mse")

    if config.get("explanations", None) is None:
        type = None
        use_scaler = False
        unobserved = False
    else:
        type = config["explanations"].get("type", 1)
        use_scaler = config["explanations"].get("use_scaler", False)
        unobserved = config["explanations"].get("unobserved", False)

    for bn_path in bn_paths:

        unknown_cls = bn_path.split("_")[1]
        full_path = os.path.join(f"results/{exp_name}/bn", bn_path)

        # load the bayesian network
        bn = load_bn(full_path)
        class_names = bn.get_cpds("class").state_names["class"]

        # load explanations
        X_test = torch.load(os.path.join(full_path, "explanations_test.pt")).numpy()

        if use_scaler:
            # load the scaler from file
            with open(os.path.join(full_path, "scaler.pkl"), "rb") as f:
                scaler = pickle.load(f)
        else:
            scaler = None

        X_test = transform_explanations(X_test, scaler=scaler, type=type, unobserved=unobserved, mode='test')
        input_dim = X_test.shape[1]

        gts = pickle.load(open(os.path.join(full_path, "gts_test.pkl"), "rb"))
        preds = pickle.load(open(os.path.join(full_path, "preds_test.pkl"), "rb"))
        
        ad_models = {}
        for cls in class_names:
            ad_models[cls] = model_cls(input_dim=input_dim, rejection_rate=rejection_rate, 
                                       EVT_rejection_rate=EVT_rejection_rate, loss=loss, 
                                       type=type, score=score)
            ad_models[cls].load(exp_name, unknown_cls, cls)

        y_gt_bin, y_pred_bin, y_gt_mul, y_pred_mul, ad_scores = model_cls.test(ad_models, X_test, gts, preds, unknown_cls)

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

        os.makedirs(f"results/{exp_name}/metrics/no_{unknown_cls}/", exist_ok=True)

        with open(f"results/{exp_name}/metrics/no_{unknown_cls}/results.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        logging.info(f"Results for unknown class {unknown_cls} saved in results/{exp_name}/metrics/no_{unknown_cls}/results.json")
