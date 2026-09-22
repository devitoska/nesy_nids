import os
import logging
import numpy as np
import torch
import pickle
import time
from sklearn.preprocessing import RobustScaler

from bn.utils import load_bn
from ad._if import IF
from ad._ae import AE
from ad._vae import VAE
from ad.utils import transform_explanations
 
def train_ad(exp_name, config, seed):

    # get all subdirectories
    bn_paths = sorted(
        d for d in os.listdir(f"results/{exp_name}/bn")
        if os.path.isdir(os.path.join(f"results/{exp_name}/bn", d))
    )
    times = {}
    
    print(f"Training anomaly detector. Config: {config}")
    
    if config["method"] == "IF":
        model_cls = IF
    elif config["method"] == "AE":
        model_cls = AE
    elif config["method"] == "VAE":
        model_cls = VAE

    loss = config.get("loss", "l2")
    rejection_rate = config.get("rejection_rate", 0.01)
    EVT_rejection_rate = config.get("EVT_rejection_rate", None)
    score = config.get("score", "mse")
    type = config["explanations"].get("type", 1)
    use_scaler = config["explanations"].get("use_scaler", False)
    unobserved = config["explanations"].get("unobserved", False)
    misclassified = config["explanations"].get("misclassified", False)

    for bn_path in bn_paths:

        unknown_cls = bn_path.split("_")[1]
        full_path = os.path.join(f"results/{exp_name}/bn", bn_path)

        # load the bayesian network
        bn = load_bn(full_path)
        class_names = bn.get_cpds("class").state_names["class"]

        # load explanations
        X_train = torch.load(os.path.join(full_path, "explanations_train.pt")).numpy()
        input_dim = X_train.shape[1]

        if use_scaler:
            scaler = RobustScaler()
        else:
            scaler = None

        X_train = transform_explanations(X_train, scaler=scaler, type=type, unobserved=unobserved, mode='train')

        # save the scaler to file
        with open(os.path.join(full_path, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
    
        gts = pickle.load(open(os.path.join(full_path, "gts_train.pkl"), "rb"))
        preds = pickle.load(open(os.path.join(full_path, "preds_train.pkl"), "rb"))

        gts = np.array([class_names.index(gt) for gt in gts])
        preds = np.array([class_names.index(pred) for pred in preds])

        times[unknown_cls] = 0
        os.makedirs(f"results/{exp_name}/ad/no_{unknown_cls}", exist_ok=True)
        
        for cls in range(len(class_names)):
            good_by_class = X_train[(gts == cls) & (preds == cls)]
            pred_by_class = X_train[(preds == cls)]
            mis_by_class = X_train[(gts != cls) & (preds == cls)]

            rr = 0.001 # redefining rejection rate for each class

            if misclassified:
                data = pred_by_class
                if rejection_rate == "auto": # setting auto rejection rate
                    if len(pred_by_class) > 0:
                        rr = max(rr, len(mis_by_class) / len(pred_by_class))
                else: # setting user-defined rejection rate
                    rr = rejection_rate
            else:
                data = good_by_class
                if rejection_rate != "auto": # setting user-defined rejection rate
                    rr = rejection_rate

            t0 = time.time()
            model = model_cls(input_dim=input_dim, rejection_rate=rr, 
                              EVT_rejection_rate=EVT_rejection_rate, loss = loss, score=score)
            model.train(data, seed)
            t1 = time.time()
            times[unknown_cls] += (t1 - t0)
            model.save(exp_name, unknown_cls, class_names[cls])

    logging.info("Time for training anomaly detector for each unknown class:")
    for cls, t in times.items():
        logging.info(f"{cls} {round(t, 2)} s")
