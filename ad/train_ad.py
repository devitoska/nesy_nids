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
 
def train_ad(exp_name, config):

    # get all subdirectories
    bn_paths = [d for d in os.listdir(f"results/{exp_name}/bn") if os.path.isdir(os.path.join(f"results/{exp_name}/bn", d))]
    times = {}

    print(f"Training anomaly detector. Config: {config}")
    
    if config["method"] == "IF":
        model_cls = IF
    elif config["method"] == "AE":
        model_cls = AE
    elif config["method"] == "VAE":
        model_cls = VAE

    use_scaler = config.get("use_scaler", False)

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
            scaler.fit(X_train)
            # save the scaler to file
            with open(os.path.join(full_path, "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
        else:
            scaler = None

        X_train = transform_explanations(X_train, scaler=scaler)

        gts = pickle.load(open(os.path.join(full_path, "gts_train.pkl"), "rb"))
        preds = pickle.load(open(os.path.join(full_path, "preds_train.pkl"), "rb"))

        gts = np.array([class_names.index(gt) for gt in gts])
        preds = np.array([class_names.index(pred) for pred in preds])

        good_by_class = {}
        
        for cls in range(len(class_names)):
            good_by_class[cls] = X_train[(gts == cls) & (preds == cls)]

        times[unknown_cls] = 0
        os.makedirs(f"results/{exp_name}/ad/no_{unknown_cls}", exist_ok=True)
        
        for cls in range(len(class_names)):
            data = good_by_class[cls]
            t0 = time.time()
            model = model_cls(input_dim=input_dim)
            model.train(data)
            t1 = time.time()
            times[unknown_cls] += (t1 - t0)
            model.save(exp_name, unknown_cls, class_names[cls])

    logging.info("Time for training anomaly detector for each unknown class:")
    for cls, t in times.items():
        logging.info(f"{cls} {round(t, 2)} s")