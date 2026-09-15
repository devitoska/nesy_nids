from tqdm import tqdm
import torch
import pickle
import logging
import os
import time
import pandas as pd

from bn.utils import load_bn
from bn.inference import InferenceEngine

def create_dataset(data, inference_engine, mb_list, bn_path, mode = "train", unobserved=False):
    
    expls = []
    preds = []
    gts = []
    
    t = tqdm(total=len(data), desc=f"Computing explanations in {bn_path}, mode = {mode} ...")

    for _, row in data.iterrows():
        t.update(1)
        # Compute explanation vector for the row, the target is the predicted class
        target_value, e = inference_engine.get_pred_expl(row, mb_list=mb_list, unobserved=unobserved)
        gts.append(row['class'])
        preds.append(target_value)
        expls.append(e)

    # convert to tensors and save
    expls = torch.tensor(expls, dtype=torch.float32)
   
    torch.save(expls, f"{bn_path}/explanations_{mode}.pt")
    pickle.dump(preds, open(f"{bn_path}/preds_{mode}.pkl", "wb")) # string labels 
    pickle.dump(gts, open(f"{bn_path}/gts_{mode}.pkl", "wb")) # string labels

def create_expl(exp_name, config, data_path, mode = "train"):
    
    # get all subdirectories in bn folder
    bn_paths = sorted(
        d for d in os.listdir(f"results/{exp_name}/bn")
        if os.path.isdir(os.path.join(f"results/{exp_name}/bn", d))
    )
    unobserved = config.get("unobserved", False)
    times = {}

    for bn_path in bn_paths:
        cls = bn_path.removeprefix("no_")
        full_path = os.path.join(f"results/{exp_name}/bn", bn_path)
        bn = load_bn(full_path)
        
        # Each BN has its own preprocessing and matching saved partitions.
        split_name = "train_2_data.csv" if mode == "train" else "test_data.csv"
        new_data = pd.read_csv(
            os.path.join(data_path, bn_path, split_name), dtype=str,
        )
        if mode == "train" and new_data["class"].eq(cls).any():
            raise ValueError(f"Unknown class {cls} found in detector training data")

        class_values = bn.get_cpds("class").state_names["class"]

        # compute markov blanket variables for "class" variable
        if mode == "train":
            mb_list = sorted(bn.get_markov_blanket("class"))
            # save markov blanket variables to file
            with open(os.path.join(full_path, "mb_list.pkl"), "wb") as f:
                pickle.dump(mb_list, f)
        else:
            # load markov blanket variables from file
            with open(os.path.join(full_path, "mb_list.pkl"), "rb") as f:
                mb_list = pickle.load(f)

        inference_engine = InferenceEngine(bn, class_values=class_values)
        
        # project data on Markov Blanket variables + class
        new_data_mb = new_data[mb_list + ["class"]]

        t0 = time.time()
        create_dataset(new_data_mb, inference_engine, mb_list, full_path, mode, unobserved=unobserved)
        t1 = time.time()
        times[cls] = round(t1 - t0, 2)
    
    logging.info(f"Times for generating explanations for each unknown class:")
    for cls, t in times.items():
        logging.info(f"{cls}: {t} s")
