from tqdm import tqdm
import torch
import pickle
import logging
import os
import time

from bn.utils import load_bn
from bn.inference import InferenceEngine

def create_dataset(data, inference_engine, mb_list, class_values, bn_path, mode = "train"):
    
    expls = []
    preds = []
    gts = []
    
    t = tqdm(total=len(data), desc=f"Computing explanations in {bn_path}, mode = {mode} ...")

    for _, row in data.iterrows():
        t.update(1)
        prob = inference_engine.infer_from_row(row)
        pred = class_values[int(prob.argmax())]
        preds.append(pred)
        gts.append(row['class'])
        # Compute explanation vector for the row, the target is the predicted class
        e = inference_engine.get_explanation_vec(row, target_value=pred, evidence_vars=mb_list)
        expls.append(e)

    # convert to tensors and save
    expls = torch.tensor(expls, dtype=torch.float32)
   
    torch.save(expls, f"{bn_path}/explanations_{mode}.pt")
    pickle.dump(preds, open(f"{bn_path}/preds_{mode}.pkl", "wb")) # string labels 
    pickle.dump(gts, open(f"{bn_path}/gts_{mode}.pkl", "wb")) # string labels

def create_expl(exp_name, data, mode = "train"):
    
    # get all subdirectories in bn folder
    bn_paths = [d for d in os.listdir(f"results/{exp_name}/bn") if os.path.isdir(os.path.join(f"results/{exp_name}/bn", d))]

    # needed for bayesian network
    for col in data.columns:
        data[col] = data[col].astype(str)

    times = {}
    for bn_path in bn_paths:
        cls = bn_path.split("_")[1]
        full_path = os.path.join(f"results/{exp_name}/bn", bn_path)
        bn = load_bn(full_path)
        
        # exclude unknown class from train set but not from test set
        if mode == "train":
            new_data = data[data['class'] != cls]
        else:
            new_data = data

        class_values = bn.get_cpds("class").state_names["class"]

        # compute markov blanket variables for "class" variable
        mb_list = bn.get_markov_blanket("class")
        inference_engine = InferenceEngine(bn, class_values=class_values)
        
        # project data on Markov Blanket variables + class
        new_data_mb = new_data[mb_list + ["class"]]

        t0 = time.time()
        create_dataset(new_data_mb, inference_engine, mb_list, class_values, full_path, mode)
        t1 = time.time()
        times[cls] = round(t1 - t0, 2)
    
    logging.info(f"Times for generating explanations for each unknown class:")
    for cls, t in times.items():
        logging.info(f"{cls}: {t} s")