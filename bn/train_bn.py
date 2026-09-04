import numpy as np
import os
import time
import logging

from bn.utils import init_bn, load_bn, save_bn_image, save_bn_model, print_bn_info
from bn.test import test_bn

def train_bn(exp_name, config, train_data, test_data):

    structure_learning_method = config.get("structure_learning").get("method")
    structure_learning_params = config.get("structure_learning").get("params", {})
    parameter_learning_method = config.get("parameter_learning").get("method")
    parameter_learning_params = config.get("parameter_learning").get("params", {})

    classes = train_data["class"].unique().tolist()
    times = {cls : 0 for cls in classes}

    # needed for bayesian network
    for col in train_data.columns:
        train_data[col] = train_data[col].astype(str)
        test_data[col] = test_data[col].astype(str)

    for cls in classes:
        
        if cls == "normal":
            continue

        logging.info(f"Creating BN without class '{cls}'...")
        
        # create partition excluding the current class
        train_data_current = train_data[train_data["class"] != cls]
        
        # Create base path for saving data model and plots
        base_save_path = f"results/{exp_name}/bn/no_{cls}"
        os.makedirs(base_save_path, exist_ok=True)

        # Initialize Bayesian Network
        t0 = time.time()
        bn, full_bn = init_bn(train_data_current, search_strategy=structure_learning_method, 
                              structure_learning_params=structure_learning_params, 
                              estimator_type=parameter_learning_method,
                              parameter_learning_params=parameter_learning_params)
        
        t1 = time.time()
        times [cls] = round(t1 - t0, 2)

        # Save BN Model, Image and Info
        save_bn_model(bn, base_save_path)
        save_bn_image(bn, base_save_path)
        print_bn_info(bn, base_save_path)

        # save full bn image only
        save_bn_image(full_bn, base_save_path, file_name="full_bn.png")

        # Reload because save_bn_model/save_bn_image might modify the bn object (unexpected)
        bn = load_bn(base_save_path)
        mb_list = bn.get_markov_blanket("class")

        # Project validation data on Markov Blanket variables + class
        test_data_mb = test_data[mb_list + ["class"]]

        # Evaluate BN Performance on Validation Set
        test_bn(bn, test_data_mb, base_save_path, unknown_class=cls)

    logging.info(f"Times for creating bayesian network for each unknown class:")
    for cls, t in times.items():
        logging.info(f"{cls}: {t} s")