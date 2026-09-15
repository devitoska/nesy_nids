import pandas as pd
import os
import time
import logging

from bn.utils import init_bn, load_bn, save_bn_image, save_bn_model, print_bn_info
from bn.test import test_bn

def train_bn(exp_name, config, data_path):

    structure_learning_method = config.get("structure_learning").get("method")
    structure_learning_params = config.get("structure_learning").get("params", {})
    parameter_learning_method = config.get("parameter_learning").get("method")
    parameter_learning_params = config.get("parameter_learning").get("params", {})

    partitions = sorted(
        name for name in os.listdir(data_path)
        if name.startswith("no_") and os.path.isdir(os.path.join(data_path, name))
    )
    if not partitions:
        raise ValueError(f"No held-out-class partitions found in {data_path}")
    times = {}

    for partition in partitions:
        cls = partition.removeprefix("no_")
        partition_path = os.path.join(data_path, partition)
        train_data_current = pd.read_csv(
            os.path.join(partition_path, "train_1_data.csv"), dtype=str,
        )
        test_data = pd.read_csv(
            os.path.join(partition_path, "test_data.csv"), dtype=str,
        )
        if train_data_current["class"].eq(cls).any():
            raise ValueError(f"Unknown class {cls} found in BN training data")

        logging.info(f"Creating BN without class '{cls}'...")

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
        print_bn_info(bn, base_save_path, 
                      search_strategy=structure_learning_method,
                      structure_learning_params=structure_learning_params,
                      estimator_type=parameter_learning_method,
                      parameter_learning_params=parameter_learning_params)

        # save full bn image only
        save_bn_image(full_bn, base_save_path, file_name="full_bn.png")

        # Reload because save_bn_model/save_bn_image might modify the bn object (unexpected)
        bn = load_bn(base_save_path)
        mb_list = sorted(bn.get_markov_blanket("class"))

        # Project validation data on Markov Blanket variables + class
        test_data_mb = test_data[mb_list + ["class"]]

        # Evaluate BN Performance on Validation Set
        test_bn(bn, test_data_mb, base_save_path, unknown_class=cls)

    logging.info(f"Times for creating bayesian network for each unknown class:")
    for cls, t in times.items():
        logging.info(f"{cls}: {t} s")
