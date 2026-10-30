import pandas as pd
import os
import logging

from bn.utils import init_bn, save_bn_image, save_bn_model, print_bn_info

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

    for partition in partitions:
        cls = partition.removeprefix("no_")
        partition_path = os.path.join(data_path, partition)
        train_data_current = pd.read_csv(
            os.path.join(partition_path, "train_1_data.csv"), dtype=str,
        )
        if train_data_current["class"].eq(cls).any():
            raise ValueError(f"Unknown class {cls} found in BN training data")

        logging.info(f"Creating BN without class '{cls}'...")

        # Create base path for saving data model and plots
        base_save_path = f"results/{exp_name}/bn/no_{cls}"
        os.makedirs(base_save_path, exist_ok=True)

        # Initialize Bayesian Network
        bn, full_bn = init_bn(train_data_current, search_strategy=structure_learning_method,
                              structure_learning_params=structure_learning_params,
                              estimator_type=parameter_learning_method,
                              parameter_learning_params=parameter_learning_params)

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