## Towards neurosymbolic network intrusion detection
This is the code base for the paper "Towards neurosymbolic network intrusion detection" Scaraggi et al. DOI: [10.1109/SmartComp-Companion70724.2026.00063](https://doi.org/10.1109/SmartComp-Companion70724.2026.00063).

### Installation
Conda environment is recommended. To install the required packages, run the following command:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda env create -f environment.yml
```

To install efc package, run the following commands:

```bash
git clone https://github.com/EnergyBasedFlowClassifier/EFC-package
cd EFC-package
pip install -r requirements.txt
python -m pip install --no-build-isolation .
```

### Usage

By default, the code will run experiments on the Ton-IoT dataset located in the `data\dataset\ton-iot_net` folder.

To generate training and testing partitions, run the following command from the root of the repository:

```bash
python data/ton_iot_preprocessing.py
```

To train the model, run the following command:

```bash
python train.py \[--config path/to/config.yaml\]
```

where `--config` is an optional argument to specify a custom configuration file. If not provided, the default configuration file `config.yml` will be used.

The configuration file contains various parameters for training the model, such as the dataset path, model architecture, and hyperparameters. You can modify the configuration file to suit your needs. The example configuration file `config.yml` is provided in the repository. Training will create a folder named `results/exp_<timestamp>` in the project folder, containing the trained model and its configuration file.

To evaluate the model, run the following command:

```bash
python test.py --exp_path path/to/experiment/folder
```

where `--exp_path` is a mandatory argument specifying the path to the folder containing the trained model and its configuration file. The evaluation results will be saved in the same folder.

To train and evaluate the EFC model, run the following commands:

```bash
python train_efc.py
python test_efc.py
```

Results will be saved in the `results` folder.
