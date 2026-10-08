# Import Cerberus for schema validation
from cerberus import Validator
from copy import deepcopy


def normalize_config(config):
    """Accept existing NeSy configs and return the dataset/algorithm layout."""
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping")
    config = deepcopy(config)
    if "algorithm" not in config:
        legacy = {key: config.pop(key) for key in ("bayesian_network", "anomaly_detection") if key in config}
        if legacy:
            config["algorithm"] = {"type": "NeSy-NIDS", **legacy}
    algorithm = config.get("algorithm")
    if isinstance(algorithm, dict):
        kind = algorithm.get("type", "NeSy-NIDS")
        if isinstance(kind, str):
            algorithm["type"] = {
                "nesy-nids": "NeSy-NIDS", "efc": "EFC", "ocn": "OCN",
            }.get(kind.casefold(), kind)
    return config

def get_allowed_structure_learning_methods():
    return ["exhaustive", "hill_climbing", "ges", "mmhc", "pc"]

def get_allowed_score_based_methods():
    return ["exhaustive", "hill_climbing", "ges", "mmhc"]

def get_allowed_scoring_methods():
    return ["k2", "bic", "bdeu"]

def get_allowed_ci_tests():
    return ["chi_square"]

def get_allowed_parameter_learning_methods():
    return ["bayesian", "mle"]

def get_allowed_prior_types():
    return ["BDeu", "dirichlet", "K2"]

def get_allowed_ad_methods():
    return ["IF", "AE", "VAE"]

def get_allowed_rejection_rates():
    return [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, "auto"]

validation_schema = {
  "name": {
    "type": "string",
    "required": False,
    "minlength": 3,
    "maxlength": 50
  },
  "seed": {
    "type": "integer",
    "min": 0,
    "required": False,
    "default": 42
  },
  "dataset_path": {
    "type": "string",
    "empty": False,
    "required": False,
    "default": "data/dataset/ton-iot_net"
  },
  "algorithm": {
    "type": "dict",
    "required": True,
    "schema": {
      "type": {
        "type": "string",
        "required": True,
        "allowed": ["NeSy-NIDS", "EFC", "OCN"],
        "default": "NeSy-NIDS"
      },
      "bayesian_network": {
        "type": "dict",
        "required": False,
        "schema": {
          "structure_learning": {
            "type": "dict",
            "required": True,
            "schema": {
              "method": {
                "type": "string",
                "required": True,
                "allowed": get_allowed_structure_learning_methods(),
                "default": "hill_climbing"
              },
              "params": {
                "type": "dict",
                "required": True,
                "schema": {
                  "scoring_method": {
                    "type": "string",
                    "required": False,
                    # "dependencies": {"methods": get_allowed_score_based_methods()},
                    "allowed": get_allowed_scoring_methods()
                  },
                  "max_indegree": {
                    "type": "integer",
                    "required": False,
                    "min": 0,
                    "default": 10
                  },
                  "tabu_length": {
                    "type": "integer",
                    "required": False,
                    "min": 0,
                    "default": 10
                  },
                  "max_iter": {
                    "type": "integer",
                    "min": 1,
                    "required": False,
                    "default": 1000
                  },
                  "epsilon": {
                    "type": "float",
                    "min": 0.0,
                    "required": False,
                    "default": 1e-6
                  },
                  "ci_test": {
                    "type": "string",
                    "required": False,
                    #"dependencies": {"method": "pc"},
                    "allowed": get_allowed_ci_tests(),
                    "default": "chi_square"
                  },
                  "significance_level": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "required": False,
                    #"dependencies": {"method": "pc"}
                  }
                }
              }
            }
          },
          "parameter_learning": {
            "type": "dict",
            "required": True,
            "schema": {
              "method": {
                "type": "string",
                "required": True,
                "default": "bayesian",
                "allowed": get_allowed_parameter_learning_methods()
              },
              "params": {
                "type": "dict",
                "required": False,
                "schema": {
                  "prior_type": {
                    "type": "string",
                    "required": False,
                    "allowed": get_allowed_prior_types(),
                    "default": "BDeu"
                  },
                  "equivalent_sample_size": {
                    "type": "integer",
                    "min": 0,
                    "required": False,
                    "default": 10
                  }
                }
              }
            }
          }
        }
      },
      "anomaly_detection":{
          "type": "dict",
          "required": False,
          "schema": {
            "method" : {
                "type": "string",
                "required": True,
                "allowed": get_allowed_ad_methods(),
                "default": "AE"
            },
            "loss" : {
                "type": "string",
                "required": False,
                "allowed": ["l2", "huber"],
                "default": "l2"
            },
            "rejection_rate": {
                "type": ["float", "string"],
                "required": False,
                "default": 0.01,
                "allowed" : get_allowed_rejection_rates()
            },
            "EVT_rejection_rate": {
                        "type": "float",
                        "nullable": True,
                        "required": False,
                        "default": None,
                        "min": 0.001,
                        "max": 1.0
            },
            "calibration_split": {
                "type": "string",
                "required": False,
                "allowed": ["same", "separate"],
                "default": "same"
            },
            "score" : {
                "type": "string",
                "required": False,
                "allowed": ["mse", "mah"],
                "default": "mse"
            },
            "explanations":{
                "type": "dict",
                "required": False,
                "default": {},
                "schema": {
                  "type": {
                      "type": "integer",
                      "required": False,
                      "default": 1
                  },
                  "use_scaler": {
                      "type": "boolean",
                      "required": False,
                      "default": False
                  },
                  "unobserved": {
                      "type": "boolean",
                      "required": False,
                      "default": False
                  },
                  "misclassified": {
                      "type": "boolean",
                      "required": False,
                      "default": False
                  }
                }
              }
          }
      }
    }
  }
} 

def validate_config(config):
    config = normalize_config(config)
    schema = deepcopy(validation_schema)
    algorithm = config.get("algorithm")
    if isinstance(algorithm, dict) and algorithm.get("type") == "NeSy-NIDS":
        for key in ("bayesian_network", "anomaly_detection"):
            schema["algorithm"]["schema"][key]["required"] = True
    v = Validator(schema)
    is_valid = v.validate(config)
    if not is_valid:
        raise ValueError(f"Configuration validation error: {v.errors}")
    return v.document
