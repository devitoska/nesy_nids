# Import Cerberus for schema validation
from cerberus import Validator

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
  "bayesian_network": {
    "type": "dict",
    "required": True,
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
      "required": True,
      "schema": {
        "method" : {
            "type": "string",
            "required": True,
            "allowed": get_allowed_ad_methods(),
            "default": "ae"
        },
        "use_scaler": {
            "type": "boolean",
            "required": False,
            "default": False
        },  
      }
  }
} 

def validate_config(config):
    v = Validator(validation_schema)
    is_valid = v.validate(config)
    if not is_valid:
        raise Exception(f"Configuration validation error: {v.errors}")