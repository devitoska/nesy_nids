import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork

estimators = {
    'maximum_likelihood': MaximumLikelihoodEstimator,
    'bayesian': BayesianEstimator
}

# Expectation Maximization (EM), missing, can be used for parameter learning with latent variables

def estimate_parameters(bn : DiscreteBayesianNetwork, data : pd.DataFrame, estimator_type : str = 'bayesian', prior_type : str = 'BDeu', equivalent_sample_size : int = 10) -> DiscreteBayesianNetwork:
    estimator_class = estimators.get(estimator_type, None)
    if estimator_class is None:
        raise ValueError("Invalid estimator type.")
    
    if estimator_type == 'bayesian':
        bn.fit(data, estimator=estimator_class, prior_type=prior_type, equivalent_sample_size=equivalent_sample_size)
    else:
        bn.fit(data, estimator=estimator_class)
    
    return bn