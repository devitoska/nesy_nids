import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, GES, MmhcEstimator, PC
from pgmpy.estimators import BDeu, BIC, K2
from pgmpy.estimators.CITests import chi_square


# available scoring functions, mapping string names to classes
scoring_functions = {
    'bdeu': BDeu,
    'bic': BIC,
    'k2': K2,
}

# available conditional independence tests, mapping string names to functions
ci_tests = {
    'chi_square': chi_square,
}

# Exhaustive Search
def exhaustive_search(data, scoring_fn, **kwargs) -> DiscreteBayesianNetwork:
    es = ExhaustiveSearch(data)
    scoring_instance = scoring_fn(data)
    best_model = es.estimate(scoring_method=scoring_instance)
    bn = DiscreteBayesianNetwork(best_model.edges())
    return bn

# Hill Climb with tabu search
def hill_climbing(data, scoring_fn, expert_knowledge = None, max_indegree=10, tabu_length=10, epsilon=1e-6, max_iter=1000, **kwargs) -> DiscreteBayesianNetwork:
    hc = HillClimbSearch(data)
    if isinstance(scoring_fn, str):
        scoring_instance = scoring_fn
    else:
        scoring_instance = scoring_fn(data)
    best_model = hc.estimate(scoring_method=scoring_instance, expert_knowledge=expert_knowledge, max_indegree=max_indegree, 
                            tabu_length=tabu_length, epsilon=epsilon, max_iter=max_iter)
    bn = DiscreteBayesianNetwork(best_model.edges())
    return bn

# Greedy Equivalence Search
def ges(data, scoring_fn, expert_knowledge = None, **kwargs) -> DiscreteBayesianNetwork:
    ges = GES(data)
    if isinstance(scoring_fn, str):
        scoring_instance = scoring_fn
    else:
        scoring_instance = scoring_fn(data)
    best_model = ges.estimate(scoring_method=scoring_instance, expert_knowledge=expert_knowledge)
    bn = DiscreteBayesianNetwork(best_model.edges())
    return bn

# Max-Min Hill Climbing
def mmhc(data, scoring_fn, expert_knowledge = None, **kwargs) -> DiscreteBayesianNetwork:
    mmhc = MmhcEstimator(data)
    if isinstance(scoring_fn, str):
        scoring_instance = scoring_fn
    else:
        scoring_instance = scoring_fn(data)
    best_model = mmhc.estimate(scoring_method=scoring_instance, expert_knowledge=expert_knowledge)
    bn = DiscreteBayesianNetwork(best_model.edges())
    return bn

# PC algorithm
def pc(data, ci_test: str = 'chi_square', significance_level: float = 0.05, expert_knowledge = None, **kwargs) -> DiscreteBayesianNetwork:
    ci_test_fn = ci_tests.get(ci_test, None)
    if ci_test_fn is None:
        raise ValueError("Invalid conditional independence test.")
    pc = PC(data)
    model = pc.estimate(return_type='dag', ci_test=ci_test_fn, significance_level=significance_level, expert_knowledge=expert_knowledge, enforce_expert_knowledge=True)
    bn = DiscreteBayesianNetwork(model.edges())
    return bn

# learn structure function
def learn_structure(data: pd.DataFrame, method: str = 'hill_climbing', scoring_function: str = "bic", **kwargs) -> DiscreteBayesianNetwork:
    # check if scoring method if present in kwargs is valid
    if scoring_function not in scoring_functions.keys():
        raise ValueError(f"Scoring function {scoring_function} not recognized.")
    else:
        scoring_fn = scoring_functions[scoring_function]

    if method == "exhaustive":
        return exhaustive_search(data, scoring_fn, **kwargs)
    elif method == "hill_climbing":
        return hill_climbing(data, scoring_fn, **kwargs)
    elif method == "ges":
        return ges(data, scoring_fn, **kwargs)
    elif method == "mmhc":
        return mmhc(data, scoring_fn, **kwargs)
    elif method == "pc": # for pc algorithm no scoring function is needed
        return pc(data, **kwargs)
    else:
        raise ValueError(f"Structure learning method {method} not implemented.")