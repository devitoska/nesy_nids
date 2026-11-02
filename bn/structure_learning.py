import pandas as pd
import networkx as nx
from copy import deepcopy
from numbers import Integral
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, GES, MmhcEstimator, PC, ExpertKnowledge
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

def _prepare_constraints(data, expert_knowledge, max_outdegree):
    """Validate the cap and required edges without mutating caller knowledge."""
    if max_outdegree is not None and (
        isinstance(max_outdegree, bool) or not isinstance(max_outdegree, Integral)
        or max_outdegree < 0
    ):
        raise ValueError("max_outdegree must be a nonnegative integer or None.")
    expert = deepcopy(expert_knowledge) if expert_knowledge is not None else ExpertKnowledge()
    required = nx.DiGraph()
    required.add_nodes_from(data.columns)
    if any(u not in data.columns or v not in data.columns for u, v in expert.required_edges):
        raise ValueError("Required edges refer to variables absent from the data.")
    required.add_edges_from(expert.required_edges)
    temporal = deepcopy(expert)
    temporal._orient_temporal_forbidden_edges(required, only_edges=False)
    if expert.required_edges & temporal.forbidden_edges:
        raise ValueError("Required edges conflict with forbidden edges or temporal order.")
    if not nx.is_directed_acyclic_graph(required):
        raise ValueError("Required edges create a cycle.")
    if max_outdegree is not None and any(d > max_outdegree for _, d in required.out_degree()):
        raise ValueError("Required edges exceed max_outdegree.")
    return expert


class _BoundedHillClimbSearch(HillClimbSearch):
    def __init__(self, data, max_outdegree):
        super().__init__(data)
        self.max_outdegree = max_outdegree

    def _legal_operations(self, model, *args, **kwargs):
        for operation, score in super()._legal_operations(model, *args, **kwargs):
            kind, (source, target) = operation
            # Reversing source -> target increases target's number of children.
            growing_node = source if kind == "+" else target
            if (self.max_outdegree is None or kind == "-"
                    or model.out_degree(growing_node) < self.max_outdegree):
                yield operation, score


class _BoundedGES(GES):
    def __init__(self, data, max_outdegree):
        super().__init__(data)
        self.max_outdegree = max_outdegree

    def _legal_edge_additions(self, model, expert):
        # pgmpy 1.0.0 starts GES empty, without inserting required edges.
        model.add_edges_from(expert.required_edges)
        return [(u, v) for u, v in super()._legal_edge_additions(model, expert)
                if self.max_outdegree is None or model.out_degree(u) < self.max_outdegree]

    def _legal_edge_flips(self, model, expert):
        # GES returns edges in their NEW direction, unlike HillClimbSearch.
        return [(u, v) for u, v in super()._legal_edge_flips(model, expert)
                if self.max_outdegree is None or model.out_degree(u) < self.max_outdegree]


class _BoundedExhaustiveSearch(ExhaustiveSearch):
    def __init__(self, data, scoring_method, max_outdegree, expert):
        super().__init__(data, scoring_method=scoring_method)
        self.max_outdegree = max_outdegree
        self.expert = expert
        self.expert._orient_temporal_forbidden_edges(nx.DiGraph(), only_edges=False)

    def all_dags(self, nodes=None):
        for model in super().all_dags(nodes):
            edges = set(model.edges())
            if (self.expert.required_edges <= edges
                    and not (edges & self.expert.forbidden_edges)
                    and (self.max_outdegree is None
                         or all(d <= self.max_outdegree for _, d in model.out_degree()))):
                yield model


class _RequiredEdgePC(PC):
    def build_skeleton(self, *args, expert_knowledge=None, **kwargs):
        # Skeleton edges are undirected, but pgmpy 1.0.0 checks required edges
        # in only the orientation yielded by graph.edges(). Protect both here;
        # estimate still receives the original directed knowledge for orientation.
        expert = deepcopy(expert_knowledge)
        expert.required_edges |= {(v, u) for u, v in expert.required_edges}
        return super().build_skeleton(*args, expert_knowledge=expert, **kwargs)


def _prune_pc_outdegree(model, data, max_outdegree, required_edges):
    """Cap PC's DAG by greedily removing edges with the smallest BIC loss.

    This changes the learned independence structure. Required edges are never
    removed; deleting edges preserves acyclicity and the existing orientations.
    """
    if max_outdegree is None:
        return
    scoring = None
    while any(d > max_outdegree for _, d in model.out_degree()):
        if scoring is None:
            scoring = BIC(data)
        candidates = []
        for source, target in model.edges():
            if model.out_degree(source) <= max_outdegree or (source, target) in required_edges:
                continue
            parents = model.get_parents(target)
            delta = (scoring.local_score(target, [p for p in parents if p != source])
                     - scoring.local_score(target, parents))
            candidates.append((delta, (source, target)))
        if not candidates:
            raise ValueError("Required edges prevent satisfying max_outdegree.")
        _, edge = max(candidates, key=lambda candidate: candidate[0])
        model.remove_edge(*edge)


# Exhaustive Search
def exhaustive_search(data, scoring_fn, expert_knowledge=None, max_outdegree=None, **kwargs) -> DiscreteBayesianNetwork:
    expert = _prepare_constraints(data, expert_knowledge, max_outdegree)
    scoring_instance = scoring_fn if isinstance(scoring_fn, str) else scoring_fn(data)
    es = _BoundedExhaustiveSearch(data, scoring_instance, max_outdegree, expert)
    best_model = es.estimate()
    bn = DiscreteBayesianNetwork(best_model.edges())
    return bn

# Hill Climb with tabu search
def hill_climbing(data, scoring_fn, expert_knowledge = None, max_indegree=10, tabu_length=10, epsilon=1e-6, max_iter=1000, max_outdegree=None, **kwargs) -> DiscreteBayesianNetwork:
    expert_knowledge = _prepare_constraints(data, expert_knowledge, max_outdegree)
    hc = _BoundedHillClimbSearch(data, max_outdegree)
    if isinstance(scoring_fn, str):
        scoring_instance = scoring_fn
    else:
        scoring_instance = scoring_fn(data)
    best_model = hc.estimate(scoring_method=scoring_instance, expert_knowledge=expert_knowledge, max_indegree=max_indegree,
                            tabu_length=tabu_length, epsilon=epsilon, max_iter=max_iter)
    bn = DiscreteBayesianNetwork(best_model.edges())
    return bn

# Greedy Equivalence Search
def ges(data, scoring_fn, expert_knowledge = None, max_outdegree=None, **kwargs) -> DiscreteBayesianNetwork:
    expert_knowledge = _prepare_constraints(data, expert_knowledge, max_outdegree)
    ges = _BoundedGES(data, max_outdegree)
    if isinstance(scoring_fn, str):
        scoring_instance = scoring_fn
    else:
        scoring_instance = scoring_fn(data)
    best_model = ges.estimate(scoring_method=scoring_instance, expert_knowledge=expert_knowledge)
    bn = DiscreteBayesianNetwork(best_model.edges())
    return bn

# Max-Min Hill Climbing
def mmhc(data, scoring_fn, expert_knowledge = None, max_outdegree=None, significance_level=0.01, **kwargs) -> DiscreteBayesianNetwork:
    expert = _prepare_constraints(data, expert_knowledge, max_outdegree)
    skeleton = MmhcEstimator(data).mmpc(significance_level=significance_level)
    # Restrict both orientations to the skeleton; explicit required edges take
    # precedence over skeleton screening.
    expert.forbidden_edges.update(
        (u, v) for u in data.columns for v in data.columns
        if u != v and not skeleton.has_edge(u, v) and (u, v) not in expert.required_edges
    )
    return hill_climbing(data, scoring_fn, expert_knowledge=expert,
                         max_outdegree=max_outdegree, **kwargs)

# PC algorithm
def pc(data, ci_test: str = 'chi_square', significance_level: float = 0.05, expert_knowledge = None, max_outdegree=None, **kwargs) -> DiscreteBayesianNetwork:
    expert_knowledge = _prepare_constraints(data, expert_knowledge, max_outdegree)
    ci_test_fn = ci_tests.get(ci_test, None)
    if ci_test_fn is None:
        raise ValueError("Invalid conditional independence test.")
    pc = _RequiredEdgePC(data) if expert_knowledge.required_edges else PC(data)
    return_type = 'pdag' if expert_knowledge.required_edges else 'dag'
    model = pc.estimate(return_type=return_type, ci_test=ci_test_fn, significance_level=significance_level, expert_knowledge=expert_knowledge, enforce_expert_knowledge=True)
    if expert_knowledge.required_edges:
        # With enforce_expert_knowledge=True and no temporal order, pgmpy 1.0.0
        # protects required adjacencies but can leave their orientation arbitrary.
        model = expert_knowledge.apply_expert_knowledge(model)
        model = pc.apply_orientation_rules(model, apply_r4=True).to_dag()
        if not expert_knowledge.required_edges <= set(model.edges()):
            raise ValueError("PC could not retain the required edge orientations.")
    _prune_pc_outdegree(model, data, max_outdegree, expert_knowledge.required_edges)
    bn = DiscreteBayesianNetwork(model.edges())
    return bn

# learn structure function
def learn_structure(data: pd.DataFrame, method: str = 'hill_climbing', **kwargs) -> DiscreteBayesianNetwork:

    scoring_method = kwargs.pop('scoring_method', None)

    # check if scoring method if present in kwargs is valid
    if method != "pc":
        if scoring_method not in scoring_functions.keys():
            raise ValueError(f"Scoring function {scoring_method} not recognized.")
        else:
            scoring_fn = scoring_functions[scoring_method]

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
