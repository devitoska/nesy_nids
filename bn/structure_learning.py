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

CLASS_NODE = "class"


def _validate_cap(value, name):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer or None.")


def _class_mb_longest_path(graph, node=CLASS_NODE):
    """Longest directed path (in edges) within the Markov blanket of `node`.

    The Markov blanket is node + parents + children + children's other parents;
    the path is measured in the subgraph induced by those nodes.
    """
    if node not in graph:
        return 0
    children = set(graph.successors(node))
    blanket = {node} | set(graph.predecessors(node)) | children
    for child in children:
        blanket.update(graph.predecessors(child))
    return nx.dag_longest_path_length(graph.subgraph(blanket))


def _edit_within_limit(graph, limit, add=None, remove=None):
    """Temporarily apply an edge edit to a scratch graph, test the limit, revert."""
    removed = added = False
    if remove is not None and graph.has_edge(*remove):
        graph.remove_edge(*remove)
        removed = True
    if add is not None and not graph.has_edge(*add):
        graph.add_edge(*add)
        added = True
    try:
        return _class_mb_longest_path(graph) <= limit
    finally:
        if added:
            graph.remove_edge(*add)
        if removed:
            graph.add_edge(*remove)


def _scratch_graph(model):
    g = nx.DiGraph()
    g.add_nodes_from(model.nodes())
    g.add_edges_from(model.edges())
    return g


def _check_class_present(data, max_lendirpath):
    if max_lendirpath is not None and CLASS_NODE not in data.columns:
        raise ValueError(f'max_lendirpath requires a "{CLASS_NODE}" column in the data.')


def _prepare_constraints(data, expert_knowledge, max_outdegree, max_lendirpath=None):
    """Validate the caps and required edges without mutating caller knowledge."""
    _validate_cap(max_outdegree, "max_outdegree")
    _validate_cap(max_lendirpath, "max_lendirpath")
    _check_class_present(data, max_lendirpath)
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
    if max_lendirpath is not None and _class_mb_longest_path(required) > max_lendirpath:
        raise ValueError("Required edges exceed max_lendirpath in the Markov blanket of the class node.")
    return expert


class _BoundedHillClimbSearch(HillClimbSearch):
    def __init__(self, data, max_outdegree, max_lendirpath=None):
        super().__init__(data)
        _check_class_present(data, max_lendirpath)
        self.max_outdegree = max_outdegree
        self.max_lendirpath = max_lendirpath

    def _legal_operations(self, model, *args, **kwargs):
        # Own scratch copy: the model is not modified while this generator is consumed.
        scratch = _scratch_graph(model) if self.max_lendirpath is not None else None
        for operation, score in super()._legal_operations(model, *args, **kwargs):
            kind, (source, target) = operation
            # Removals can only shrink the Markov blanket, so they are always legal.
            if kind == "-":
                yield operation, score
                continue
            # Reversing source -> target increases target's number of children.
            growing_node = source if kind == "+" else target
            if (self.max_outdegree is not None
                    and model.out_degree(growing_node) >= self.max_outdegree):
                continue
            if scratch is not None:
                if kind == "+":
                    ok = _edit_within_limit(scratch, self.max_lendirpath, add=(source, target))
                else:  # flip: source -> target becomes target -> source
                    ok = _edit_within_limit(scratch, self.max_lendirpath,
                                            add=(target, source), remove=(source, target))
                if not ok:
                    continue
            yield operation, score


class _BoundedGES(GES):
    def __init__(self, data, max_outdegree, max_lendirpath=None):
        super().__init__(data)
        _check_class_present(data, max_lendirpath)
        self.max_outdegree = max_outdegree
        self.max_lendirpath = max_lendirpath

    def _legal_edge_additions(self, model, expert):
        # pgmpy 1.0.0 starts GES empty, without inserting required edges.
        model.add_edges_from(expert.required_edges)
        candidates = [(u, v) for u, v in super()._legal_edge_additions(model, expert)
                      if self.max_outdegree is None or model.out_degree(u) < self.max_outdegree]
        if self.max_lendirpath is None:
            return candidates
        scratch = _scratch_graph(model)
        return [(u, v) for u, v in candidates
                if _edit_within_limit(scratch, self.max_lendirpath, add=(u, v))]

    def _legal_edge_flips(self, model, expert):
        # GES returns edges in their NEW direction, unlike HillClimbSearch.
        candidates = [(u, v) for u, v in super()._legal_edge_flips(model, expert)
                      if self.max_outdegree is None or model.out_degree(u) < self.max_outdegree]
        if self.max_lendirpath is None:
            return candidates
        scratch = _scratch_graph(model)
        return [(u, v) for u, v in candidates
                if _edit_within_limit(scratch, self.max_lendirpath, add=(u, v), remove=(v, u))]


class _BoundedExhaustiveSearch(ExhaustiveSearch):
    def __init__(self, data, scoring_method, max_outdegree, expert, max_lendirpath=None):
        super().__init__(data, scoring_method=scoring_method)
        _check_class_present(data, max_lendirpath)
        self.max_outdegree = max_outdegree
        self.max_lendirpath = max_lendirpath
        self.expert = expert
        self.expert._orient_temporal_forbidden_edges(nx.DiGraph(), only_edges=False)

    def all_dags(self, nodes=None):
        for model in super().all_dags(nodes):
            edges = set(model.edges())
            if (self.expert.required_edges <= edges
                    and not (edges & self.expert.forbidden_edges)
                    and (self.max_outdegree is None
                         or all(d <= self.max_outdegree for _, d in model.out_degree()))
                    and (self.max_lendirpath is None
                         or _class_mb_longest_path(model) <= self.max_lendirpath)):
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
def exhaustive_search(data, scoring_fn, expert_knowledge=None, max_outdegree=None, max_lendirpath=None, **kwargs) -> DiscreteBayesianNetwork:
    expert = _prepare_constraints(data, expert_knowledge, max_outdegree, max_lendirpath)
    scoring_instance = scoring_fn if isinstance(scoring_fn, str) else scoring_fn(data)
    es = _BoundedExhaustiveSearch(data, scoring_instance, max_outdegree, expert, max_lendirpath)
    best_model = es.estimate()
    bn = DiscreteBayesianNetwork(best_model.edges())
    return bn

# Hill Climb with tabu search
def hill_climbing(data, scoring_fn, expert_knowledge = None, max_indegree=10, tabu_length=10, epsilon=1e-6, max_iter=1000, max_outdegree=None, max_lendirpath=None, **kwargs) -> DiscreteBayesianNetwork:
    expert_knowledge = _prepare_constraints(data, expert_knowledge, max_outdegree, max_lendirpath)
    hc = _BoundedHillClimbSearch(data, max_outdegree, max_lendirpath)
    if isinstance(scoring_fn, str):
        scoring_instance = scoring_fn
    else:
        scoring_instance = scoring_fn(data)
    best_model = hc.estimate(scoring_method=scoring_instance, expert_knowledge=expert_knowledge, max_indegree=max_indegree,
                            tabu_length=tabu_length, epsilon=epsilon, max_iter=max_iter)
    bn = DiscreteBayesianNetwork(best_model.edges())
    return bn

# Greedy Equivalence Search
def ges(data, scoring_fn, expert_knowledge = None, max_outdegree=None, max_lendirpath=None, **kwargs) -> DiscreteBayesianNetwork:
    expert_knowledge = _prepare_constraints(data, expert_knowledge, max_outdegree, max_lendirpath)
    ges = _BoundedGES(data, max_outdegree, max_lendirpath)
    if isinstance(scoring_fn, str):
        scoring_instance = scoring_fn
    else:
        scoring_instance = scoring_fn(data)
    best_model = ges.estimate(scoring_method=scoring_instance, expert_knowledge=expert_knowledge)
    bn = DiscreteBayesianNetwork(best_model.edges())
    return bn

# Max-Min Hill Climbing
def mmhc(data, scoring_fn, expert_knowledge = None, max_outdegree=None, max_lendirpath=None, significance_level=0.01, **kwargs) -> DiscreteBayesianNetwork:
    expert = _prepare_constraints(data, expert_knowledge, max_outdegree, max_lendirpath)
    skeleton = MmhcEstimator(data).mmpc(significance_level=significance_level)
    # Restrict both orientations to the skeleton; explicit required edges take
    # precedence over skeleton screening.
    expert.forbidden_edges.update(
        (u, v) for u in data.columns for v in data.columns
        if u != v and not skeleton.has_edge(u, v) and (u, v) not in expert.required_edges
    )
    return hill_climbing(data, scoring_fn, expert_knowledge=expert,
                         max_outdegree=max_outdegree, max_lendirpath=max_lendirpath, **kwargs)

# PC algorithm
def pc(data, ci_test: str = 'chi_square', significance_level: float = 0.05, expert_knowledge = None, max_outdegree=None, max_lendirpath=None, **kwargs) -> DiscreteBayesianNetwork:
    if max_lendirpath is not None:
        raise ValueError("max_lendirpath is not supported by the PC algorithm.")
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