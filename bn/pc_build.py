from collections.abc import Sequence
from numbers import Integral
from typing import Hashable
import pickle
import os
import logging

import networkx as nx
import torch
import pyjuice as juice
from pyjuice.nodes.distributions import Categorical
from pyjuice.nodes import CircuitNodes
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from bn.pc_utils import (
    _atomic_pc_file, _pc_cache_key, _attach_cache_location,
    _save_compiled_pc, _choose_elimination_order, _elimination_plan,
    _align_factor, _project_assignments, _pc_postorder,
    # Old sidecars pickle these names under bn.pc_build. Keep them importable.
    _restore_cached_node, _restore_cached_region,
)

from bn.utils import load_bn

def bn_to_pc(
    bn: DiscreteBayesianNetwork,
    *,
    variable_order: Sequence[Hashable] | None = None,
    elimination_order: Sequence[Hashable] | None = None,
    max_nodes: int | None = 1_000_000,
) -> "CircuitNodes":
    """Compile the full BN joint law by exact, packed bucket elimination.

    ``variable_order`` controls input columns and retains the historical
    topological-order requirement. ``elimination_order`` may be any permutation;
    by default a weighted factor-scope heuristic minimizes estimated scalar
    circuit size. No BN edges, CPT entries, states, or evidence are approximated.

    Each bucket emits a vector of normalized subcircuits, indexed by its
    separator assignments, plus log probability masses for subsequent buckets.
    Keeping masses separate is essential: PyJuice normalizes sum weights when
    compiling. Products combine disjoint eliminated-variable scopes; sums mix
    equal scopes. The final circuit supports arbitrary marginal/conditional
    queries, not only a fixed evidence pattern.

    Packing uses indexed multi-node groups with block_size=1, avoiding a Python
    object per scalar node and dense all-to-all sum connectivity. Complexity is
    still exponential in induced separator cardinality, not bounded by degree.
    ``max_nodes`` limits scalar nodes, not groups or bytes; None disables it.
    Metadata records input/elimination orders, state labels, actual node counts,
    and dense factor-size estimates. ``bn_log_scale`` retains numerical mass
    accumulated during normalization and is applied by PCInference.
    """
    if not isinstance(bn, DiscreteBayesianNetwork):
        raise TypeError("bn must be a pgmpy DiscreteBayesianNetwork.")
    if not bn.nodes:
        raise ValueError("Cannot convert an empty Bayesian network.")
    if max_nodes is not None:
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, Integral):
            raise TypeError("max_nodes must be a positive integer or None.")
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive or None.")
    if not nx.is_directed_acyclic_graph(bn):
        raise ValueError("The Bayesian network must be acyclic.")
    for node in bn.nodes:
        if not isinstance(bn.get_cpds(node), TabularCPD):
            raise TypeError(f"Variable {node!r} must have a fitted TabularCPD.")
    if not bn.check_model():
        raise ValueError("The Bayesian network failed check_model().")

    order = tuple(nx.topological_sort(bn) if variable_order is None else variable_order)
    if len(order) != len(bn.nodes) or set(order) != set(bn.nodes):
        raise ValueError("variable_order must contain every BN node exactly once.")
    indices = {node: i for i, node in enumerate(order)}
    if any(indices[u] >= indices[v] for u, v in bn.edges):
        raise ValueError("variable_order must be a topological ordering of the BN.")
    cards = tuple(int(bn.get_cpds(node).variable_card) for node in order)
    scopes = [tuple(indices[v] for v in bn.get_cpds(node).variables) for node in order]
    if elimination_order is None:
        elimination = _choose_elimination_order(scopes, cards)
    else:
        elimination_order = tuple(elimination_order)
        if len(elimination_order) != len(order) or set(elimination_order) != set(order):
            raise ValueError("elimination_order must contain every BN node exactly once.")
        elimination = tuple(indices[v] for v in elimination_order)
    plan = _elimination_plan(scopes, cards, elimination)
    if max_nodes is not None and plan["scalar_nodes"] > max_nodes:
        raise RuntimeError(
            f"Exact BN conversion exceeded max_nodes={max_nodes}: selected elimination "
            f"order requires {plan['scalar_nodes']} scalar nodes in {plan['node_groups']} "
            "packed groups. Try another elimination_order or increase max_nodes; "
            "no approximate circuit was returned."
        )

    factors, state_names = [], {}
    for node, scope in zip(order, scopes):
        cpd = bn.get_cpds(node)
        values = torch.as_tensor(cpd.get_values()).detach().to(
            device="cpu", dtype=torch.float64
        ).clone().reshape(tuple(cards[v] for v in scope))
        if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"CPD for {node!r} has invalid probabilities.")
        totals = values.sum(dim=0)
        if not torch.allclose(totals, torch.ones_like(totals), rtol=1e-7, atol=1e-8):
            raise ValueError(f"CPD for {node!r} is not normalized.")
        factors.append((scope, values.log(), None))
        state_names[node] = tuple(cpd.state_names[node])

    def inputs(var, probabilities):
        ns = juice.inputs(var, num_nodes=probabilities.shape[0], block_size=1,
                          dist=Categorical(num_cats=cards[var]))
        ns.set_params(probabilities, normalize=False)
        return ns

    for var, separator, entries, _ in plan["steps"]:
        bucket = [factor for factor in factors if var in factor[0]]
        factors = [factor for factor in factors if var not in factor[0]]
        target = (*separator, var)
        coefficients = torch.zeros(tuple(cards[v] for v in target), dtype=torch.float64)
        messages = []
        for scope, log_values, node in bucket:
            coefficients.add_(_align_factor(log_values, scope, target, cards))
            if node is not None:
                messages.append((scope, node))
        log_mass = torch.logsumexp(coefficients, dim=-1)
        # An unreachable separator row may use any normalized subcircuit: its
        # outgoing mass remains exactly zero. Avoid -inf - -inf and NaNs.
        reachable = torch.isfinite(log_mass)
        log_weights = coefficients - torch.where(reachable, log_mass, 0.0).unsqueeze(-1)
        weights = log_weights.exp().reshape(-1, cards[var])
        if ((weights == 0) & torch.isfinite(log_weights.reshape_as(weights))).any():
            raise ValueError("Elimination weights underflow float64; no approximate circuit was returned.")
        weights[~reachable.reshape(-1), 0] = 1.0
        if not messages:
            node = inputs(var, weights)
        else:
            indicator = inputs(var, torch.eye(cards[var], dtype=torch.float64))
            assignments = torch.arange(entries)
            edges = [assignments % cards[var]]
            edges.extend(_project_assignments(assignments, target, scope, cards)
                         for scope, _ in messages)
            product = juice.multiply(indicator, *(node for _, node in messages),
                                     edge_ids=torch.stack(edges, dim=1))
            contexts = entries // cards[var]
            node = juice.summate(
                product, num_nodes=contexts, block_size=1,
                edge_ids=torch.stack((assignments // cards[var], assignments)),
            )
            node.set_params(weights.flatten(), normalize=False)
        factors.append((separator, log_mass, node))

    roots = [node for _, _, node in factors]
    root = roots[0]
    if len(roots) > 1 or root.is_input():
        product = juice.multiply(*roots)
        root = juice.summate(product, num_nodes=1, block_size=1)
        root.set_params(torch.ones(1, dtype=torch.float64), normalize=False)
    root.bn_variable_order = order
    root.bn_elimination_order = tuple(order[v] for v in elimination)
    root.bn_state_names = state_names
    root.bn_state_to_index = {
        node: {state: i for i, state in enumerate(states)}
        for node, states in state_names.items()
    }
    root.bn_num_nodes = plan["scalar_nodes"]
    root.bn_num_node_groups = plan["node_groups"]
    root.bn_log_scale = sum(float(mass) for _, mass, _ in factors)
    root.bn_compilation_stats = {key: value for key, value in plan.items() if key != "steps"}
    logging.info("Exact packed VE: %s scalar nodes in %s groups; peak factor %s entries",
                 root.bn_num_nodes, root.bn_num_node_groups, plan["peak_factor_entries"])
    return root

def save_pc(root: "CircuitNodes", path: str):
    """Save a ``bn_to_pc`` root, its exact weights, and BN metadata to one file.

    Uses a versioned pickle containing a flat node table: shared subcircuits are
    stored once, without pickling PyJuice's cyclic region graph or GPU buffers.
    This format supports packed categorical circuits with scalar blocks;
    pass the root, not a compiled TensorCircuit or PCInference instance. The
    parent directory must exist. A ``.pkl`` suffix is recommended, not required.
    Call construct_cache(root, path) during training to also write the
    ``<path>.compiled.pt`` sidecar required for CUDA inference.
    Saving new weights invalidates the old sidecar through a content hash.
    """
    if not isinstance(root, CircuitNodes):
        raise TypeError("root must be a CircuitNodes root returned by bn_to_pc.")
    metadata_names = (
        "bn_variable_order", "bn_state_names", "bn_state_to_index", "bn_num_nodes",
    )
    if any(not hasattr(root, name) for name in metadata_names):
        raise ValueError("The circuit is missing bn_to_pc metadata.")
    metadata_names += tuple(name for name in (
        "bn_elimination_order", "bn_num_node_groups", "bn_log_scale", "bn_compilation_stats",
    ) if hasattr(root, name))

    # Use the evaluator's iterative postorder traversal, independent of the
    # mutable traversal-order flag on PyJuice nodes.
    nodes, indices = [], {}
    for node in _pc_postorder(root):
        if node.block_size != 1:
            raise ValueError("Only block_size=1 circuits produced by bn_to_pc are supported.")
        record = {"children": tuple(indices[child] for child in node.chs),
                  "num_nodes": node.num_nodes}
        if node.is_input():
            if not isinstance(node.dist, Categorical) or len(node.scope) != 1:
                raise ValueError("Only single-variable categorical leaves are supported.")
            record.update(kind="input", variable=node.scope.to_list()[0],
                          num_cats=node.dist.num_cats)
        else:
            record["kind"] = "product" if node.is_prod() else "sum"
            record["edge_ids"] = node.edge_ids.detach().cpu().numpy().copy()
        if not node.is_prod():
            params = node.get_params()
            if params is None:
                raise ValueError("All circuit parameters must be initialized.")
            record["params"] = params.detach().cpu().numpy().copy()
        indices[node] = len(nodes)
        nodes.append(record)

    payload = {
        "format": "bn_to_pc", "version": 2, "nodes": nodes,
        "metadata": {name: getattr(root, name) for name in metadata_names},
    }
    with _atomic_pc_file(path) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _attach_cache_location(root, path, _pc_cache_key(path))

def construct_cache(root: CircuitNodes, path: str, *, device="cuda"):
    """Save the model and its compiled CUDA cache during training.

    Requires a CUDA device matching the inference host. Returns the sidecar
    path; load_pc(path) loads it for inference.
    Compilation and cache-write errors propagate to the training caller.
    CPU inference only needs save_pc and does not require this cache.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Compiled PyJuice caches require a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to construct a compiled PC cache.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    for node in root:
        if node.is_input() or node.is_sum():
            parameters = node.get_params()
            if ((parameters > 0) & (parameters.float() == 0)).any():
                raise ValueError(
                    "Some circuit probabilities underflow in float32; "
                    "use save_pc and PCInference(..., device='cpu') for this BN."
                )
    save_pc(root, path)
    circuit = juice.compile(root, device=device, verbose=False)
    _save_compiled_pc(root, circuit, device)
    return root._pc_cache_path


def bn_to_pc_loop(exp_name, max_nodes = 1_000_000, *, device=None):
    """Rebuild saved BNs' circuits; also construct caches when using CUDA.

    Always rebuild after training so existing PCs cannot retain old BN weights.
    On CPU-only hosts save the exact model for CPU inference; CUDA deployment
    requires an explicit construct_cache call on a compatible CUDA host.
    """
    device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type not in ("cpu", "cuda"):
        raise ValueError("PC construction supports CPU and CUDA devices only.")
    path = f"results/{exp_name}/bn"
    for partition in sorted(os.listdir(path)):
        partition_path = os.path.join(path, partition)
        if not partition.startswith("no_") or not os.path.isdir(partition_path):
            continue
        bn_path = os.path.join(partition_path, "bn_model.xmlbif")
        pc_path = os.path.join(partition_path, "pc_root.pkl")
        if not os.path.exists(bn_path):
            raise FileNotFoundError(f"BN model file not found: {bn_path}")
        bn = load_bn(partition_path)
        root = bn_to_pc(bn, max_nodes=max_nodes)
        if device.type == "cuda":
            construct_cache(root, pc_path, device=device)
        else:
            save_pc(root, pc_path)
