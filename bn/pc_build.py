from collections.abc import Sequence
from numbers import Integral
from typing import Hashable
import pickle
import os

import networkx as nx
import torch
import pyjuice as juice
from pyjuice.nodes.distributions import Categorical
from pyjuice.nodes import CircuitNodes
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from bn.pc_utils import (
    _atomic_pc_file, _pc_cache_key, _attach_cache_location,
    _save_compiled_pc,
    # Old sidecars pickle these names under bn.pc_build. Keep them importable.
    _restore_cached_node, _restore_cached_region,
)
from bn.utils import load_bn

def bn_to_pc(
    bn: DiscreteBayesianNetwork,
    *,
    variable_order: Sequence[Hashable] | None = None,
    max_nodes: int | None = 1_000_000,
) -> "CircuitNodes":
    """Return a fully parameterized PyJuice circuit for ``bn``'s joint law.

    Args:
        bn: Nonempty fitted pgmpy DiscreteBayesianNetwork with TabularCPDs.
            The model and its CPDs are never modified.
        variable_order: Optional topological ordering containing every BN node
            exactly once. Controls both conditioning order and input columns.
            Defaults to NetworkX's topological order of the supplied model.
        max_nodes: Maximum number of scalar circuit nodes constructed, including
            leaves and wrapper nodes. None disables the limit. This is not a
            memory limit on subsequent PyJuice compilation.

    Returns:
        A PyJuice SumNodes root accepted by ``pyjuice.compile``. Metadata:
        ``bn_variable_order`` is the tuple of original variable names;
        ``bn_state_names`` maps names to tuples of original state labels;
        ``bn_state_to_index`` maps names to label-to-integer dictionaries;
        ``bn_num_nodes`` records the number of circuit nodes constructed.
        Metadata belongs to this root, not automatically to a TensorCircuit or
        a serialized/reloaded copy.

    Raises:
        TypeError: Unsupported model/CPDs or a noninteger node limit.
        ValueError: Empty/invalid BN, invalid order, or invalid probabilities.
        ImportError: PyJuice is unavailable.
        RuntimeError: Construction would exceed max_nodes.

    Each sum branches on one variable with its original conditional weights.
    Products combine disjoint scopes, and every branch of a sum has the same
    scope. Disconnected remaining components factor given their boundary
    parents. These properties preserve the full joint distribution, hence also
    all marginal/conditional queries with nonzero conditioning probability.
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

    # Keep the CPD's parent-axis order, which need not match graph insertion order.
    parents, tables, state_names = {}, {}, {}
    for i, node in enumerate(order):
        cpd = bn.get_cpds(node)
        parents[i] = tuple(indices[parent] for parent in cpd.variables[1:])
        shape = tuple(int(card) for card in cpd.cardinality)
        values = torch.as_tensor(cpd.get_values()).detach().to(
            device="cpu", dtype=torch.float64
        ).clone().reshape(shape)
        if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"CPD for {node!r} has invalid probabilities.")
        totals = values.sum(dim=0)
        if not torch.allclose(totals, torch.ones_like(totals), rtol=1e-7, atol=1e-8):
            raise ValueError(f"CPD for {node!r} is not normalized.")
        tables[i] = values
        state_names[node] = tuple(cpd.state_names[node])

    graph = nx.Graph()
    graph.add_nodes_from(range(len(order)))
    graph.add_edges_from((indices[u], indices[v]) for u, v in bn.edges)
    memo, layouts, leaves = {}, {}, {}
    node_count = 0

    def reserve_node():
        nonlocal node_count
        if max_nodes is not None and node_count >= max_nodes:
            raise RuntimeError(
                f"Exact BN conversion exceeded max_nodes={max_nodes}. "
                "Try another topological variable_order or increase the limit; "
                "no approximate circuit was returned."
            )
        node_count += 1

    def leaf(var, probabilities):
        key = (var, tuple(probabilities.tolist()))
        if key not in leaves:
            reserve_node()
            ns = juice.inputs(
                var, num_nodes=1, block_size=1,
                dist=Categorical(num_cats=len(probabilities)),
            )
            # Do not silently renormalize CPTs or add probability to zero states.
            ns.set_params(probabilities.clone(), normalize=False)
            leaves[key] = ns
        return leaves[key]

    def product(children):
        reserve_node()
        return juice.multiply(*children)

    def mixture(children, weights):
        reserve_node()
        ns = juice.summate(*children, num_nodes=1, block_size=1)
        ns.set_params(torch.tensor(weights, dtype=torch.float64), normalize=False)
        return ns

    def build(scope, assignments):
        if scope not in layouts:
            remaining = set(scope)
            boundary = tuple(sorted({
                parent for var in scope for parent in parents[var]
                if parent not in remaining
            }))
            components = tuple(sorted(
                (tuple(sorted(c)) for c in nx.connected_components(graph.subgraph(scope))),
                key=lambda c: c[0],
            ))
            layouts[scope] = boundary, components
        boundary, components = layouts[scope]
        context = {var: assignments[var] for var in boundary}
        key = scope, tuple(context[var] for var in boundary)
        if key in memo:
            return memo[key]

        if len(components) > 1:
            # A unit-weight sum keeps the product/sum alternation PyJuice requires.
            children = [build(component, context) for component in components]
            ns = mixture([product(children)], [1.0])
        else:
            var = scope[0]  # Topological order: every parent is already assigned.
            probabilities = tables[var][
                (slice(None),) + tuple(context[parent] for parent in parents[var])
            ]
            if len(scope) == 1:
                ns = leaf(var, probabilities)
            else:
                branches, weights = [], []
                for state, probability in enumerate(probabilities.tolist()):
                    if probability == 0.0:
                        continue  # Only mathematically zero branches are removed.
                    indicator = torch.zeros_like(probabilities)
                    indicator[state] = 1.0
                    suffix = build(scope[1:], {**context, var: state})
                    branches.append(product([leaf(var, indicator), suffix]))
                    weights.append(probability)
                ns = mixture(branches, weights)
        memo[key] = ns
        return ns

    root = build(tuple(range(len(order))), {})
    if root.is_input():
        root = mixture([product([root])], [1.0])
    root.bn_variable_order = order
    root.bn_state_names = state_names
    root.bn_state_to_index = {
        node: {state: i for i, state in enumerate(states)}
        for node, states in state_names.items()
    }
    root.bn_num_nodes = node_count
    return root

def save_pc(root: "CircuitNodes", path: str):
    """Save a ``bn_to_pc`` root, its exact weights, and BN metadata to one file.

    Uses a versioned pickle containing a flat node table: shared subcircuits are
    stored once, without pickling PyJuice's cyclic region graph or GPU buffers.
    This format supports the scalar categorical circuits built by this module;
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

    # Use the evaluator's iterative postorder traversal, independent of the
    # mutable traversal-order flag on PyJuice nodes.
    nodes, indices = [], {}
    pending = [(root, False)]
    while pending:
        node, ready = pending.pop()
        if node in indices:
            continue
        if not ready:
            pending.append((node, True))
            pending.extend((child, False) for child in reversed(node.chs))
            continue
        if node.num_nodes != 1 or node.block_size != 1:
            raise ValueError("Only scalar circuits produced by bn_to_pc are supported.")
        record = {"children": tuple(indices[child] for child in node.chs)}
        if node.is_input():
            if not isinstance(node.dist, Categorical) or len(node.scope) != 1:
                raise ValueError("Only single-variable categorical leaves are supported.")
            record.update(kind="input", variable=node.scope.to_list()[0])
        else:
            record["kind"] = "product" if node.is_prod() else "sum"
        if not node.is_prod():
            params = node.get_params()
            if params is None:
                raise ValueError("All circuit parameters must be initialized.")
            record["params"] = params.detach().cpu().numpy().copy()
        indices[node] = len(nodes)
        nodes.append(record)

    payload = {
        "format": "bn_to_pc", "version": 1, "nodes": nodes,
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
                    "Some CPT probabilities underflow in float32; "
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
