"""Build exact PyJuice circuits and evaluate batched BN probability queries.

The converter uses recursive conditioning, splits independent components, and
memoizes subcircuits by their remaining variables and assigned boundary parents.
It does not learn parameters or approximate the BN. Circuit size can nevertheless
be exponential; ``max_nodes`` stops construction rather than truncating it.

Example (PyJuice must be installed separately)::

    import pyjuice as juice
    from bn.pc_build import bn_to_pc, encode_evidence

    root = bn_to_pc(bn)
    values, missing = encode_evidence(root, [{"class": "normal"}, {}])
    # Variable IDs are columns in root.bn_variable_order. Values are state IDs,
    # not the original string/integer labels; missing entries use a separate mask.
    pc = juice.compile(root)
    pc.to("cuda")

``root`` is an initialized PyJuice SumNodes DAG, not a compiled TensorCircuit.
Construction runs on CPU. Compile/move/evaluate it separately for the installed
PyJuice backend. All BN variables, including isolated and latent nodes, remain in
the joint distribution. Do not reinitialize or train the circuit if equivalence
to the original BN is required. Parameters are stored as float64 during building;
PyJuice compilation/execution may use float32, so exactness is algebraic, subject
to floating-point rounding, not bitwise identity.

For ready-to-use probability queries, ``PCInference(root).posterior(evidence)``
handles backend selection, missing values, batching and normalization.
Pass a list of evidence dictionaries to query multiple observations together.
``save_pc`` stores the exact graph; ``load_pc`` also restores a compatible
compiled sidecar when available. The first CUDA inference builds that cache.
"""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from numbers import Integral
from typing import TYPE_CHECKING, Hashable
import pickle
import hashlib
import io
import logging
import os
import sys
import tempfile
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import pyjuice as juice
from pyjuice.nodes.distributions import Categorical
from pyjuice.nodes import CircuitNodes as _CircuitNodes
from pyjuice.graph import RegionGraph
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

if TYPE_CHECKING:
    from pyjuice.nodes import CircuitNodes


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
    The first CUDA PCInference created from the saved/loaded root also writes a
    ``<path>.compiled.pt`` sidecar. Later loads reuse that compiled structure.
    Saving new weights invalidates the old sidecar through a content hash.
    """
    from pyjuice.nodes import CircuitNodes

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


def load_pc(path: str) -> "CircuitNodes":
    """Load a saved root and, when compatible, its compiled inference cache.

    Restores shared subcircuits, parameter dtypes and values (without
    renormalization), and all BN metadata required by ``encode_evidence``.
    A cache hit bypasses the PyJuice node constructors. PCInference consumes the
    attached TensorCircuit without calling juice.compile on a matching GPU.
    Legacy files and missing/stale/corrupt caches fall back to reconstruction;
    the next CUDA PCInference rebuilds the sidecar. CPU-only loading is supported.
    Treat loaded roots as snapshots; call save_pc again after editing a root
    before constructing PCInference, so its cached parameters are invalidated.
    Only load trusted files and sidecars: pickle can execute code while loading.
    """
    key = _pc_cache_key(path)
    cache_path = str(path) + ".compiled.pt"
    try:
        with open(cache_path, "rb") as handle:
            header = pickle.load(handle)
            if header.get("key") == key:
                circuit = torch.load(
                    io.BytesIO(handle.read()), map_location="cpu", weights_only=False,
                )
                root = circuit.root_ns
                _attach_cache_location(root, path, key)
                root._pc_cached_circuit = circuit
                root._pc_cached_gpu = header["gpu"]
                return root
    except FileNotFoundError:
        pass
    except Exception as exc:
        logging.warning("Ignoring unusable PC cache %s: %s", cache_path, exc)

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if (not isinstance(payload, dict) or payload.get("format") != "bn_to_pc"
            or payload.get("version") != 1):
        raise ValueError("Unsupported PC file format or version.")
    records = payload.get("nodes")
    metadata = payload.get("metadata")
    metadata_names = (
        "bn_variable_order", "bn_state_names", "bn_state_to_index", "bn_num_nodes",
    )
    if (not isinstance(records, list) or not records
            or not isinstance(metadata, dict)
            or any(name not in metadata for name in metadata_names)):
        raise ValueError("PC file is missing its node table or BN metadata.")

    nodes = []
    for record in records:
        child_ids = record["children"]
        if any(not isinstance(i, int) or i < 0 or i >= len(nodes) for i in child_ids):
            raise ValueError("PC file contains an invalid child reference.")
        children = [nodes[i] for i in child_ids]
        kind = record["kind"]
        if kind == "input":
            params = torch.from_numpy(record["params"])
            node = juice.inputs(
                record["variable"], num_nodes=1, block_size=1,
                dist=Categorical(num_cats=params.numel()),
            )
        elif kind == "product":
            node = juice.multiply(*children)
        elif kind == "sum":
            node = juice.summate(*children, num_nodes=1, block_size=1)
        else:
            raise ValueError(f"Unknown circuit node kind {kind!r}.")
        if kind != "product":
            node.set_params(torch.from_numpy(record["params"]), normalize=False)
        nodes.append(node)

    root = nodes[-1]
    for name in metadata_names:
        setattr(root, name, metadata[name])
    _attach_cache_location(root, path, key)
    return root


@contextmanager
def _atomic_pc_file(path):
    """Never expose an incomplete model/cache to another reader."""
    path = Path(path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp_path = handle.name
            yield handle
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


@lru_cache(maxsize=1)
def _pc_runtime_key():
    # Include source content: local PyJuice modifications need not change its
    # package version. Cached objects depend on private PyJuice class layouts.
    digest = hashlib.sha256()
    package = Path(juice.__file__).parent
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return (sys.version_info[:3], str(torch.__version__), torch.version.cuda,
            version("pyjuice"), version("triton"), digest.hexdigest())


def _pc_cache_key(path):
    with open(path, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"format": 1, "source": digest, "runtime": _pc_runtime_key()}


def _attach_cache_location(root, path, key):
    # A root is a snapshot. Re-save it after editing its structure/parameters.
    root.__dict__.pop("_pc_cached_circuit", None)
    root.__dict__.pop("_pc_cached_gpu", None)
    root._pc_cache_path = str(Path(path).resolve()) + ".compiled.pt"
    root._pc_cache_key = key


def _restore_cached_region(cls, scope, children):
    region = cls.__new__(cls)
    region.__dict__.update(scope=scope, children=children, num_chs=len(children), parents=[])
    for child in children:
        child.parents.append(region)
    return region


def _restore_cached_node(cls, state, tensor_names):
    node = cls.__new__(cls)
    for name in tensor_names:
        state[name] = torch.from_numpy(state[name])
    node.__dict__.update(state)
    return node


class _PCPickler(pickle.Pickler):
    """Restore validated graph objects without re-running their constructors."""

    def reducer_override(self, obj):
        if isinstance(obj, RegionGraph):
            # Parent backreferences cause recursion overflow for large DAGs.
            # Reconstruct them from children during loading instead.
            return _restore_cached_region, (type(obj), obj.scope, obj.children)
        if isinstance(obj, _CircuitNodes):
            state = {k: v for k, v in vars(obj).items() if not k.startswith("_pc_")}
            tensor_names = []
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    # Avoid one torch ZIP storage entry for every scalar node.
                    state[name] = value.detach().cpu().numpy().copy()
                    tensor_names.append(name)
            return _restore_cached_node, (type(obj), state, tensor_names)
        return NotImplemented


class _PCPickleModule:
    __name__ = "pickle"
    Pickler = _PCPickler
    Unpickler = pickle.Unpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


def _gpu_cache_key(device):
    return (torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))


def _save_compiled_pc(root, circuit, device):
    """Cache immediately after compilation, before runtime buffers/JIT kernels exist."""
    if not hasattr(root, "_pc_cache_path"):
        return
    try:
        stream = io.BytesIO()
        torch.save(circuit, stream, pickle_module=_PCPickleModule)
        with _atomic_pc_file(root._pc_cache_path) as handle:
            pickle.dump({"key": root._pc_cache_key, "gpu": _gpu_cache_key(device)}, handle)
            handle.write(stream.getbuffer())
    except Exception as exc:
        # Read-only directories or cache failures must not prevent inference.
        logging.warning("Could not save compiled PC cache %s: %s", root._pc_cache_path, exc)


def encode_evidence(
    root: "CircuitNodes",
    evidence: Mapping | Sequence[Mapping],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode original BN labels as integer values and a Boolean missing mask.

    Accept one evidence mapping or a sequence of mappings. Return CPU tensors of
    shape ``(batch_size, num_variables)``: int64 values and bool mask (True means
    unobserved). Omit a variable to marginalize it; unknown variables/state labels
    raise ValueError rather than silently changing the query. Placeholder zeroes
    in missing columns are not observations: pass the mask to the query backend.
    """
    values, missing, _ = _encode_query_batch(root, evidence, deduplicate=False)
    return values, missing


def _encode_query_batch(root, evidence, *, deduplicate=True):
    """Encode on NumPy arrays and return an index restoring original row order.

    Canonical keys use column/state IDs, so dictionary insertion order and
    equivalent state-label representations do not affect deduplication. Every
    input row is validated, but repeated evidence is encoded/evaluated once.
    """
    rows = [evidence] if isinstance(evidence, Mapping) else list(evidence)
    order = root.bn_variable_order
    columns = {node: i for i, node in enumerate(order)}
    values = np.zeros((len(rows), len(order)), dtype=np.int64)
    missing = np.ones(values.shape, dtype=np.bool_)
    inverse = np.empty(len(rows), dtype=np.intp)
    seen = {}
    count = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("Each evidence row must be a mapping of variables to states.")
        encoded = []
        for node, state in row.items():
            if node not in columns:
                raise ValueError(f"Unknown BN variable {node!r}.")
            try:
                state_index = root.bn_state_to_index[node][state]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Unknown state {state!r} for variable {node!r}.") from exc
            encoded.append((columns[node], state_index))
        if deduplicate:
            key = tuple(sorted(encoded))
            if key in seen:
                inverse[row_index] = seen[key]
                continue
            seen[key] = count
        inverse[row_index] = count
        for column, state_index in encoded:
            values[count, column] = state_index
            missing[count, column] = False
        count += 1
    return torch.from_numpy(values[:count]), torch.from_numpy(missing[:count]), inverse


class _CPUCircuit:
    """Batch evaluator for the scalar categorical DAG produced by bn_to_pc.

    PyJuice's TensorCircuit compilation requires CUDA even when some compilation
    steps run on CPU. This evaluator instead plans the same DAG once and performs
    its arithmetic in float64 log space. It does not invoke BN inference.
    """

    def __init__(self, root):
        self.plan = []
        indices = {}
        pending = [(root, False)]
        while pending:
            ns, ready = pending.pop()
            if ns in indices:
                continue
            if not ready:
                pending.append((ns, True))
                pending.extend((child, False) for child in reversed(ns.chs))
                continue
            if ns.is_input():
                variable, = ns.scope.to_list()
                weights = ns.get_params().detach().cpu().double().clone()
                operation = ("input", variable, weights.log())
            else:
                children = tuple(indices[child] for child in ns.chs)
                weights = (
                    ns.get_params().detach().cpu().double().flatten().clone().log()
                    if ns.is_sum() else None
                )
                operation = ("sum" if ns.is_sum() else "product", children, weights)
            indices[ns] = len(self.plan)
            self.plan.append(operation)

    def __call__(self, values, missing):
        outputs = []
        for kind, inputs, weights in self.plan:
            if kind == "input":
                result = weights[values[:, inputs]]
                result = torch.where(
                    missing[:, inputs], torch.logsumexp(weights, dim=0), result
                )
            else:
                children = torch.stack([outputs[index] for index in inputs])
                result = (
                    torch.logsumexp(children + weights[:, None], dim=0)
                    if kind == "sum" else children.sum(dim=0)
                )
            outputs.append(result)
        return outputs[-1]


class PCInference:
    """Reusable, batched probability queries on a converted discrete BN.

    CUDA uses a PyJuice TensorCircuit; CPU uses the identical circuit with a
    float64 log-space evaluator. Both accept different evidence masks per row.
    Parameters are snapshots of the BN at construction time. Rebuild this object
    after changing the BN. Instances are intended for sequential calls, since
    PyJuice reuses evaluation buffers internally.

    Example::

        root = load_pc("model.pkl")
        infer = PCInference(root)  # CUDA if available, otherwise CPU
        probs = infer.posterior([{"feature": "value"}, {}], variable="class")
        # probs.shape == (2, number_of_class_states)

    ``batch_size`` bounds the number of joint queries evaluated together (each
    posterior row generates one joint query per target state). CPU chunks are
    additionally limited to roughly 64 MiB of stored node values, excluding
    temporary tensors. Compiled caches are specific to the software and GPU.
    CUDA execution also requires a working Triton/CUDA compiler toolchain; pass
    ``device="cpu"`` explicitly to use CPU even when a CUDA device is visible.
    """

    def __init__(self, root, *, device=None, batch_size=1024):
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be a positive integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("PC inference supports CPU and CUDA devices only.")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable.")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        self.root = root
        self.batch_size = int(batch_size)
        cached = root.__dict__.pop("_pc_cached_circuit", None)
        cached_gpu = root.__dict__.pop("_pc_cached_gpu", None)
        if self.device.type == "cuda":
            if cached is not None and cached_gpu == _gpu_cache_key(self.device):
                self.circuit = cached
            else:
                # Check before compiling/caching: float32 must preserve every
                # positive CPT weight. A compatible cache already passed this.
                for node in self.root:
                    if node.is_input() or node.is_sum():
                        parameters = node.get_params()
                        if ((parameters > 0) & (parameters.float() == 0)).any():
                            raise ValueError(
                                "Some CPT probabilities underflow in float32; "
                                "use PCInference(..., device='cpu') for this BN."
                            )
                self.circuit = juice.compile(self.root, device=self.device, verbose=False)
                _save_compiled_pc(root, self.circuit, self.device)
            self.circuit.to(self.device)
            self.circuit.eval()
        else:
            self.circuit = _CPUCircuit(self.root)
            self.batch_size = min(
                self.batch_size, max(1, (64 * 1024**2) // (8 * self.root.bn_num_nodes))
            )

    def _forward(self, values, missing):
        if self.device.type == "cpu":
            return self.circuit(values, missing)
        # Both tensors are batch-first; the flag also resolves square-mask
        # ambiguity in PyJuice when batch size equals the number of variables.
        return self.circuit(
            values.to(self.device), missing_mask=missing.to(self.device),
            _batch_first=True, force_use_fp32=True, apply_cudagraph=False,
        ).reshape(-1).to(device="cpu", dtype=torch.float64)

    @torch.inference_mode()
    def log_prob(self, evidence: Mapping | Sequence[Mapping]) -> np.ndarray:
        """Return one log P(evidence) per row; impossible evidence gives -inf."""
        values, missing, inverse = _encode_query_batch(self.root, evidence)
        result = torch.empty(len(values), dtype=torch.float64)
        for start in range(0, len(values), self.batch_size):
            stop = start + self.batch_size
            result[start:stop] = self._forward(values[start:stop], missing[start:stop])
        return result.numpy()[inverse]

    @torch.inference_mode()
    def posterior_for_states(
        self, evidence: Sequence[Mapping], states: Sequence[Hashable],
        variable: Hashable = "class",
    ) -> np.ndarray:
        """Return P(variable=state | evidence) for each paired evidence/state.

        Uses two likelihood queries per row, independent of class cardinality:
        P(evidence, state) / P(evidence). Repeated queries are evaluated once by
        log_prob. The target variable must be absent from each evidence mapping.
        Impossible evidence raises ValueError; an impossible target has mass 0.
        """
        if variable not in self.root.bn_state_names:
            raise ValueError(f"Unknown BN variable {variable!r}.")
        rows, states = list(evidence), list(states)
        if len(rows) != len(states):
            raise ValueError("Provide one target state per evidence row.")
        queries = []
        for row, state in zip(rows, states):
            if not isinstance(row, Mapping):
                raise TypeError("Each evidence row must be a mapping of variables to states.")
            if variable in row:
                raise ValueError(f"Query variable {variable!r} must not appear in evidence.")
            queries.extend((row, {**row, variable: state}))
        likelihoods = self.log_prob(queries).reshape(-1, 2)
        denominator, numerator = likelihoods[:, 0], likelihoods[:, 1]
        invalid = ~np.isfinite(denominator) | np.isnan(numerator) | np.isposinf(numerator)
        if invalid.any():
            raise ValueError(
                f"Evidence has zero probability or a nonfinite likelihood "
                f"in batch rows {np.flatnonzero(invalid).tolist()}; the posterior is undefined."
            )
        # Roundoff in float32 log likelihoods can put a unit probability just
        # above one; the mathematical log ratio cannot be positive.
        return np.exp(np.minimum(numerator - denominator, 0.0))

    @torch.inference_mode()
    def posterior(
        self, evidence: Mapping | Sequence[Mapping], variable: Hashable = "class",
    ) -> np.ndarray:
        """Compute P(variable | evidence), in the original CPD state order.

        A single mapping returns a 1-D array; a sequence returns a 2-D array,
        including shape (0, cardinality) for an empty batch. The target must be
        absent from every evidence mapping. Unknown states are rejected; callers
        that treat them as unobserved must remove them first. Zero-probability
        evidence raises ValueError because its posterior is undefined.
        """
        if variable not in self.root.bn_state_names:
            raise ValueError(f"Unknown BN variable {variable!r}.")
        single = isinstance(evidence, Mapping)
        rows = [evidence] if single else list(evidence)
        values, missing, inverse = _encode_query_batch(self.root, rows)
        column = self.root.bn_variable_order.index(variable)
        if (~missing[:, column]).any():
            raise ValueError(f"Query variable {variable!r} must not appear in evidence.")
        cardinality = len(self.root.bn_state_names[variable])
        log_joint = torch.empty((len(values), cardinality), dtype=torch.float64)
        flat = log_joint.view(-1)
        # Construct each chunk directly, avoiding a full batch x states x vars
        # expansion. Class-state queries for a row can cross a chunk boundary.
        for start in range(0, flat.numel(), self.batch_size):
            stop = min(start + self.batch_size, flat.numel())
            query_ids = torch.arange(start, stop)
            row_ids = query_ids // cardinality
            query_values = values[row_ids].clone()
            query_missing = missing[row_ids].clone()
            query_values[:, column] = query_ids % cardinality
            query_missing[:, column] = False
            flat[start:stop] = self._forward(query_values, query_missing)
        normalizer = torch.logsumexp(log_joint, dim=1, keepdim=True)
        invalid = ~torch.isfinite(normalizer[:, 0])
        if invalid.any():
            indices = np.flatnonzero(invalid.numpy()[inverse]).tolist()
            raise ValueError(
                f"Evidence has zero probability or a nonfinite likelihood "
                f"in batch rows {indices}; the posterior is undefined."
            )
        probabilities = (log_joint - normalizer).exp().numpy()[inverse]
        return probabilities[0] if single else probabilities
