from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Hashable
import pickle

import numpy as np
import torch
import pyjuice as juice
from pyjuice.nodes.distributions import Categorical


from bn.pc_utils import (
    _pc_cache_key, _attach_cache_location, _gpu_cache_key,
    _encode_query_batch, _load_compiled_pc,
)

if TYPE_CHECKING:
    from pyjuice.nodes import CircuitNodes

def load_pc(path: str) -> "CircuitNodes":
    """Load a saved root and, when compatible, its compiled inference cache.

    Restores shared subcircuits, parameter dtypes and values (without
    renormalization), and all BN metadata required by ``encode_evidence``.
    A cache hit bypasses the PyJuice node constructors. PCInference consumes the
    attached TensorCircuit without calling juice.compile on a matching GPU.
    Legacy files and missing/stale/corrupt caches fall back to reconstruction;
    CPU inference can use that fallback, but CUDA inference requires an explicit
    construct_cache(root, path) call from bn.pc_build beforehand.
    Treat loaded roots as snapshots; call save_pc again after editing a root
    before constructing PCInference, so its cached parameters are invalidated.
    Only load trusted files and sidecars: pickle can execute code while loading.
    """
    key = _pc_cache_key(path)
    circuit, gpu = _load_compiled_pc(str(path) + ".compiled.pt", key)
    if circuit is not None:
        root = circuit.root_ns
        _attach_cache_location(root, path, key)
        root._pc_cached_circuit = circuit
        root._pc_cached_gpu = gpu
        return root

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
    CUDA requires a compatible cache prepared by construct_cache during training.
    This class never compiles a circuit or writes a cache.
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
            if cached is None and hasattr(root, "_pc_cache_path"):
                cached, cached_gpu = _load_compiled_pc(root._pc_cache_path, root._pc_cache_key)
            if cached is not None and cached_gpu == _gpu_cache_key(self.device):
                self.circuit = cached
            else:
                raise RuntimeError(
                    "No compatible compiled PC cache is attached to this root. "
                    "Run bn.pc_build.construct_cache(root, model_path, device=...) "
                    "on the target CUDA device during preparation, then load_pc(model_path) "
                    "before CUDA inference. Inference does not rebuild missing, stale, "
                    "corrupt, or GPU-incompatible caches."
                )
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
