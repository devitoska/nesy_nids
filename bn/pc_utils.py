from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING
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

import numpy as np
import torch
import pyjuice as juice
from pyjuice.nodes import CircuitNodes as _CircuitNodes
from pyjuice.graph import RegionGraph

if TYPE_CHECKING:
    from pyjuice.nodes import CircuitNodes

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

def _gpu_cache_key(device):
    return (torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))


def _load_compiled_pc(path, key):
    """Read a compatible sidecar without constructing or compiling a graph."""
    try:
        with open(path, "rb") as handle:
            header = pickle.load(handle)
            if header.get("key") == key:
                circuit = torch.load(
                    io.BytesIO(handle.read()), map_location="cpu", weights_only=False,
                )
                return circuit, header["gpu"]
    except FileNotFoundError:
        pass
    except Exception as exc:
        logging.warning("Ignoring unusable PC cache %s: %s", path, exc)
    return None, None


def _save_compiled_pc(root, circuit, device):
    """Cache immediately after compilation, before runtime buffers/JIT kernels exist."""
    if not hasattr(root, "_pc_cache_path"):
        raise ValueError("Save the PC model before writing its compiled cache.")
    stream = io.BytesIO()
    torch.save(circuit, stream, pickle_module=_PCPickleModule)
    with _atomic_pc_file(root._pc_cache_path) as handle:
        pickle.dump({"key": root._pc_cache_key, "gpu": _gpu_cache_key(device)}, handle)
        handle.write(stream.getbuffer())


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
