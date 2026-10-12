"""Checkpoint admission for the selected-expert mmap reader."""
from contextlib import ExitStack, contextmanager
import hashlib
import mmap
from types import MappingProxyType
import threading

import numpy as np

from ..native_checkpoint import _open_local, _stamp
from .storage import positive


class K3MmapExpertProvider:
    """Reuse MappedMXFP4Experts, with checkpoint admission and lease budgets.

    Arrays are borrowed read-only views and must not escape a lease. One lease
    is admitted at a time. No resident expert bank or second mmap reader exists.
    """

    def __init__(self, checkpoint, *, max_expert_bytes, max_mapped_bytes=None):
        from ..k3_execution.packed import MappedMXFP4Experts, PackedTensorSpan

        positive(max_expert_bytes, "max_expert_bytes")
        if checkpoint.report.family != "kimi_k3":
            raise ValueError("exact packed K3 checkpoint required")
        rules = tuple(r for r in checkpoint.report.contracts
                      if r.encoding == "mxfp4-g32" and ".block_sparse_moe.experts." in r.sources[0])
        if not rules:
            raise ValueError("checkpoint has no explicitly mapped packed K3 expert bank")
        checkpoint._select(tuple(r.target for r in rules))
        self._metadata = checkpoint._metadata
        self.max_expert_bytes = max_expert_bytes
        self.max_mapped_bytes = positive(max_mapped_bytes if max_mapped_bytes is not None
                                        else max_expert_bytes + 6*mmap.ALLOCATIONGRANULARITY,
                                        "max_mapped_bytes")
        groups, shapes, spans = {}, {}, {}
        for rule in rules:
            for name in rule.sources:
                parts = name.split(".")
                groups.setdefault((int(parts[3]), int(parts[6])), []).append(name)
                tensor = checkpoint.tensors[name]
                shapes[name] = tensor.shape
                spans[name] = PackedTensorSpan(str(self._metadata.root / tensor.shard),
                                              tensor.base+tensor.offsets[0], tensor.shape)
        for names in groups.values():
            if len(names) != 6:
                raise ValueError("complete six-table expert lease required")
            tensors = [checkpoint.tensors[n] for n in names]
            if sum(t.nbytes for t in tensors) > self.max_expert_bytes:
                raise MemoryError("selected expert exceeds provider byte budget")
            mapped = sum(((t.nbytes+(t.base+t.offsets[0]) % mmap.ALLOCATIONGRANULARITY+
                           mmap.PAGESIZE-1)//mmap.PAGESIZE)*mmap.PAGESIZE for t in tensors)
            if mapped > self.max_mapped_bytes:
                raise MemoryError("aligned selected-expert maps exceed provider mapping budget")
        self._metadata.assert_unchanged()
        self._reader = MappedMXFP4Experts(spans, max_expert_bytes=max_expert_bytes,
                                         max_mapped_bytes=self.max_mapped_bytes)
        self._metadata.assert_unchanged()
        if self._reader.tensor_shapes != shapes:
            raise ValueError("packed reader inventory differs from admitted checkpoint")
        self.tensor_shapes = MappingProxyType(shapes)
        self._groups = MappingProxyType({key: tuple(value) for key, value in groups.items()})
        digest = hashlib.sha256((checkpoint.report.config_sha256 + self._reader.identity).encode("ascii"))
        self.identity = "k3-mmap-metadata-sha256:" + digest.hexdigest()
        self._lock, self._closed = threading.Lock(), False

    def _control_stamps(self):
        for filename in ("config.json", "model.safetensors.index.json"):
            expected = self._metadata.files.get(filename)
            if expected is None:
                if (self._metadata.root / filename).exists():
                    raise ValueError("publisher index presence changed")
                continue
            with ExitStack() as stack:
                source, stamp = _open_local(self._metadata.root, self._metadata.root / filename, stack)
                if stamp != expected or _stamp(source) != expected:
                    raise ValueError("publisher control metadata changed since planning")

    @contextmanager
    def lease(self, layer, expert):
        if type(layer) is not int or type(expert) is not int or (layer, expert) not in self._groups:
            raise ValueError("unknown exact K3 layer/expert selection")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("provider already has an active lease")
        tables = None
        try:
            if self._closed:
                raise RuntimeError("provider is closed")
            self._control_stamps()
            with ExitStack() as stack:
                handles = {}
                for name in self._groups[(layer, expert)]:
                    tensor = self._metadata.tensors[name]
                    if tensor.shard not in handles:
                        source, stamp = _open_local(self._metadata.root, self._metadata.root / tensor.shard, stack)
                        if stamp != tensor.stamp:
                            raise ValueError("expert shard changed since metadata planning")
                        handles[tensor.shard] = (source, stamp)
                with self._reader.lease(layer, expert) as tables:
                    prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}."
                    try:
                        for matrix in ("w1", "w3", "w2"):
                            for suffix in ("weight_packed", "weight_scale"):
                                name = matrix + "." + suffix
                                if (tables[name].dtype != np.uint8 or tables[name].shape != self.tensor_shapes[prefix+name]
                                        or not tables[name].flags.c_contiguous or tables[name].flags.writeable):
                                    raise ValueError("reader did not yield admitted readonly packed tables")
                            for start in range(0, tables[matrix + ".weight_scale"].size, 65536):
                                if np.any(tables[matrix + ".weight_scale"].reshape(-1)[start:start+65536] == 255):
                                    raise ValueError("selected expert has nonfinite UE8M0 scales")
                        yield tables
                        if any(_stamp(source) != stamp for source, stamp in handles.values()):
                            raise ValueError("expert shard changed during lease")
                        self._control_stamps()
                    finally:
                        # Invalidate the borrowed dictionary before reader unmaps its views.
                        tables.clear()
        finally:
            if tables is not None:
                tables.clear()
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close provider during an active lease")
        try:
            self._closed = True
            self._reader.close()
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("provider is closed")
        return self

    def __exit__(self, *exc):
        self.close()
