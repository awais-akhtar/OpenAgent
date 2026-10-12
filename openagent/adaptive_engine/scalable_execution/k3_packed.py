"""Selected-expert mmap leases for the explicit K3 provider contract."""
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import mmap

import numpy as np

from ..k3_execution.executor import tensor_shapes
from .resources import PackedStore, check_cancel, positive


class K3PackedExpertProvider:
    """Borrow a PackedStore; expose only six uint8 tables per routed expert.

    Region keys are exact decoder tensor names. No whole-shard reads or dense
    expert conversion occur here. The caller owns the store and its map budget.
    """

    def __init__(self, config, store, *, max_expert_bytes, cancelled=None):
        if not isinstance(store, PackedStore) or config.expert_format == 'fp32':
            raise ValueError('packed K3 config and PackedStore required')
        self.tensor_shapes = {name: shape for name, shape in tensor_shapes(config).items()
                              if '.experts.' in name}
        self.max_expert_bytes = positive(max_expert_bytes, 'max_expert_bytes')
        self.store, self.config, self.cancelled = store, config, cancelled
        inventory = []
        expert_bytes = {}
        mapped_bytes = {}
        self._regions = {}
        for name, shape in sorted(self.tensor_shapes.items()):
            region = store.regions.get(name)
            if region is None or region.size != math.prod(shape) or region.packed_format != config.expert_format:
                raise ValueError('packed K3 region inventory/extent/format mismatch')
            prefix = name.rsplit('.', 2)[0]
            expert_bytes[prefix] = expert_bytes.get(prefix, 0) + region.size
            extent = region.offset % mmap.ALLOCATIONGRANULARITY + region.size
            mapped_bytes[prefix] = mapped_bytes.get(prefix, 0) + (
                (extent + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE)
            inventory.append((name, shape, region.size, region.sha256, region.packed_format))
            self._regions[name] = region
        if any(size > self.max_expert_bytes for size in expert_bytes.values()):
            raise MemoryError('selected K3 expert exceeds provider payload budget')
        if any(size > store.budget.capacity_bytes for size in mapped_bytes.values()):
            raise MemoryError('selected K3 expert aligned mappings exceed store budget')
        self.identity = hashlib.sha256(json.dumps(
            ['k3-selected-expert-v1', inventory], separators=(',', ':'),
            allow_nan=False).encode('ascii')).hexdigest()

    @contextmanager
    def lease(self, layer, expert):
        c = self.config
        if (type(layer) is not int or not c.dense_prefix <= layer < len(c.layer_types) or
                type(expert) is not int or not 0 <= expert < c.experts):
            raise ValueError('invalid routed K3 layer/expert')
        check_cancel(self.cancelled)
        prefix = f'language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.'
        tables = {}
        with ExitStack() as stack:
            try:
                for matrix in ('w1', 'w3', 'w2'):
                    for suffix in ('weight_packed', 'weight_scale'):
                        local = matrix + '.' + suffix
                        name = prefix + local
                        if self.store.regions.get(name) != self._regions[name]:
                            raise ValueError('K3 packed region descriptor changed')
                        payload = stack.enter_context(self.store.acquire(name, cancelled=self.cancelled))
                        tables[local] = np.frombuffer(payload, dtype=np.uint8).reshape(self.tensor_shapes[name])
                check_cancel(self.cancelled)
                yield tables
            finally:
                # Drop ndarray buffer exports before ExitStack unmaps the leases.
                tables.clear()
