"""Selected packed experts for the explicit V4.1 equation schema."""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
import threading
from types import MappingProxyType

import numpy as np


def packed_tensor_shapes(config):
    result = {}
    for layer in range(config.layers):
        for name,(rows,cols) in {'gate':(config.intermediate,config.hidden),
                                'up':(config.intermediate,config.hidden),
                                'down':(config.hidden,config.intermediate)}.items():
            result[f'layers.{layer}.moe.{name}.codes']=(config.experts,rows,cols//2)
            result[f'layers.{layer}.moe.{name}.scales']=(config.experts,rows,cols//32)
    return result


@dataclass(frozen=True)
class PackedExpertBinding:
    layer: int
    expert: int
    matrix: str
    part: str
    region_key: str


class V41PackedExperts:
    """Borrow a hashed PackedStore; six immutable regions per expert.

    Bindings identify the declared equation schema, not publisher name aliases.
    The caller owns the store and its mapping budget. No files are converted,
    fetched, or automatically admitted as publisher-compatible.
    """
    format='e2m1-low-first-e8m0-g32'

    def __init__(self, config, store, bindings, *, max_expert_bytes):
        if config.hidden%32 or config.intermediate%32 or type(max_expert_bytes) is not int or max_expert_bytes<=0:
            raise ValueError('group-32 widths and positive selected-expert budget required')
        self.max_expert_bytes,self.store=max_expert_bytes,store
        self.tensor_shapes=MappingProxyType(packed_tensor_shapes(config))
        self._records={}
        self._shapes={}
        digest=hashlib.sha256(repr(config).encode('ascii'))
        expected_count=config.layers*config.experts*6
        from itertools import islice
        records=tuple(islice(iter(bindings),expected_count+1))
        if len(records)!=expected_count: raise ValueError('exact six-region expert inventory required')
        dims={'gate':(config.intermediate,config.hidden),'up':(config.intermediate,config.hidden),
              'down':(config.hidden,config.intermediate)}
        seen=set()
        for record in records:
            if (type(record) is not PackedExpertBinding or type(record.layer) is not int or
                type(record.expert) is not int or not 0<=record.layer<config.layers or
                not 0<=record.expert<config.experts or record.matrix not in dims or
                record.part not in ('codes','scales') or type(record.region_key) is not str):
                raise ValueError('invalid packed expert binding')
            key=(record.layer,record.expert,record.matrix+'.'+record.part)
            if key in seen: raise ValueError('duplicate selected-expert binding')
            seen.add(key)
            rows,cols=dims[record.matrix]
            shape=(rows,cols//(2 if record.part=='codes' else 32))
            region=store.regions[record.region_key]
            if region.packed_format!=self.format or region.size!=shape[0]*shape[1]:
                raise ValueError('packed region format/extent mismatch')
            self._records[key]=(record.region_key,region)
            self._shapes[key]=shape
            digest.update(json.dumps((key,record.region_key,region.path,region.offset,
                                     region.size,region.sha256,region.storage)).encode('utf-8'))
        selected_bytes=sum(rows*cols//2+rows*cols//32 for rows,cols in dims.values())
        if selected_bytes>max_expert_bytes: raise MemoryError('selected packed expert exceeds budget')
        self.identity='v41-packed-regions:'+digest.hexdigest()
        self._lock=threading.Lock()

    @contextmanager
    def lease(self,layer,expert):
        if type(layer) is not int or type(expert) is not int or (layer,expert,'gate.codes') not in self._records:
            raise ValueError('unknown packed expert')
        if not self._lock.acquire(blocking=False): raise RuntimeError('expert provider lease already active')
        tables={}
        try:
            with ExitStack() as stack:
                try:
                    for name in ('gate.codes','gate.scales','up.codes','up.scales','down.codes','down.scales'):
                        key=(layer,expert,name)
                        region_key,region=self._records[key]
                        if self.store.regions.get(region_key)!=region: raise ValueError('packed manifest changed')
                        payload=stack.enter_context(self.store.acquire(region_key))
                        tables[name]=np.frombuffer(payload,dtype=np.uint8).reshape(self._shapes[key])
                    yield tables
                finally: tables.clear()
        finally:
            self._lock.release()
