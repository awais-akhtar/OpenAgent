"""Persistent cache transactions: mutable containers, immutable row payloads."""
import ctypes as C
from dataclasses import dataclass, field

import numpy as np

from .retrieved_engram import RetrievedEngram


@dataclass(frozen=True)
class V41LayerSnapshot:
    local: tuple
    pending_kv: tuple
    pending_z: tuple
    global_kv: tuple
    index_keys: tuple
    owner: object
    selector: object
    selected: tuple
    candidates: tuple


@dataclass(frozen=True)
class V41CacheSnapshot:
    layers: tuple
    engram: tuple


def immutable_row(row):
    return np.frombuffer(row.tobytes(),dtype=np.float32)


def _session_snapshot(model, session):
    if isinstance(model,RetrievedEngram):
        return ('retrieved',session.snapshot())
    return ('legacy',bytes(session._state))


def _restore_session(model, session, payload, compressed, *, validate):
    if type(payload) is not tuple or len(payload)!=2:
        raise ValueError('invalid Engram cache snapshot')
    kind,data = payload
    if isinstance(model,RetrievedEngram):
        if kind!='retrieved': raise ValueError('Engram cache kind mismatch')
        session.restore(data)
        if validate and data!=(len(compressed),compressed[-3:]):
            raise ValueError('retrieved Engram history differs from token history')
    else:
        if kind!='legacy' or type(data) is not bytes or len(data)!=C.sizeof(session._state):
            raise ValueError('legacy Engram cache ABI/extent mismatch')
        candidate = type(session._state).from_buffer_copy(data)
        if validate and (candidate.abi!=1 or candidate.reserved or
                         candidate.capacity!=session._state.capacity or candidate.length!=len(compressed) or
                         tuple(candidate.tokens[:candidate.length])!=compressed or
                         any(candidate.tokens[candidate.length:])):
            raise ValueError('invalid legacy Engram cached history')
        C.memmove(C.byref(session._state),C.byref(candidate),C.sizeof(candidate))


@dataclass
class _Layer:
    local: list = field(default_factory=list)
    pending_kv: list = field(default_factory=list)
    pending_z: list = field(default_factory=list)
    global_kv: list = field(default_factory=list)
    index_keys: list = field(default_factory=list)
    owner: object = None
    selector: object = None
    selected: tuple = ()
    candidates: tuple = ()

    def clone(self):
        # Row bytes and FP32 vectors backed by immutable bytes can be shared.
        return _Layer(list(self.local),list(self.pending_kv),list(self.pending_z),
                      list(self.global_kv),list(self.index_keys),self.owner,self.selector,
                      self.selected,self.candidates)

    def snapshot(self):
        return V41LayerSnapshot(tuple(self.local),tuple(x.tobytes() for x in self.pending_kv),
             tuple(x.tobytes() for x in self.pending_z),tuple(self.global_kv),
             tuple(x.tobytes() for x in self.index_keys),self.owner,self.selector,
             self.selected,self.candidates)


class ExecutionState:
    def __init__(self, owner):
        self.tokens = ()
        self.layers = [_Layer() for _ in range(owner.config.layers)]
        self.sessions = [model.session(owner.config.capacity) for model in owner._engram]

    def clone(self, owner):
        candidate = ExecutionState(owner)
        candidate.tokens = self.tokens
        candidate.layers = [layer.clone() for layer in self.layers]
        for model,source,target in zip(owner._engram,self.sessions,candidate.sessions):
            _restore_session(model,target,_session_snapshot(model,source),(),validate=False)
        return candidate

    def snapshot(self, owner):
        return V41CacheSnapshot(tuple(layer.snapshot() for layer in self.layers),
            tuple(_session_snapshot(model,session) for model,session in zip(owner._engram,self.sessions)))

    @classmethod
    def restore(cls, owner, tokens, snapshot):
        c = owner.config
        if (type(snapshot) is not V41CacheSnapshot or type(snapshot.layers) is not tuple or
            type(snapshot.engram) is not tuple or len(snapshot.layers)!=c.layers or len(snapshot.engram)!=c.layers):
            raise ValueError('cache snapshot layer inventory mismatch')
        candidate = cls(owner)
        candidate.tokens = tokens
        compressed = tuple(owner._compressed[t] for t in tokens)
        for model,session,payload in zip(owner._engram,candidate.sessions,snapshot.engram):
            _restore_session(model,session,payload,compressed,validate=True)
        packed_width = c.hidden//16*9

        def packed(row):
            if type(row) is not bytes or len(row)!=packed_width or any(s>126 for s in row[::9]):
                raise ValueError('invalid cached private FP4 row')
            return row

        def vector(row):
            if type(row) is not bytes or len(row)!=c.hidden*4:
                raise ValueError('invalid cached FP32 row extent')
            result = np.frombuffer(row,dtype=np.float32)
            if not np.isfinite(result).all(): raise ValueError('nonfinite cached FP32 row')
            return result

        def indices(values, maximum, limit):
            if (type(values) is not tuple or len(values)>limit or
                any(type(v) is not int or not 0<=v<maximum for v in values) or len(set(values))!=len(values)):
                raise ValueError('invalid cache selection/candidate indices')
            return values

        owner_layer = selector = None
        for l,(saved,mode) in enumerate(zip(snapshot.layers,c.modes)):
            if type(saved) is not V41LayerSnapshot or any(type(v) is not tuple for v in
                (saved.local,saved.pending_kv,saved.pending_z,saved.global_kv,saved.index_keys)):
                raise ValueError('invalid layer cache snapshot')
            if l==c.encoder: owner_layer=selector=None
            if mode=='full': owner_layer=l
            if mode in ('full','reindex'): selector=l
            entries = len(tokens)//c.ratio if mode=='full' else 0
            pending = len(tokens)%c.ratio if mode=='full' else 0
            if (len(saved.local)!=min(len(tokens),c.window) or len(saved.pending_kv)!=pending or
                len(saved.pending_z)!=pending or len(saved.global_kv)!=entries or len(saved.index_keys)!=entries):
                raise ValueError('cache geometry does not match token position')
            layer = candidate.layers[l]
            for pair in saved.local:
                if type(pair) is not tuple or len(pair)!=2: raise ValueError('invalid local KV pair')
                layer.local.append((packed(pair[0]),packed(pair[1])))
            layer.pending_kv = [vector(x) for x in saved.pending_kv]
            layer.pending_z = [vector(x) for x in saved.pending_z]
            layer.global_kv = [packed(x) for x in saved.global_kv]
            layer.index_keys = [vector(x) for x in saved.index_keys]
            expected_owner = owner_layer if tokens and mode!='swa' else None
            expected_selector = selector if tokens and mode!='swa' else None
            if (type(saved.owner) not in (int,type(None)) or type(saved.selector) not in (int,type(None)) or
                saved.owner!=expected_owner or saved.selector!=expected_selector):
                raise ValueError('cache index ownership mismatch')
            available = len(tokens)//c.ratio if expected_owner is not None else 0
            layer.owner,layer.selector = expected_owner,expected_selector
            layer.selected = indices(saved.selected,available,min(c.topk,available))
            blocks = (available+c.block_size-1)//c.block_size
            candidate_limit = min(c.candidate_blocks,blocks) if l>=c.encoder and mode!='swa' else 0
            layer.candidates = indices(saved.candidates,blocks,candidate_limit)
            if l>=c.encoder and mode=='reindex' and any(i//c.block_size not in layer.candidates for i in layer.selected):
                raise ValueError('reindexed selection lies outside candidate blocks')
        return candidate
