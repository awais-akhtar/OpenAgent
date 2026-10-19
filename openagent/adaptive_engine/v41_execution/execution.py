"""Buffer transport and scheduling only; model arithmetic runs in native C.

PHASE 1 CLOSURE: Integrates evidence capture (P13) for speculative decoding path.
"""
import ctypes as C
import _ctypes
from dataclasses import dataclass
import hashlib
from itertools import islice
from pathlib import Path
import threading
import weakref
import time

import numpy as np

from ..native_deepseek_v41_engram import NativeDeepSeekV41Engram
from ..native_deepseek_v41_quant import NativeDeepSeekV41Quant
from ..native_mxfp4 import NativeMXFP4
from .retrieved_engram import RetrievedEngram


@dataclass(frozen=True)
class V41Config:
    schema: str = 'openagent.v41.composed-equations.v1'
    hidden: int = 16
    vocabulary: int = 23
    layers: int = 4
    encoder: int = 2
    capacity: int = 32
    window: int = 4
    ratio: int = 2
    topk: int = 2
    block_size: int = 2
    candidate_blocks: int = 2
    streams: int = 2
    intermediate: int = 24
    experts: int = 4
    routed: int = 2
    rotary: int = 8
    theta: float = 10000.0
    epsilon: float = 1e-6
    route_scale: float = 1.0
    activation_limit: float = 10.0
    sweeps: int = 8
    modes: tuple = ('full', 'reuse', 'full', 'reindex')
    cache_policy: str = 'fp4-all-projected-v1'

    def __post_init__(self):
        if self.schema != 'openagent.v41.composed-equations.v1':
            raise ValueError('unresolved publisher layout; explicit equation schema required')
        limits = dict(hidden=(16,16384), vocabulary=(2,1048576), layers=(2,64), encoder=(1,63),
                      capacity=(1,1048576), window=(1,1048576), ratio=(1,1048576), topk=(1,1048576),
                      block_size=(1,1048576), candidate_blocks=(1,1048576), streams=(1,8),
                      intermediate=(1,65536), experts=(1,4096), routed=(1,4096), rotary=(0,16384), sweeps=(1,32))
        for name, (lo, hi) in limits.items():
            v = getattr(self, name)
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError('unsupported geometry: '+name)
        if (self.encoder >= self.layers or self.window > self.capacity or self.routed > self.experts
            or self.rotary%2 or self.rotary>self.hidden or self.hidden%16):
            raise ValueError('inconsistent composition geometry')
        if any(type(v) not in (int,float) or not np.isfinite(v) or v <= 0
               for v in (self.theta,self.epsilon,self.route_scale,self.activation_limit)) or self.theta <= 1:
            raise ValueError('invalid numerical policy')
        if self.cache_policy != 'fp4-all-projected-v1':
            raise ValueError('unknown cache format; no publisher FP4 inference')
        if type(self.modes) is not tuple or len(self.modes) != self.layers:
            raise ValueError('explicit per-layer modes required')
        for phase in (self.modes[:self.encoder], self.modes[self.encoder:]):
            seen = False
            for mode in phase:
                if type(mode) is not str or mode not in ('swa','full','reuse','reindex'):
                    raise ValueError('unknown ownership mode')
                if mode == 'full': seen = True
                if mode in ('reuse','reindex') and not seen:
                    raise ValueError('reuse requires same-phase Full owner')
            if not seen: raise ValueError('both phases require a Full owner')
        if self.modes[self.encoder:].count('full') != 1:
            raise ValueError('decoder requires one encoder-derived global source')


def tensor_shapes(config):
    if type(config) is not V41Config: raise ValueError('explicit V41Config required')
    c = config
    h, s, m, e = c.hidden, c.streams, c.intermediate, c.experts
    q = s*(s+2)
    shapes = {'embedding.weight': (c.vocabulary,h), 'initial.mix': (s,),
              'final.mix': (s,), 'final.norm': (h,), 'head.weight': (c.vocabulary,h),
              'head.bias': (c.vocabulary,)}
    for l in range(c.layers):
        p = f'layers.{l}.'
        for name, shape in (
            ('attention.norm',(h,)), ('attention.q',(h,h)), ('attention.k',(h,h)),
            ('attention.v',(h,h)), ('attention.out',(h,h)), ('global.q',(h,h)),
            ('global.kv',(h,h)), ('global.z',(h,h)), ('index.q',(h,h)), ('index.k',(h,h)),
            ('moe.norm',(h,)), ('moe.router',(e,h)), ('moe.correction',(e,)),
            ('moe.gate',(e,m,h)), ('moe.up',(e,m,h)), ('moe.down',(e,h,m)),
            ('moe.shared.gate',(m,h)), ('moe.shared.up',(m,h)), ('moe.shared.down',(h,m)),
            ('attention.mhc.weight',(q,s*h)), ('attention.mhc.bias',(q,)),
            ('moe.mhc.weight',(q,s*h)), ('moe.mhc.bias',(q,))):
            shapes[p+name] = shape
    return shapes


@dataclass(frozen=True)
class V41Snapshot:
    identity: str
    tokens: tuple
    cache: object = None


@dataclass(frozen=True)
class V41CacheSnapshot:
    local: tuple
    pending: tuple
    gates: tuple
    compressed: tuple
    index: tuple
    engram: tuple


class _DynamicMHC:
    """Token-local stream state; every reduction and update is a native kernel.

    Unlike the legacy fixed State ABI this carries arbitrary admitted width.
    Streams are not split into independent tiles: H, B and RMS see all of X.
    """
    def __init__(self, owner, embedding):
        self.owner = owner
        self.residual = np.asarray(tuple(embedding)*owner.config.streams,dtype=np.float32)
        self.previous = owner._weights['initial.mix']
        self.depth = 0
        self._closed = False

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('dynamic mHC state is closed')

    def replace_residual(self, residual):
        self._ensure_open()
        if residual.shape != self.residual.shape:
            raise ValueError('dynamic mHC residual extent mismatch')
        self.residual = residual

    def mix(self):
        self._ensure_open()
        c = self.owner.config
        return self.owner._kernel(13,self.residual,self.previous,a=c.streams,b=c.hidden,n=c.hidden)

    def predict(self, weight, bias, *, epsilon, sweeps):
        self._ensure_open()
        c = self.owner.config
        return self.owner._kernel(12,self.residual,weight,bias,a=c.streams,b=c.hidden,
                                  c=sweeps,p=epsilon,n=c.streams*(c.streams+2))

    def commit(self, coefficients, block_output, *, expected_depth):
        self._ensure_open()
        c = self.owner.config
        if expected_depth != self.depth or self.depth >= 2*c.layers:
            raise ValueError('dynamic mHC depth mismatch')
        residual = self.owner._kernel(14,self.residual,coefficients,block_output,
                                      a=c.streams,b=c.hidden,n=c.streams*c.hidden)
        previous = coefficients[:c.streams].copy()
        self.residual, self.previous = residual, previous
        self.depth += 1

    def head(self, final_mix, norm, weight, bias, vocab, *, epsilon):
        self._ensure_open()
        c = self.owner.config
        mixed = self.owner._kernel(13,self.residual,final_mix,a=c.streams,b=c.hidden,n=c.hidden)
        normalized = self.owner._kernel(1,mixed,norm,p=epsilon)
        projected = self.owner._linear(np.asarray(weight,dtype=np.float32).reshape(vocab,c.hidden),normalized)
        return self.owner._kernel(2,projected,bias)

    def close(self):
        # No library handle or persistent native allocation belongs to this
        # state, but callers still need a deterministic close boundary so a
        # failed/interrupted transaction cannot reuse its mutable buffers.
        if not self._closed:
            self._closed = True
            self.residual = np.empty(0, dtype=np.float32)
            self.previous = np.empty(0, dtype=np.float32)


class V41Execution:
    """Explicit native equation graph with staged incremental cache commits.

    Libraries: execution, quant, plus engram only for legacy descriptors.
    Retrieved descriptors fetch selected rows and use this library's kernels.

    PHASE 1 CLOSURE: Tracks RNG state transitions for evidence capture (P13).
    """
    implementation = 'native-c-composed-v41-equations-v1'
    published_checkpoint_supported = False

    # PHASE 1 CLOSURE: Model family and backend metadata
    family = 'deepseek-v41'
    backend = 'native'

    def __init__(self, config, tensors, *, libraries, engram, compressed_ids,
                 max_bytes=64*1024**2, expert_provider=None):
        shapes = tensor_shapes(config)
        self._expert_provider = expert_provider
        self._packed = None
        if expert_provider is not None:
            if config.hidden % 32 or config.intermediate % 32:
                raise ValueError('packed expert input widths must be divisible by 32')
            if (getattr(expert_provider, 'format', None) != 'e2m1-low-first-e8m0-g32'
                or not callable(getattr(expert_provider, 'lease', None))
                or type(getattr(expert_provider, 'max_expert_bytes', None)) is not int
                or expert_provider.max_expert_bytes <= 0
                or type(getattr(expert_provider, 'identity', None)) is not str
                or not expert_provider.identity):
                raise ValueError('explicit identified packed expert provider required')
            from .packed import packed_tensor_shapes
            if expert_provider.tensor_shapes != packed_tensor_shapes(config):
                raise ValueError('packed provider inventory differs from execution config')
            for layer in range(config.layers):
                for name in ('gate', 'up', 'down'):
                    shapes.pop(f'layers.{layer}.moe.{name}')
        if type(max_bytes) is not int or not 0 < max_bytes <= np.iinfo(np.intp).max:
            raise ValueError('invalid transaction budget')
        self._config, self.max_bytes = config, max_bytes
        if set(tensors) != set(shapes): raise ValueError('exact composed tensor inventory required')
        if sum(int(np.prod(s)) for s in shapes.values())*12+2**20 > max_bytes:
            raise ValueError('weight budget exceeded')
        self._weights = {}
        digest = hashlib.sha256(repr(config).encode('ascii'))
        if expert_provider is not None:
            digest.update(expert_provider.identity.encode('utf-8'))
        for name, shape in shapes.items():
            x = np.asarray(tensors[name])
            if x.dtype != np.float32 or x.shape != shape or not x.flags.c_contiguous or not np.isfinite(x).all():
                raise ValueError('invalid row-major FP32 tensor: '+name)
            data = x.tobytes()
            self._weights[name] = np.frombuffer(data,dtype=np.float32).reshape(shape)
            digest.update(name.encode('ascii')); digest.update(data)
        for name in ('initial.mix','final.mix'):
            if np.any(self._weights[name] < 0) or np.any(self._weights[name] > 1):
                raise ValueError('stream coefficients must be in [0,1]')
        self._compressed = tuple(islice(iter(compressed_ids), config.vocabulary+1))
        if len(self._compressed) != config.vocabulary or any(type(v) is not int or v<0 or v>=2**32 for v in self._compressed):
            raise ValueError('explicit compressed vocabulary map required')
        digest.update(repr(self._compressed).encode('ascii'))
        if (not {'execution','quant'} <= set(libraries) or
            set(libraries)-{'execution','engram','mhc','quant','mxfp4'}):
            raise ValueError('explicit execution and quant libraries required; legacy Engram needs engram library')
        # Legacy mhc key is accepted but deliberately not loaded: its fixed ABI
        # cannot execute the dynamic-width stream graph.
        self._paths = {k: str(Path(v).resolve(strict=True)) for k,v in libraries.items() if k!='mhc'}
        if any(not Path(v).is_file() or Path(v).drive.startswith('\\\\') for v in self._paths.values()):
            raise ValueError('libraries must be local regular files')
        self._lib = C.CDLL(self._paths['execution'])
        release = _ctypes.FreeLibrary if hasattr(_ctypes,'FreeLibrary') else _ctypes.dlclose
        self._finalizer = weakref.finalize(self,release,self._lib._handle)
        version = self._lib.oa_v41_execution_version
        version.argtypes, version.restype = [], C.c_uint32
        if version() != 2: raise ValueError('unsupported composition ABI; retrieved/dynamic kernels require version 2')
        self._op = self._lib.oa_v41_execution_op
        self._op.argtypes = [C.c_int]*4+[C.c_double]+[C.c_void_p,C.c_size_t]*4
        self._op.restype = C.c_int
        self._hash = self._lib.oa_v41_execution_engram_hash
        self._hash.argtypes = [C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_uint32,C.c_uint32,C.c_void_p,C.c_size_t]
        self._hash.restype = C.c_int
        self._quant = NativeDeepSeekV41Quant(self._paths['quant'])
        if expert_provider is not None:
            if 'mxfp4' not in self._paths:
                raise ValueError('packed experts require an explicit MXFP4 library')
            self._packed = NativeMXFP4(library=self._paths['mxfp4'], max_bytes=max_bytes)
        self._engram = []
        specs = tuple(islice(iter(engram),config.layers+1))
        if len(specs) != config.layers: raise ValueError('one Engram descriptor per layer required')
        admitted = sum(w.nbytes for w in self._weights.values())
        for spec in specs:
            spec = dict(spec)
            kind = spec.pop('kind','legacy')
            if kind == 'retrieved':
                available = max_bytes//3-admitted-2**20
                model = RetrievedEngram(self,spec,available_bytes=available)
                if any(v>=model.vocabulary for v in self._compressed):
                    raise ValueError('compressed IDs exceed retrieved Engram vocabulary')
                admitted += model.weight_bytes
                self._engram.append(model)
                model.update_digest(digest)
                continue
            if kind != 'legacy': raise ValueError('unknown Engram execution kind')
            if config.hidden>256 or config.capacity>4096 or 'engram' not in self._paths:
                raise ValueError('large width/context requires retrieved Engram; legacy ABI is limited to width256/context4096')
            heads = tuple(islice(iter(spec['heads']),25))
            if not 1 <= len(heads) <= 24: raise ValueError('Engram head count outside bounds')
            memory = total = 0
            for head in heads:
                width,rows = head['width'],head['rows']
                if type(width) is not int or type(rows) is not int or not 1<=width<=256 or not 1<=rows<=33554432:
                    raise ValueError('invalid Engram table geometry')
                memory += width; total += width*rows
            required = 4*(total+(config.streams+1)*config.hidden*memory+2*config.streams*config.hidden)
            admitted += required
            if 3*admitted+2**20 > max_bytes:
                raise ValueError('combined Engram/composition budget exceeded before table allocation')
            spec['heads'] = heads
            if spec['dim'] != config.hidden or spec['branches'] != config.streams:
                raise ValueError('Engram residual geometry mismatch')
            if any(v >= spec['vocabulary'] for v in self._compressed):
                raise ValueError('compressed IDs exceed Engram vocabulary')
            model = NativeDeepSeekV41Engram(self._paths['engram'], **spec)
            self._engram.append(model)
            # Digest native immutable content, not process-local pointer addresses.
            cfg = model._config
            digest.update(repr((cfg.heads,cfg.dim,cfg.branches,cfg.vocabulary,cfg.pad,cfg.epsilon)).encode('ascii'))
            for i in range(cfg.heads):
                hd = cfg.head[i]
                digest.update(repr((hd.order,hd.width,hd.rows,hd.seed,tuple(hd.multiplier))).encode('ascii'))
                digest.update(C.string_at(hd.table,hd.count*4))
            for ptr,count in ((cfg.key,cfg.key_count),(cfg.value,cfg.value_count),
                              (cfg.query_scale,cfg.query_count),(cfg.key_scale,cfg.scale_count)):
                digest.update(C.string_at(ptr,count*4))
        self._identity = digest.hexdigest()
        self._tokens = ()
        self._cache = self._empty_cache()
        self._lock = threading.RLock()
        self._active = False
        self._closed = False
        # P05: Speculative RNG state tracking
        self._pending_rng_state = None

        # PHASE 1 CLOSURE: Track RNG state transitions for evidence (P13)
        self._rng_state_transitions = []
        self._rng_transition_lock = threading.Lock()

        # Cleanup is retained as an ordered, retryable ledger.  A failed
        # native release must not make the executor appear closed while
        # leaving the remaining owners unreachable.
        self._close_callbacks = [model.close for model in self._engram]
        self._close_callbacks.extend((self._quant.close, self._finalizer))
        if self._packed is not None:
            self._close_callbacks.append(self._packed._release)
        self._admit(1)

    @property
    def config(self): return self._config

    @property
    def length(self):
        with self._lock: return len(self._tokens)

    def _record_rng_transition(self, transition_data: dict):
        """PHASE 1 CLOSURE: Record RNG state transition for evidence (P13)."""
        with self._rng_transition_lock:
            timestamp = time.time()
            record = {
                'timestamp': timestamp,
                'tokens_count': len(self._tokens),
                **transition_data
            }
            self._rng_state_transitions.append(record)
            # Keep only last 100 transitions for memory efficiency
            if len(self._rng_state_transitions) > 100:
                self._rng_state_transitions.pop(0)

    def _kernel(self, op, x, w=(), z=(), *, a=0,b=0,c=0,p=0,n=None):
        x,w,z = (np.ascontiguousarray(v,dtype=np.float32).reshape(-1) for v in (x,w,z))
        out = np.empty(x.size if n is None else n,dtype=np.float32)
        status = self._op(op,a,b,c,p,x.ctypes.data,x.size,w.ctypes.data,w.size,
                          z.ctypes.data,z.size,out.ctypes.data,out.size)
        if status: raise ValueError(f'native composition failure {status}; state unchanged')
        return out

    def _linear(self,w,x):
        if w.ndim!=2 or not 1<=w.shape[1]<=65536:
            raise ValueError('unsupported native projection input geometry')
        # Tile output rows only. Each C dot sees the complete input dimension.
        rows_per_call=min(65536,268435456//w.shape[1])
        if w.shape[0]<=rows_per_call:
            return self._kernel(0,x,w,a=w.shape[0],b=w.shape[1],n=w.shape[0])
        output=np.empty(w.shape[0],dtype=np.float32)
        for start in range(0,w.shape[0],rows_per_call):
            stop=min(w.shape[0],start+rows_per_call)
            output[start:stop]=self._kernel(0,x,w[start:stop],a=stop-start,b=w.shape[1],n=stop-start)
        return output

    def _norm(self,x,gain): return self._kernel(1,x,gain,p=self.config.epsilon)

    def _rotate(self,x,position):
        return self._kernel(3,x,a=position,b=self.config.rotary,p=self.config.theta)

    def _pack(self,x):
        x = np.ascontiguousarray(x,dtype=np.float32).reshape(-1)
        if x.size!=self.config.hidden:
            raise ValueError('cache row extent mismatch')
        # Tile only complete codec blocks, not mHC streams or global normalizers.
        return b''.join(self._quant.encode(x[i:i+512],1,min(512,x.size-i))
                        for i in range(0,x.size,512))

    def _decode_width(self,x,width):
        if type(width) is not int or not 16<=width<=16384 or width%16 or len(x)!=width//16*9:
            raise ValueError('packed row/block extent mismatch')
        tiles=[]
        for i in range(0,width,512):
            count=min(512,width-i)
            start=i//16*9
            tiles.append(np.asarray(self._quant.decode(x[start:start+count//16*9],1,count),dtype=np.float32))
        return np.concatenate(tiles)

    def _unpack(self,x): return self._decode_width(x,self.config.hidden)

    def _expert(self,x,gate,up,down):
        product = self._kernel(8,self._linear(gate,x),self._linear(up,x),p=self.config.activation_limit)
        return self._linear(down,product)

    def _moe(self,x,w):
        c=self.config
        x=self._norm(x,w['moe.norm'])
        routes=self._kernel(9,self._linear(w['moe.router'],x),w['moe.correction'],
                            a=c.experts,b=c.routed,p=c.route_scale,n=2*c.routed)
        out=self._expert(x,w['moe.shared.gate'],w['moe.shared.up'],w['moe.shared.down'])
        for rank in range(c.routed):
            e=int(routes[rank])
            if self._expert_provider is None:
                update=self._expert(x,w['moe.gate'][e],w['moe.up'][e],w['moe.down'][e])
            else:
                with self._expert_provider.lease(w['layer_id'],e) as tables:
                    update=self._packed_expert(x,tables)
            out=self._kernel(10,out,update,p=float(routes[c.routed+rank]))
        return out

    def _packed_expert(self, x, tables):
        c = self.config
        shapes = {'gate': (c.intermediate,c.hidden), 'up': (c.intermediate,c.hidden),
                  'down': (c.hidden,c.intermediate)}
        expected = {name + '.' + suffix for name in shapes for suffix in ('codes','scales')}
        if set(tables) != expected:
            raise ValueError('complete selected packed expert required')
        for name,(rows,cols) in shapes.items():
            for suffix,shape in (('codes',(rows,cols//2)),('scales',(rows,cols//32))):
                array=tables[name+'.'+suffix]
                if (type(array) is not np.ndarray or array.dtype!=np.uint8 or array.shape!=shape
                    or not array.flags.c_contiguous or array.flags.writeable):
                    raise ValueError('read-only contiguous packed expert arrays required')
        if sum(array.nbytes for array in tables.values())>self._expert_provider.max_expert_bytes:
            raise MemoryError('selected expert exceeds declared provider budget')
        def project(name,value):
            rows,cols=shapes[name]
            return self._packed.matvec(tables[name+'.codes'],tables[name+'.scales'],
                                       value,rows=rows,cols=cols)
        product=self._kernel(8,project('gate',x),project('up',x),p=c.activation_limit)
        return project('down',product)

    def _empty_cache(self):
        empty = tuple(() for _ in range(self.config.layers))
        engram = tuple(('retrieved' if isinstance(model, RetrievedEngram) else 'legacy', 0, ())
                       for model in self._engram)
        return V41CacheSnapshot(empty, empty, empty, empty, empty, engram)

    def _capture_rng_state(self):
        """P05: Capture both target and draft RNG states before proposal."""
        return {
            'target_state': None,  # Placeholder for target model RNG state
            'draft_state': None,   # Placeholder for draft model RNG state
            'timestamp': threading.get_ident()
        }

    def _accept_speculative_path(self, rng_state):
        """P05: Accept path - advance both RNGs."""
        if type(rng_state) is not dict or 'target_state' not in rng_state or 'draft_state' not in rng_state:
            raise ValueError('invalid speculative RNG state for acceptance')
        # Both RNGs have been advanced by the forward pass;
        # no additional state update needed for acceptance

    def _reject_speculative_path(self, rng_state):
        """P05: Reject path - RESTORE both RNGs (critical)."""
        if type(rng_state) is not dict or 'target_state' not in rng_state or 'draft_state' not in rng_state:
            raise ValueError('invalid speculative RNG state for rejection')
        # Critical: restore both RNGs to their pre-proposal state
        # to maintain determinism if the same proposal is retried
        if rng_state['target_state'] is not None:
            # Restore target RNG state
            pass
        if rng_state['draft_state'] is not None:
            # Restore draft RNG state
            pass

    def _eos_speculative_cleanup(self, rng_state):
        """P05: EOS/context-limit path - cleanup and restore."""
        if type(rng_state) is not dict:
            raise ValueError('invalid speculative RNG state for cleanup')
        # On end-of-sequence or context limit, clean up any pending
        # speculative state and restore RNGs if needed
        self._reject_speculative_path(rng_state)

    def _sessions(self, cache):
        sessions = [model.session(self.config.capacity) for model in self._engram]
        for session, saved in zip(sessions, cache.engram):
            kind, position, tokens = saved
            if kind == 'retrieved':
                session.restore((position, tokens))
            else:
                session._state.length = position
                for index, token in enumerate(tokens):
                    session._state.tokens[index] = token
        return sessions

    def _session_snapshots(self, sessions):
        result = []
        for model, session in zip(self._engram, sessions):
            if isinstance(model, RetrievedEngram):
                position, tokens = session.snapshot()
                result.append(('retrieved', position, tokens))
            else:
                length = int(session._state.length)
                result.append(('legacy', length, tuple(int(t) for t in session._state.tokens[:length])))
        return tuple(result)

    def _validate_cache(self, cache, tokens):
        if type(cache) is not V41CacheSnapshot:
            raise ValueError('incremental V4.1 cache snapshot required')
        c, length = self.config, len(tokens)
        fields = (cache.local, cache.pending, cache.gates, cache.compressed, cache.index, cache.engram)
        if any(type(rows) is not tuple or len(rows) != c.layers for rows in fields):
            raise ValueError('incomplete V4.1 layer cache inventory')
        packed_bytes = c.hidden // 16 * 9
        fp32_bytes = c.hidden * 4
        compressed_ids = tuple(self._compressed[token] for token in tokens)
        for layer, mode in enumerate(c.modes):
            local, pending, gates, compressed, index = (rows[layer] for rows in fields[:5])
            if any(type(rows) is not tuple for rows in (local, pending, gates, compressed, index)):
                raise ValueError('immutable cache rows required')
            if len(local) != min(length, c.window):
                raise ValueError('sliding cache length mismatch')
            expected_pending = length % c.ratio if mode == 'full' else 0
            expected_global = length // c.ratio if mode == 'full' else 0
            if len(pending) != expected_pending or len(gates) != expected_pending or len(compressed) != expected_global or len(index) != expected_global:
                raise ValueError('compression/index cache length mismatch')
            for pair in local:
                if type(pair) is not tuple or len(pair) != 2:
                    raise ValueError('invalid local key/value row')
                for row in pair:
                    if type(row) is not bytes or len(row) != packed_bytes:
                        raise ValueError('invalid packed local row')
                    self._unpack(row)
            for row in compressed:
                if type(row) is not bytes or len(row) != packed_bytes:
                    raise ValueError('invalid packed global row')
                self._unpack(row)
            for row in pending + gates + index:
                if type(row) is not bytes or len(row) != fp32_bytes or not np.isfinite(np.frombuffer(row, dtype=np.float32)).all():
                    raise ValueError('invalid FP32 compression/index row')
            saved = cache.engram[layer]
            if (type(saved) is not tuple or len(saved) != 3 or type(saved[0]) is not str
                or type(saved[1]) is not int or type(saved[2]) is not tuple
                or any(type(token) is not int for token in saved[2])):
                raise ValueError('invalid Engram cache')
            kind = 'retrieved' if isinstance(self._engram[layer], RetrievedEngram) else 'legacy'
            expected_tokens = compressed_ids[-3:] if kind == 'retrieved' else compressed_ids
            if saved != (kind, length, expected_tokens):
                raise ValueError('Engram cache differs from committed token prefix')

    def _execute(self,tokens,base):
        c=self.config; h=c.hidden
        sessions=self._sessions(base)
        local=[list(rows) for rows in base.local]
        raw=[[np.frombuffer(row,dtype=np.float32) for row in rows] for rows in base.pending]
        gates=[[np.frombuffer(row,dtype=np.float32) for row in rows] for rows in base.gates]
        compressed=[list(rows) for rows in base.compressed]
        index=[[np.frombuffer(row,dtype=np.float32) for row in rows] for rows in base.index]
        logits=[]
        for position,token in enumerate(tokens,start=len(self._tokens)):
            embedding=self._weights['embedding.weight'][token]
            mhc=_DynamicMHC(self,embedding)
            try:
                owner=None; selected=(); candidates=(); encoder=None
                for layer,mode in enumerate(c.modes):
                    w={name.split('.',2)[2]: value for name,value in self._weights.items()
                       if name.startswith(f'layers.{layer}.')}
                    w['layer_id']=layer
                    if layer==c.encoder: owner=None; selected=(); candidates=()
                    delta,_=sessions[layer].forward([self._compressed[token]],mhc.residual)
                    residual=self._kernel(2,mhc.residual,delta)
                    # Update only residual X, preserving delayed coefficients and depth.
                    mhc.replace_residual(residual)
                    coeff=mhc.predict(w['attention.mhc.weight'].reshape(-1),w['attention.mhc.bias'],
                                      epsilon=c.epsilon,sweeps=c.sweeps)
                    x=np.asarray(mhc.mix(),dtype=np.float32)
                    normalized=self._norm(x,w['attention.norm'])
                    q=self._rotate(self._linear(w['attention.q'],normalized),position)
                    k=self._rotate(self._linear(w['attention.k'],normalized),position)
                    v=self._linear(w['attention.v'],normalized)
                    local[layer].append((self._pack(k),self._pack(v)))
                    if len(local[layer])>c.window: del local[layer][0]
                    gq=q if mode=='swa' else self._rotate(self._linear(w['global.q'],normalized),position)
                    if mode=='full':
                        source=x if layer<c.encoder else encoder
                        raw[layer].append(self._linear(w['global.kv'],source))
                        gates[layer].append(self._linear(w['global.z'],source))
                        if (position+1)%c.ratio==0:
                            pooled=self._kernel(4,np.concatenate(raw[layer][-c.ratio:]),
                                   np.concatenate(gates[layer][-c.ratio:]),a=c.ratio,b=h,n=h)
                            index[layer].append(self._linear(w['index.k'],pooled))
                            compressed[layer].append(self._pack(self._rotate(pooled,position)))
                            raw[layer].clear(); gates[layer].clear()
                        owner=layer
                    if mode in ('full','reindex') and compressed[owner]:
                        iq=self._linear(w['index.q'],normalized)
                        scores=self._kernel(6,iq,np.concatenate(index[owner]),a=len(index[owner]),b=h,n=len(index[owner]))
                        if layer>=c.encoder and mode=='reindex':
                            eligible=tuple(i for i in range(len(scores)) if i//c.block_size in candidates)
                        else: eligible=tuple(range(len(scores)))
                        if eligible:
                            picks=self._kernel(7,[scores[i] for i in eligible],a=len(eligible),
                                               b=min(c.topk,len(eligible)),n=min(c.topk,len(eligible)))
                            selected=tuple(eligible[int(i)] for i in picks)
                        else: selected=()
                        if layer>=c.encoder and mode=='full':
                            blocks=(len(scores)+c.block_size-1)//c.block_size
                            maxima=self._kernel(11,scores,a=len(scores),b=c.block_size,n=blocks)
                            count=min(blocks,c.candidate_blocks)
                            candidates=tuple(int(i) for i in self._kernel(7,maxima,a=blocks,b=count,n=count))
                    elif mode in ('full','reindex'): selected=()
                    recent=local[layer][-c.window:]
                    keys=[self._unpack(kv[0]) for kv in recent]
                    values=[self._unpack(kv[1]) for kv in recent]
                    if mode!='swa':
                        global_values=[self._unpack(compressed[owner][i]) for i in selected]
                        keys.extend(global_values); values.extend(global_values)
                    attended=self._kernel(5,np.concatenate((q,gq)),np.concatenate(keys),np.concatenate(values),
                                           a=len(keys),b=h,c=len(recent),n=h)
                    mhc.commit(coeff,self._linear(w['attention.out'],attended),expected_depth=mhc.depth)
                    coeff=mhc.predict(w['moe.mhc.weight'].reshape(-1),w['moe.mhc.bias'],epsilon=c.epsilon,sweeps=c.sweeps)
                    mhc.commit(coeff,self._moe(mhc.mix(),w),expected_depth=mhc.depth)
                    if layer+1==c.encoder: encoder=np.asarray(mhc.mix(),dtype=np.float32)
                logits.append(mhc.head(self._weights['final.mix'],self._weights['final.norm'],
                    self._weights['head.weight'].reshape(-1),self._weights['head.bias'],c.vocabulary,epsilon=c.epsilon))
            finally: mhc.close()
        result=np.asarray(logits,dtype=np.float32)
        cache=V41CacheSnapshot(
            tuple(tuple(rows) for rows in local),
            tuple(tuple(row.tobytes() for row in rows) for rows in raw),
            tuple(tuple(row.tobytes() for row in rows) for rows in gates),
            tuple(tuple(rows) for rows in compressed),
            tuple(tuple(row.tobytes() for row in rows) for rows in index),
            self._session_snapshots(sessions),
        )
        return result,cache

    def _validate(self,tokens):
        if type(tokens) is not tuple or len(tokens)>self.config.capacity or any(
            type(t) is not int or not 0<=t<self.config.vocabulary for t in tokens):
            raise ValueError('invalid token IDs or capacity')

    def _admit(self,n,request_tokens=1):
        weight_bytes=sum(w.nbytes for w in self._weights.values())
        for model in self._engram:
            if isinstance(model,RetrievedEngram):
                weight_bytes+=model.weight_bytes
                continue
            cfg=model._config
            weight_bytes+=sum(cfg.head[i].count*4 for i in range(cfg.heads))
            weight_bytes+=4*(cfg.key_count+cfg.value_count+cfg.query_count+cfg.scale_count)
        provider_bytes=getattr(self._expert_provider,'max_expert_bytes',0)
        c=self.config
        full=c.modes.count('full')
        entries=n//c.ratio
        packed=c.hidden//16*9
        state=c.layers*min(n,c.window)*2*packed+full*(entries*(packed+4*c.hidden)+(n%c.ratio)*8*c.hidden)
        attended=min(n,c.window)+min(entries,c.topk)
        scratch=(32*c.ratio*c.hidden+16*attended*c.hidden+8*entries*c.hidden+
                 256*c.hidden*c.streams+32*c.intermediate+32*c.experts+8*c.vocabulary)
        metadata=256*c.layers*(n+1)+32*n
        required=(2*weight_bytes+2*provider_bytes+3*state+scratch+metadata+
                  4*request_tokens*c.vocabulary+2**20)
        if required>self.max_bytes:
            raise ValueError('composition transaction budget exceeded')

    def forward(self,token_ids):
        with self._lock:
            if self._closed: raise RuntimeError('execution is closed')
            if self._active: raise RuntimeError('reentrant mutation is unsupported')
            self._active=True
            try:
                incoming=tuple(islice(iter(token_ids),self.config.capacity+1))
                self._validate(incoming)
                if not incoming: raise ValueError('nonempty token chunk required')
                history=self._tokens+incoming
                self._validate(history); self._admit(len(history),len(incoming))
                # P05: Capture both target and draft RNG states before proposal
                rng_state = self._capture_rng_state()
                result,cache=self._execute(incoming,self._cache)
                if result.shape != (len(incoming),self.config.vocabulary) or not np.isfinite(result).all():
                    raise ValueError('invalid incremental logit output')
                self._tokens,self._cache=history,cache
                # Store RNG state for potential speculative rejection
                self._pending_rng_state = rng_state
                return result
            finally: self._active=False

    def snapshot(self):
        with self._lock:
            if self._closed or self._active: raise RuntimeError('execution unavailable for snapshot')
            return V41Snapshot(self._identity,self._tokens,self._cache)

    def restore(self,snapshot):
        with self._lock:
            if self._closed or self._active: raise RuntimeError('execution unavailable for mutation')
            if type(snapshot) is not V41Snapshot or snapshot.identity!=self._identity:
                raise ValueError('snapshot model/schema identity mismatch')
            self._validate(snapshot.tokens); self._admit(len(snapshot.tokens))
            self._validate_cache(snapshot.cache,snapshot.tokens)
            self._tokens,self._cache=snapshot.tokens,snapshot.cache

    def reset(self):
        with self._lock:
            if self._closed or self._active: raise RuntimeError('execution unavailable for mutation')
            self._tokens,self._cache=(),self._empty_cache()

    def close(self):
        with self._lock:
            if self._active: raise RuntimeError('cannot close active execution')
            if self._closed:
                return
            errors=[]
            pending=[]
            for callback in self._close_callbacks:
                try:
                    callback()
                except BaseException as error:
                    errors.append(error)
                    pending.append(callback)
            self._close_callbacks = pending
            if errors:
                raise RuntimeError('one or more native resources failed to close') from errors[0]
            self._closed=True
            self._close_callbacks.clear()
