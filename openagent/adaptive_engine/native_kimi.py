"""Original resident Kimi Linear decoder and projected KDA private CPU kernel."""
import ctypes as C
from contextlib import ExitStack
from pathlib import Path
import threading

import numpy as np

from .independent_kimi import IndependentKimiLinearDecoder


class NativeKimiKDA:
    """Batch-one token mixer with resident FP32 state and short-conv history.

    No checkpoint loading, projections, MLA, MoE, logits or model aliases.
    beta and gate must already be sigmoid probabilities; decay is log(alpha).
    """
    implementation = 'original-c-projected-kda-fp32'
    full_decoder = False

    def __init__(self, taps, norm, *, library, capacity, epsilon=1e-6,
                 q_scale=1., l2_epsilon=None, max_bytes=64*1024*1024):
        if type(capacity) is not int or not 0 < capacity <= 1048576:
            raise ValueError('invalid capacity')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('invalid memory budget')
        l2_epsilon = epsilon if l2_epsilon is None else l2_epsilon
        for x in (epsilon, q_scale, l2_epsilon):
            if isinstance(x, bool) or not np.isscalar(x) or not np.isfinite(x) or x <= 0:
                raise ValueError('invalid epsilon or query scale')
        if len(taps) != 3:
            raise ValueError('three Q/K/V convolution tensors required')
        taps = tuple(np.asarray(x) for x in taps)
        if taps[0].ndim != 3:
            raise ValueError('taps must be [heads, channels, kernel]')
        h, d, kernel = taps[0].shape
        if taps[2].ndim != 3:
            raise ValueError('invalid value taps')
        vd = taps[2].shape[1]
        if not (0 < h <= 128 and 0 < d <= 256 and 0 < vd <= 256 and 0 < kernel <= 64):
            raise ValueError('native geometry bound exceeded')
        if taps[1].shape != (h, d, kernel) or taps[2].shape != (h, vd, kernel):
            raise ValueError('inconsistent convolution tensors')
        norm = np.asarray(norm)
        if norm.shape != (vd,):
            raise ValueError('norm must be [value channels]')
        for a in (*taps, norm):
            self._buffer(a)
        self.heads, self.key, self.value, self.kernel = h, d, vd, kernel
        self.max_bytes, self.capacity = max_bytes, capacity
        self.epsilon, self.q_scale = float(epsilon), float(q_scale)
        self.l2_epsilon = float(l2_epsilon)
        self._resident = 4*(sum(a.size for a in taps)+vd+h*d*vd+(kernel-1)*h*(2*d+vd))
        self._check_budget(self._resident*2)
        self._taps = tuple(np.frombuffer(a.tobytes(), dtype=np.float32).reshape(a.shape) for a in taps)
        self._norm = np.frombuffer(norm.tobytes(), dtype=np.float32)
        self._lock = threading.RLock()
        self.library = C.CDLL(str(Path(library).resolve(strict=True)))
        version = self.library.oa_kimi_kda_version
        version.argtypes, version.restype = [], C.c_uint64
        if version() != 1:
            raise ValueError('unsupported KDA private ABI')
        self._run = self.library.oa_kimi_kda
        pointer = C.POINTER(C.c_float)
        self._run.argtypes = [C.c_uint64]*5 + [C.c_double]*3 + [pointer]*15 + [C.POINTER(C.c_uint64)]
        self._run.restype = C.c_int
        self.reset()

    @staticmethod
    def _buffer(a):
        if a.dtype != np.float32 or not a.flags.c_contiguous or not np.isfinite(a).all():
            raise ValueError('finite contiguous FP32 buffers required')

    def _check_budget(self, amount):
        if amount > self.max_bytes:
            raise ValueError('resident plus transaction memory budget exceeded')

    def reset(self):
        with self._lock:
            self._state = np.zeros((self.heads, self.key, self.value), np.float32)
            self._history = tuple(np.zeros((self.kernel-1, self.heads, d), np.float32)
                                  for d in (self.key, self.key, self.value))
            self._position = 0

    @property
    def position(self):
        with self._lock:
            return self._position

    def snapshot(self):
        """Detached state for numerical inspection; cannot mutate live cache."""
        with self._lock:
            return self._position, self._state.copy(), tuple(a.copy() for a in self._history)

    def forward(self, q, k, v, log_decay, beta, gate):
        with self._lock:
            arrays = tuple(np.asarray(a) for a in (q, k, v, log_decay, beta, gate))
            q = arrays[0]
            if q.ndim != 3 or not 0 < q.shape[0] <= 4096:
                raise ValueError('Q must be [tokens, heads, key] with 1..4096 tokens')
            n = q.shape[0]
            if self._position+n > self.capacity:
                raise ValueError('KDA cache capacity exceeded')
            shapes = [(n,self.heads,self.key)]*2 + [(n,self.heads,self.value),
                       (n,self.heads,self.key), (n,self.heads), (n,self.heads,self.value)]
            for a, shape in zip(arrays, shapes):
                if a.shape != shape:
                    raise ValueError('projected tensor shape mismatch')
                self._buffer(a)
            if (arrays[3] > 0).any() or any(((a < 0) | (a > 1)).any() for a in arrays[4:]):
                raise ValueError('log decay must be <=0; beta and gate in [0,1]')
            # Include input snapshots, staged caches, output, and C double scratch.
            scratch = 8*(2*self.heads*self.key+self.heads*self.value+2*self.value)
            self._check_budget(self._resident*2 + sum(a.nbytes for a in arrays) +
                               n*self.heads*self.value*4 + scratch)
            arrays = tuple(a.copy() for a in arrays)
            state = self._state.copy()
            history = tuple(a.copy() for a in self._history)
            out = np.empty((n,self.heads,self.value), np.float32)
            buffers = (*arrays, self._norm, *self._taps, state, *history, out)
            counts = (C.c_uint64*15)(*(a.size for a in buffers))
            ptr = C.POINTER(C.c_float)
            status = self._run(n, self.heads, self.key, self.value, self.kernel,
                               self.epsilon, self.q_scale, self.l2_epsilon,
                               *(a.ctypes.data_as(ptr) for a in buffers), counts)
            if status:
                raise ValueError(f'native KDA execution status {status}; cache unchanged')
            if not np.isfinite(out).all() or not np.isfinite(state).all():
                raise ValueError('nonfinite KDA result; cache unchanged')
            self._state, self._history = state, history
            self._position += n
            return out


class NativeKimiLinearDecoder(IndependentKimiLinearDecoder):
    """Checkpoint-name FP32 Linear decoder; C arithmetic, Python scheduling.

    All tensor contracts are inherited, never reference code. No NumPy forward
    fallback. Whole input calls stage every KDA/MLA cache before atomic commit.
    This private CPU implementation has not been verified by a vendor runtime.
    """
    implementation = 'original-c-kimi-linear-resident-fp32'

    @classmethod
    def from_checkpoint(cls, directory, *, library, capacity, max_bytes=256*1024*1024):
        """Load local unquantized SafeTensors using existing safe file parsers.

        No registry changes, remote code, downloads, tensor renaming or wrappers.
        Conversion of F16/BF16 weights to resident FP32 is explicit.
        """
        if type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('positive Kimi Linear checkpoint admission required')
        from .source_scope import require_checkpoint_inventory_admission
        require_checkpoint_inventory_admission(
            directory,family='kimi-k3',model_type='kimi_linear',backend='native',
            arithmetic='fp32-approximate',max_metadata_bytes=max_bytes)
        from .native_checkpoint import _open_local, _json, _header, _read_tensor, _stamp
        root = Path(directory).resolve(strict=True)
        with ExitStack() as stack:
            source,identity = _open_local(root,root/'config.json',stack)
            config = _json(source,identity)
            if _stamp(source)!=identity:
                raise ValueError('checkpoint config changed during read')
            cls.validate_config(config,capacity=capacity,max_bytes=max_bytes)
            expected = cls.tensor_shapes(config)
            if config['tie_word_embeddings']:
                expected.pop('lm_head.weight')
            index_path = root/'model.safetensors.index.json'
            if index_path.exists():
                source,identity = _open_local(root,index_path,stack)
                index = _json(source,identity)
                if _stamp(source)!=identity or not isinstance(index,dict) or not isinstance(index.get('weight_map'),dict):
                    raise ValueError('invalid or changed checkpoint index')
                weight_map = index['weight_map']
                if set(weight_map)!=set(expected) and not (config['tie_word_embeddings'] and set(weight_map)==set(expected)|{'lm_head.weight'}):
                    raise ValueError('checkpoint index tensor names mismatch')
                filenames = set(weight_map.values())
                if any(not isinstance(name,str) or Path(name).name!=name for name in filenames):
                    raise ValueError('shards must be local basename files')
            else:
                filenames,weight_map = {'model.safetensors'},None
            records = {}
            for filename in sorted(filenames):
                source,identity = _open_local(root,root/filename,stack)
                header,base = _header(source,identity)
                for name,metadata in header.items():
                    if name=='__metadata__':
                        continue
                    if name in records or (weight_map is not None and weight_map.get(name)!=filename):
                        raise ValueError('duplicate tensor or index/shard disagreement')
                    records[name] = source,identity,base,metadata
            if config['tie_word_embeddings'] and 'lm_head.weight' in records:
                expected['lm_head.weight'] = expected['model.embed_tokens.weight']
            if set(records)!=set(expected):
                raise ValueError('checkpoint tensor names mismatch')
            sizes = []
            for name,shape in expected.items():
                if tuple(records[name][3]['shape'])!=shape:
                    raise ValueError(f'checkpoint tensor shape mismatch: {name}')
                sizes.append(int(np.prod(shape,dtype=object))*4)
            # Includes one model-owned copy and worst-case source conversion.
            if 2*sum(sizes)+3*max(sizes)>max_bytes:
                raise ValueError('checkpoint FP32 conversion memory budget exceeded')
            tensors = {name:_read_tensor(source,metadata,base)
                       for name,(source,identity,base,metadata) in records.items()}
            if any(_stamp(source)!=identity for source,identity,_,_ in records.values()):
                raise ValueError('checkpoint shard changed during read')
        return cls(config,tensors,library=library,capacity=capacity,max_bytes=max_bytes)

    def __init__(self, config, tensors, *, library, capacity, max_bytes=256*1024*1024):
        super().__init__(config,tensors,capacity=capacity,max_bytes=max_bytes)
        self.library = C.CDLL(str(Path(library).resolve(strict=True)))
        ptr = C.POINTER(C.c_float)
        version = self.library.oa_kimi_kda_version
        version.argtypes,version.restype = [],C.c_uint64
        if version()!=1:
            raise ValueError('unsupported Kimi private ABI')
        specs = {
            '_c_mat': ('oa_kimi_matvec',[C.c_uint64]*2+[ptr]*3),
            '_c_vec': ('oa_kimi_vector',[C.c_uint64]*2+[C.c_double]+[ptr]*3),
            '_c_att': ('oa_kimi_attention',[C.c_uint64]*4+[ptr]*4),
            '_c_route': ('oa_kimi_route',[C.c_uint64]*3+[C.c_double]+[ptr]*2+[C.POINTER(C.c_uint64),ptr]),
            '_c_kda': ('oa_kimi_kda',[C.c_uint64]*5+[C.c_double]*3+[ptr]*15+[C.POINTER(C.c_uint64)])}
        for name,(symbol,args) in specs.items():
            f = getattr(self.library,symbol)
            f.argtypes,f.restype = args,C.c_int
            setattr(self,name,f)
        if getattr(self.config, 'tensor_layout', None) == 'upstream-fused':
            self._c_group_route = self.library.oa_kimi_group_route
            self._c_group_route.argtypes = [C.c_uint64]*5+[C.c_double]+[ptr]*2+[C.POINTER(C.c_uint64),ptr]
            self._c_group_route.restype = C.c_int

    @staticmethod
    def _ptr(a):
        if a is None:
            return None
        if a.dtype!=np.float32 or not a.flags.c_contiguous:
            raise ValueError('native arithmetic requires contiguous FP32 buffers')
        return a.ctypes.data_as(C.POINTER(C.c_float))

    @staticmethod
    def _status(status):
        if status:
            raise ValueError(f'native decoder status {status}; cache unchanged')

    def _mat(self,w,x):
        out = np.empty(w.shape[0],np.float32)
        self._status(self._c_mat(*w.shape,self._ptr(w),self._ptr(x),self._ptr(out)))
        return out

    def _vec(self,op,x,w=None,param=1.):
        out = np.empty_like(x)
        self._status(self._c_vec(x.size,op,float(param),self._ptr(x),self._ptr(w),self._ptr(out)))
        return out

    def _attention(self,q,k,v):
        out = np.empty((q.shape[0],v.shape[-1]),np.float32)
        self._status(self._c_att(k.shape[0],q.shape[0],q.shape[1],v.shape[-1],
            *(self._ptr(a) for a in (q,k,v,out))))
        return out

    def _route(self,scores,bias):
        c = self.config
        ids = np.empty(c.num_experts_per_token,np.uint64)
        weights = np.empty(c.num_experts_per_token,np.float32)
        if getattr(c, 'tensor_layout', None) == 'upstream-fused':
            self._status(self._c_group_route(c.num_experts,c.num_experts_per_token,c.num_expert_group,
                c.topk_group,int(c.moe_renormalize),c.routed_scaling_factor,self._ptr(scores),self._ptr(bias),
                ids.ctypes.data_as(C.POINTER(C.c_uint64)),self._ptr(weights)))
        else:
            self._status(self._c_route(c.num_experts,c.num_experts_per_token,int(c.moe_renormalize),
                c.routed_scaling_factor,self._ptr(scores),self._ptr(bias),
                ids.ctypes.data_as(C.POINTER(C.c_uint64)),self._ptr(weights)))
        return ids,weights

    def _kda(self,streams,taps,decay,beta,gate,norm,cache):
        h,d = streams[0].shape[1:]
        kernel = taps[0].shape[-1]
        out = np.empty((1,h,d),np.float32)
        buffers = (*streams,decay,beta,gate,norm,*taps,*cache,out)
        counts = (C.c_uint64*15)(*(a.size for a in buffers))
        self._status(self._c_kda(1,h,d,d,kernel,self.config.rms_norm_eps,d**-.5,1e-6,
            *(self._ptr(a) for a in buffers),counts))
        return out,cache
