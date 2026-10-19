"""Owned native Mixtral hidden ranges with transactional continuation."""
import ctypes as C
import hashlib
import json
import math
import threading
from dataclasses import dataclass

import numpy as np

from .native_mixtral_decoder import Config, Layer, Tensor, ExpertFFN, LAYER_NAMES, GLOBAL_NAMES, tensor, configuration


@dataclass(frozen=True)
class MixtralRangeSnapshot:
    identity: str
    position: int
    caches: tuple
    mask: bytes


class NativeMixtralRange:
    """Own selected fused layer weights; borrow an explicitly loaded library.

    Arrays are named with original checkpoint layer indices. Embeddings and
    head tensors are excluded. The owner of the library keeps it alive until
    the range closes. This class performs no Python decoder arithmetic.
    """

    @classmethod
    def from_directory(cls, directory, *, begin, end, library, context_tokens,
                       max_weight_bytes, max_bytes, batch_size=1, read_limits=None, cancel=None,
                       expert_provider=None):
        """Admit the whole source inventory, then read only selected layers."""
        if expert_provider is not None:
            raise ValueError(
                'Mixtral packed experts are restricted until source-content identity is bound')
        from .native_checkpoint import read_native_text_checkpoint, _finish_decoder
        selection_layout = []
        def select(config, inventory):
            c, capacity = configuration(config, context_tokens)
            if type(begin) is not int or type(end) is not int or not 0 <= begin < end <= c['num_hidden_layers']:
                raise ValueError('valid Mixtral checkpoint layer interval required')
            h,m,e,d = (c[key] for key in ('hidden_size','intermediate_size','num_local_experts','head_dim'))
            q,k = c['num_attention_heads']*d,c['num_key_value_heads']*d
            globals_ = dict(zip(GLOBAL_NAMES,((c['vocab_size'],h),(h,),(c['vocab_size'],h))))
            if c['tie_word_embeddings'] and 'lm_head.weight' not in inventory:
                globals_.pop('lm_head.weight')
            fused,split = dict(globals_),dict(globals_)
            selected_fused,selected_split = set(),set()
            shapes = ((h,),(h,),(q,h),(k,h),(k,h),(h,q),(e,h),(e,2*m,h),(e,h,m))
            for index in range(c['num_hidden_layers']):
                prefix = f'model.layers.{index}.'
                for ordinal,(name,shape) in enumerate(zip(LAYER_NAMES,shapes)):
                    fused[prefix+name] = shape
                    if begin <= index < end and (expert_provider is None or ordinal < 7):
                        selected_fused.add(prefix+name)
                    if ordinal < 6:
                        split[prefix+name] = shape
                        if begin <= index < end:
                            selected_split.add(prefix+name)
                name = prefix+'block_sparse_moe.gate.weight'
                split[name] = (e,h)
                if begin <= index < end:
                    selected_split.add(name)
                for expert in range(e):
                    for leaf,shape in (('w1',(m,h)),('w3',(m,h)),('w2',(h,m))):
                        name = prefix+f'block_sparse_moe.experts.{expert}.{leaf}.weight'
                        split[name] = shape
                        if begin <= index < end and expert_provider is None:
                            selected_split.add(name)
            if set(inventory) == set(fused):
                expected,selected,layout = fused,selected_fused,'fused'
            elif set(inventory) == set(split):
                expected,selected,layout = split,selected_split,'split'
            else:
                raise ValueError('one exact complete fused or split Mixtral source inventory required')
            for name,shape in expected.items():
                if tuple(inventory[name]['shape']) != shape or inventory[name].get('dtype') not in ('F32','F16','BF16'):
                    raise ValueError('Mixtral source shape or dtype mismatch: '+name)
            weight_bytes = 4*(end-begin)*sum(math.prod(shape) for shape in
                                          (shapes if expert_provider is None else shapes[:7]))
            state_bytes = 8*(end-begin)*batch_size*capacity*k
            scratch = 4*(5*h+2*q+3*m+2*e)+8*(capacity+e)
            if 3*weight_bytes+3*state_bytes+scratch+getattr(expert_provider,'max_bytes',0) > max_bytes:
                raise MemoryError('selected checkpoint conversion and range ownership exceed admission')
            selection_layout.append(layout)
            return tuple(sorted(selected))
        config,weights = read_native_text_checkpoint(directory,model_type='mixtral',
            max_weight_bytes=max_weight_bytes,read_limits=read_limits,cancel=cancel,
            tensor_selection=select,source_backend='public-range')
        if selection_layout != ['fused']:
            c,_ = configuration(config,context_tokens)
            e,m,h = c['num_local_experts'],c['intermediate_size'],c['hidden_size']
            converted = {}
            for index in range(begin,end):
                prefix = f'model.layers.{index}.'
                for name in LAYER_NAMES[:6]:
                    converted[prefix+name] = weights[prefix+name]
                source = prefix+'block_sparse_moe.'
                converted[prefix+'mlp.gate.weight'] = weights[source+'gate.weight']
                if expert_provider is not None:
                    continue
                gate_up = np.empty((e,2*m,h),np.float32)
                down = np.empty((e,h,m),np.float32)
                for expert in range(e):
                    expert_prefix = source+f'experts.{expert}.'
                    gate_up[expert,:m] = weights[expert_prefix+'w1.weight']
                    gate_up[expert,m:] = weights[expert_prefix+'w3.weight']
                    down[expert] = weights[expert_prefix+'w2.weight']
                converted[prefix+'mlp.experts.gate_up_proj'] = gate_up
                converted[prefix+'mlp.experts.down_proj'] = down
            weights = converted
        model = cls(config,weights,begin=begin,end=end,library=library,context_tokens=context_tokens,
                    max_bytes=max_bytes,batch_size=batch_size,expert_provider=expert_provider)
        return _finish_decoder(model,cancel)

    def __init__(self, config, tensors, *, begin, end, library, context_tokens,
                 max_bytes, batch_size=1, expert_provider=None):
        if expert_provider is not None:
            raise ValueError(
                'Mixtral packed experts are restricted until source-content identity is bound')
        self.config, self.capacity = configuration(config, context_tokens)
        c = self.config
        if (type(begin) is not int or type(end) is not int or
                not 0 <= begin < end <= c['num_hidden_layers'] or
                type(batch_size) is not int or not 1 <= batch_size <= 16 or
                type(max_bytes) is not int or max_bytes <= 0 or not isinstance(library, C.CDLL)):
            raise ValueError('exact layer range, batch, budget and borrowed native library required')
        self.begin, self.end, self.batch_size, self.max_bytes = begin, end, batch_size, max_bytes
        self.library, self._lock = library, threading.RLock()
        self._expert_provider = expert_provider
        self._expert_identity = None
        self._expert_error = None
        self._expert_callback = ExpertFFN()
        self._expert_bytes = 0
        self._busy = self._closed = False
        self._position = 0
        h, m, e, d = (c[key] for key in ('hidden_size', 'intermediate_size', 'num_local_experts', 'head_dim'))
        q, k = c['num_attention_heads']*d, c['num_key_value_heads']*d
        if expert_provider is not None:
            from .mixtral_streamed_experts import MixtralStreamedExperts
            from .native_mixtral_decoder import mixtral_metadata_fingerprint
            if (type(expert_provider) is not MixtralStreamedExperts or
                    expert_provider.runtime.manifest.fingerprint != mixtral_metadata_fingerprint(config) or
                    len(expert_provider.layer_ids) != c['num_hidden_layers'] or
                    expert_provider.hidden_size != h or expert_provider.intermediate_size != m or
                    expert_provider.expert_count != e or expert_provider.selected_experts != c['num_experts_per_tok']):
                raise ValueError('exact full-model streamed Mixtral provider required')
            self._expert_identity = expert_provider.identity
            self._expert_bytes = expert_provider.max_bytes
            def selected_expert(_user, layer, selected, incoming, outgoing, hidden_width, intermediate):
                try:
                    if hidden_width != h or intermediate != m or not 0 <= layer < end-begin:
                        raise ValueError('native streamed range expert geometry mismatch')
                    vector = np.ctypeslib.as_array(incoming,shape=(h,)).copy()
                    result = expert_provider.execute(begin+int(layer),int(selected),vector)
                    if result.dtype != np.float32 or result.shape != (h,) or not np.isfinite(result).all():
                        raise ValueError('invalid streamed range expert output')
                    C.memmove(outgoing,result.ctypes.data,h*4)
                    return 0
                except BaseException as error:
                    self._expert_error = error
                    return 1
            self._expert_callback = ExpertFFN(selected_expert)
        sizes = ((h,), (h,), (q,h), (k,h), (k,h), (h,q), (e,h), (e,2*m,h), (e,h,m))
        shapes = {f'model.layers.{index}.{name}': shape for index in range(begin,end)
                  for ordinal,(name, shape) in enumerate(zip(LAYER_NAMES,sizes))
                  if expert_provider is None or ordinal < 7}
        self._weight_bytes = 4*sum(math.prod(shape) for shape in shapes.values())
        self._state_bytes = 8*(end-begin)*batch_size*self.capacity*k
        self._scratch = 4*(5*h+2*q+3*m+2*e)+8*(self.capacity+e)
        if 2*self._weight_bytes+3*self._state_bytes+self._scratch+self._expert_bytes > max_bytes:
            raise MemoryError('Mixtral range residency and continuation exceed admission')
        if set(tensors) != set(shapes):
            raise ValueError('exact selected fused Mixtral layer inventory required')
        self.weights = {}
        digest = hashlib.sha256(json.dumps(dict(c),sort_keys=True,separators=(',',':')).encode())
        digest.update(repr((begin,end,self.capacity,batch_size)).encode())
        if self._expert_identity is not None:
            digest.update(self._expert_identity.encode())
        for name, shape in shapes.items():
            value = np.asarray(tensors[name])
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError('invalid Mixtral range tensor: '+name)
            owned = np.frombuffer(value.tobytes(order='C'),dtype=np.float32).reshape(shape)
            self.weights[name] = owned
            digest.update(name.encode())
            digest.update(memoryview(owned).cast('B'))
        self.identity = 'oa-mixtral-range-v1:'+digest.hexdigest()
        version = library.oa_mixtral_range_version
        version.argtypes, version.restype = [], C.c_uint64
        if version() != 1:
            raise ValueError('Mixtral hidden range ABI version 1 required')
        self._run = library.oa_mixtral_range_run
        self._run.argtypes = [C.POINTER(Config), C.POINTER(Layer), Tensor,
            C.c_uint64,C.c_uint64,C.c_uint64,Tensor,Tensor,Tensor,
            C.POINTER(C.c_uint8),C.c_uint64,C.POINTER(C.c_int64),C.c_uint64,ExpertFFN,C.c_void_p]
        self._run.restype = C.c_int
        self._rope_arguments=()
        if c['rope_type']=='linear':
            capability=getattr(library,'oa_mixtral_scaled_rope_version',None)
            if capability is None:raise ValueError('scaled Mixtral hidden range ABI required')
            capability.argtypes,capability.restype=[],C.c_uint64
            if capability()!=1:raise ValueError('unsupported scaled Mixtral range ABI')
            arguments=self._run.argtypes+[C.c_double]
            self._run=library.oa_mixtral_range_run_scaled_rope
            self._run.argtypes,self._run.restype=arguments,C.c_int
            self._rope_arguments=(c['rope_factor'],)
        self._c = Config(h,m,end-begin,c['num_attention_heads'],c['num_key_value_heads'],d,e,
            c['num_experts_per_tok'],c['vocab_size'],self.capacity,c['sliding_window'],c['rms_norm_eps'],c['rope_theta'])
        self._shape = (batch_size,self.capacity,k)
        self._cache = tuple((np.zeros(self._shape,np.float32),np.zeros(self._shape,np.float32))
                            for _ in range(end-begin))
        self._mask = np.zeros((batch_size,self.capacity),np.uint8)

    def _check(self):
        if self._busy or self._closed:
            raise RuntimeError('Mixtral range is unavailable')

    @property
    def position(self):
        with self._lock:
            self._check()
            return self._position

    def forward(self, hidden, *, attention_mask=None, position_ids=None, cancelled=None):
        with self._lock:
            self._check()
            if cancelled is not None and not callable(cancelled):
                raise TypeError('callable range cancellation required')
            if cancelled is not None and cancelled():
                raise InterruptedError('Mixtral range cancelled before staging')
            values = np.asarray(hidden)
            h, e = self.config['hidden_size'], self.config['num_local_experts']
            if (values.dtype != np.float32 or values.ndim != 3 or values.shape[0] != self.batch_size or
                    values.shape[2] != h or not np.isfinite(values).all()):
                raise ValueError('finite FP32 [batch,tokens,hidden] range input required')
            count = values.shape[1]
            stop = self._position+count
            if not 0 < count <= 4096 or stop > self.capacity:
                raise ValueError('Mixtral range input exceeds continuation capacity')
            work = values.size*12+values.shape[0]*count*(self.end-self.begin)*e*8
            if self._weight_bytes+3*self._state_bytes+self._scratch+work+self._expert_bytes > self.max_bytes:
                raise MemoryError('Mixtral range transaction exceeds admission')
            mask = self._mask.copy()
            if attention_mask is None:
                mask[:,self._position:stop] = 1
            else:
                observed = np.asarray(attention_mask)
                if observed.shape != (self.batch_size,stop) or not np.isin(observed,(0,1)).all():
                    raise ValueError('binary full occupied-prefix attention mask required')
                if not np.array_equal(observed[:,:self._position],mask[:,:self._position]):
                    raise ValueError('committed range padding mask cannot change')
                mask[:,:stop] = observed
            positions = np.tile(np.arange(self._position,stop,dtype=np.int64),(self.batch_size,1))
            if position_ids is not None:
                observed = np.asarray(position_ids)
                if observed.shape != positions.shape or observed.dtype.kind not in 'iu' or np.any(observed >= 1048576) or np.any(observed < 0):
                    raise ValueError('valid explicit range positions required')
                positions = np.ascontiguousarray(observed,dtype=np.int64)
            incoming = np.ascontiguousarray(values)
            output = np.empty_like(incoming)
            routers = np.empty((self.batch_size,count,self.end-self.begin,e),np.float32)
            routes = np.empty_like(routers)
            staged = tuple((keys.copy(),vals.copy()) for keys,vals in self._cache)
            layers = (Layer*(self.end-self.begin))()
            for local,(keys,vals) in enumerate(staged):
                weights = (Tensor*9)(*[tensor(self.weights[f'model.layers.{self.begin+local}.{name}'])
                    if f'model.layers.{self.begin+local}.{name}' in self.weights else Tensor() for name in LAYER_NAMES])
                layers[local] = Layer(weights,tensor(keys),tensor(vals))
            occupied = np.ascontiguousarray(mask[:,:stop])
            self._busy = True
            try:
                if self._expert_provider is not None and self._expert_provider.identity != self._expert_identity:
                    raise ValueError('streamed range provider changed after admission')
                self._expert_error = None
                status = self._run(C.byref(self._c),layers,tensor(incoming),self.batch_size,count,self._position,
                    tensor(output),tensor(routers),tensor(routes),occupied.ctypes.data_as(C.POINTER(C.c_uint8)),
                    occupied.size,positions.ctypes.data_as(C.POINTER(C.c_int64)),positions.size,self._expert_callback,None,
                    *self._rope_arguments)
                if status:
                    if self._expert_error is not None:
                        error,self._expert_error = self._expert_error,None
                        raise RuntimeError('streamed Mixtral range expert failed; continuation unchanged') from error
                    raise RuntimeError('native Mixtral range failure '+str(status)+'; continuation unchanged')
                if not np.isfinite(output).all() or any(not np.isfinite(a).all() for pair in staged for a in pair):
                    raise ArithmeticError('nonfinite Mixtral range output or continuation')
                if cancelled is not None and cancelled():
                    raise InterruptedError('Mixtral range cancelled; continuation unchanged')
                self._cache,self._mask,self._position = staged,mask,stop
                return output
            finally:
                self._busy = False

    def write_cache(self, output, *, max_bytes):
        from .mixtral_range_cache import write_range_cache
        return write_range_cache(self,output,max_bytes=max_bytes)

    def restore_cache(self, source, *, max_bytes):
        from .mixtral_range_cache import restore_range_cache
        return restore_range_cache(self,source,max_bytes=max_bytes)

    def save_cache(self, destination, *, max_bytes):
        from .cache_file_recovery import save_executor_cache
        return save_executor_cache(self,destination,max_bytes=max_bytes)

    def load_cache(self, source, *, max_bytes):
        from .cache_file_recovery import load_executor_cache
        return load_executor_cache(self,source,max_bytes=max_bytes)

    def snapshot(self):
        with self._lock:
            self._check()
            return MixtralRangeSnapshot(self.identity,self._position,
                tuple((k[:,:self._position].tobytes(),v[:,:self._position].tobytes()) for k,v in self._cache),
                np.ascontiguousarray(self._mask[:,:self._position]).tobytes())

    def restore(self, state):
        with self._lock:
            self._check()
            if (type(state) is not MixtralRangeSnapshot or state.identity != self.identity or
                    type(state.position) is not int or not 0 <= state.position <= self.capacity or
                    len(state.caches) != self.end-self.begin):
                raise ValueError('Mixtral range snapshot identity or extent mismatch')
            shape = (self.batch_size,state.position,self._shape[2])
            expected = math.prod(shape)*4
            staged = []
            for pair in state.caches:
                if type(pair) is not tuple or len(pair) != 2:
                    raise ValueError('invalid range cache inventory')
                arrays = []
                for raw in pair:
                    if type(raw) is not bytes or len(raw) != expected:
                        raise ValueError('invalid range cache byte extent')
                    active = np.frombuffer(raw,dtype=np.float32).reshape(shape)
                    if not np.isfinite(active).all():
                        raise ValueError('nonfinite range snapshot cache')
                    full = np.zeros(self._shape,np.float32)
                    full[:,:state.position] = active
                    arrays.append(full)
                staged.append(tuple(arrays))
            if type(state.mask) is not bytes or len(state.mask) != self.batch_size*state.position:
                raise ValueError('invalid range snapshot mask')
            active = np.frombuffer(state.mask,dtype=np.uint8).reshape(self.batch_size,state.position)
            if np.any(active > 1):
                raise ValueError('invalid range snapshot mask values')
            mask = np.zeros_like(self._mask)
            mask[:,:state.position] = active
            self._cache,self._mask,self._position = tuple(staged),mask,state.position

    def reset(self):
        with self._lock:
            self._check()
            for keys,values in self._cache:
                keys.fill(0)
                values.fill(0)
            self._mask.fill(0)
            self._position = 0

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('cannot close active Mixtral range')
            self._closed = True
            self._cache = ()
            self.weights.clear()
