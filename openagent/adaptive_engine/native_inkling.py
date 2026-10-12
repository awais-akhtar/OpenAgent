"""Resident compiled C Inkling decoder; no Transformers forward dependency."""
import ctypes as C
import _ctypes
from pathlib import Path
from types import SimpleNamespace
from types import MappingProxyType
import weakref

import numpy as np

from .independent_inkling import IndependentInklingDecoder, GLOBAL_NAMES, ALL_LAYER_NAMES, logits_selection


class Tensor(C.Structure):
    _fields_ = [('data', C.POINTER(C.c_float)), ('count', C.c_uint64)]


class Config(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
                ('hidden', 'vocab', 'output_vocab', 'layers', 'width', 'kernel',
                 'relative', 'capacity', 'floor', 'moe_width', 'experts', 'shared', 'topk')]
    _fields_ += [(name, C.c_float) for name in ('epsilon', 'multiplier', 'alpha', 'route_scale')]


class Layer(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ('heads', 'kv_heads', 'dim', 'extent', 'window', 'sparse')]
    _fields_ += [('weights', Tensor * 26), ('state', Tensor)]

ExpertCall=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_uint64,C.c_uint64,
                      C.POINTER(C.c_float),C.POINTER(C.c_float))


def tensor(a):
    if a.dtype != np.float32 or not a.flags.c_contiguous:
        raise ValueError('private C ABI requires contiguous FP32')
    return Tensor(a.ctypes.data_as(C.POINTER(C.c_float)), a.size)


def _release_library(handle):
    release = _ctypes.FreeLibrary if hasattr(_ctypes, 'FreeLibrary') else _ctypes.dlclose
    release(handle)


class NativeInklingDecoder(IndependentInklingDecoder):
    implementation = 'compiled-c-inkling-token-stream'
    native_range_execution = True
    native_streamed_range_execution = False

    def __init__(self, config, tensors, *, library, max_bytes=512 * 1024 * 1024,_borrow_globals=False,expert_provider=None):
        if expert_provider is not None:
            from .inkling_publisher_experts import InklingPublisherExperts
            if type(expert_provider) is not InklingPublisherExperts:
                raise ValueError('explicit original Inkling expert provider required')
        reserve=0 if expert_provider is None else expert_provider.max_bytes
        if reserve>=max_bytes: raise MemoryError('Inkling expert reservation exhausts native budget')
        super().__init__(config, tensors, max_bytes=max_bytes-reserve,_borrow_globals=_borrow_globals,
                         _omit_experts=expert_provider is not None)
        self.max_bytes=max_bytes
        self._expert_provider=expert_provider;self._expert_reserve=reserve
        self._callback=None;self._callback_error=None;self._active_cancel=None;self._executing=False
        self._resident_bytes=(sum(value.nbytes for value in self.weights.values())+reserve+
                              8*sum(self._state_size(g) for g in self.geometry))
        self._closed = False
        self._library_release = None
        self.library = library if isinstance(library, C.CDLL) else C.CDLL(
            str(Path(library).resolve(strict=True)))
        if not isinstance(library, C.CDLL):
            self._library_release = weakref.finalize(self, _release_library, self.library._handle)
        try:
            self._bind_native()
        except BaseException:
            self.close()
            raise

    def _bind_native(self):
        self.library.oa_inkling_version.argtypes = []
        self.library.oa_inkling_version.restype = C.c_uint64
        if self.library.oa_inkling_version() != 2:
            raise ValueError('unsupported private Inkling ABI')
        self.library.oa_inkling_run.argtypes = [C.POINTER(Config), C.POINTER(Layer),
            C.POINTER(Tensor), C.POINTER(C.c_int64), C.c_uint64, C.c_uint64,
            C.c_uint64, C.c_uint64, Tensor, Tensor, Tensor]
        self.library.oa_inkling_run.restype = C.c_int
        c = self.config
        self._c = Config(c['hidden_size'], c['vocab_size'], c['unpadded_vocab_size'] or c['vocab_size'],
                         c['num_hidden_layers'], c['intermediate_size'], c['conv_kernel_size'],
                         c['d_rel'], c['max_position_embeddings'], c['log_scaling_n_floor'] or 0,
                         c['moe_intermediate_size'], c['n_routed_experts'], c['n_shared_experts'], c['num_experts_per_tok'],
                         c['rms_norm_eps'], c['logits_mup_width_multiplier'], c['log_scaling_alpha'], c['route_scale'])
        if self._expert_provider is not None:
            provider=self._expert_provider
            fields=('hidden_size','moe_intermediate_size','n_routed_experts','mlp_layer_types')
            if any(provider.config[name]!=c[name] for name in fields): raise ValueError('Inkling streamed expert geometry differs')
            self.library.oa_inkling_streamed_version.argtypes=[]
            self.library.oa_inkling_streamed_version.restype=C.c_uint64
            if self.library.oa_inkling_streamed_version()!=1: raise ValueError('unsupported Inkling streamed expert ABI')
            def call(user,layer,expert,incoming,outgoing):
                try:
                    x=np.ctypeslib.as_array(incoming,shape=(c['hidden_size'],))
                    value=provider.evaluate(int(layer),int(expert),x,cancelled=self._active_cancel)
                    np.ctypeslib.as_array(outgoing,shape=(c['hidden_size'],))[:]=value
                    return 0
                except BaseException as error: self._callback_error=error;return 1
            self._callback=ExpertCall(call)

    def _check_open(self):
        if self._closed or self._executing:
            raise RuntimeError('native Inkling decoder is closed')

    def reset(self):
        with self._lock:
            self._check_open()
            super().reset()

    def cache_snapshot(self):
        with self._lock:
            self._check_open()
            return super().cache_snapshot()

    def status(self):
        with self._lock:
            result = super().status()
            result['closed'] = self._closed
            return result

    def close(self):
        with self._lock:
            if self._executing: raise RuntimeError('Inkling native callback owns resources')
            if self._closed:
                return
            self._closed = True
            self._cache, self.position = None, 0
            self.weights = MappingProxyType({})
            if self._library_release is not None:
                self._library_release()
            self.library = None
            self._callback=self._expert_provider=None

    def __enter__(self):
        with self._lock:
            self._check_open()
            return self

    def __exit__(self, *_):
        self.close()

    def _execute(self, ids, pending, hidden, logits, routing, start=None, embeddings=None,raw_hidden=False,position_origin=0):
        layers = (Layer * len(self.geometry))()
        for i, g in enumerate(self.geometry):
            weights = (Tensor * 26)(*[tensor(self.weights[f'model.layers.{i}.{name}'])
                if f'model.layers.{i}.{name}' in self.weights else Tensor() for name in ALL_LAYER_NAMES])
            layers[i] = Layer(g.heads, g.kv_heads, g.dim, g.extent, g.window,
                              int(self.config['mlp_layer_types'][i] == 'sparse'), weights, tensor(pending[i]))
        global_weights = (Tensor * 4)(*[tensor(self.weights[name]) for name in GLOBAL_NAMES])
        function=self.library.oa_inkling_run
        extra=()
        if embeddings is not None:
            try:
                version=self.library.oa_inkling_embeddings_version
                version.argtypes=[];version.restype=C.c_uint64
                function=self.library.oa_inkling_run_embeddings
            except AttributeError as error:
                raise ValueError('Inkling media composition requires the native embeddings ABI') from error
            if version()!=1:
                raise ValueError('unsupported Inkling embeddings ABI')
            function.argtypes=self.library.oa_inkling_run.argtypes+[Tensor]
            function.restype=C.c_int
            extra=(tensor(embeddings),)
        if raw_hidden:
            version=self.library.oa_inkling_block_version
            version.argtypes=[];version.restype=C.c_uint64
            if version()!=1: raise ValueError('unsupported Inkling decoder block ABI')
            function=self.library.oa_inkling_run_block
            function.argtypes=self.library.oa_inkling_run.argtypes+[Tensor,C.c_uint64]
            function.restype=C.c_int
            extra=(tensor(embeddings),position_origin)
        if self._expert_provider is not None:
            if raw_hidden: raise ValueError('routed expert streaming is not a dense MTP block')
            function=self.library.oa_inkling_run_streamed
            function.argtypes=self.library.oa_inkling_run.argtypes+[Tensor,ExpertCall,C.c_void_p]
            function.restype=C.c_int
            extra=(Tensor() if embeddings is None else tensor(embeddings),self._callback,None)
        return function(C.byref(self._c), layers, global_weights,
            ids.ctypes.data_as(C.POINTER(C.c_int64)), ids.size, ids.shape[0], ids.shape[1],
            self.position if start is None else start, tensor(hidden), Tensor() if logits is None else tensor(logits), tensor(routing),*extra)

    def forward(self, token_ids, *, attention_mask=None, position_ids=None, logits_to_keep=0, use_cache=True, cancelled=None, inputs_embeds=None,
                _raw_hidden=False,_position_origin=0):
        with self._lock:
            self._check_open()
            if type(_raw_hidden) is not bool or type(_position_origin) is not int or not 0<=_position_origin<=65536 or (_raw_hidden and inputs_embeds is None):
                raise ValueError('explicit native block inputs and position origin required')
            if cancelled is not None and not callable(cancelled):
                raise TypeError('callable Inkling cancellation required')
            if cancelled is not None and cancelled():
                raise InterruptedError('Inkling cancelled before staging')
            source=None if inputs_embeds is None else np.asarray(inputs_embeds)
            extra_bytes=0 if source is None else 2*source.nbytes
            ids, pending = self._prepare(token_ids, attention_mask, position_ids, use_cache,
                additional_bytes=extra_bytes+self._expert_reserve)
            start = self.position if use_cache else 0
            selection = logits_selection(logits_to_keep, ids.shape[1])
            c = self.config
            embeddings=None
            if inputs_embeds is not None:
                if source.shape!=(*ids.shape,c['hidden_size']) or source.dtype!=np.float32 or not np.isfinite(source).all():
                    raise ValueError('finite post-embedding-norm Inkling media embeddings required')
                embeddings=np.array(source,dtype=np.float32,order='C',copy=True)
            hidden = np.empty((*ids.shape, c['hidden_size']), np.float32)
            logits = None if _raw_hidden else np.empty((*ids.shape, c['unpadded_vocab_size'] or c['vocab_size']), np.float32)
            routing = np.empty((*ids.shape, len(self.geometry), c['n_routed_experts'] + c['n_shared_experts']), np.float32)
            self._active_cancel=cancelled;self._callback_error=None;self._executing=True
            try: code = self._execute(ids, pending, hidden, logits, routing, start,embeddings,_raw_hidden,_position_origin)
            finally: self._executing=False;self._active_cancel=None
            if self._callback_error is not None:
                error=self._callback_error;self._callback_error=None;raise error
            if code:
                raise RuntimeError(f'Inkling C kernel failed ({code}); state not committed')
            if not np.isfinite(hidden).all() or (logits is not None and not np.isfinite(logits).all()) or not np.isfinite(routing).all() or any(not np.isfinite(a).all() for a in pending):
                raise FloatingPointError('nonfinite native result; state not committed')
            result = SimpleNamespace(logits=None if logits is None else logits[:, selection], last_hidden_state=hidden, routing_weights=routing)
            if use_cache:
                if cancelled is not None and cancelled():
                    raise InterruptedError('Inkling cancelled; continuation unchanged')
                self._cache, self.position = pending, start + ids.shape[1]
            return result
