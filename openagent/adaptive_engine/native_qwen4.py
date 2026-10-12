"""Caller-built C arithmetic for exact mixed linear/QSA/PLE Qwen4 text.

Python owns buffers and transaction commit. No model arithmetic falls back to
NumPy. Resident FP32, default/linear 2D text RoPE and bounded C geometry.
"""
import ctypes as C
import _ctypes
import os
from pathlib import Path
import weakref

import numpy as np

from .independent_qwen4 import IndependentQwen4TextDecoder
from .native_qwen35 import Config, Tensor, tensor
from .native_qwen4_hyper import CompiledQwen4HyperKernel


class NativeQwen4TextDecoder(IndependentQwen4TextDecoder):
    implementation = 'c-qwen4-text-resident'
    native_range_execution = True

    def __init__(self, config, tensors, *, library, max_bytes=512*1024*1024, max_snapshots=4,
                 layer_range=None, _edge_only=False, expert_provider=None,ple_provider=None,
                 native_composition=False,native_expert_bank=None,native_ple_bank=None):
        if type(native_composition) is not bool:raise ValueError('boolean native composition policy required')
        self._composition=None
        self._composition_experts=native_expert_bank
        self._composition_ple=native_ple_bank
        if native_ple_bank is not None:
            from .native_original_ple import NativeOriginalPLE
            from .qwen4_ple_store import Qwen4PLEStore
            if (not native_composition or type(native_ple_bank) is not NativeOriginalPLE
                    or type(ple_provider) is not Qwen4PLEStore
                    or native_ple_bank.provider_identity!=ple_provider.identity):
                raise ValueError('matching borrowed native PLE bank and composition required')
            native_ple_bank.callback
        if native_expert_bank is not None:
            from .native_original_experts import NativeOriginalExperts
            from .qwen4_publisher_experts import Qwen4PublisherExperts
            if (not native_composition or type(native_expert_bank) is not NativeOriginalExperts
                    or type(expert_provider) is not Qwen4PublisherExperts
                    or native_expert_bank.family!='qwen4_exp_text'
                    or native_expert_bank.provider_identity!=expert_provider.identity):
                raise ValueError('matching borrowed native Qwen4 original expert bank and composition required')
            native_expert_bank.handle
        self._closed = False
        self._source_config = dict(config)
        self._executing = False
        self._expert_provider,self._expert_identity,self._expert_error = expert_provider,None,None
        self._expert_bytes = 0
        self._ple_provider=ple_provider;self._ple_identity=None;self._ple_error=None;self._ple_bytes=0
        if ple_provider is not None:
            from .qwen4_ple_store import Qwen4PLEStore
            if type(ple_provider) is not Qwen4PLEStore or _edge_only or dict(ple_provider.config)!=dict(config):
                raise ValueError('exact Qwen4 text PLE provider contract required')
            self._ple_identity=ple_provider.identity;self._ple_bytes=ple_provider.max_bytes
            if native_ple_bank is not None:self._ple_bytes+=native_ple_bank.max_bytes
            if self._ple_bytes>=max_bytes:raise MemoryError('native Qwen4 PLE owners exceed aggregate admission')
        if expert_provider is not None:
            from .qwen4_streamed_experts import Qwen4StreamedExperts,qwen4_metadata_fingerprint
            from .qwen4_publisher_experts import Qwen4PublisherExperts
            if type(expert_provider) not in (Qwen4StreamedExperts,Qwen4PublisherExperts):
                raise ValueError('explicit Qwen4 packed or original publisher expert provider required')
            fingerprint=(expert_provider.fingerprint if type(expert_provider) is Qwen4PublisherExperts else
                         expert_provider.runtime.manifest.fingerprint)
            if (_edge_only or
                    expert_provider.hidden_size!=config['hidden_size'] or
                    expert_provider.intermediate_size!=config['moe_intermediate_size'] or
                    expert_provider.expert_count!=config['num_experts'] or
                    expert_provider.selected_experts!=config['num_experts_per_tok'] or
                    len(expert_provider.layer_ids)!=config['num_hidden_layers'] or
                    fingerprint!=qwen4_metadata_fingerprint(config)):
                raise ValueError('exact Qwen4 packed provider geometry/fingerprint required')
            self._expert_identity = expert_provider.identity
            self._expert_bytes = expert_provider.max_bytes
            if native_expert_bank is not None:self._expert_bytes+=native_expert_bank.max_bytes
            if self._expert_bytes>=max_bytes:raise MemoryError('native Qwen4 expert owners exceed aggregate admission')
        if self._expert_bytes+self._ple_bytes>=max_bytes:
            raise MemoryError('native Qwen4 borrowed stores exceed aggregate admission')
        super().__init__(config, tensors, max_bytes=max_bytes, max_snapshots=max_snapshots,
                         layer_range=layer_range,_edge_only=_edge_only,_omit_experts=expert_provider is not None,
                         _omit_ple=ple_provider is not None)
        self.library = C.CDLL(str(Path(library).resolve(strict=True)))
        release = _ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose
        self._release = weakref.finalize(self, release, self.library._handle)
        try:
            self._bind_native()
            if native_composition:
                if (_edge_only or (expert_provider is not None and native_expert_bank is None)
                        or (ple_provider is not None and native_ple_bank is None)):
                    raise ValueError('native composition requires resident complete/range weights or native store attachment')
                from .native_qwen4_composition import NativeQwen4Composition
                self._composition=NativeQwen4Composition(self.library)
            import hashlib
            digest = hashlib.sha256()
            with Path(library).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            self.identity += ':library:'+digest.hexdigest()
            if self._expert_identity is not None:
                self.identity += ':experts:'+self._expert_identity
            if self._ple_identity is not None:
                self.identity += ':ple:'+self._ple_identity
            if self._composition is not None:self.identity+=':native-composition-v1'
            if native_expert_bank is not None:self.identity+=':native-experts:'+native_expert_bank.identity
            if native_ple_bank is not None:self.identity+=':native-ple:'+native_ple_bank.identity
        except BaseException:
            self.close()
            raise

    def _bind_native(self):
        self.hyper_kernel = CompiledQwen4HyperKernel.__new__(CompiledQwen4HyperKernel)
        self.hyper_kernel.library = self.library
        self.hyper_kernel.function = self.library.oa_qwen4_hyper
        pointer = C.POINTER(C.c_float)
        self.hyper_kernel.function.argtypes = [C.c_uint64]*4+[C.c_double]+[pointer]*7
        self.hyper_kernel.function.restype = C.c_int
        c = self.config
        if not 2<=c.hc_count<=64:
            raise ValueError('unsupported native Qwen4 hyper geometry')
        if self.rotary_dim and (c.indexer_n_heads>128 or c.indexer_head_dim>4096 or
                c.indexer_compress_ratio>c.max_position_embeddings or
                c.indexer_budget>c.max_position_embeddings):
            raise ValueError('unsupported native Qwen4 indexer geometry')
        if c.ple_layer_ids and (c.ngram_size>64 or c.heads_per_ngram>128 or
                c.ple_embed_dim>65536 or c.ple_conv_kernel_size>64):
            raise ValueError('unsupported native Qwen4 PLE geometry')
        sparse = 'qwen_sparse_attention' in c.layer_types
        # The shared C record requires positive attention fields even when no
        # sparse layer executes. These lanes are unused by the linear kernels.
        heads, kv, dim = ((c.num_attention_heads, c.num_key_value_heads, c.head_dim)
                          if sparse else (1, 1, 2))
        self._c = Config(c.hidden_size, heads, kv, dim,
            c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim,
            c.linear_conv_kernel_dim, c.num_experts, c.num_experts_per_tok,
            c.moe_intermediate_size, c.shared_expert_intermediate_size, c.max_position_embeddings,
            self.rotary_dim or 2, c.rms_norm_eps, getattr(self, 'theta', 10000.0))
        self.library.oa_qwen4_version.argtypes = []
        self.library.oa_qwen4_version.restype = C.c_uint64
        self.library.oa_qwen4_geometry.argtypes = [C.POINTER(Config)]
        self.library.oa_qwen4_geometry.restype = C.c_int
        if self.library.oa_qwen4_version() != 1 or self.library.oa_qwen4_geometry(C.byref(self._c)):
            raise ValueError('unsupported private Qwen4 ABI or bounded geometry')
        pointer = C.POINTER(C.c_int64)
        config_pointer, weights_pointer = C.POINTER(Config), C.POINTER(Tensor)
        self.library.oa_qwen4_linear.argtypes = [config_pointer, weights_pointer, C.c_uint64, C.c_uint64,
            C.POINTER(C.c_uint8), C.c_int, Tensor, Tensor, Tensor, Tensor]
        self.library.oa_qwen4_moe.argtypes = [config_pointer, weights_pointer, C.c_uint64, C.c_int,
            Tensor, Tensor, Tensor, pointer, Tensor]
        self.library.oa_qwen4_inject.argtypes = [C.c_uint64]*3+[Tensor]*4
        self.library.oa_qwen4_head.argtypes = [C.c_uint64]*3+[Tensor]*3
        self.library.oa_qwen35_embed.argtypes = [Tensor, C.c_uint64, C.c_uint64, pointer, C.c_uint64, Tensor]
        self.library.oa_qwen4_qsa.argtypes = [config_pointer, weights_pointer]+[C.c_uint64]*7+[
            C.POINTER(C.c_uint8), pointer]+[Tensor]*6+[C.POINTER(C.c_uint8)]
        self._qsa_function = self.library.oa_qwen4_qsa
        if self.rope_scale != 1:
            try:
                version = self.library.oa_qwen4_linear_rope_version
                version.argtypes = []
                version.restype = C.c_uint64
                scaled = self.library.oa_qwen4_qsa_scaled
            except AttributeError as error:
                raise ValueError('linear RoPE requires the additive Qwen4 scaled ABI') from error
            if version() != 1:
                raise ValueError('unsupported Qwen4 linear RoPE ABI')
            scaled.argtypes = self.library.oa_qwen4_qsa.argtypes + [C.c_double]
            scaled.restype = C.c_int
            self._qsa_function = scaled
        if self.mrope_section is not None:
            try:
                version = self.library.oa_qwen4_mrope_version
                version.argtypes = []
                version.restype = C.c_uint64
                mrope = self.library.oa_qwen4_qsa_mrope
            except AttributeError as error:
                raise ValueError('multimodal Qwen4 requires the native MRoPE ABI') from error
            if version() != 1:
                raise ValueError('unsupported Qwen4 MRoPE ABI')
            mrope.argtypes = self.library.oa_qwen4_qsa.argtypes + [C.c_double,C.POINTER(C.c_int32)]
            mrope.restype = C.c_int
            self._qsa_function = mrope
            self._mrope_sections = (C.c_int32*3)(*self.mrope_section)
        self.library.oa_qwen4_ple.argtypes = [C.c_uint64]*9+[C.c_double, weights_pointer]+[
            pointer]*5+[C.POINTER(C.c_uint8)]+[Tensor]*3
        if self._ple_provider is not None:
            try:
                version=self.library.oa_qwen4_streamed_ple_version
                version.argtypes=[];version.restype=C.c_uint64
                self._streamed_ple=self.library.oa_qwen4_ple_streamed
            except AttributeError as error:
                raise ValueError('PLE row streaming requires the additive native ABI') from error
            if version()!=1: raise ValueError('unsupported streamed PLE ABI')
            self._ple_callback_type=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_uint64,C.c_uint64,C.POINTER(C.c_float))
            self._streamed_ple.argtypes=self.library.oa_qwen4_ple.argtypes+[
                self._ple_callback_type,C.c_void_p,C.c_uint64]
            self._streamed_ple.restype=C.c_int
        self.library.oa_qwen4_add.argtypes = [Tensor]*3
        for name in ('oa_qwen4_linear', 'oa_qwen4_moe', 'oa_qwen4_inject', 'oa_qwen4_head',
                     'oa_qwen35_embed', 'oa_qwen4_qsa', 'oa_qwen4_ple', 'oa_qwen4_add'):
            getattr(self.library, name).restype = C.c_int
        self._streamed_moe = None
        self._expert_callback = None
        if self._expert_provider is not None:
            from .native_qwen35 import ExpertFFN
            version = self.library.oa_qwen4_streamed_version
            version.argtypes,version.restype = [],C.c_uint64
            if version()!=1:
                raise ValueError('unsupported Qwen4 additive selected-expert ABI')
            self._streamed_moe = self.library.oa_qwen4_moe_streamed
            self._streamed_moe.argtypes = self.library.oa_qwen4_moe.argtypes+[ExpertFFN,C.c_void_p,C.c_uint64]
            self._streamed_moe.restype = C.c_int
            def selected_expert(user,layer,expert,incoming,count,outgoing,out_count):
                try:
                    if count!=c.hidden_size or out_count!=c.hidden_size:
                        raise ValueError('Qwen4 selected-expert callback geometry mismatch')
                    vector = np.array(np.ctypeslib.as_array(incoming,shape=(count,)),copy=True)
                    from .qwen4_publisher_experts import Qwen4PublisherExperts
                    if type(self._expert_provider) is Qwen4PublisherExperts:
                        result=self._expert_provider.execute(int(layer),int(expert),vector,
                            cancelled=getattr(self,'_cancelled',None))
                    else:
                        self._cancellation_boundary()
                        result = self._expert_provider.execute(int(layer),int(expert),vector)
                    if result.dtype!=np.float32 or result.shape!=(c.hidden_size,) or not np.isfinite(result).all():
                        raise ArithmeticError('invalid Qwen4 selected expert result')
                    C.memmove(outgoing,result.ctypes.data,result.nbytes)
                    return 0
                except BaseException as error:
                    self._expert_error = error
                    return 1
            self._expert_callback = ExpertFFN(selected_expert)

    def _check_open(self):
        if getattr(self, '_transport', False) or self._executing:
            raise RuntimeError('Qwen4 cache transport is active')
        if self._closed:
            raise RuntimeError('native Qwen4 decoder is closed')

    def reset(self):
        self._check_open()
        with self._lock:
            self._check_open()
            super().reset()

    def close(self):
        with self._lock:
            if getattr(self, '_transport', False) or self._executing:
                raise RuntimeError('cannot close Qwen4 during cache transport')
            if not self._closed:
                self._closed = True
                try:
                    self.cache = self.ple_cache = ()
                    self.mask = self.positions = None
                    self.position = 0
                    self.weights = {}
                    self._snapshots.clear()
                    self._snapshot_bytes = 0
                finally:
                    self.hyper_kernel = None
                    self._composition = None
                    self._composition_experts = None
                    self._composition_ple = None
                    self._streamed_moe = None
                    self._expert_callback = None
                    self._expert_provider = None
                    self._ple_provider = None
                    self.library = None
                    self._release()

    routing_observation_supported = False  # Native composition returns executed traces directly.

    def forward(self, token_ids, *, attention_mask=None, position_ids=None, inputs_embeds=None, cancelled=None):
        return self._forward(token_ids,attention_mask=attention_mask,position_ids=position_ids,
                             inputs_embeds=inputs_embeds,cancelled=cancelled)

    def _forward(self, token_ids, *, attention_mask, position_ids, incoming_hidden=None, cancelled=None, inputs_embeds=None):
        with self._lock:
            self._check_open()
            ids = np.asarray(token_ids)
            if ids.ndim == 1:
                ids = ids[None]
            c = self.config
            if (ids.ndim != 2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or
                    np.any(ids < 0) or np.any(ids >= c.vocab_size)):
                raise ValueError('invalid token_ids')
            b,t = ids.shape
            end = self.position+t
            if b>128 or b*t>1048576 or end>c.max_position_embeddings:
                raise ValueError('unsupported bounded native batch/context geometry')
            self._admit_native(b,t,end)
            ids = np.array(ids, dtype=np.int64, order='C', copy=True)
            committed = (self.cache,self.mask,self.position,self.positions,self.ple_cache)
            try:
                if cancelled is not None and not callable(cancelled):
                    raise TypeError('callable Qwen4 cancellation required')
                self._cancelled = cancelled
                self._cancellation_boundary()
                if self._composition is not None:
                    return self._composition.forward_staged(self,ids,attention_mask=attention_mask,
                        position_ids=position_ids,incoming_hidden=incoming_hidden,inputs_embeds=inputs_embeds)
                return super()._forward(ids, attention_mask=attention_mask, position_ids=position_ids,
                                        incoming_hidden=incoming_hidden,inputs_embeds=inputs_embeds)
            except BaseException:
                self.cache,self.mask,self.position,self.positions,self.ple_cache = committed
                raise
            finally:
                self._cancelled = None

    def _cancellation_boundary(self):
        callback = getattr(self, '_cancelled', None)
        if callback is not None:
            self._executing = True
            try:
                if callback():
                    raise InterruptedError('Qwen4 cancelled; continuation unchanged')
            finally:
                self._executing = False

    def _admit_native(self, b, t, end):
        c = self.config
        rows,width = b*t,c.hc_count*c.hidden_size
        linear = c.layer_types.count('linear_attention')
        sparse = c.num_hidden_layers-linear
        vw = c.linear_num_value_heads*c.linear_value_head_dim
        channels = 2*c.linear_num_key_heads*c.linear_key_head_dim+vw
        candidate = b*linear*(channels*c.linear_conv_kernel_dim+
            c.linear_num_value_heads*c.linear_key_head_dim*c.linear_value_head_dim)*4
        old = sum(a.nbytes for entries in (self.cache,self.ple_cache) for entry in entries if entry for a in entry)
        old += sum(a.nbytes for a in (self.mask,self.positions) if a is not None)
        retained = 4*rows*((c.num_hidden_layers+1)*width+c.hidden_size+c.vocab_size+
            c.num_hidden_layers*c.num_experts)+12*rows*c.num_hidden_layers*c.num_experts_per_tok
        workspace = 4*(2*channels+3*vw+2*c.linear_num_value_heads+
            2*c.linear_key_head_dim+c.linear_value_head_dim+c.hidden_size)
        workspace += 4*(c.num_experts+2*c.moe_intermediate_size+2*c.shared_expert_intermediate_size+c.hidden_size+1)
        workspace += 8*(width+c.hc_lowrank)+4*rows*(c.hidden_size+c.hc_count)
        if sparse:
            candidate += 4*b*sparse*end*(2*c.num_key_value_heads*c.head_dim+c.indexer_head_dim)
            retained += sparse*rows*end*(4*c.num_attention_heads+1)
            ip = max((c.indexer_n_heads+1)*c.indexer_head_dim,2*c.num_key_value_heads*c.head_dim)
            workspace += 4*(3*c.num_attention_heads*c.head_dim+ip+2*c.indexer_head_dim+3*end)+24*end
            workspace += rows*end*2
        if c.ple_layer_ids:
            span = (c.ple_conv_kernel_size-1)*c.ngram_size
            candidate += len(c.ple_layer_ids)*b*(8*(c.ngram_size-1)+4*width*span)
            workspace += 4*(c.ple_embed_dim+4*width+c.hidden_size)+8*c.ngram_size
        temporary = 4*rows*(4*width+4*c.hidden_size+c.num_experts)+12*rows*c.num_experts_per_tok
        metadata = 8*rows+32*b*end+4*b*end
        if self.weight_bytes+self._snapshot_bytes+old+candidate+retained+workspace+temporary+metadata+self._expert_bytes+self._ple_bytes>self.max_bytes:
            raise MemoryError('native Qwen4 transaction exceeds memory budget')

    def _mix(self, x, prefix, inject):
        self._check_open()
        return super()._mix(x,prefix,inject)

    def status(self):
        with self._lock:
            result = super().status()
            result.update(implementation=self.implementation, native_range_execution=True,
                          supported_contract='linear-QSA-PLE-default-linear-MRoPE-FP32-boundaries',
                          native_streamed_range_execution=self._composition is not None and
                              (self._composition_experts is not None or self._composition_ple is not None),
                          native_composition=self._composition is not None,
                          native_original_experts=self._composition_experts is not None,
                          native_original_ple=self._composition_ple is not None,
                          composition_abi=1 if self._composition is not None else None,closed=self._closed)
            return result

    @staticmethod
    def _result(status):
        if status:
            raise ValueError(f'native Qwen4 execution failed ({status})')

    def _embed(self, ids):
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        c = self.config
        output = np.empty((*ids.shape, c.hidden_size), np.float32)
        self._result(self.library.oa_qwen35_embed(tensor(self.weights['model.embed_tokens.weight']),
            c.vocab_size, c.hidden_size, ids.ctypes.data_as(C.POINTER(C.c_int64)), ids.size, tensor(output)))
        return np.tile(output, (1, 1, c.hc_count))

    def _inject(self, x, value, gates):
        output = np.empty_like(x)
        c = self.config
        self._result(self.library.oa_qwen4_inject(value.size//c.hidden_size, c.hc_count, c.hidden_size,
            tensor(x), tensor(value), tensor(gates), tensor(output)))
        return output

    def _add(self, x, value):
        output = np.empty_like(x)
        self._result(self.library.oa_qwen4_add(tensor(x), tensor(value), tensor(output)))
        return output

    def _head(self, x):
        c = self.config
        output = np.empty((*x.shape[:-1], c.vocab_size), np.float32)
        self._result(self.library.oa_qwen4_head(x.size//c.hidden_size, c.hidden_size, c.vocab_size,
            tensor(x), tensor(self.weights['lm_head.weight']), tensor(output)))
        return output

    def _delta(self, x, prefix, old, mask):
        c = self.config
        b, t, _ = x.shape
        width = 2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
        history = np.zeros((b, width, c.linear_conv_kernel_dim), np.float32) if old is None else old[0].copy()
        state = np.zeros((b, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim), np.float32) if old is None else old[1].copy()
        names = ('in_proj_qkv.weight', 'in_proj_z.weight', 'in_proj_b.weight', 'in_proj_a.weight',
                 'conv1d.weight', 'norm.weight', 'dt_bias', 'A_log', 'out_proj.weight')
        weights = (Tensor*9)(*(tensor(self.weights[prefix+'linear_attn.'+name]) for name in names))
        output = np.empty_like(x)
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        self._result(self.library.oa_qwen4_linear(C.byref(self._c), weights, b, t,
            mask.ctypes.data_as(C.POINTER(C.c_uint8)), int(getattr(c, 'output_gate_type', None) == 'sigmoid'),
            tensor(x), tensor(history), tensor(state), tensor(output)))
        return output, (history, state)

    def _moe(self, x, prefix):
        c = self.config
        rows = x.size//c.hidden_size
        names = ('gate.weight', 'experts.gate_up_proj', 'experts.down_proj', 'shared_expert.gate_proj.weight',
                 'shared_expert.up_proj.weight', 'shared_expert.down_proj.weight', 'shared_expert_gate.weight')
        weights = (Tensor*7)(*(Tensor() if self._expert_provider is not None and slot in (1,2)
            else tensor(self.weights[prefix+'mlp.'+name]) for slot,name in enumerate(names)))
        output = np.empty_like(x)
        router = np.empty((rows, c.num_experts), np.float32)
        selected = np.empty((rows, c.num_experts_per_tok), np.int64)
        scores = np.empty(selected.shape, np.float32)
        args = (C.byref(self._c),weights,rows,int(getattr(c,'norm_topk_prob',True)),
            tensor(x),tensor(output),tensor(router),selected.ctypes.data_as(C.POINTER(C.c_int64)),tensor(scores))
        self._executing = True
        try:
            if self._expert_provider is None:
                status = self.library.oa_qwen4_moe(*args)
            else:
                if self._expert_provider.identity!=self._expert_identity:
                    raise ValueError('Qwen4 packed provider changed after admission')
                layer = int(prefix.split('.')[2])
                self._expert_error = None
                status = self._streamed_moe(*args,self._expert_callback,None,layer)
                if status and self._expert_error is not None:
                    error,self._expert_error = self._expert_error,None
                    raise RuntimeError('Qwen4 selected expert failed; continuation unchanged') from error
            self._result(status)
        finally:
            self._executing = False
        return output, router, selected, scores

    def _qsa(self, x, prefix, old, positions, full_positions, mask):
        c = self.config
        b, t, _ = x.shape
        end = self.position+t
        keys = np.zeros((b, c.num_key_value_heads, end, c.head_dim), np.float32)
        values = np.zeros_like(keys)
        raw = np.zeros((b, end, c.indexer_head_dim), np.float32)
        if old is not None:
            keys[:, :, :self.position] = old[0]
            values[:, :, :self.position] = old[1]
            raw[:, :self.position] = old[2]
        names = ('q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight', 'q_norm.weight',
                 'k_norm.weight', 'indexer.index_qk_proj.weight', 'indexer.q_layernorm.weight', 'indexer.k_layernorm.weight')
        weights = (Tensor*9)(*(tensor(self.weights[prefix+'self_attn.'+name]) for name in names))
        output = np.empty_like(x)
        attention = np.empty((b, c.num_attention_heads, t, end), np.float32)
        chosen = np.empty((b, 1, t, end), np.uint8)
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        full_positions = np.ascontiguousarray(full_positions, dtype=np.int64)
        extra = ((self.rope_scale,self._mrope_sections) if self.mrope_section is not None else
                 (() if self.rope_scale == 1 else (self.rope_scale,)))
        self._result(self._qsa_function(C.byref(self._c), weights, b, t, self.position,
            c.indexer_n_heads, c.indexer_head_dim, c.indexer_budget, c.indexer_compress_ratio,
            mask.ctypes.data_as(C.POINTER(C.c_uint8)), full_positions.ctypes.data_as(C.POINTER(C.c_int64)),
            tensor(x), tensor(keys), tensor(values), tensor(raw), tensor(output), tensor(attention),
            chosen.ctypes.data_as(C.POINTER(C.c_uint8)), *extra))
        return output, (keys, values, raw), attention, chosen.astype(bool)

    def _ple(self, x, ids, prefix, old, mask):
        c = self.config
        b, t, width = x.shape
        span = (c.ple_conv_kernel_size-1)*c.ngram_size
        eos = c.eos_token_id[0] if isinstance(c.eos_token_id, list) else c.eos_token_id
        previous = np.full((b, c.ngram_size-1), eos, np.int64) if old is None else old[0].copy()
        history = np.zeros((b, width, span), np.float32) if old is None else old[1].copy()
        p = prefix+'ple.'
        names = ('ple_embedding.ngram_embedding.weight', 'key_proj.weight', 'value_proj.weight',
                 'norm_key.weight', 'norm_query.weight', 'norm_conv.weight', 'conv1d.weight')
        weights = (Tensor*7)(*(Tensor() if index==0 and self._ple_provider is not None else tensor(self.weights[p+name])
                               for index,name in enumerate(names)))
        metadata = [self.weights[p+'ple_embedding.'+name] for name in
                    ('layer_multipliers', 'ngram_heads_vocab_sizes', 'ngram_heads_offsets')]
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        output = np.empty_like(x)
        pointers = [value.ctypes.data_as(C.POINTER(C.c_int64)) for value in (*metadata, ids, previous)]
        arguments=(b, t, c.hc_count, c.hidden_size, c.ple_embed_dim,
            c.ple_conv_kernel_size, c.ngram_size, c.heads_per_ngram, eos, c.rms_norm_eps,
            weights, *pointers, mask.ctypes.data_as(C.POINTER(C.c_uint8)),
            tensor(x), tensor(history), tensor(output))
        self._executing=True
        try:
            if self._ple_provider is None:
                status=self.library.oa_qwen4_ple(*arguments)
            else:
                if self._ple_provider.identity!=self._ple_identity:
                    raise ValueError('PLE provider identity changed after admission')
                layer=int(prefix.split('.')[2]);self._ple_error=None
                def lookup(user,index,part,destination):
                    try:
                        callback=getattr(self,'_cancelled',None)
                        if callback is not None and callback():
                            raise InterruptedError('Qwen4 PLE row read cancelled')
                        value=self._ple_provider.row(layer,int(index))
                        if value.shape!=(part,): raise ValueError('streamed PLE row extent differs')
                        C.memmove(destination,value.ctypes.data,int(part)*4)
                        return 0
                    except BaseException as error:
                        self._ple_error=error;return 1
                callback=self._ple_callback_type(lookup)
                status=self._streamed_ple(*arguments,callback,None,self._ple_provider.shapes[layer][0])
                if status and self._ple_error is not None:
                    error,self._ple_error=self._ple_error,None
                    raise RuntimeError('Qwen4 PLE row streaming failed; continuation unchanged') from error
            self._result(status)
        finally:
            self._executing=False
        return output, (previous, history)
