"""Exact Qwen4 text, resident linear/QSA/PLE with default or linear text RoPE.

Hyper-connections are Qwen4-specific. The gated-delta equation is shared with
Qwen3.5; reusing that primitive does not change model identity or tensor names.
"""
from threading import RLock
from types import SimpleNamespace

import numpy as np
from dataclasses import dataclass
import hashlib
import json
from uuid import uuid4

from .independent_qwen35 import IndependentQwen35TextDecoder, sigmoid, silu
from .native_mixtral import rmsnorm, rotary, softmax


@dataclass(frozen=True)
class Qwen4Snapshot:
    identity: str
    owner: str
    handle: str
    position: int


class IndependentQwen4TextDecoder:
    implementation = 'numpy-qwen4-linear-hyperconnections'
    native_range_execution = False

    def __init__(self, config, tensors, *, hyper_kernel=None, delta_kernel=None,
                 max_bytes=512*1024*1024, max_snapshots=4, layer_range=None,
                 _inventory_only=False, _edge_only=False, _omit_experts=False, _omit_ple=False):
        raw = dict(config)
        if raw.get('model_type') != 'qwen4_exp_text':
            raise ValueError('expected exact qwen4_exp_text')
        fields = ('hidden_size', 'vocab_size', 'num_hidden_layers', 'hc_count', 'hc_lowrank',
                  'linear_conv_kernel_dim', 'linear_key_head_dim', 'linear_value_head_dim',
                  'linear_num_key_heads', 'linear_num_value_heads', 'num_experts',
                  'num_experts_per_tok', 'moe_intermediate_size',
                  'shared_expert_intermediate_size', 'max_position_embeddings')
        for key in fields:
            if type(raw.get(key)) is not int or raw[key] <= 0:
                raise ValueError(f'invalid {key}')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('invalid memory budget')
        if (raw['hc_count'] <= 1 or raw['linear_num_value_heads'] % raw['linear_num_key_heads']
                or raw['num_experts_per_tok'] > raw['num_experts']):
            raise ValueError('invalid hyper/head/expert geometry')
        schedule = raw.get('layer_types')
        if not isinstance(schedule, list) or len(schedule) != raw['num_hidden_layers'] or any(
                kind not in ('linear_attention', 'qwen_sparse_attention') for kind in schedule):
            raise ValueError('explicit exact Qwen4 layer schedule required')
        ple = raw.get('ple_layer_ids') or []
        if not isinstance(ple, list) or ple != sorted(set(ple)) or any(
                type(i) is not int or not 1 <= i <= raw['num_hidden_layers'] or schedule[i-1] != 'linear_attention' for i in ple):
            raise ValueError('invalid PLE layer ownership')
        raw['ple_layer_ids'] = ple
        if ple:
            for key in ('ple_embed_dim', 'ple_conv_kernel_size', 'ngram_size', 'heads_per_ngram',
                        'ngram_vocab_size_base', 'make_ngram_vocab_size_divisible_by'):
                if type(raw.get(key)) is not int or raw[key] <= 0:
                    raise ValueError(f'invalid PLE {key}')
            if type(raw.get('seed')) is not int or raw['ngram_size'] < 2 or raw['ple_embed_dim'] % ((raw['ngram_size']-1)*raw['heads_per_ngram']):
                raise ValueError('invalid PLE hash geometry')
            eos = raw.get('eos_token_id')
            eos = eos[0] if isinstance(eos, list) and eos else eos
            if type(eos) is not int or not 0 <= eos < raw['vocab_size']:
                raise ValueError('PLE requires a valid EOS token')
        self.rotary_dim = 0
        self.rope_scale = 1.0
        self.mrope_section = None
        if 'qwen_sparse_attention' in schedule:
            for key in ('num_attention_heads', 'num_key_value_heads', 'head_dim',
                        'indexer_n_heads', 'indexer_kv_heads', 'indexer_head_dim',
                        'indexer_budget', 'indexer_compress_ratio'):
                if type(raw.get(key)) is not int or raw[key] <= 0:
                    raise ValueError(f'invalid QSA {key}')
            rope = raw.get('rope_parameters')
            if not isinstance(rope, dict) or rope.get('rope_type') not in ('default', 'linear'):
                raise ValueError('QSA requires an explicit default or linear text RoPE contract')
            if set(rope) - {'rope_type', 'rope_theta', 'partial_rotary_factor', 'factor',
                            'mrope_section', 'mrope_interleaved'}:
                raise ValueError('unadmitted QSA rotary parameter')
            scale = rope.get('factor', 1.0)
            if (isinstance(scale, bool) or not isinstance(scale, (float, int)) or
                    not np.isfinite(scale) or scale < 1 or
                    (rope['rope_type'] == 'default' and scale != 1)):
                raise ValueError('invalid QSA linear rotary scaling factor')
            if rope['rope_type'] == 'linear' and 'factor' not in rope:
                raise ValueError('linear RoPE requires an explicit factor')
            self.rope_scale = float(scale)
            theta, factor = rope.get('rope_theta'), rope.get('partial_rotary_factor', 1.0)
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or v <= 0 for v in (theta, factor)):
                raise ValueError('invalid QSA rotary parameters')
            dimension = raw['head_dim']*factor
            if int(dimension) != dimension or int(dimension) % 2 or not 2 <= dimension <= min(raw['head_dim'], raw['indexer_head_dim']):
                raise ValueError('invalid QSA rotary dimension')
            if raw['indexer_kv_heads'] != 1 or raw['num_attention_heads'] % raw['num_key_value_heads'] or raw['indexer_budget'] % raw['indexer_compress_ratio']:
                raise ValueError('invalid QSA head or block geometry')
            self.rotary_dim, self.theta = int(dimension), theta
            if 'mrope_section' in rope:
                sections = rope['mrope_section']
                if (type(sections) is not list or len(sections) != 3 or
                        any(type(v) is not int or v < 0 for v in sections) or
                        sum(sections) != self.rotary_dim//2 or
                        rope.get('mrope_interleaved') is not True):
                    raise ValueError('explicit interleaved Qwen4 MRoPE sections required')
                self.mrope_section = tuple(sections)
            elif 'mrope_interleaved' in rope:
                raise ValueError('MRoPE interleaving requires an explicit section inventory')
        elif any(raw.get(key) is not None for key in ('indexer_n_heads', 'indexer_kv_heads',
                'indexer_head_dim', 'indexer_budget', 'indexer_compress_ratio')):
            raise ValueError('indexer geometry without a QSA layer')
        if (raw.get('hidden_act', 'silu') != 'silu' or
                raw.get('output_gate_type') not in (None, 'silu', 'sigmoid') or
                raw.get('quantization_config') is not None or raw.get('attention_bias', False)):
            raise ValueError('unsupported activation, quantization or bias')
        if type(raw.get('norm_topk_prob', True)) is not bool:
            raise ValueError('invalid norm_topk_prob')
        eps = raw.get('rms_norm_eps')
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not np.isfinite(eps) or eps <= 0:
            raise ValueError('invalid rms_norm_eps')
        self.config = c = SimpleNamespace(**raw)
        if layer_range is None:
            self.layer_begin,self.layer_end = 0,c.num_hidden_layers
            self.range_mode = False
        else:
            if (type(layer_range) is not tuple or len(layer_range)!=2 or
                    any(type(value) is not int for value in layer_range) or
                    not 0<=layer_range[0]<layer_range[1]<=c.num_hidden_layers):
                raise ValueError('exact nonempty Qwen4 layer interval required')
            self.layer_begin,self.layer_end = layer_range
            self.range_mode = True
        if type(max_snapshots) is not int or not 1 <= max_snapshots <= 64:
            raise ValueError('bounded Qwen4 snapshot inventory required')
        self.max_snapshots = max_snapshots
        self._snapshot_owner = uuid4().hex
        self._snapshots = {}
        self._snapshot_bytes = 0
        self.max_bytes = max_bytes
        self._lock = RLock()
        h, g, r = c.hidden_size, c.hc_count*c.hidden_size, c.hc_lowrank
        selected_ple = [layer for layer in ple if self.layer_begin<=layer-1<self.layer_end]
        if (ple and not _edge_only and not _inventory_only and not _omit_ple and
                len(selected_ple)*c.ngram_vocab_size_base*c.ple_embed_dim*4 > max_bytes):
            raise MemoryError('PLE table lower bound exceeds resident budget')
        width = 2*c.linear_num_key_heads*c.linear_key_head_dim + c.linear_num_value_heads*c.linear_value_head_dim
        shapes = {'model.embed_tokens.weight': (c.vocab_size, h), 'lm_head.weight': (c.vocab_size, h)}
        integer_metadata = {}
        def hyper(prefix, inject):
            shapes.update({prefix+'hc_norm.weight': (g,),
                prefix+'input_mix_weight_down.weight': (r, g),
                prefix+'input_mix_weight_up.weight': (g, r)})
            if inject:
                shapes[prefix+'block_inject_weight.weight'] = (c.hc_count, g)
        hyper('model.hyper_connection_mixer.', False)
        for i in range(c.num_hidden_layers):
            p = f'model.layers.{i}.'
            hyper(p+'attn_hyper_connection.', True)
            hyper(p+'mlp_hyper_connection.', True)
            if i+1 in ple:
                from .qwen4_ple import metadata
                multipliers, sizes, offsets = metadata(c, ple.index(i+1))
                lane = p+'ple.ple_embedding.'
                integer_metadata.update({lane+'layer_multipliers': multipliers,
                    lane+'ngram_heads_vocab_sizes': sizes, lane+'ngram_heads_offsets': offsets})
                count = int(sizes.sum())
                divisor = c.make_ngram_vocab_size_divisible_by
                count = (count+divisor-1)//divisor*divisor
                shapes[lane+'ngram_embedding.weight'] = (count, c.ple_embed_dim//sizes.size)
                shapes.update({p+'ple.key_proj.weight': (g, c.ple_embed_dim),
                    p+'ple.value_proj.weight': (h, c.ple_embed_dim),
                    p+'ple.conv1d.weight': (g, 1, c.ple_conv_kernel_size)})
                for name in ('norm_key', 'norm_query', 'norm_conv'):
                    shapes[p+'ple.'+name+'.weight'] = (g,)
            a = p+'linear_attn.'
            shapes.update({a+'dt_bias': (c.linear_num_value_heads,), a+'A_log': (c.linear_num_value_heads,),
                a+'conv1d.weight': (width, 1, c.linear_conv_kernel_dim),
                a+'norm.weight': (c.linear_value_head_dim,), a+'in_proj_qkv.weight': (width, h),
                a+'in_proj_z.weight': (c.linear_num_value_heads*c.linear_value_head_dim, h),
                a+'in_proj_a.weight': (c.linear_num_value_heads, h),
                a+'in_proj_b.weight': (c.linear_num_value_heads, h),
                a+'out_proj.weight': (h, c.linear_num_value_heads*c.linear_value_head_dim),
                p+'mlp.gate.weight': (c.num_experts, h),
                p+'mlp.experts.gate_up_proj': (c.num_experts, 2*c.moe_intermediate_size, h),
                p+'mlp.experts.down_proj': (c.num_experts, h, c.moe_intermediate_size),
                p+'mlp.shared_expert_gate.weight': (1, h)})
            if schedule[i] == 'qwen_sparse_attention':
                for name in list(shapes):
                    if name.startswith(a):
                        del shapes[name]
                attention_shapes = {'q_proj.weight': (2*c.num_attention_heads*c.head_dim, h),
                    'k_proj.weight': (c.num_key_value_heads*c.head_dim, h),
                    'v_proj.weight': (c.num_key_value_heads*c.head_dim, h),
                    'o_proj.weight': (h, c.num_attention_heads*c.head_dim),
                    'q_norm.weight': (c.head_dim,), 'k_norm.weight': (c.head_dim,),
                    'indexer.index_qk_proj.weight': ((c.indexer_n_heads+1)*c.indexer_head_dim, h),
                    'indexer.q_layernorm.weight': (c.indexer_head_dim,),
                    'indexer.k_layernorm.weight': (c.indexer_head_dim,)}
                shapes.update({p+'self_attn.'+name: shape for name, shape in attention_shapes.items()})
            for name, shape in {'gate_proj': (c.shared_expert_intermediate_size, h),
                    'up_proj': (c.shared_expert_intermediate_size, h),
                    'down_proj': (h, c.shared_expert_intermediate_size)}.items():
                shapes[p+'mlp.shared_expert.'+name+'.weight'] = shape
        if _inventory_only:
            self.tensor_shapes = dict(shapes)
            self.integer_metadata = dict(integer_metadata)
            return
        if self.range_mode:
            prefixes = tuple(f'model.layers.{index}.' for index in range(self.layer_begin,self.layer_end))
            shapes = {name:shape for name,shape in shapes.items() if name.startswith(prefixes)}
            integer_metadata = {name:value for name,value in integer_metadata.items() if name.startswith(prefixes)}
        elif _edge_only:
            shapes = {name:shape for name,shape in shapes.items() if not name.startswith('model.layers.')}
            integer_metadata = {}
        if _omit_experts:
            shapes = {name:shape for name,shape in shapes.items() if not name.endswith(
                ('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
        if _omit_ple:
            shapes={name:shape for name,shape in shapes.items() if not name.endswith('ple.ple_embedding.ngram_embedding.weight')}
        tensors = dict(tensors)
        if raw.get('tie_word_embeddings', False) and not self.range_mode:
            embed = tensors.get('model.embed_tokens.weight')
            if 'lm_head.weight' in tensors and not np.array_equal(embed, tensors['lm_head.weight']):
                raise ValueError('inconsistent tied head')
            tensors['lm_head.weight'] = embed
        if set(tensors) != set(shapes) | set(integer_metadata):
            raise ValueError('exact Qwen4 tensor names mismatch')
        weight_bytes = sum(int(np.prod(shape))*4 for shape in shapes.values())
        if weight_bytes > max_bytes:
            raise MemoryError('resident weights exceed budget')
        self.weights = {}
        for name, expected in integer_metadata.items():
            source = np.asarray(tensors[name])
            if source.dtype.kind not in 'iu' or not np.array_equal(source, expected):
                raise ValueError(f'invalid PLE hash metadata {name}')
            array = expected.copy()
            array.flags.writeable = False
            self.weights[name] = array
            weight_bytes += array.nbytes
        if weight_bytes > max_bytes:
            raise MemoryError('resident PLE metadata exceeds budget')
        for name, shape in shapes.items():
            source = np.asarray(tensors[name])
            if source.shape != shape or source.dtype.kind != 'f' or not np.isfinite(source).all():
                raise ValueError(f'invalid tensor {name}')
            array = np.array(source, dtype=np.float32, order='C', copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f'FP32 overflow {name}')
            array.flags.writeable = False
            self.weights[name] = array
        self.weight_bytes = weight_bytes
        self.hyper_kernel, self.delta_kernel = hyper_kernel, delta_kernel
        if hyper_kernel is not None:
            from .native_qwen4_hyper import CompiledQwen4HyperKernel
            if not isinstance(hyper_kernel, CompiledQwen4HyperKernel):
                raise ValueError('expected compiled Qwen4 hyper kernel')
            self.implementation = 'numpy+c-qwen4-hyperconnections'
        if delta_kernel is not None:
            from .native_mixtral_delta import CompiledGatedDeltaKernel
            if not isinstance(delta_kernel, CompiledGatedDeltaKernel):
                raise ValueError('expected compiled gated delta kernel')
            if hyper_kernel is not None:
                self.implementation = 'numpy+c-qwen4-hyperconnections+c-gated-delta'
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'),
                                           allow_nan=False).encode('utf-8'))
        digest.update(self.implementation.encode('ascii'))
        if self.range_mode:
            digest.update(repr(('hidden-range',self.layer_begin,self.layer_end)).encode('ascii'))
        for name, array in sorted(self.weights.items()):
            digest.update(name.encode('utf-8'))
            digest.update(memoryview(array).cast('B'))
        self.identity = 'openagent-qwen4-v1:'+digest.hexdigest()
        self.reset()

    def _state_bytes(self):
        return (sum(a.nbytes for entries in (self.cache, self.ple_cache)
                    for entry in entries if entry for a in entry) +
                sum(a.nbytes for a in (self.mask, self.positions) if a is not None))

    def _check_snapshot_access(self):
        if getattr(self, '_transport', False):
            raise RuntimeError('Qwen4 cache transport is active')
        check = getattr(self, '_check_open', None)
        if callable(check):
            check()

    def snapshot(self):
        with self._lock:
            self._check_snapshot_access()
            if len(self._snapshots) >= self.max_snapshots:
                raise MemoryError('Qwen4 snapshot inventory exceeds admission')
            size = self._state_bytes()
            if self.weight_bytes + self._snapshot_bytes + 2*size + getattr(self,'_expert_bytes',0)+getattr(self,'_ple_bytes',0) > self.max_bytes:
                raise MemoryError('Qwen4 snapshot exceeds memory admission')
            def freeze(array):
                result = array.copy()
                result.flags.writeable = False
                return result
            cache = tuple(None if entry is None else tuple(freeze(a) for a in entry) for entry in self.cache)
            ple = tuple(None if entry is None else tuple(freeze(a) for a in entry) for entry in self.ple_cache)
            mask = None if self.mask is None else freeze(self.mask)
            positions = None if self.positions is None else freeze(self.positions)
            handle = Qwen4Snapshot(self.identity, self._snapshot_owner, uuid4().hex, self.position)
            self._snapshots[handle.handle] = (handle, cache, ple, mask, positions, size)
            self._snapshot_bytes += size
            return handle

    def _owned_snapshot(self, handle):
        if (type(handle) is not Qwen4Snapshot or handle.owner != self._snapshot_owner or
                handle.identity != self.identity):
            raise ValueError('foreign Qwen4 snapshot')
        owned = self._snapshots.get(handle.handle)
        if owned is None or owned[0] is not handle:
            raise ValueError('unknown or released Qwen4 snapshot')
        return owned

    def restore(self, handle):
        with self._lock:
            self._check_snapshot_access()
            _, cache, ple, mask, positions, size = self._owned_snapshot(handle)
            if self.weight_bytes + self._snapshot_bytes + self._state_bytes() + size + getattr(self,'_expert_bytes',0)+getattr(self,'_ple_bytes',0) > self.max_bytes:
                raise MemoryError('Qwen4 restore exceeds memory admission')
            cache = [None if entry is None else tuple(a.copy() for a in entry) for entry in cache]
            ple = [None if entry is None else tuple(a.copy() for a in entry) for entry in ple]
            mask = None if mask is None else mask.copy()
            positions = None if positions is None else positions.copy()
            self.cache, self.ple_cache, self.mask, self.positions, self.position = (
                cache, ple, mask, positions, handle.position)

    def release_snapshot(self, handle):
        with self._lock:
            self._check_snapshot_access()
            owned = self._owned_snapshot(handle)
            self._snapshot_bytes -= owned[-1]
            del self._snapshots[handle.handle]

    def reset(self):
        with self._lock:
            self._check_snapshot_access()
            self.position = 0
            self.cache = [None]*self.config.num_hidden_layers
            self.mask = None
            self.positions = None
            self.ple_cache = [None]*self.config.num_hidden_layers

    def status(self):
        return dict(model_type='qwen4_exp_text', implementation=self.implementation,
                    supported_contract='linear-QSA-PLE-default-or-linear-text-RoPE-FP32-resident',
                    native_range_execution=False, position=self.position)

    def _mix(self, x, prefix, inject):
        c, w = self.config, self.weights
        if self.hyper_kernel is not None:
            return self.hyper_kernel.mix(x, w[prefix+'hc_norm.weight'],
                w[prefix+'input_mix_weight_down.weight'], w[prefix+'input_mix_weight_up.weight'],
                w[prefix+'block_inject_weight.weight'] if inject else None,
                heads=c.hc_count, epsilon=c.rms_norm_eps)
        z = rmsnorm(x.reshape(*x.shape[:-1], c.hc_count, c.hidden_size),
                    1+w[prefix+'hc_norm.weight'].reshape(c.hc_count, c.hidden_size), c.rms_norm_eps)
        flat = z.reshape(x.shape)
        low = silu((flat @ w[prefix+'input_mix_weight_down.weight'].T)/c.hc_count)
        gate = sigmoid(low @ w[prefix+'input_mix_weight_up.weight'].T).reshape(z.shape)
        mixed = (gate*z).mean(axis=-2)
        injection = 2*sigmoid((flat @ w[prefix+'block_inject_weight.weight'].T)/c.hc_count) if inject else None
        return mixed, injection

    routing_observation_supported = False  # Forward returns its routing trace directly.

    def forward(self, token_ids, *, attention_mask=None, position_ids=None, inputs_embeds=None):
        with self._lock:
            self._check_snapshot_access()
            return self._forward(token_ids, attention_mask=attention_mask, position_ids=position_ids,
                                 inputs_embeds=inputs_embeds)

    def write_cache(self, output, *, max_bytes):
        from .qwen4_cache_recovery import write_qwen4_cache
        return write_qwen4_cache(self, output, max_bytes=max_bytes)

    def restore_cache(self, source, *, max_bytes):
        from .qwen4_cache_recovery import restore_qwen4_cache
        return restore_qwen4_cache(self, source, max_bytes=max_bytes)

    def _forward(self, token_ids, *, attention_mask, position_ids, incoming_hidden=None, inputs_embeds=None):
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None]
        c = self.config
        if ids.ndim != 2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= c.vocab_size):
            raise ValueError('invalid token_ids')
        b, t = ids.shape
        end = self.position+t
        if end > c.max_position_embeddings:
            raise ValueError('context capacity exceeded')
        # Admission precedes allocation of defaults and pending recurrent state.
        width = 2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
        state = b*c.num_hidden_layers*(width*c.linear_conv_kernel_dim+
            c.linear_num_value_heads*c.linear_key_head_dim*c.linear_value_head_dim)*4
        old = sum(a.nbytes for entry in self.cache if entry for a in entry)
        old += sum(a.nbytes for entry in self.ple_cache if entry for a in entry)
        old += sum(a.nbytes for a in (self.mask, self.positions) if a is not None)
        rows = b*t
        scratch = rows*(12*c.hc_count*c.hidden_size+4*c.hc_lowrank+8*width+
            8*c.moe_intermediate_size+8*c.shared_expert_intermediate_size+
            4*c.num_experts+2*c.vocab_size)*8
        scratch += rows*c.num_hidden_layers*(c.hc_count*c.hidden_size*4+
            c.num_experts*4+c.num_experts_per_tok*12)
        if self.rotary_dim:
            sparse = c.layer_types.count('qwen_sparse_attention')
            state += b*sparse*end*(2*c.num_key_value_heads*c.head_dim+c.indexer_head_dim)*4
            scratch += rows*end*(c.num_attention_heads+4)*8
        if c.ple_layer_ids:
            state += len(c.ple_layer_ids)*b*(8*(c.ngram_size-1)+4*c.hc_count*c.hidden_size*(c.ple_conv_kernel_size-1)*c.ngram_size)
            scratch += rows*(8*c.ple_embed_dim+16*c.hc_count*c.hidden_size)
        scratch += (24 if self.mrope_section is not None else 8)*b*end
        if inputs_embeds is not None:
            scratch += 4*rows*c.hidden_size
        if self.weight_bytes+self._snapshot_bytes+state+old+scratch+4*b*end+getattr(self,'_expert_bytes',0)+getattr(self,'_ple_bytes',0) > self.max_bytes:
            raise MemoryError('Qwen4 request exceeds memory budget')
        mask = np.ones((b, end), bool) if attention_mask is None else np.asarray(attention_mask)
        if mask.shape != (b, end) or not np.isin(mask, [0, 1]).all():
            raise ValueError('invalid attention_mask')
        mask = mask.astype(bool)
        if self.mask is not None and not np.array_equal(self.mask, mask[:, :self.position]):
            raise ValueError('cached batch or mask changed')
        positions = np.broadcast_to(np.arange(self.position, end), ids.shape) if position_ids is None else np.asarray(position_ids)
        if self.mrope_section is not None and positions.shape == ids.shape:
            positions = np.broadcast_to(positions, (3, *ids.shape))
        expected_positions = (3, *ids.shape) if self.mrope_section is not None else ids.shape
        if positions.shape != expected_positions or positions.dtype.kind not in 'iu' or np.any(positions < 0) or np.any(positions >= c.max_position_embeddings):
            raise ValueError('bounded positions must match the text or three-axis MRoPE contract')
        full_positions = positions.copy() if self.positions is None else np.concatenate((self.positions, positions), axis=-1)
        if self.range_mode:
            if inputs_embeds is not None:
                raise ValueError('multimodal embeddings belong to the complete decoder input edge')
            value = np.asarray(incoming_hidden)
            if value.dtype != np.float32 or value.shape != (b,t,c.hc_count*c.hidden_size) or not np.isfinite(value).all():
                raise ValueError('finite FP32 Qwen4 hyper-channel range input required')
            x = np.array(value,dtype=np.float32,order='C',copy=True)
        else:
            if incoming_hidden is not None:
                raise ValueError('hidden range input requires an explicit selected layer owner')
            if inputs_embeds is None:
                x = self._embed(ids)
            else:
                embedding = np.asarray(inputs_embeds)
                if (embedding.dtype != np.float32 or embedding.shape != (b,t,c.hidden_size)
                        or not np.isfinite(embedding).all()):
                    raise ValueError('finite FP32 embeddings with original PLE token ids required')
                x = np.tile(embedding, (1,1,c.hc_count))
        hidden, routers, routes, pending, attentions, selections = [], [], [], [], [], []
        pending = [None]*c.num_hidden_layers
        pending_ple = [None]*c.num_hidden_layers
        for i in range(self.layer_begin,self.layer_end):
            boundary = getattr(self, '_cancellation_boundary', None)
            if boundary is not None:
                boundary()
            hidden.append(x.copy())
            p = f'model.layers.{i}.'
            if i+1 in c.ple_layer_ids:
                injection, pending_ple[i] = self._ple(x, ids, p, self.ple_cache[i], mask[:, self.position:end])
                x = self._add(x, injection)
            mixed, injection = self._mix(x, p+'attn_hyper_connection.', True)
            if c.layer_types[i] == 'linear_attention':
                value, new_state = self._delta(mixed, p, self.cache[i], mask[:, self.position:end])
            else:
                value, new_state, probability, selected_mask = self._qsa(mixed, p, self.cache[i], positions, full_positions, mask)
                attentions.append(probability)
                selections.append(selected_mask)
            pending[i] = new_state
            x = self._inject(x, value, injection)
            mixed, injection = self._mix(x, p+'mlp_hyper_connection.', True)
            out, router, selected, scores = self._moe(mixed, p)
            x = self._inject(x, out, injection)
            routers.append(router)
            routes.append((selected, scores))
        if self.range_mode:
            if not np.isfinite(x).all() or any(not np.isfinite(a).all() for state in pending+pending_ple if state for a in state):
                raise ValueError('nonfinite Qwen4 range execution')
            boundary = getattr(self, '_cancellation_boundary', None)
            if boundary is not None:
                boundary()
            self.cache,self.mask,self.position,self.positions = pending,mask.copy(),end,full_positions
            self.ple_cache = pending_ple
            return SimpleNamespace(hidden=x,hidden_states=tuple(hidden),router_logits=tuple(routers),
                                   routes=tuple(routes),attentions=tuple(attentions),selections=tuple(selections))
        x, _ = self._mix(x, 'model.hyper_connection_mixer.', False)
        hidden.append(x.copy())
        logits = self._head(x)
        if not np.isfinite(logits).all() or any(not np.isfinite(a).all() for state in pending+pending_ple if state for a in state):
            raise ValueError('nonfinite Qwen4 execution')
        boundary = getattr(self, '_cancellation_boundary', None)
        if boundary is not None:
            boundary()
        self.cache, self.mask, self.position, self.positions = pending, mask.copy(), end, full_positions
        self.ple_cache = pending_ple
        return SimpleNamespace(logits=logits, hidden_states=tuple(hidden), router_logits=tuple(routers), routes=tuple(routes),
                               attentions=tuple(attentions), selections=tuple(selections))

    def _embed(self, ids):
        return np.tile(self.weights['model.embed_tokens.weight'][ids], (1, 1, self.config.hc_count))

    def _ple(self, x, ids, prefix, old, mask):
        from .qwen4_ple import execute
        return execute(self, x, ids, prefix, old, mask)

    def _add(self, x, value):
        return x+value

    def _inject(self, x, value, gates):
        return x+(value[..., None, :]*gates[..., :, None]).reshape(x.shape)

    def _head(self, x):
        return x @ self.weights['lm_head.weight'].T

    def _delta(self, x, prefix, old, mask):
        if getattr(self.config, 'output_gate_type', None) != 'sigmoid':
            return IndependentQwen35TextDecoder._delta(self, x, prefix, old, mask)
        # Same delta transition, but Qwen4 may select a sigmoid output gate.
        c, w = self.config, self.weights
        b, t, _ = x.shape
        a = prefix+'linear_attn.'
        x = x*mask[..., None]
        projected = x @ w[a+'in_proj_qkv.weight'].T
        width, kernel = projected.shape[-1], c.linear_conv_kernel_dim
        history = np.zeros((b, width, kernel), np.float32) if old is None else old[0].copy()
        state = np.zeros((b, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim), np.float32) if old is None else old[1].copy()
        conv = np.empty_like(projected)
        for step in range(t):
            history = np.concatenate((history[..., 1:], projected[:, step, :, None]), axis=-1)
            conv[:, step] = silu((history*w[a+'conv1d.weight'][:, 0]).sum(axis=-1))
        kw = c.linear_num_key_heads*c.linear_key_head_dim
        q, k, v = np.split(conv, [kw, 2*kw], axis=-1)
        q, k = (z.reshape(b, t, c.linear_num_key_heads, c.linear_key_head_dim) for z in (q, k))
        q, k = (z/np.sqrt((z*z).sum(axis=-1, keepdims=True)+np.float32(1e-6)) for z in (q, k))
        repeat = c.linear_num_value_heads//c.linear_num_key_heads
        q = np.repeat(q, repeat, axis=2)/np.float32(np.sqrt(c.linear_key_head_dim))
        k = np.repeat(k, repeat, axis=2)
        v = v.reshape(b, t, c.linear_num_value_heads, c.linear_value_head_dim)
        beta = sigmoid(x @ w[a+'in_proj_b.weight'].T)
        decay = np.exp(-np.exp(w[a+'A_log'])*np.logaddexp(np.float32(0), x @ w[a+'in_proj_a.weight'].T+w[a+'dt_bias']))
        output = np.empty_like(v)
        for step in range(t):
            if self.delta_kernel is not None:
                output[:, step] = self.delta_kernel.step(*(np.ascontiguousarray(z) for z in
                    (q[:, step], k[:, step], v[:, step], beta[:, step], decay[:, step])), state)
            else:
                state *= decay[:, step, :, None, None]
                error = v[:, step]-np.einsum('bhk,bhkv->bhv', k[:, step], state)
                state += k[:, step, :, :, None]*(error*beta[:, step, :, None])[:, :, None, :]
                output[:, step] = np.einsum('bhk,bhkv->bhv', q[:, step], state)
        gate = (x @ w[a+'in_proj_z.weight'].T).reshape(output.shape)
        output = rmsnorm(output, w[a+'norm.weight'], c.rms_norm_eps)*sigmoid(gate)
        return output.reshape(b, t, -1) @ w[a+'out_proj.weight'].T, (history, state)

    def _moe(self, mixed, p):
        c, w = self.config, self.weights
        z = mixed.reshape(-1, c.hidden_size)
        router = z @ w[p+'mlp.gate.weight'].T
        prob = softmax(router)
        selected = np.argsort(-prob, axis=-1, kind='stable')[:, :c.num_experts_per_tok]
        scores = np.take_along_axis(prob, selected, axis=-1)
        if getattr(c, 'norm_topk_prob', True):
            scores = scores/scores.sum(axis=-1, keepdims=True)
        out = np.zeros_like(z)
        for expert in range(c.num_experts):
            rows, ranks = np.nonzero(selected == expert)
            if rows.size:
                gate, up = np.split(z[rows] @ w[p+'mlp.experts.gate_up_proj'][expert].T, 2, axis=-1)
                out[rows] += ((silu(gate)*up) @ w[p+'mlp.experts.down_proj'][expert].T)*scores[rows, ranks, None]
        shared = silu(z @ w[p+'mlp.shared_expert.gate_proj.weight'].T)*(z @ w[p+'mlp.shared_expert.up_proj.weight'].T)
        out += (shared @ w[p+'mlp.shared_expert.down_proj.weight'].T)*sigmoid(z @ w[p+'mlp.shared_expert_gate.weight'].T)
        return out.reshape(mixed.shape), router, selected, scores

    def _rotate(self, value, positions):
        if self.mrope_section is not None:
            if positions.ndim != 3 or positions.shape[0] != 3:
                raise ValueError('three-axis MRoPE positions required')
            dimension = self.rotary_dim
            frequency = self.theta ** (-np.arange(0,dimension,2,dtype=np.float32)/dimension)
            angles = positions.astype(np.float32)[...,None]/np.float32(self.rope_scale)*frequency
            mixed = angles[0].copy()
            for axis in (1,2):
                lanes = slice(axis, self.mrope_section[axis]*3, 3)
                mixed[...,lanes] = angles[axis,...,lanes]
            cosine,sine = np.cos(mixed)[...,None,:],np.sin(mixed)[...,None,:]
            first,second = value[...,:dimension//2],value[...,dimension//2:dimension]
            return np.concatenate((first*cosine-second*sine,second*cosine+first*sine,
                                   value[...,dimension:]),axis=-1).astype(np.float32)
        if self.rope_scale != 1:
            positions = positions.astype(np.float64) / self.rope_scale
        dimension = self.rotary_dim
        return np.concatenate((rotary(value[..., :dimension], positions, self.theta), value[..., dimension:]), axis=-1)

    def _qsa(self, x, prefix, old, positions, full_positions, mask):
        c, w = self.config, self.weights
        b, t, _ = x.shape
        a = prefix+'self_attn.'
        projected = x @ w[a+'indexer.index_qk_proj.weight'].T
        query_width = c.indexer_n_heads*c.indexer_head_dim
        query = projected[..., :query_width].reshape(b, t, c.indexer_n_heads, c.indexer_head_dim)
        query = self._rotate(rmsnorm(query, 1+w[a+'indexer.q_layernorm.weight'], c.rms_norm_eps), positions)
        raw = projected[..., query_width:]
        if old is not None:
            raw = np.concatenate((old[2], raw), axis=1)
        chosen = np.zeros((b, 1, t, raw.shape[1]), bool)
        ratio, budget = c.indexer_compress_ratio, c.indexer_budget//c.indexer_compress_ratio
        for row in range(b):
            for step in range(t):
                visible = np.flatnonzero(mask[row, :self.position+step+1])
                if not visible.size:
                    raise ValueError('QSA query has no visible keys')
                blocks = visible.size//ratio
                if blocks:
                    groups = visible[:blocks*ratio].reshape(blocks, ratio)
                    pooled = raw[row, groups].mean(axis=1)
                    normalized = rmsnorm(pooled, 1+w[a+'indexer.k_layernorm.weight'], c.rms_norm_eps)
                    selected_positions = (full_positions[:,row:row+1,groups[:,0]] if self.mrope_section is not None
                                          else full_positions[row,groups[:,0]][None])
                    keys = self._rotate(normalized[None, :, None, :], selected_positions)[0, :, 0]
                    score = np.maximum(query[row, step] @ keys.T, 0).sum(axis=0)/np.float32(np.sqrt(c.indexer_head_dim))
                    selected = np.argsort(-score, kind='stable')[:min(budget, blocks)]
                    chosen[row, 0, step, groups[selected].ravel()] = True
                chosen[row, 0, step, visible[blocks*ratio:]] = True
        qg = (x @ w[a+'q_proj.weight'].T).reshape(b, t, c.num_attention_heads, 2*c.head_dim)
        q, gate = np.split(qg, 2, axis=-1)
        q = self._rotate(rmsnorm(q, 1+w[a+'q_norm.weight'], c.rms_norm_eps), positions).transpose(0, 2, 1, 3)
        k = (x @ w[a+'k_proj.weight'].T).reshape(b, t, c.num_key_value_heads, c.head_dim)
        k = self._rotate(rmsnorm(k, 1+w[a+'k_norm.weight'], c.rms_norm_eps), positions).transpose(0, 2, 1, 3)
        v = (x @ w[a+'v_proj.weight'].T).reshape(b, t, c.num_key_value_heads, c.head_dim).transpose(0, 2, 1, 3)
        if old is not None:
            k, v = np.concatenate((old[0], k), axis=2), np.concatenate((old[1], v), axis=2)
        repeats = c.num_attention_heads//c.num_key_value_heads
        score = (q @ np.repeat(k, repeats, axis=1).swapaxes(-1, -2))/np.float32(np.sqrt(c.head_dim))
        probability = softmax(np.where(chosen, score, np.finfo(np.float32).min))
        value = (probability @ np.repeat(v, repeats, axis=1)).transpose(0, 2, 1, 3)*sigmoid(gate)
        output = value.reshape(b, t, -1) @ w[a+'o_proj.weight'].T
        return output, (k, v, raw), probability, chosen

    def generate_ids(self, prompt_ids, *, max_tokens):
        with self._lock:
            return IndependentQwen35TextDecoder.generate_ids(self, prompt_ids, max_tokens=max_tokens)
