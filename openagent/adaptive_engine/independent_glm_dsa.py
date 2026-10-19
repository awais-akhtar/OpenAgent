"""Independent exact GLM-MoE-DSA equations and compressed latent state.

NumPy numerical baseline, not compiled execution or an alias to older GLM.
Default/YaRN RoPE, low-rank Q, resident unquantized expert banks; inference only.
"""
from types import SimpleNamespace
from copy import deepcopy
import math

import numpy as np

from .native_mixtral import rmsnorm, softmax
from .independent_qwen35 import sigmoid, silu


def rope_contract(rope):
    """Validate the exact static rotary subset; return owned, explicit parameters."""
    if not isinstance(rope, dict) or rope.get('rope_type') not in ('default', 'yarn'):
        raise ValueError('only explicit default or yarn RoPE supported')
    rope = dict(rope)
    allowed = {'rope_type', 'rope_theta'}
    if rope['rope_type'] == 'yarn':
        allowed |= {'factor', 'original_max_position_embeddings', 'attention_factor',
                    'beta_fast', 'beta_slow', 'mscale', 'mscale_all_dim', 'truncate',
                    'partial_rotary_factor'}
    if set(rope)-allowed:
        raise ValueError('unsupported RoPE extension')
    def positive(name, minimum=0):
        value = rope.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= minimum or not 1e-12 <= value <= 1e12):
            raise ValueError(f'invalid RoPE {name}')
    positive('rope_theta')
    if rope['rope_type'] == 'yarn':
        positive('rope_theta', 1)
        positive('factor')
        if rope['factor'] < 1:
            raise ValueError('YaRN factor must be at least one')
        original = rope.get('original_max_position_embeddings')
        if type(original) is not int or not 1 <= original <= 1048576:
            raise ValueError('invalid YaRN original context')
        for name, default in (('beta_fast',32.), ('beta_slow',1.)):
            rope.setdefault(name, default)
            positive(name)
        if rope['beta_fast'] < rope['beta_slow']:
            raise ValueError('YaRN beta_fast must be at least beta_slow')
        for name in ('mscale', 'mscale_all_dim'):
            rope.setdefault(name, 0.)
            value = rope[name]
            if (isinstance(value, bool) or not isinstance(value, (int,float)) or
                    not math.isfinite(value) or not 0 <= value <= 1e12):
                raise ValueError(f'invalid YaRN {name}')
        rope.setdefault('truncate', True)
        if type(rope['truncate']) is not bool:
            raise ValueError('YaRN truncate must be boolean')
        if type(rope.get('partial_rotary_factor',1.)) not in (float,int) or rope.get('partial_rotary_factor',1.) != 1:
            raise ValueError('partial YaRN rotary dimensions unsupported')
        if rope.get('attention_factor') is not None:
            positive('attention_factor')
    return rope


def yarn_magnitude(factor, multiplier=1.):
    return 1. if factor <= 1 else 1.+0.1*multiplier*math.log(factor)


def rotary_parameters(n, rope):
    """Independent frequency blend and amplitude for the NumPy baseline only."""
    base = np.float32(rope['rope_theta'])**(np.arange(0,n,2,dtype=np.float32)/np.float32(n))
    frequency = np.float32(1)/base
    amplitude = 1.
    if rope['rope_type'] == 'yarn':
        factor = rope['factor']
        original = rope['original_max_position_embeddings']
        bounds = [n*math.log(original/(2*math.pi*rope[name]))/(2*math.log(rope['rope_theta']))
                  for name in ('beta_fast','beta_slow')]
        if rope['truncate']:
            bounds = [math.floor(bounds[0]),math.ceil(bounds[1])]
        low, high = max(bounds[0],0),min(bounds[1],n-1)
        if low == high:
            high += .001
        ramp = np.clip((np.arange(n//2,dtype=np.float32)-np.float32(low))/np.float32(high-low),0,1)
        extrapolation = np.float32(1)-ramp
        frequency = (np.float32(1)/(np.float32(factor)*base))*(np.float32(1)-extrapolation)+frequency*extrapolation
        amplitude = rope.get('attention_factor')
        if amplitude is None:
            amplitude = (yarn_magnitude(factor,rope['mscale'])/yarn_magnitude(factor,rope['mscale_all_dim'])
                         if rope['mscale'] and rope['mscale_all_dim'] else yarn_magnitude(factor))
    return frequency, amplitude


def interleaved_rotary(x, positions, theta, rope=None):
    n = x.shape[-1]
    frequency, amplitude = rotary_parameters(n, rope or {'rope_type':'default','rope_theta':theta})
    angle = positions.astype(np.float32)[..., None, None]*frequency
    pairs = x.reshape(*x.shape[:-1], n//2, 2)
    out = np.empty_like(pairs)
    cosine, sine = np.cos(angle)*np.float32(amplitude),np.sin(angle)*np.float32(amplitude)
    out[..., 0] = pairs[..., 0]*cosine-pairs[..., 1]*sine
    out[..., 1] = pairs[..., 1]*cosine+pairs[..., 0]*sine
    # Adjacent input pairs rotate into the oracle/checkpoint's real-half/imag-half layout.
    return np.concatenate((out[..., 0], out[..., 1]), axis=-1)


class IndependentGlmDsaDecoder:
    implementation = 'numpy-glm-dsa-compressed-latent'
    native_range_execution = False

    def __init__(self, config, tensors, *, max_weight_bytes=None, _inventory_only=False):
        raw = deepcopy(dict(config))
        if raw.get('model_type') != 'glm_moe_dsa':
            raise ValueError('expected exact glm_moe_dsa identity')
        # This implementation admits the GLM-DSA tensor contract by model_type,
        # but has no pinned architecture-class identity for this route.  Do not
        # silently select it when a checkpoint explicitly names another (or an
        # as-yet-unreviewed) architecture.  Architecture-less configs remain
        # admitted only under the exact model_type and tensor/schedule checks
        # below; a future reviewed architecture must be added deliberately.
        if raw.get('architectures') is not None:
            raise ValueError('GLM DSA architecture declarations are not admitted by this implementation')
        for key in ('hidden_size', 'vocab_size', 'intermediate_size', 'moe_intermediate_size',
                    'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
                    'n_shared_experts', 'n_routed_experts', 'num_experts_per_tok', 'q_lora_rank',
                    'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim',
                    'index_head_dim', 'index_n_heads', 'index_topk', 'n_group', 'topk_group',
                    'max_position_embeddings'):
            if type(raw.get(key)) is not int or not 0 < raw[key] <= 2**31-1:
                raise ValueError(f'invalid {key}')
        if raw['num_hidden_layers'] > 4096 or raw['max_position_embeddings'] > 1048576:
            raise ValueError('GLM DSA layer or context inventory exceeds admission')
        if raw['num_attention_heads'] != raw['num_key_value_heads']:
            raise ValueError('MLA requires equal attention/key-value head counts')
        if (raw['qk_rope_head_dim'] % 2 or raw['qk_rope_head_dim'] > raw['index_head_dim'] or
                raw['n_routed_experts'] % raw['n_group'] or raw['n_routed_experts']//raw['n_group'] < 2 or
                raw['topk_group'] > raw['n_group'] or
                raw['num_experts_per_tok'] > raw['topk_group']*(raw['n_routed_experts']//raw['n_group'])):
            raise ValueError('invalid rotary/indexer/group-routing geometry')
        for key, allowed in (('mlp_layer_types', ('dense', 'sparse')), ('indexer_types', ('full', 'shared'))):
            if not isinstance(raw.get(key), list) or len(raw[key]) != raw['num_hidden_layers'] or any(t not in allowed for t in raw[key]):
                raise ValueError(f'explicit exact {key} required')
        if raw['indexer_types'][0] != 'full':
            raise ValueError('shared indexer requires a preceding full layer')
        if raw.get('layer_types') != ['deepseek_sparse_attention']*raw['num_hidden_layers']:
            raise ValueError('only exact DSA layer schedule supported')
        if raw.get('hidden_act', 'silu') != 'silu' or raw.get('attention_bias', False) or raw.get('mlp_bias', False) or raw.get('quantization_config') is not None:
            raise ValueError('unsupported activation, bias or quantization')
        if type(raw.get('norm_topk_prob')) is not bool or type(raw.get('tie_word_embeddings', False)) is not bool:
            raise ValueError('normalization/tied weight switches must be boolean')
        raw.setdefault('tie_word_embeddings', False)
        rope = raw['rope_parameters'] = rope_contract(raw.get('rope_parameters'))
        if raw.get('head_dim', raw['qk_rope_head_dim']) != raw['qk_rope_head_dim']:
            raise ValueError('inconsistent rotary head dimension')
        for value in (raw.get('rms_norm_eps'), raw.get('routed_scaling_factor'), rope.get('rope_theta')):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
                raise ValueError('invalid positive numerical parameter')
        self.config = c = SimpleNamespace(**raw)
        self.theta = rope['rope_theta']
        self.attention_multiplier = (yarn_magnitude(rope['factor'],rope['mscale_all_dim'])**2
                                     if rope['rope_type'] == 'yarn' else 1.)
        h, heads, qr, kr = c.hidden_size, c.num_attention_heads, c.q_lora_rank, c.kv_lora_rank
        nd, rd, vd = c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        shapes = {'model.embed_tokens.weight': (c.vocab_size, h), 'model.norm.weight': (h,), 'lm_head.weight': (c.vocab_size, h)}
        for i in range(c.num_hidden_layers):
            p = f'model.layers.{i}.'
            shapes[p+'input_layernorm.weight'] = (h,)
            shapes[p+'post_attention_layernorm.weight'] = (h,)
            for name, shape in {'q_a_proj.weight': (qr, h), 'q_a_layernorm.weight': (qr,),
                'q_b_proj.weight': (heads*(nd+rd), qr), 'kv_a_proj_with_mqa.weight': (kr+rd, h),
                'kv_a_layernorm.weight': (kr,), 'kv_b_proj.weight': (heads*(nd+vd), kr),
                'o_proj.weight': (h, heads*vd)}.items():
                shapes[p+'self_attn.'+name] = shape
            if c.indexer_types[i] == 'full':
                for name, shape in {'wq_b.weight': (c.index_n_heads*c.index_head_dim, qr),
                    'wk.weight': (c.index_head_dim, h), 'k_norm.weight': (c.index_head_dim,),
                    'k_norm.bias': (c.index_head_dim,), 'weights_proj.weight': (c.index_n_heads, h)}.items():
                    shapes[p+'self_attn.indexer.'+name] = shape
            if c.mlp_layer_types[i] == 'dense':
                for name, shape in {'gate_proj.weight': (c.intermediate_size, h),
                    'up_proj.weight': (c.intermediate_size, h), 'down_proj.weight': (h, c.intermediate_size)}.items():
                    shapes[p+'mlp.'+name] = shape
            else:
                shapes.update({p+'mlp.gate.weight': (c.n_routed_experts, h),
                    p+'mlp.gate.e_score_correction_bias': (c.n_routed_experts,),
                    p+'mlp.experts.gate_up_proj': (c.n_routed_experts, 2*c.moe_intermediate_size, h),
                    p+'mlp.experts.down_proj': (c.n_routed_experts, h, c.moe_intermediate_size)})
                shared = c.n_shared_experts*c.moe_intermediate_size
                for name, shape in {'gate_proj.weight': (shared, h), 'up_proj.weight': (shared, h), 'down_proj.weight': (h, shared)}.items():
                    shapes[p+'mlp.shared_experts.'+name] = shape
        self.tensor_shapes = shapes
        if type(_inventory_only) is not bool:
            raise ValueError('boolean GLM inventory policy required')
        if _inventory_only:
            self.weights = {}
            return
        tensors = dict(tensors)
        if raw.get('tie_word_embeddings', False):
            embedding = tensors.get('model.embed_tokens.weight')
            if 'lm_head.weight' in tensors and not np.array_equal(tensors['lm_head.weight'], embedding):
                raise ValueError('inconsistent tied weights')
            tensors['lm_head.weight'] = embedding
        if set(tensors) != set(shapes):
            raise ValueError(f'tensor names mismatch: missing={set(shapes)-set(tensors)}, extra={set(tensors)-set(shapes)}')
        if max_weight_bytes is not None:
            if type(max_weight_bytes) is not int or max_weight_bytes <= 0:
                raise ValueError('positive GLM weight budget required')
            sizes = [4*math.prod(shape) for shape in shapes.values()]
            if sum(sizes)+max(sizes, default=0) > max_weight_bytes:
                raise MemoryError('GLM weight ownership and conversion exceed admission')
        self.weights = {}
        for name, shape in shapes.items():
            value = np.asarray(tensors[name])
            if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
                raise ValueError(f'invalid tensor {name}')
            value = np.array(value, dtype=np.float32, order='C', copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f'FP32 overflow in {name}')
            value.flags.writeable = False
            self.weights[name] = value
        self.reset()

    def reset(self):
        self.position = 0
        self.cache = [None]*self.config.num_hidden_layers
        self.mask = None

    def status(self):
        return dict(model_type='glm_moe_dsa', implementation=self.implementation,
                    native_range_execution=False, cache_layout='latent+rotary+indexer', position=self.position)

    def _inputs(self, token_ids, attention_mask, position_ids):
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        c = self.config
        if ids.ndim != 2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= c.vocab_size):
            raise ValueError('invalid token_ids')
        end = self.position+ids.shape[1]
        if end > c.max_position_embeddings:
            raise ValueError('context capacity exceeded')
        mask = np.ones((ids.shape[0], end), bool) if attention_mask is None else np.asarray(attention_mask)
        if mask.shape != (ids.shape[0], end) or not np.isin(mask, [0, 1]).all():
            raise ValueError('invalid attention_mask')
        mask = mask.astype(bool)
        if self.mask is not None and not np.array_equal(mask[:, :self.position], self.mask):
            raise ValueError('cached mask or batch changed')
        positions = np.broadcast_to(np.arange(self.position, end), ids.shape) if position_ids is None else np.asarray(position_ids)
        if positions.shape != ids.shape or positions.dtype.kind not in 'iu' or np.any(positions < 0) or np.any(positions >= c.max_position_embeddings):
            raise ValueError('invalid position_ids')
        return ids, mask, positions

    def forward(self, token_ids, *, attention_mask=None, position_ids=None):
        ids, mask, positions = self._inputs(token_ids, attention_mask, position_ids)
        c, w = self.config, self.weights
        x = w['model.embed_tokens.weight'][ids].copy()
        hidden, cache, attentions, indexers, routers, routes = [], [], [], [], [], []
        selected = None
        for i in range(c.num_hidden_layers):
            p = f'model.layers.{i}.'
            hidden.append(x.copy())
            z = rmsnorm(x, w[p+'input_layernorm.weight'], c.rms_norm_eps)
            mixed, state, attention, selected = self._attention(z, p, self.cache[i], mask, positions, selected, c.indexer_types[i])
            x = x+mixed
            cache.append(state)
            attentions.append(attention)
            indexers.append(selected.copy())
            z = rmsnorm(x, w[p+'post_attention_layernorm.weight'], c.rms_norm_eps)
            if c.mlp_layer_types[i] == 'dense':
                output = (silu(z @ w[p+'mlp.gate_proj.weight'].T)*(z @ w[p+'mlp.up_proj.weight'].T)) @ w[p+'mlp.down_proj.weight'].T
                routers.append(None)
                routes.append(None)
            else:
                flat = z.reshape(-1, c.hidden_size)
                logits = flat @ w[p+'mlp.gate.weight'].T
                scores = sigmoid(logits)
                choice = scores+w[p+'mlp.gate.e_score_correction_bias']
                groups = choice.reshape(-1, c.n_group, c.n_routed_experts//c.n_group)
                group_scores = np.sort(groups, axis=-1)[..., -2:].sum(axis=-1)
                group_indices = np.argsort(-group_scores, axis=-1, kind='stable')[:, :c.topk_group]
                allowed = np.zeros_like(group_scores, bool)
                np.put_along_axis(allowed, group_indices, True, axis=-1)
                choice = np.where(np.repeat(allowed, c.n_routed_experts//c.n_group, axis=-1), choice, -np.inf)
                experts = np.argsort(-choice, axis=-1, kind='stable')[:, :c.num_experts_per_tok]
                weights = np.take_along_axis(scores, experts, axis=-1)
                if c.norm_topk_prob:
                    weights /= weights.sum(axis=-1, keepdims=True)+np.float32(1e-20)
                weights *= np.float32(c.routed_scaling_factor)
                output = np.zeros_like(flat)
                for expert in range(c.n_routed_experts):
                    rows, ranks = np.nonzero(experts == expert)
                    gate, up = np.split(flat[rows] @ w[p+'mlp.experts.gate_up_proj'][expert].T, 2, axis=-1)
                    result = (silu(gate)*up) @ w[p+'mlp.experts.down_proj'][expert].T
                    output[rows] += result*weights[rows, ranks, None]
                shared = silu(flat @ w[p+'mlp.shared_experts.gate_proj.weight'].T)*(flat @ w[p+'mlp.shared_experts.up_proj.weight'].T)
                output = (output+shared @ w[p+'mlp.shared_experts.down_proj.weight'].T).reshape(x.shape)
                routers.append(logits)
                routes.append((experts, weights))
            x = x+output
        x = rmsnorm(x, w['model.norm.weight'], c.rms_norm_eps)
        hidden.append(x.copy())
        logits = x @ w['lm_head.weight'].T
        self.cache, self.mask, self.position = cache, mask.copy(), self.position+ids.shape[1]
        return SimpleNamespace(logits=logits, hidden_states=tuple(hidden), attentions=tuple(attentions),
                               indexer_indices=tuple(indexers), router_logits=tuple(routers), routes=tuple(routes))

    def _attention(self, x, p, old, mask, positions, selected, indexer_kind):
        c, w = self.config, self.weights
        b, t, _ = x.shape
        a = p+'self_attn.'
        residual = rmsnorm(x @ w[a+'q_a_proj.weight'].T, w[a+'q_a_layernorm.weight'], 1e-6)
        q = (residual @ w[a+'q_b_proj.weight'].T).reshape(b, t, c.num_attention_heads, -1)
        q_nope, q_rope = np.split(q, [c.qk_nope_head_dim], axis=-1)
        q_rope = interleaved_rotary(q_rope, positions, self.theta, c.rope_parameters)
        projected = x @ w[a+'kv_a_proj_with_mqa.weight'].T
        latent, key_rope = np.split(projected, [c.kv_lora_rank], axis=-1)
        latent = rmsnorm(latent, w[a+'kv_a_layernorm.weight'], 1e-6)
        key_rope = interleaved_rotary(key_rope[:, :, None, :], positions, self.theta, c.rope_parameters)[:, :, 0]
        if old is not None:
            latent, key_rope = np.concatenate((old[0], latent), axis=1), np.concatenate((old[1], key_rope), axis=1)
        allowed = (np.arange(latent.shape[1])[None, :] <= np.arange(self.position, self.position+t)[:, None])[None] & mask[:, None]
        index_keys = None
        if indexer_kind == 'full':
            prefix = a+'indexer.'
            iq = (residual @ w[prefix+'wq_b.weight'].T).reshape(b, t, c.index_n_heads, c.index_head_dim)
            iq = np.concatenate((interleaved_rotary(iq[..., :c.qk_rope_head_dim], positions, self.theta, c.rope_parameters), iq[..., c.qk_rope_head_dim:]), axis=-1)
            ik = x @ w[prefix+'wk.weight'].T
            centered = ik-ik.mean(axis=-1, keepdims=True)
            ik = centered/np.sqrt(np.mean(centered*centered, axis=-1, keepdims=True)+np.float32(1e-6))*w[prefix+'k_norm.weight']+w[prefix+'k_norm.bias']
            ik = np.concatenate((interleaved_rotary(ik[:, :, None, :c.qk_rope_head_dim], positions, self.theta, c.rope_parameters)[:, :, 0], ik[..., c.qk_rope_head_dim:]), axis=-1)
            index_keys = ik if old is None else np.concatenate((old[2], ik), axis=1)
            dot = np.einsum('bthi,bsi->bths', iq, index_keys)/np.float32(np.sqrt(c.index_head_dim))
            head_weights = (x @ w[prefix+'weights_proj.weight'].T)/np.float32(np.sqrt(c.index_n_heads))
            score = np.sum(np.maximum(dot, 0)*head_weights[..., None], axis=2)
            score = np.where(allowed, score, np.finfo(np.float32).min)
            selected = np.argsort(-score, axis=-1, kind='stable')[..., :min(c.index_topk, latent.shape[1])]
        sparse = np.zeros_like(allowed)
        np.put_along_axis(sparse, selected, True, axis=-1)
        projection = w[a+'kv_b_proj.weight'].reshape(c.num_attention_heads, c.qk_nope_head_dim+c.v_head_dim, c.kv_lora_rank)
        latent_query = np.einsum('bthd,hdr->bthr', q_nope, projection[:, :c.qk_nope_head_dim])
        scores = np.einsum('bthr,bsr->bhts', latent_query, latent)+np.einsum('bthd,bsd->bhts', q_rope, key_rope)
        scores /= np.float32(np.sqrt(c.qk_nope_head_dim+c.qk_rope_head_dim))
        scores *= np.float32(self.attention_multiplier)
        prob = softmax(np.where((allowed & sparse)[:, None], scores, np.finfo(np.float32).min))
        attended_latent = np.einsum('bhts,bsr->bhtr', prob, latent)
        output = np.einsum('bhtr,hdr->bthd', attended_latent, projection[:, c.qk_nope_head_dim:])
        return output.reshape(b, t, -1) @ w[a+'o_proj.weight'].T, (latent, key_rope, index_keys), prob, selected

    def expanded_cache(self, layer):
        c = self.config
        latent, rotary, _ = self.cache[layer]
        w = self.weights[f'model.layers.{layer}.self_attn.kv_b_proj.weight']
        kv = (latent @ w.T).reshape(latent.shape[0], latent.shape[1], c.num_attention_heads, -1)
        key = np.concatenate((kv[..., :c.qk_nope_head_dim], np.broadcast_to(rotary[:, :, None], (*kv.shape[:-1], c.qk_rope_head_dim))), axis=-1)
        return key.transpose(0, 2, 1, 3), kv[..., c.qk_nope_head_dim:].transpose(0, 2, 1, 3)

    def generate_ids(self, prompt_ids, *, max_tokens):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('invalid max_tokens')
        ids = np.asarray(prompt_ids)
        if ids.ndim != 1 or not ids.size or ids.size+max_tokens-1 > self.config.max_position_embeddings:
            raise ValueError('invalid generation length')
        self.reset()
        logits = self.forward(ids).logits[0, -1]
        result = []
        eos = getattr(self.config, 'eos_token_id', None)
        eos = eos if isinstance(eos, list) else [eos]
        for step in range(max_tokens):
            token = int(logits.argmax())
            if token in eos:
                break
            result.append(token)
            if step+1 < max_tokens:
                logits = self.forward([token]).logits[0, -1]
        return result
