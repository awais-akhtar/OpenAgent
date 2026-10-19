"""Independent Qwen3.5 MoE text equations, not a Mixtral identity alias.

NumPy FP32 inference; default partial RoPE and packed unquantized expert banks.
Gated delta state uses key-by-value orientation and the published delta equation.
"""
from types import SimpleNamespace

import numpy as np

from .native_mixtral import rmsnorm, rotary, softmax


def sigmoid(x):
    return np.exp(-np.logaddexp(np.float32(0), -x))


def silu(x):
    return x * sigmoid(x)


class IndependentQwen35TextDecoder:
    implementation = 'numpy'
    native_range_execution = False

    def __init__(self, config, tensors, *, delta_kernel=None, layer_range=None,
                 _inventory_only=False, _edge_only=False, _omit_experts=False):
        raw = dict(config)
        if raw.get('model_type') != 'qwen3_5_moe_text':
            raise ValueError('expected exact qwen3_5_moe_text; multimodal wrappers unsupported')
        if raw.get('architectures') not in (None, ['Qwen3_5MoeForCausalLM'],
                                             ('Qwen3_5MoeForCausalLM',)):
            raise ValueError('unsupported Qwen3.5 text architecture declaration')
        # This switch changes the admitted tensor inventory: a tied checkpoint
        # may omit lm_head.weight and the decoder then aliases the embedding.
        # Do not let truthy strings/integers silently select that alternate
        # inventory in either header planning or execution.
        if type(raw.get('tie_word_embeddings', False)) is not bool:
            raise ValueError('Qwen3.5 tie_word_embeddings must be an explicit boolean')
        raw.setdefault('tie_word_embeddings', False)
        for key in ('hidden_size', 'vocab_size', 'num_hidden_layers', 'num_attention_heads',
                    'num_key_value_heads', 'head_dim', 'linear_conv_kernel_dim',
                    'linear_key_head_dim', 'linear_value_head_dim', 'linear_num_key_heads',
                    'linear_num_value_heads', 'moe_intermediate_size',
                    'shared_expert_intermediate_size', 'num_experts', 'num_experts_per_tok',
                    'max_position_embeddings'):
            if type(raw.get(key)) is not int or raw[key] <= 0:
                raise ValueError(f'invalid {key}')
        if raw['max_position_embeddings']>2**63-1:
            raise ValueError('Qwen3.5 rotary position bound exceeds signed int64 transport')
        if (raw['num_attention_heads'] % raw['num_key_value_heads'] or
                raw['linear_num_value_heads'] % raw['linear_num_key_heads'] or
                raw['num_experts_per_tok'] > raw['num_experts']):
            raise ValueError('invalid head or expert geometry')
        layers = raw.get('layer_types')
        if not isinstance(layers, list) or len(layers) != raw['num_hidden_layers'] or any(t not in ('linear_attention', 'full_attention') for t in layers):
            raise ValueError('explicit exact layer_types required')
        rope = raw.get('rope_parameters')
        if not isinstance(rope, dict) or rope.get('rope_type') != 'default':
            raise ValueError('only explicit default rope_parameters supported')
        if raw.get('hidden_act', 'silu') != 'silu' or raw.get('attention_bias', False) or raw.get('quantization_config') is not None:
            raise ValueError('unsupported activation, bias or quantization')
        if raw.get('attn_output_gate',True) is not True or raw.get('mlp_only_layers',[]):
            raise ValueError('Qwen3.5 requires its output gate and routed/shared expert schedule')
        for value in (rope.get('rope_theta'), rope.get('partial_rotary_factor'), raw.get('rms_norm_eps')):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
                raise ValueError('invalid normalization or rotary parameters')
        rd = raw['head_dim'] * rope['partial_rotary_factor']
        if int(rd) != rd or int(rd) % 2 or not 2 <= rd <= raw['head_dim']:
            raise ValueError('invalid partial rotary dimension')
        self.rotary_dim = int(rd)
        section=rope.get('mrope_section')
        if section is not None and (type(section) not in (list,tuple) or len(section)!=3 or
                any(type(value) is not int or value<0 for value in section) or sum(section)!=int(rd)//2):
            raise ValueError('explicit three-axis rotary sections must cover the partial rotary frequencies')
        self.mrope_section=None if section is None else tuple(section)
        if section is not None and rope.get('mrope_interleaved',True) is not True:
            raise ValueError('explicit interleaved three-axis Qwen3.5 rotary layout required')
        self.theta = rope['rope_theta']
        self.config = c = SimpleNamespace(**raw)
        self.layer_begin,self.layer_end = 0,c.num_hidden_layers
        self.range_mode = layer_range is not None
        if self.range_mode:
            if (type(layer_range) is not tuple or len(layer_range)!=2 or
                    any(type(value) is not int for value in layer_range) or
                    not 0<=layer_range[0]<layer_range[1]<=c.num_hidden_layers):
                raise ValueError('valid exact Qwen3.5 layer interval required')
            self.layer_begin,self.layer_end = layer_range
        h, e, m = c.hidden_size, c.num_experts, c.moe_intermediate_size
        shapes = {'model.embed_tokens.weight': (c.vocab_size, h), 'model.norm.weight': (h,),
                  'lm_head.weight': (c.vocab_size, h)}
        for i, kind in enumerate(layers):
            p = f'model.layers.{i}.'
            shapes.update({p+'input_layernorm.weight': (h,), p+'post_attention_layernorm.weight': (h,),
                p+'mlp.gate.weight': (e, h), p+'mlp.experts.gate_up_proj': (e, 2*m, h),
                p+'mlp.experts.down_proj': (e, h, m), p+'mlp.shared_expert_gate.weight': (1, h)})
            for name, shape in {'gate_proj': (c.shared_expert_intermediate_size, h),
                                'up_proj': (c.shared_expert_intermediate_size, h),
                                'down_proj': (h, c.shared_expert_intermediate_size)}.items():
                shapes[p+'mlp.shared_expert.'+name+'.weight'] = shape
            if kind == 'full_attention':
                for name, shape in {'q_proj': (2*c.num_attention_heads*c.head_dim, h),
                                    'k_proj': (c.num_key_value_heads*c.head_dim, h),
                                    'v_proj': (c.num_key_value_heads*c.head_dim, h),
                                    'o_proj': (h, c.num_attention_heads*c.head_dim),
                                    'q_norm': (c.head_dim,), 'k_norm': (c.head_dim,)}.items():
                    shapes[p+'self_attn.'+name+'.weight'] = shape
            else:
                width = 2*c.linear_num_key_heads*c.linear_key_head_dim + c.linear_num_value_heads*c.linear_value_head_dim
                a = p+'linear_attn.'
                shapes.update({a+'dt_bias': (c.linear_num_value_heads,), a+'A_log': (c.linear_num_value_heads,),
                    a+'conv1d.weight': (width, 1, c.linear_conv_kernel_dim),
                    a+'norm.weight': (c.linear_value_head_dim,),
                    a+'in_proj_qkv.weight': (width, h),
                    a+'in_proj_z.weight': (c.linear_num_value_heads*c.linear_value_head_dim, h),
                    a+'in_proj_a.weight': (c.linear_num_value_heads, h),
                    a+'in_proj_b.weight': (c.linear_num_value_heads, h),
                    a+'out_proj.weight': (h, c.linear_num_value_heads*c.linear_value_head_dim)})
        if _inventory_only:
            self.tensor_shapes = shapes
            return
        if self.range_mode:
            prefixes = tuple(f'model.layers.{index}.' for index in range(self.layer_begin,self.layer_end))
            shapes = {name:shape for name,shape in shapes.items() if name.startswith(prefixes)}
        elif _edge_only:
            shapes = {name:shape for name,shape in shapes.items() if not name.startswith('model.layers.')}
        if _omit_experts:
            shapes = {name:shape for name,shape in shapes.items()
                      if not name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
        tensors = dict(tensors)
        if raw.get('tie_word_embeddings', False) and not self.range_mode:
            embedding = tensors.get('model.embed_tokens.weight')
            if 'lm_head.weight' in tensors and not np.array_equal(embedding, tensors['lm_head.weight']):
                raise ValueError('inconsistent tied weights')
            tensors['lm_head.weight'] = embedding
        if set(tensors) != set(shapes):
            raise ValueError(f'tensor names mismatch: missing={set(shapes)-set(tensors)}, extra={set(tensors)-set(shapes)}')
        self.weights = {}
        for name, shape in shapes.items():
            array = np.asarray(tensors[name])
            if array.shape != shape or array.dtype.kind != 'f' or not np.isfinite(array).all():
                raise ValueError(f'invalid tensor {name}')
            array = np.array(array, dtype=np.float32, copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f'FP32 overflow {name}')
            array.flags.writeable = False
            self.weights[name] = array
        self.delta_kernel = delta_kernel
        if delta_kernel is not None:
            from .native_mixtral_delta import CompiledGatedDeltaKernel
            if not isinstance(delta_kernel, CompiledGatedDeltaKernel):
                raise ValueError('expected explicit CompiledGatedDeltaKernel')
            self.implementation = 'numpy+c-gated-delta-step'
        self.reset()

    def reset(self):
        self.position = 0
        self.cache = [None] * self.config.num_hidden_layers
        self.mask = None

    def status(self):
        return dict(model_type='qwen3_5_moe_text', implementation=self.implementation,
                    native_range_execution=False, position=self.position)

    def forward(self, token_ids, *, attention_mask=None, position_ids=None):
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        c, w = self.config, self.weights
        if ids.ndim != 2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= c.vocab_size):
            raise ValueError('invalid token_ids')
        batch, length = ids.shape
        end = self.position + length
        if end > c.max_position_embeddings:
            raise ValueError('context capacity exceeded')
        mask = np.ones((batch, end), bool) if attention_mask is None else np.asarray(attention_mask)
        if mask.shape != (batch, end) or not np.isin(mask, [0, 1]).all():
            raise ValueError('invalid attention_mask')
        mask = mask.astype(bool)
        if self.mask is not None and not np.array_equal(self.mask, mask[:, :self.position]):
            raise ValueError('cached mask or batch changed')
        positions = np.broadcast_to(np.arange(self.position, end), ids.shape) if position_ids is None else np.asarray(position_ids)
        if positions.shape != ids.shape or positions.dtype.kind not in 'iu' or np.any(positions < 0) or np.any(positions >= c.max_position_embeddings):
            raise ValueError('only text position_ids supported')
        x = w['model.embed_tokens.weight'][ids].copy()
        hidden, routers, routes, cache, attentions = [], [], [], [], []
        for i, kind in enumerate(c.layer_types):
            hidden.append(x.copy())
            p = f'model.layers.{i}.'
            z = rmsnorm(x, 1+w[p+'input_layernorm.weight'], c.rms_norm_eps)
            if kind == 'linear_attention':
                mixed, state = self._delta(z, p, self.cache[i], mask[:, self.position:end])
            else:
                mixed, state, probabilities = self._attention(z, p, self.cache[i], positions, mask)
                attentions.append(probabilities)
            cache.append(state)
            x = x + mixed
            z = rmsnorm(x, 1+w[p+'post_attention_layernorm.weight'], c.rms_norm_eps).reshape(-1, c.hidden_size)
            router = z @ w[p+'mlp.gate.weight'].T
            routers.append(router)
            prob = softmax(router)
            indices = np.argsort(-prob, axis=-1, kind='stable')[:, :c.num_experts_per_tok]
            scores = np.take_along_axis(prob, indices, axis=-1)
            scores /= scores.sum(axis=-1, keepdims=True)
            routes.append((indices, scores))
            output = np.zeros_like(z)
            for expert in range(c.num_experts):
                rows, ranks = np.nonzero(indices == expert)
                if not rows.size:
                    continue
                gate, up = np.split(z[rows] @ w[p+'mlp.experts.gate_up_proj'][expert].T, 2, axis=-1)
                result = (silu(gate)*up) @ w[p+'mlp.experts.down_proj'][expert].T
                output[rows] += result * scores[rows, ranks, None]
            shared = silu(z @ w[p+'mlp.shared_expert.gate_proj.weight'].T) * (z @ w[p+'mlp.shared_expert.up_proj.weight'].T)
            shared = (shared @ w[p+'mlp.shared_expert.down_proj.weight'].T) * sigmoid(z @ w[p+'mlp.shared_expert_gate.weight'].T)
            x = x + (output+shared).reshape(x.shape)
        x = rmsnorm(x, 1+w['model.norm.weight'], c.rms_norm_eps)
        hidden.append(x.copy())
        logits = x @ w['lm_head.weight'].T
        self.cache, self.mask, self.position = cache, mask.copy(), end
        return SimpleNamespace(logits=logits, hidden_states=tuple(hidden), router_logits=tuple(routers),
                               routes=tuple(routes), attentions=tuple(attentions))

    def _attention(self, x, p, old, positions, mask):
        c, w = self.config, self.weights
        b, t, _ = x.shape
        a = p+'self_attn.'
        query_gate = (x @ w[a+'q_proj.weight'].T).reshape(b, t, c.num_attention_heads, 2*c.head_dim)
        q, gate = np.split(query_gate, 2, axis=-1)
        q = rmsnorm(q, 1+w[a+'q_norm.weight'], c.rms_norm_eps)
        k = (x @ w[a+'k_proj.weight'].T).reshape(b, t, c.num_key_value_heads, c.head_dim)
        k = rmsnorm(k, 1+w[a+'k_norm.weight'], c.rms_norm_eps)
        v = (x @ w[a+'v_proj.weight'].T).reshape(b, t, c.num_key_value_heads, c.head_dim).transpose(0, 2, 1, 3)
        def rotate(z):
            r = self.rotary_dim
            return np.concatenate((rotary(z[..., :r], positions, self.theta), z[..., r:]), axis=-1).transpose(0, 2, 1, 3)
        q, k = rotate(q), rotate(k)
        if old is not None:
            k, v = np.concatenate((old[0], k), axis=2), np.concatenate((old[1], v), axis=2)
        groups = c.num_attention_heads // c.num_key_value_heads
        scores = q @ np.repeat(k, groups, axis=1).swapaxes(-1, -2) / np.float32(np.sqrt(c.head_dim))
        allowed = (np.arange(k.shape[2])[None, :] <= np.arange(self.position, self.position+t)[:, None])[None, None] & mask[:, None, None]
        prob = softmax(np.where(allowed, scores, np.finfo(np.float32).min))
        context = (prob @ np.repeat(v, groups, axis=1)).transpose(0, 2, 1, 3)
        return (context * sigmoid(gate)).reshape(b, t, -1) @ w[a+'o_proj.weight'].T, (k, v), prob

    def _delta(self, x, p, old, mask):
        c, w = self.config, self.weights
        b, t, _ = x.shape
        a = p+'linear_attn.'
        # Mask the mixer input, retaining physical convolution slots for padding.
        x = x * mask[..., None]
        projected = x @ w[a+'in_proj_qkv.weight'].T
        width = projected.shape[-1]
        kernel = c.linear_conv_kernel_dim
        history = np.zeros((b, width, kernel), np.float32) if old is None else old[0].copy()
        state = np.zeros((b, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim), np.float32) if old is None else old[1].copy()
        conv = np.empty_like(projected)
        for row in range(t):
            history = np.concatenate((history[..., 1:], projected[:, row, :, None]), axis=-1)
            conv[:, row] = silu(np.sum(history*w[a+'conv1d.weight'][:, 0, :], axis=-1))
        keywidth = c.linear_num_key_heads*c.linear_key_head_dim
        q, k, v = np.split(conv, [keywidth, 2*keywidth], axis=-1)
        q, k = (z.reshape(b, t, c.linear_num_key_heads, c.linear_key_head_dim) for z in (q, k))
        q = q / np.sqrt(np.sum(q*q, axis=-1, keepdims=True)+1e-6)
        k = k / np.sqrt(np.sum(k*k, axis=-1, keepdims=True)+1e-6)
        repeat = c.linear_num_value_heads // c.linear_num_key_heads
        q, k = np.repeat(q, repeat, axis=2), np.repeat(k, repeat, axis=2)
        q = q / np.float32(np.sqrt(c.linear_key_head_dim))
        v = v.reshape(b, t, c.linear_num_value_heads, c.linear_value_head_dim)
        beta = sigmoid(x @ w[a+'in_proj_b.weight'].T)
        decay = np.exp(-np.exp(w[a+'A_log']) * np.logaddexp(np.float32(0), x @ w[a+'in_proj_a.weight'].T+w[a+'dt_bias']))
        output = np.empty_like(v)
        for row in range(t):
            if self.delta_kernel is not None:
                output[:, row] = self.delta_kernel.step(*(np.ascontiguousarray(z) for z in
                    (q[:, row], k[:, row], v[:, row], beta[:, row], decay[:, row])), state)
                continue
            state *= decay[:, row, :, None, None]
            error = v[:, row] - np.einsum('bhk,bhkv->bhv', k[:, row], state)
            state += k[:, row, :, :, None] * (error*beta[:, row, :, None])[:, :, None, :]
            output[:, row] = np.einsum('bhk,bhkv->bhv', q[:, row], state)
        gate = (x @ w[a+'in_proj_z.weight'].T).reshape(output.shape)
        output = rmsnorm(output, w[a+'norm.weight'], c.rms_norm_eps) * silu(gate)
        return output.reshape(b, t, -1) @ w[a+'out_proj.weight'].T, (history, state)

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
