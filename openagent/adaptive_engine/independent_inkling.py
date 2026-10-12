"""Independently designed FP32 Inkling decoder and resident state contract.

Architecture learned from Apache-2.0 Transformers 5.16.1; no imported model code.
This NumPy implementation is a diagnostic baseline, not native execution.
"""
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType, SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class Geometry:
    heads: int
    kv_heads: int
    dim: int
    extent: int
    window: int

    @property
    def q_width(self):
        return self.heads * self.dim

    @property
    def kv_width(self):
        return self.kv_heads * self.dim


LAYER_NAMES = (
    'input_layernorm.weight', 'post_attention_layernorm.weight',
    'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
    'self_attn.r_proj.weight', 'self_attn.o_proj.weight',
    'self_attn.k_sconv.conv1d.weight', 'self_attn.v_sconv.conv1d.weight',
    'self_attn.q_norm.weight', 'self_attn.k_norm.weight',
    'self_attn.rel_logits_proj.proj', 'attn_sconv.conv1d.weight',
    'mlp_sconv.conv1d.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight',
    'mlp.down_proj.weight', 'mlp.global_scale',
)
GLOBAL_NAMES = ('model.embed_tokens.weight', 'model.embed_norm.weight',
                'model.norm.weight', 'lm_head.weight')
MOE_NAMES = ('mlp.gate.weight', 'mlp.gate.global_scale', 'mlp.gate.e_score_correction_bias',
             'mlp.experts.gate_up_proj', 'mlp.experts.down_proj',
             'mlp.shared_experts.gate_proj', 'mlp.shared_experts.up_proj',
             'mlp.shared_experts.down_proj')
ALL_LAYER_NAMES = LAYER_NAMES + MOE_NAMES


def configuration(config):
    raw = dict(config.to_dict() if hasattr(config, 'to_dict') else config)
    if raw.get('model_type') != 'inkling_text':
        raise ValueError('expected exact inkling_text identity; no aliases')
    if raw.get('architectures') not in (None, ['InklingForCausalLM'], ('InklingForCausalLM',)):
        raise ValueError('unsupported architecture declaration')
    if (raw.get('use_cache', True) is not True or raw.get('return_dict', True) is not True or
            raw.get('output_hidden_states', False) is not False or
            raw.get('output_attentions', False) is not False):
        raise ValueError('unsupported cache/output mode; use the explicit native API')
    metadata = set('transformers_version architectures output_hidden_states return_dict dtype '
                   'id2label label2id problem_type output_attentions _name_or_path '
                   'initializer_range pad_token_id bos_token_id eos_token_id use_cache'.split())
    moe_fields = set('shared_expert_sink route_scale rms_norm_eps_moe_gate'.split())
    integers = ('vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
                'num_key_value_heads', 'head_dim', 'swa_num_attention_heads',
                'swa_num_key_value_heads', 'swa_head_dim', 'sliding_window_size',
                'd_rel', 'rel_extent', 'max_position_embeddings', 'conv_kernel_size',
                'intermediate_size', 'moe_intermediate_size', 'n_routed_experts',
                'num_experts_per_tok', 'n_shared_experts')
    recognized = metadata | moe_fields | set(integers) | set(
        'unpadded_vocab_size layer_types mlp_layer_types local_layer_ids hidden_act '
        'log_scaling_n_floor log_scaling_alpha rms_norm_eps logits_mup_width_multiplier '
        'attention_dropout num_mtp_layers chain_hidden_post_norm mtp_hidden_states_first '
        'mtp_local_layer_ids number_of_conv_states tie_word_embeddings is_encoder_decoder '
        'chunk_size_feed_forward model_type'.split())
    if set(raw) - recognized:
        raise ValueError(f'unsupported config fields: {sorted(set(raw) - recognized)}')
    for key in integers:
        if type(raw.get(key)) is not int or not 0 < raw[key] <= 1048576:
            raise ValueError(f'invalid or unbounded {key}')
    if (raw['num_hidden_layers'] > 128 or raw['conv_kernel_size'] > 64 or
            raw['max_position_embeddings'] > 65536):
        raise ValueError('unsupported resident geometry')
    n = raw['num_hidden_layers']
    kinds = raw.get('layer_types')
    if not isinstance(kinds, (list, tuple)) or len(kinds) != n or any(
            k not in ('hybrid', 'hybrid_sliding') for k in kinds):
        raise ValueError('explicit full/sliding layer schedule required')
    mlps = raw.get('mlp_layer_types')
    if not isinstance(mlps, (list, tuple)) or len(mlps) != n or any(k not in ('dense', 'sparse') for k in mlps):
        raise ValueError('explicit dense/sparse MLP schedule required')
    if raw['num_experts_per_tok'] > raw['n_routed_experts'] or raw['n_routed_experts'] + raw['n_shared_experts'] > 65536:
        raise ValueError('unsupported expert geometry')
    if raw.get('shared_expert_sink', True) is not True or raw.get('rms_norm_eps_moe_gate', 1e-6) != 1e-6:
        raise ValueError('only default shared sink and gate-epsilon settings supported')
    if (raw.get('local_layer_ids') is not None or raw.get('dtype') not in (None, 'float32') or
            raw.get('hidden_act', 'silu') != 'silu' or raw.get('attention_dropout', 0) != 0 or
            raw.get('tie_word_embeddings', False) is not False or
            raw.get('is_encoder_decoder', False) is not False or
            raw.get('chunk_size_feed_forward', 0) != 0 or
            raw.get('num_mtp_layers') is not None or raw.get('mtp_local_layer_ids') is not None or
            raw.get('chain_hidden_post_norm', False) is not False or
            raw.get('mtp_hidden_states_first', True) is not True or
            raw.get('number_of_conv_states', 4) != 4):
        raise ValueError('unsupported activation, dropout, tying, MTP, or model mode')
    for key in ('rms_norm_eps', 'logits_mup_width_multiplier', 'route_scale'):
        v = raw.get(key)
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or v <= 0:
            raise ValueError(f'invalid {key}')
        if not np.isfinite(np.float32(v)) or np.float32(v) <= 0:
            raise ValueError(f'{key} is outside FP32 range')
    alpha = raw.get('log_scaling_alpha', 0.1)
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not np.isfinite(alpha) or not 0 <= alpha <= 100:
        raise ValueError('invalid log_scaling_alpha')
    floor = raw.get('log_scaling_n_floor')
    if floor is not None and (type(floor) is not int or not 0 < floor <= 1048576):
        raise ValueError('invalid log_scaling_n_floor')
    unpadded = raw.get('unpadded_vocab_size')
    if unpadded is not None and (type(unpadded) is not int or not 0 < unpadded <= raw['vocab_size']):
        raise ValueError('invalid unpadded_vocab_size')
    geometries = []
    for kind in kinds:
        sliding = kind == 'hybrid_sliding'
        prefix = 'swa_' if sliding else ''
        g = Geometry(raw[prefix + 'num_attention_heads'], raw[prefix + 'num_key_value_heads'],
                     raw[prefix + 'head_dim'], raw['sliding_window_size'] if sliding else raw['rel_extent'],
                     raw['sliding_window_size'] if sliding else 0)
        if g.heads % g.kv_heads or max(g.q_width, g.kv_width, g.heads * raw['d_rel']) > 65536:
            raise ValueError('unsupported grouped-head geometry')
        geometries.append(g)
    if any(g.window == 1 for g in geometries):
        raise ValueError('window=1 oracle cached/full paths disagree; unsupported')
    raw['layer_types'], raw['mlp_layer_types'] = tuple(kinds), tuple(mlps)
    raw['log_scaling_alpha'] = alpha
    # Only scalar numerical settings and immutable schedules are retained.
    selected = set(integers) | set('layer_types mlp_layer_types rms_norm_eps '
        'logits_mup_width_multiplier log_scaling_alpha log_scaling_n_floor '
        'unpadded_vocab_size model_type route_scale'.split())
    return MappingProxyType({k: raw.get(k) for k in selected}), tuple(geometries)


def rms(x, weight, eps):
    variance = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(variance + np.float32(eps))) * weight


def fir(x, bank, history):
    history[:-1] = history[1:]
    history[-1] = x
    return x + np.sum(history * bank[:, 0, :].T, axis=0, dtype=np.float32)


def logits_selection(logits_to_keep, tokens):
    if type(logits_to_keep) is int:
        if logits_to_keep < 0:
            raise ValueError('logits_to_keep must be nonnegative')
        return slice(-logits_to_keep, None) if logits_to_keep else slice(None)
    indices = np.asarray(logits_to_keep)
    if indices.ndim != 1 or indices.size > tokens or indices.dtype.kind not in 'iu' or np.any(indices >= tokens) or np.any(indices < 0):
        raise ValueError('logits_to_keep requires in-range one-dimensional integer indices')
    return indices.astype(np.int64, copy=True)


class IndependentInklingDecoder:
    implementation = 'numpy-inkling-token-stream'
    native_range_execution = False

    def __init__(self, config, tensors, *, max_bytes=512 * 1024 * 1024,_borrow_globals=False,_omit_experts=False,_inventory_only=False):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('invalid max_bytes')
        self._lock = RLock()
        self.max_bytes = max_bytes
        self.config, self.geometry = configuration(config)
        c = self.config
        h, k, m = c['hidden_size'], c['conv_kernel_size'], c['intermediate_size']
        shapes = dict(zip(GLOBAL_NAMES, ((c['vocab_size'], h), (h,), (h,), (c['vocab_size'], h))))
        for i, g in enumerate(self.geometry):
            sizes = ((h,), (h,), (g.q_width, h), (g.kv_width, h), (g.kv_width, h),
                     (g.heads * c['d_rel'], h), (h, g.q_width),
                     (g.kv_width, 1, k), (g.kv_width, 1, k), (g.dim,), (g.dim,),
                     (c['d_rel'], g.extent), (h, 1, k), (h, 1, k),
                     (m, h), (m, h), (h, m), (1,))
            shapes.update({f'model.layers.{i}.{name}': shape for name, shape in zip(LAYER_NAMES[:14], sizes[:14])})
            if c['mlp_layer_types'][i] == 'dense':
                shapes.update({f'model.layers.{i}.{name}': shape for name, shape in zip(LAYER_NAMES[14:], sizes[14:])})
            else:
                e, s, z = c['n_routed_experts'], c['n_shared_experts'], c['moe_intermediate_size']
                moe_shapes = ((e + s, h), (1,), (e,), (e, 2 * z, h), (e, h, z),
                              (s, z, h), (s, z, h), (s, h, z))
                shapes.update({f'model.layers.{i}.{name}': shape for name, shape in zip(MOE_NAMES, moe_shapes)})
        if _omit_experts:
            shapes={name:shape for name,shape in shapes.items() if not name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
        self.tensor_shapes=MappingProxyType(shapes)
        if _inventory_only:
            self.weights=MappingProxyType({});self._cache=None;self.position=0
            return
        if set(tensors) != set(shapes):
            raise ValueError(f'weight names mismatch; missing={sorted(set(shapes)-set(tensors))}; '
                             f'extra={sorted(set(tensors)-set(shapes))}')
        expected_bytes = sum(int(np.prod(shape)) * 4 for shape in shapes.values())
        if expected_bytes > max_bytes:
            raise MemoryError('resident weights exceed max_bytes')
        weights = {}
        for name, shape in shapes.items():
            a = np.asarray(tensors[name])
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError(f'{name} requires finite FP32 shape {shape}')
            if _borrow_globals and name in GLOBAL_NAMES:
                if a.flags.writeable or not a.flags.c_contiguous:
                    raise ValueError('borrowed immutable contiguous global weights required')
                weights[name]=a
            else:
                weights[name] = np.frombuffer(a.tobytes(order='C'), dtype=np.float32).reshape(shape)
        self.weights = MappingProxyType(weights)
        self._cache = None
        self.position = 0

    def _state_size(self, g):
        c = self.config
        return 2 * c['max_position_embeddings'] * g.kv_width + 2 * c['conv_kernel_size'] * (g.kv_width + c['hidden_size'])

    def _views(self, state, g):
        c = self.config
        cap, k, h, v = c['max_position_embeddings'], c['conv_kernel_size'], c['hidden_size'], g.kv_width
        sizes = (cap * v, cap * v, k * v, k * v, k * h, k * h)
        shapes = ((cap, g.kv_heads, g.dim), (cap, g.kv_heads, g.dim),
                  (k, v), (k, v), (k, h), (k, h))
        result, offset = [], 0
        for size, shape in zip(sizes, shapes):
            result.append(state[offset:offset + size].reshape(shape))
            offset += size
        return result

    def reset(self):
        with self._lock:
            self._cache, self.position = None, 0

    def cache_snapshot(self):
        with self._lock:
            return tuple(a.copy() for a in self._cache) if self._cache is not None else ()

    def status(self):
        with self._lock:
            return dict(model_type='inkling_text', implementation=self.implementation,
                        position=self.position, mlp=self.config['mlp_layer_types'], cache='resident-kv+four-fir-histories')

    def _prepare(self, token_ids, attention_mask, position_ids, use_cache=True, *, additional_bytes=0):
        if type(use_cache) is not bool:
            raise ValueError('use_cache must be boolean')
        if attention_mask is not None or position_ids is not None:
            raise ValueError('only unmasked contiguous implicit positions supported')
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or ids.shape[0] > 128:
            raise ValueError('expected nonempty integer [batch, tokens]')
        if np.any(ids >= self.config['vocab_size']) or np.any(ids < 0):
            raise ValueError('token outside vocabulary')
        b, t = ids.shape
        start = self.position if use_cache else 0
        if start + t > self.config['max_position_embeddings']:
            raise ValueError('context capacity exceeded')
        if use_cache and self._cache is not None and b != self._cache[0].shape[0]:
            raise ValueError('cached batch size changed')
        c = self.config
        state_bytes = b * sum(self._state_size(g) for g in self.geometry) * 4
        outputs = b * t * (c['hidden_size'] + (c['unpadded_vocab_size'] or c['vocab_size']) +
                          c['num_hidden_layers'] * (c['n_routed_experts'] + c['n_shared_experts'])) * 4
        scratch = 4 * (8 * c['hidden_size'] + 4 * max(c['intermediate_size'], c['moe_intermediate_size']) +
            8 * (c['n_routed_experts'] + c['n_shared_experts']) +
            max(3 * g.q_width + 4 * g.kv_width + g.heads * c['d_rel'] for g in self.geometry) +
            c['max_position_embeddings'] + max(g.extent for g in self.geometry))
        old_bytes = sum(a.nbytes for a in self._cache) if self._cache is not None else 0
        needed = sum(a.nbytes for a in self.weights.values()) + old_bytes + 2 * state_bytes + 2 * outputs + scratch + ids.size * 16
        if type(additional_bytes) is not int or additional_bytes<0:
            raise ValueError('nonnegative additional transaction budget required')
        if needed+additional_bytes > self.max_bytes:
            raise MemoryError(f'resident call needs {needed} bytes; limit {self.max_bytes}')
        pending = ([a.copy() for a in self._cache] if use_cache and self._cache is not None else
                   [np.zeros((b, self._state_size(g)), np.float32) for g in self.geometry])
        return np.ascontiguousarray(ids, dtype=np.int64), pending

    def _moe(self, x, i):
        c = self.config
        a = [self.weights[f'model.layers.{i}.{name}'] for name in MOE_NAMES]
        e, s, top = c['n_routed_experts'], c['n_shared_experts'], c['num_experts_per_tok']
        logits = a[0] @ x
        sigmoid = np.exp(-np.logaddexp(np.float32(0), -logits))
        choice = sigmoid[:e] + a[2]
        order = np.argsort(-choice, kind='stable')
        if not np.isfinite(logits).all() or not np.isfinite(choice).all():
            raise FloatingPointError('nonfinite routing; state not committed')
        if top < e and choice[order[top - 1]] == choice[order[top]]:
            raise ValueError('ambiguous top-k boundary tie; state not committed')
        chosen = np.concatenate((order[:top], np.arange(e, e + s)))
        logs = -np.logaddexp(np.float32(0), -logits[chosen])
        probability = np.exp(logs - logs.max())
        probability /= probability.sum(dtype=np.float32)
        probability *= np.float32(c['route_scale']) * a[1][0]
        gammas = np.zeros(e + s, np.float32)
        gammas[chosen] = probability
        routed, shared = np.zeros_like(x), np.zeros_like(x)
        for expert in sorted(order[:top]):
            gate, up = np.split(a[3][expert] @ x, 2)
            activated = gate * np.exp(-np.logaddexp(np.float32(0), -gate)) * up
            routed += (a[4][expert] @ activated) * gammas[expert]
        for expert in range(s):
            gate, up = a[5][expert] @ x, a[6][expert] @ x
            activated = gate * np.exp(-np.logaddexp(np.float32(0), -gate)) * up * gammas[e + expert]
            shared += a[7][expert] @ activated
        return routed + shared, gammas

    def forward(self, token_ids, *, attention_mask=None, position_ids=None, logits_to_keep=0, use_cache=True):
        with self._lock:
            ids, pending = self._prepare(token_ids, attention_mask, position_ids, use_cache)
            start = self.position if use_cache else 0
            c, w = self.config, self.weights
            b, t = ids.shape
            selection = logits_selection(logits_to_keep, t)
            hidden = np.empty((b, t, c['hidden_size']), np.float32)
            routing = np.zeros((b, t, len(self.geometry), c['n_routed_experts'] + c['n_shared_experts']), np.float32)
            eps = c['rms_norm_eps']
            for batch in range(b):
                for j, token in enumerate(ids[batch]):
                    pos = start + j
                    x = rms(w[GLOBAL_NAMES[0]][token], w[GLOBAL_NAMES[1]], eps)
                    for i, g in enumerate(self.geometry):
                        a = [w.get(f'model.layers.{i}.{name}') for name in LAYER_NAMES]
                        keys, values, kc, vc, ac, mc = self._views(pending[i][batch], g)
                        n = rms(x, a[0], eps)
                        q = rms((a[2] @ n).reshape(g.heads, g.dim), a[9], eps)
                        keys[pos] = rms(fir(a[3] @ n, a[7], kc).reshape(g.kv_heads, g.dim), a[10], eps)
                        values[pos] = fir(a[4] @ n, a[8], vc).reshape(g.kv_heads, g.dim)
                        r = (a[5] @ n).reshape(g.heads, c['d_rel'])
                        floor = c['log_scaling_n_floor']
                        tau = np.float32(1)
                        if not g.window and floor is not None:
                            tau += np.float32(c['log_scaling_alpha']) * np.log(max(np.float32(pos + 1) / np.float32(floor), np.float32(1)))
                        q *= tau
                        first = max(0, pos - g.window + 1) if g.window else 0
                        context = np.empty((g.heads, g.dim), np.float32)
                        distances = pos - np.arange(first, pos + 1)
                        for head in range(g.heads):
                            kh = head // (g.heads // g.kv_heads)
                            scores = (keys[first:pos + 1, kh] @ q[head]) / np.float32(g.dim)
                            valid = distances < g.extent
                            profile = r[head] @ a[11]
                            scores[valid] += profile[distances[valid]] * tau
                            probability = np.exp(scores - scores.max())
                            probability /= probability.sum(dtype=np.float32)
                            context[head] = probability @ values[first:pos + 1, kh]
                        x = x + fir(a[6] @ context.ravel(), a[12], ac)
                        n = rms(x, a[1], eps)
                        if c['mlp_layer_types'][i] == 'dense':
                            gate, up = a[14] @ n, a[15] @ n
                            sigmoid = np.exp(-np.logaddexp(np.float32(0), -gate))
                            feedforward = (a[16] @ (gate * sigmoid * up)) * a[17][0]
                        else:
                            feedforward, routing[batch, j, i] = self._moe(n, i)
                        x = x + fir(feedforward, a[13], mc)
                    hidden[batch, j] = rms(x, w[GLOBAL_NAMES[2]], eps)
            rows = c['unpadded_vocab_size'] or c['vocab_size']
            logits = (hidden / np.float32(c['logits_mup_width_multiplier'])) @ w[GLOBAL_NAMES[3]][:rows].T
            if not np.isfinite(logits).all() or not np.isfinite(hidden).all() or any(not np.isfinite(a).all() for a in pending):
                raise FloatingPointError('nonfinite result; state not committed')
            result = SimpleNamespace(logits=logits[:, selection], last_hidden_state=hidden, routing_weights=routing)
            if use_cache:
                self._cache, self.position = pending, start + t
            return result
