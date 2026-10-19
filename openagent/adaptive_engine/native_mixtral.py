"""Independent inference-only Mixtral equations; NumPy, not Transformers execution.

Resident FP32 weights and append-only KV state. No quantization or family aliases.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def softmax(x):
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def rmsnorm(x, weight, epsilon):
    return x * (1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon)) * weight


def rotary(x, positions, theta):
    half = x.shape[-1] // 2
    frequencies = np.float32(theta) ** (-np.arange(half, dtype=np.float32) / np.float32(half))
    angles = positions.astype(np.float32)[..., None, None] * frequencies
    a, b = x[..., :half], x[..., half:]
    return np.concatenate((a * np.cos(angles) - b * np.sin(angles),
                           b * np.cos(angles) + a * np.sin(angles)), axis=-1)


class IndependentMixtralDecoder:
    """Strict exact-identity decoder with batched prefill and cached continuation.

    Masks passed to forward cover the entire cache plus the current chunk.
    Positions default to physical cache slots, as in direct Transformers forward.
    Padding-aware positions can be supplied explicitly.
    """

    implementation = 'numpy'
    native_range_execution = False

    def __init__(self, config, tensors):
        raw = dict(config)
        if raw.get("model_type") != "mixtral":
            raise ValueError("expected exact model_type mixtral")
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "num_local_experts",
                    "num_experts_per_tok", "vocab_size", "max_position_embeddings"):
            if type(raw.get(key)) is not int or raw[key] <= 0:
                raise ValueError(f"invalid {key}")
        h, heads, kv = raw['hidden_size'], raw['num_attention_heads'], raw['num_key_value_heads']
        d = raw.get('head_dim') or h // heads
        if h % heads or heads % kv or type(d) is not int or d <= 0 or d % 2:
            raise ValueError("invalid attention geometry")
        if raw['num_experts_per_tok'] > raw['num_local_experts']:
            raise ValueError("invalid top-k geometry")
        for key, default in (("rms_norm_eps", 1e-5), ("rope_theta", 1e6)):
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {key}")
            raw[key] = value
        if raw.get('hidden_act', 'silu') != 'silu' or raw.get('rope_scaling') is not None or raw.get('quantization_config') is not None:
            raise ValueError("unsupported activation, scaled RoPE or quantization")
        if raw.get('attention_bias', False):
            raise ValueError("attention bias unsupported")
        window = raw.get('sliding_window')
        if window is not None and (type(window) is not int or window <= 0):
            raise ValueError("invalid sliding_window")
        raw['head_dim'] = d
        raw['sliding_window'] = window
        self.config = SimpleNamespace(**raw)
        shapes = {'model.embed_tokens.weight': (raw['vocab_size'], h),
                  'model.norm.weight': (h,), 'lm_head.weight': (raw['vocab_size'], h)}
        for i in range(raw['num_hidden_layers']):
            p = f'model.layers.{i}.'
            for name in ('input_layernorm', 'post_attention_layernorm'):
                shapes[p + name + '.weight'] = (h,)
            for name, shape in {'q': (heads*d, h), 'k': (kv*d, h), 'v': (kv*d, h), 'o': (h, heads*d)}.items():
                shapes[p + 'self_attn.' + name + '_proj.weight'] = shape
            shapes[p + 'block_sparse_moe.gate.weight'] = (raw['num_local_experts'], h)
            for e in range(raw['num_local_experts']):
                for name, shape in {'w1': (raw['intermediate_size'], h), 'w3': (raw['intermediate_size'], h), 'w2': (h, raw['intermediate_size'])}.items():
                    shapes[p + f'block_sparse_moe.experts.{e}.{name}.weight'] = shape
        tensors = dict(tensors)
        if raw.get('tie_word_embeddings', False):
            embedding = tensors.get('model.embed_tokens.weight')
            if 'lm_head.weight' in tensors and not np.array_equal(tensors['lm_head.weight'], embedding):
                raise ValueError("inconsistent tied weights")
            tensors['lm_head.weight'] = embedding
        if set(tensors) != set(shapes):
            raise ValueError(f"tensor names mismatch: missing={set(shapes)-set(tensors)}, extra={set(tensors)-set(shapes)}")
        self.weights = {}
        for name, shape in shapes.items():
            value = np.asarray(tensors[name])
            if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
                raise ValueError(f"invalid tensor {name}")
            value = np.array(value, dtype=np.float32, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"FP32 overflow in {name}")
            value.flags.writeable = False
            self.weights[name] = value
        self.reset()

    @classmethod
    def from_directory(cls, directory, *, max_weight_bytes=512*1024*1024,
                       read_limits=None, cancel=None):
        """Load admitted local shards without downloads or checkpoint code."""
        from .native_checkpoint import read_native_text_checkpoint, _cancelled
        config, weights = read_native_text_checkpoint(
            directory, model_type='mixtral', max_weight_bytes=max_weight_bytes,
            read_limits=read_limits, cancel=cancel,source_backend='native')
        _cancelled(cancel)
        decoder = cls(config, weights)
        _cancelled(cancel)
        return decoder

    def reset(self):
        self.position = 0
        self.cache = []
        self.mask = None

    def status(self):
        return dict(model_type='mixtral', implementation=self.implementation,
                    native_range_execution=self.native_range_execution, position=self.position)

    def forward(self, token_ids, *, attention_mask=None, position_ids=None):
        c = self.config
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= c.vocab_size):
            raise ValueError("invalid token_ids")
        batch, length = ids.shape
        end = self.position + length
        if end > c.max_position_embeddings:
            raise ValueError("context capacity exceeded")
        if self.cache and batch != self.cache[0][0].shape[0]:
            raise ValueError("cache batch mismatch; reset first")
        mask = np.ones((batch, end), dtype=bool) if attention_mask is None else np.asarray(attention_mask)
        if mask.shape != (batch, end) or not np.isin(mask, [0, 1]).all():
            raise ValueError("invalid attention_mask")
        mask = mask.astype(bool)
        if self.mask is not None and not np.array_equal(mask[:, :self.position], self.mask):
            raise ValueError("cached padding mask changed")
        positions = np.broadcast_to(np.arange(self.position, end), ids.shape) if position_ids is None else np.asarray(position_ids)
        if positions.shape != ids.shape or positions.dtype.kind not in 'iu' or np.any(positions < 0) or np.any(positions >= c.max_position_embeddings):
            raise ValueError("invalid position_ids")
        w = self.weights
        x = w['model.embed_tokens.weight'][ids].copy()
        hidden, routers, routes, attentions, caches = [], [], [], [], []
        for i in range(c.num_hidden_layers):
            hidden.append(x.copy())
            p = f'model.layers.{i}.'
            z = rmsnorm(x, w[p+'input_layernorm.weight'], c.rms_norm_eps)
            def projection(name, heads):
                return (z @ w[p+'self_attn.'+name+'_proj.weight'].T).reshape(batch, length, heads, c.head_dim)
            q = rotary(projection('q', c.num_attention_heads), positions, c.rope_theta).transpose(0, 2, 1, 3)
            k = rotary(projection('k', c.num_key_value_heads), positions, c.rope_theta).transpose(0, 2, 1, 3)
            v = projection('v', c.num_key_value_heads).transpose(0, 2, 1, 3)
            if self.cache:
                k = np.concatenate((self.cache[i][0], k), axis=2)
                v = np.concatenate((self.cache[i][1], v), axis=2)
            caches.append((k, v))
            groups = c.num_attention_heads // c.num_key_value_heads
            keys, values = np.repeat(k, groups, axis=1), np.repeat(v, groups, axis=1)
            scores = (q @ keys.swapaxes(-1, -2)) / np.float32(np.sqrt(c.head_dim))
            slots = np.arange(end)
            query_slots = np.arange(self.position, end)
            allowed = slots[None, :] <= query_slots[:, None]
            if c.sliding_window is not None:
                allowed &= slots[None, :] > query_slots[:, None] - c.sliding_window
            allowed = allowed[None, None, :, :] & mask[:, None, None, :]
            probabilities = softmax(np.where(allowed, scores, np.finfo(np.float32).min))
            attentions.append(probabilities)
            context = (probabilities @ values).transpose(0, 2, 1, 3).reshape(batch, length, -1)
            x = x + context @ w[p+'self_attn.o_proj.weight'].T
            z = rmsnorm(x, w[p+'post_attention_layernorm.weight'], c.rms_norm_eps).reshape(-1, c.hidden_size)
            router = z @ w[p+'block_sparse_moe.gate.weight'].T
            routers.append(router)
            prob = softmax(router)
            selected = np.argsort(-prob, axis=-1, kind='stable')[:, :c.num_experts_per_tok]
            weights = np.take_along_axis(prob, selected, axis=-1)
            weights /= weights.sum(axis=-1, keepdims=True)
            routes.append((selected, weights))
            output = np.zeros_like(z)
            for expert in range(c.num_local_experts):
                rows, ranks = np.nonzero(selected == expert)
                if not rows.size:
                    continue
                ep = p + f'block_sparse_moe.experts.{expert}.'
                gate = z[rows] @ w[ep+'w1.weight'].T
                sigmoid = np.exp(-np.logaddexp(np.float32(0), -gate))
                result = (gate * sigmoid * (z[rows] @ w[ep+'w3.weight'].T)) @ w[ep+'w2.weight'].T
                output[rows] += result * weights[rows, ranks, None]
            x = x + output.reshape(x.shape)
        x = rmsnorm(x, w['model.norm.weight'], c.rms_norm_eps)
        hidden.append(x.copy())
        logits = x @ w['lm_head.weight'].T
        self.cache, self.mask, self.position = caches, mask.copy(), end
        return SimpleNamespace(logits=logits, hidden_states=tuple(hidden), router_logits=tuple(routers),
                               routes=tuple(routes), attentions=tuple(attentions))

    def generate_ids(self, prompt_ids, *, max_tokens):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        ids = np.asarray(prompt_ids)
        if ids.ndim != 1 or not ids.size or ids.size + max_tokens - 1 > self.config.max_position_embeddings:
            raise ValueError("invalid generation length")
        self.reset()
        logits = self.forward(ids).logits[0, -1]
        generated = []
        for step in range(max_tokens):
            token = int(np.argmax(logits))
            if token == getattr(self.config, 'eos_token_id', None):
                break
            generated.append(token)
            if step + 1 < max_tokens:
                logits = self.forward([token]).logits[0, -1]
        return generated
