"""Original bounded FP32 V4 equations; no Transformers dependency at runtime.

The state is a token log, not a compressed KV cache. Calls replay that log and
commit only after successful finite inference. See ENGINE_NATIVE_DEEPSEEK_V4.md.
"""

from threading import RLock
from types import MappingProxyType, SimpleNamespace

import numpy as np


KINDS = ("sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention")
GLOBAL_NAMES = ("model.embed_tokens.weight", "model.norm.weight", "model.hc_head.hc_fn",
                "model.hc_head.hc_base", "model.hc_head.hc_scale", "lm_head.weight")


def layer_spec(c, kind, routing):
    h, a, d, q, g, o, u = (c.hidden_size, c.num_attention_heads, c.head_dim,
                            c.q_lora_rank, c.o_groups, c.o_lora_rank, c.hc_mult)
    e, m, i, j = c.n_routed_experts, c.moe_intermediate_size, c.index_head_dim, c.index_n_heads
    spec = [("input_layernorm.weight", (h,)), ("post_attention_layernorm.weight", (h,)),
            ("self_attn.q_a_proj.weight", (q, h)), ("self_attn.q_a_norm.weight", (q,)),
            ("self_attn.q_b_proj.weight", (a*d, q)), ("self_attn.kv_proj.weight", (d, h)),
            ("self_attn.kv_norm.weight", (d,)), ("self_attn.o_a_proj.weight", (g*o, a*d//g)),
            ("self_attn.o_b_proj.weight", (h, g*o)), ("self_attn.sinks", (a,))]
    rate = c.compress_rates.get(kind, 1)
    width = d * (2 if kind == KINDS[1] else 1)
    spec += [("self_attn.compressor."+name, shape) if kind != KINDS[0] else (None, ())
             for name, shape in (("kv_proj.weight", (width, h)), ("gate_proj.weight", (width, h)),
                                 ("position_bias", (rate, width)), ("kv_norm.weight", (d,)))]
    spec += [("self_attn.compressor.indexer."+name, shape) if kind == KINDS[1] else (None, ())
             for name, shape in (("kv_proj.weight", (2*i, h)), ("gate_proj.weight", (2*i, h)),
                                 ("position_bias", (rate, 2*i)), ("kv_norm.weight", (i,)),
                                 ("q_b_proj.weight", (j*i, q)), ("scorer.weights_proj.weight", (j, h)))]
    for site in ("attn_hc", "ffn_hc"):
        spec += [(site+".fn", ((u+2)*u, u*h)), (site+".base", ((u+2)*u,)), (site+".scale", (3,))]
    spec += [("mlp.gate.weight", (e, h)),
             ("mlp.gate.e_score_correction_bias", (e,)) if routing == "moe" else (None, ()),
             ("mlp.experts.gate_up_proj", (e, 2*m, h)), ("mlp.experts.down_proj", (e, h, m)),
             ("mlp.shared_experts.gate_proj.weight", (m, h)),
             ("mlp.shared_experts.up_proj.weight", (m, h)),
             ("mlp.shared_experts.down_proj.weight", (h, m))]
    return spec


def rms(x, epsilon, weight=None):
    y = x * np.float32(1 / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + epsilon))
    return y if weight is None else y * weight


def sigmoid(x):
    return np.exp(-np.logaddexp(np.float32(0), -x))


def softmax(x, axis=-1):
    z = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return z / np.sum(z, axis=axis, keepdims=True)


def rotate(x, position, theta, rotary):
    y = x.copy()
    angle = np.float32(position) / np.power(np.float32(theta),
                  np.arange(0, rotary, 2, dtype=np.float32) / np.float32(rotary))
    co, si = np.cos(angle), np.sin(angle)
    even, odd = x[..., -rotary::2], x[..., -rotary+1::2]
    y[..., -rotary::2] = even*co - odd*si
    y[..., -rotary+1::2] = even*si + odd*co
    return y


def choose(scores, count):
    if scores.ndim != 1 or type(count) is not int or not 1 <= count <= len(scores):
        raise ValueError("invalid top-k geometry")
    if not np.isfinite(scores).all():
        raise FloatingPointError("nonfinite top-k scores; state not committed")
    order = np.argsort(-scores, kind="stable")
    if count < len(order) and scores[order[count-1]] == scores[order[count]]:
        raise ValueError("top-k cutoff tie is outside the exact subset")
    return order[:count]


def positive_fp32(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"invalid {name}: expected positive FP32 scalar")
    with np.errstate(over="ignore", under="ignore"):
        converted = np.float32(value)
    if not np.isfinite(converted) or converted <= 0:
        raise ValueError(f"invalid {name}: expected positive FP32 scalar")
    return float(converted)


class IndependentDeepseekV4Decoder:
    implementation = "independent-numpy-deepseek-v4-token-replay"
    native_range_execution = False
    native_streamed_range_execution = False

    def __init__(self, config, tensors, *, max_context=128, max_bytes=512*1024*1024):
        raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        self._lock = RLock()
        if type(max_context) is not int or not 1 <= max_context <= 256:
            raise ValueError("max_context must be in 1..256")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_context, self.max_bytes = max_context, max_bytes
        defaults = dict(num_key_value_heads=1, n_shared_experts=1, hidden_act="silu",
                        scoring_func="sqrtsoftplus", norm_topk_prob=True, hc_eps=1e-6,
                        rms_norm_eps=1e-6, hc_sinkhorn_iters=20, swiglu_limit=10.,
                        routed_scaling_factor=1.5, attention_bias=False, mlp_bias=False,
                        attention_dropout=0., router_jitter_noise=0., num_nextn_predict_layers=0,
                        tie_word_embeddings=False, max_position_embeddings=max_context)
        cdict = defaults | raw
        required = ("hidden_size", "num_attention_heads", "head_dim", "q_lora_rank", "o_groups",
                    "o_lora_rank", "index_head_dim", "index_n_heads", "index_topk", "n_routed_experts",
                    "num_experts_per_tok", "moe_intermediate_size", "hc_mult", "num_hidden_layers",
                    "vocab_size", "qk_rope_head_dim", "sliding_window", "layer_types", "mlp_layer_types",
                    "compress_rates", "rope_parameters")
        if any(name not in cdict for name in required):
            raise ValueError("missing V4 numerical configuration")
        if cdict.get("model_type", "deepseek_v4") != "deepseek_v4":
            raise ValueError("model_type must be deepseek_v4")
        c = SimpleNamespace(**cdict)
        bounds = dict(hidden_size=128, num_attention_heads=8, head_dim=64, q_lora_rank=128,
                      o_groups=8, o_lora_rank=128, index_head_dim=64, index_n_heads=8,
                      index_topk=256, n_routed_experts=32, num_experts_per_tok=8,
                      moe_intermediate_size=256, hc_mult=4, num_hidden_layers=8,
                      vocab_size=4096, qk_rope_head_dim=64, sliding_window=256,
                      hc_sinkhorn_iters=64, max_position_embeddings=1048576)
        for name, bound in bounds.items():
            value = getattr(c, name)
            if type(value) is not int or not 1 <= value <= bound:
                raise ValueError(f"unsupported {name}: expected 1..{bound}")
        for name, value in dict(num_key_value_heads=1, n_shared_experts=1,
                                scoring_func="sqrtsoftplus", norm_topk_prob=True,
                                attention_bias=False, mlp_bias=False, attention_dropout=0.,
                                router_jitter_noise=0., num_nextn_predict_layers=0).items():
            if getattr(c, name) != value:
                raise ValueError(f"unsupported {name}")
        if type(c.tie_word_embeddings) is not bool:
            raise ValueError("tie_word_embeddings must be boolean")
        if c.hidden_act not in ("silu", "relu"):
            raise ValueError("unsupported hidden_act")
        if raw.get("quantization_config") is not None:
            raise ValueError("quantization is unsupported")
        if (c.num_attention_heads % c.o_groups or c.qk_rope_head_dim % 2 or
                c.qk_rope_head_dim > min(c.head_dim, c.index_head_dim) or
                c.num_experts_per_tok > c.n_routed_experts or max_context > c.max_position_embeddings):
            raise ValueError("inconsistent V4 geometry")
        kinds, routing = tuple(c.layer_types), tuple(c.mlp_layer_types)
        if len(kinds) != c.num_hidden_layers or len(routing) != len(kinds):
            raise ValueError("layer schedule length mismatch")
        if any(k not in KINDS for k in kinds) or any(k not in ("moe", "hash_moe") for k in routing):
            raise ValueError("unsupported layer schedule")
        rates = dict(c.compress_rates)
        for kind in KINDS[1:]:
            if type(rates.get(kind)) is not int or not 1 <= rates[kind] <= 256:
                raise ValueError("compression rate must be in 1..256")
        thetas = []
        for name in ("main", "compress"):
            p = c.rope_parameters.get(name, {})
            if p.get("rope_type") != "default" or set(p) - {"rope_type", "rope_theta", "partial_rotary_factor"}:
                raise ValueError("only default unscaled RoPE is supported")
            if int(c.head_dim*p.get("partial_rotary_factor", 1.)) != c.qk_rope_head_dim:
                raise ValueError("rotary metadata disagrees with qk_rope_head_dim")
            theta = p.get("rope_theta", 0.)
            thetas.append(positive_fp32(theta, "RoPE theta"))
        for name in ("rms_norm_eps", "hc_eps", "swiglu_limit", "routed_scaling_factor"):
            value = getattr(c, name)
            converted = positive_fp32(value, name)
            if value > 1e6:
                raise ValueError(f"invalid {name}")
            setattr(c, name, converted)
        c.layer_types, c.mlp_layer_types = kinds, routing
        c.compress_rates = MappingProxyType(rates)
        c.rope_parameters = MappingProxyType({k: MappingProxyType(dict(v))
                              for k, v in c.rope_parameters.items() if isinstance(v, dict)})
        self.config = c
        self.thetas = tuple(thetas)
        h, u = c.hidden_size, c.hc_mult
        specs = dict(zip(GLOBAL_NAMES, ((c.vocab_size, h), (h,), (u, u*h), (u,), (1,), (c.vocab_size, h))))
        self.layer_specs = []
        for index, (kind, mode) in enumerate(zip(kinds, routing)):
            spec = layer_spec(c, kind, mode)
            self.layer_specs.append(spec)
            prefix = f"model.layers.{index}."
            specs.update({prefix+n: shape for n, shape in spec if n})
            if mode == "hash_moe":
                specs[prefix+"mlp.gate.tid2eid"] = (c.vocab_size, c.num_experts_per_tok)
        expected = set(specs)
        if c.tie_word_embeddings and "lm_head.weight" not in tensors:
            expected.remove("lm_head.weight")
        if set(tensors) != expected:
            raise ValueError(f"tensor manifest mismatch; missing={sorted(expected-set(tensors))}; "
                             f"unexpected={sorted(set(tensors)-set(specs))}")
        resident = sum(int(np.prod(shape))* (8 if n.endswith("tid2eid") else 4)
                       for n, shape in specs.items() if not (c.tie_word_embeddings and n == "lm_head.weight"))
        if 2*resident + self.workspace_bytes(max_context) > max_bytes:
            raise MemoryError("resident V4 construction exceeds max_bytes")
        weights = {}
        for name, shape in specs.items():
            if c.tie_word_embeddings and name == "lm_head.weight" and name not in tensors:
                weights[name] = weights["model.embed_tokens.weight"]
                continue
            value = tensors[name]
            if hasattr(value, "detach"):
                value = value.detach().numpy()
            value = np.asarray(value)
            dtype = np.int64 if name.endswith("tid2eid") else np.float32
            if value.dtype != dtype or value.shape != shape:
                raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite tensor {name}")
            if c.tie_word_embeddings and name == "lm_head.weight":
                if not np.array_equal(value, weights["model.embed_tokens.weight"]):
                    raise ValueError("tied lm_head.weight must equal model.embed_tokens.weight exactly")
                weights[name] = weights["model.embed_tokens.weight"]
                continue
            if dtype == np.int64:
                if (value < 0).any() or (value >= c.n_routed_experts).any():
                    raise ValueError("hash expert outside range")
                if any(len(set(row)) != len(row) for row in value):
                    raise ValueError("duplicate hash experts are outside the exact subset")
            # Immutable bytes backing prevents callers from re-enabling writes.
            weights[name] = np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(shape)
        self.weights = MappingProxyType(weights)
        self.resident_bytes = resident
        self._history = np.empty(0, dtype=np.int64)

    def workspace_bytes(self, tokens):
        c = self.config
        # Conservative bound covers arena, NumPy intermediates, output and stack.
        return 4*(tokens*(8*c.hc_mult*c.hidden_size + 16*c.hidden_size +
                  8*c.num_attention_heads*c.head_dim + 8*c.q_lora_rank +
                  24*c.head_dim + 24*c.index_head_dim + 4*c.index_n_heads*c.index_head_dim +
                  8*c.hc_mult*c.hc_mult + 12*c.moe_intermediate_size +
                  8*c.n_routed_experts + 4*c.vocab_size) + 65536) + tokens*32

    @property
    def position(self):
        return len(self._history)

    def reset(self):
        with self._lock:
            self._history = np.empty(0, dtype=np.int64)

    def status(self):
        with self._lock:
            return dict(model_type="deepseek_v4", implementation=self.implementation,
                        position=self.position, max_context=self.max_context,
                        cache_layout="transactional-token-history-replay",
                        native_range_execution=False, native_streamed_range_execution=False)

    def forward(self, token_ids, *, attention_mask=None, position_ids=None):
        with self._lock:
            if attention_mask is not None or position_ids is not None:
                raise ValueError("only unmasked contiguous positions from zero are supported")
            ids = np.asarray(token_ids)
            if ids.ndim == 2 and ids.shape[0] == 1:
                ids = ids[0]
            if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
                raise ValueError("token_ids must be a nonempty integer sequence, batch size one")
            if (ids < 0).any() or (ids >= self.config.vocab_size).any():
                raise ValueError("token outside vocabulary")
            end = self.position + len(ids)
            if end > self.max_context:
                raise ValueError("context capacity exceeded")
            if self.resident_bytes + self.workspace_bytes(end) > self.max_bytes:
                raise MemoryError("resident V4 call exceeds max_bytes")
            pending = np.concatenate((self._history, ids.astype(np.int64)))
            logits = self._evaluate(pending)
            if not np.isfinite(logits).all():
                raise FloatingPointError("nonfinite V4 output; state not committed")
            output = SimpleNamespace(logits=logits[None, self.position:].copy())
            self._history = pending
            return output

    def _hyper(self, streams, w, head=False):
        c, u = self.config, self.config.hc_mult
        mix = rms(streams.reshape(len(streams), -1), c.rms_norm_eps) @ w[0].T
        if head:
            pre = sigmoid(mix*w[2][0] + w[1]) + c.hc_eps
            return np.sum(streams*pre[..., None], axis=1)
        pre = sigmoid(mix[:, :u]*w[2][0] + w[1][:u]) + c.hc_eps
        post = 2*sigmoid(mix[:, u:2*u]*w[2][1] + w[1][u:2*u])
        comb = softmax((mix[:, 2*u:]*w[2][2] + w[1][2*u:]).reshape(-1, u, u)) + c.hc_eps
        comb /= np.sum(comb, axis=1, keepdims=True) + c.hc_eps
        for _ in range(c.hc_sinkhorn_iters-1):
            comb /= np.sum(comb, axis=2, keepdims=True) + c.hc_eps
            comb /= np.sum(comb, axis=1, keepdims=True) + c.hc_eps
        return np.sum(streams*pre[..., None], axis=1), post, comb

    def _pool(self, x, w, rate, width, overlap, theta):
        c, n = self.config, len(x)//rate
        projections, gates = x @ w[0].T, x @ w[1].T
        result = np.empty((n, width), dtype=np.float32)
        for entry in range(n):
            start = entry*rate
            if overlap:
                values = projections[start:start+rate, width:]
                scores = gates[start:start+rate, width:] + w[2][:, width:]
                if entry:
                    values = np.concatenate((projections[start-rate:start, :width], values))
                    scores = np.concatenate((gates[start-rate:start, :width] + w[2][:, :width], scores))
            else:
                values = projections[start:start+rate]
                scores = gates[start:start+rate] + w[2]
            pooled = np.sum(values*softmax(scores, axis=0), axis=0)
            result[entry] = rotate(rms(pooled, c.rms_norm_eps, w[3]), start, theta, c.qk_rope_head_dim)
        return result

    def _attention(self, x, w, kind):
        c, n = self.config, len(x)
        a, d, r = c.num_attention_heads, c.head_dim, c.qk_rope_head_dim
        theta = self.thetas[0 if kind == KINDS[0] else 1]
        qr = rms(x @ w[2].T, c.rms_norm_eps, w[3])
        q = rms((qr @ w[4].T).reshape(n, a, d), c.rms_norm_eps)
        kv = rms(x @ w[5].T, c.rms_norm_eps, w[6])
        for t in range(n):
            q[t], kv[t] = rotate(q[t], t, theta, r), rotate(kv[t], t, theta, r)
        pool = index = None
        if kind != KINDS[0]:
            rate = c.compress_rates[kind]
            pool = self._pool(x, w[10:14], rate, d, kind == KINDS[1], theta)
            if kind == KINDS[1]:
                index = self._pool(x, w[14:18], rate, c.index_head_dim, True, theta)
                iq = (qr @ w[18].T).reshape(n, c.index_n_heads, c.index_head_dim)
                iw = x @ w[19].T
        out = np.empty((n, a*d), dtype=np.float32)
        for t in range(n):
            keys = kv[max(0, t+1-c.sliding_window):t+1]
            if pool is not None:
                ready = (t+1)//rate
                selected = np.arange(ready)
                if index is not None and ready:
                    scores = np.sum(np.maximum(rotate(iq[t], t, theta, r) @ index[:ready].T, 0)
                              * (iw[t]/np.sqrt(np.float32(c.index_n_heads)))[:, None], axis=0)
                    scores /= np.sqrt(np.float32(c.index_head_dim))
                    selected = choose(scores, min(c.index_topk, ready))
                keys = np.concatenate((keys, pool[selected]))
            scores = q[t] @ keys.T / np.sqrt(np.float32(d))
            probabilities = softmax(np.concatenate((scores, w[9][:, None]), axis=1))[:, :-1]
            out[t] = rotate(probabilities @ keys, -t, theta, r).reshape(-1)
        g, o = c.o_groups, c.o_lora_rank
        grouped = np.concatenate([out[:, k*a*d//g:(k+1)*a*d//g] @ w[7][k*o:(k+1)*o].T
                                  for k in range(g)], axis=1)
        return grouped @ w[8].T

    def _moe(self, x, w, ids, table):
        c, m = self.config, self.config.moe_intermediate_size
        scores = np.sqrt(np.logaddexp(np.float32(0), x @ w[26].T))
        limit = c.swiglu_limit
        def activated(gate, up):
            gate = np.minimum(gate, limit)
            active = np.maximum(gate, np.float32(0)) if c.hidden_act == "relu" else gate*sigmoid(gate)
            return active*np.clip(up, -limit, limit)
        result = activated(x @ w[30].T, x @ w[31].T) @ w[32].T
        for t in range(len(x)):
            selected = table[ids[t]] if table is not None else choose(scores[t]+w[27], c.num_experts_per_tok)
            weights = scores[t, selected]
            weights = weights/(weights.sum()+np.float32(1e-20))*c.routed_scaling_factor
            for expert, weight in zip(selected, weights):
                gu = w[28][expert] @ x[t]
                result[t] += (w[29][expert] @ activated(gu[:m], gu[m:]))*weight
        return result

    def _evaluate(self, ids):
        c = self.config
        streams = np.repeat(self.weights[GLOBAL_NAMES[0]][ids, None, :], c.hc_mult, axis=1)
        for i, (kind, mode) in enumerate(zip(c.layer_types, c.mlp_layer_types)):
            prefix = f"model.layers.{i}."
            w = [self.weights[prefix+n] if n else None for n, _ in self.layer_specs[i]]
            for site in (0, 1):
                offset = 20+3*site
                collapsed, post, comb = self._hyper(streams, w[offset:offset+3])
                normalized = rms(collapsed, c.rms_norm_eps, w[site])
                branch = self._attention(normalized, w, kind) if site == 0 else self._moe(
                    normalized, w, ids, self.weights.get(prefix+"mlp.gate.tid2eid"))
                streams = np.einsum("njk,njd->nkd", comb, streams) + post[..., None]*branch[:, None, :]
        head = self._hyper(streams, [self.weights[k] for k in GLOBAL_NAMES[2:5]], head=True)
        return rms(head, c.rms_norm_eps, self.weights[GLOBAL_NAMES[1]]) @ self.weights[GLOBAL_NAMES[5]].T
