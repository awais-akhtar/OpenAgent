"""Independent GLM5-next text contract, tensor layout and transaction ownership.

There is deliberately no Python numerical decoder. Execution requires the
separate compiled implementation; NumPy is used only for owned FP32 buffers.
"""

from dataclasses import dataclass
from collections.abc import Mapping
import math
from types import MappingProxyType
import numpy as np


@dataclass(frozen=True)
class Glm5NextTextContract:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    hc_mult: int
    hc_sinkhorn_iters: int
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    hc_eps: float
    swiglu_limit: float
    routed_scaling_factor: float
    linear_lower_bound: float | None
    norm_topk_prob: bool
    tie_word_embeddings: bool
    layer_types: tuple
    mlp_layer_types: tuple
    indexer_types: tuple

    @classmethod
    def parse(cls, config):
        data = config.to_dict() if hasattr(config, "to_dict") else config
        if not isinstance(data, Mapping) or data.get("model_type") != "glm5_next_text":
            raise ValueError("requires explicit glm5_next_text config, not an outer/other-family config")
        data = dict(data)
        # The current source contract has no pinned Transformers architecture
        # class for this specialized GLM5 text route.  Do not silently accept
        # a declared class merely because `architectures` is treated as
        # metadata below: architecture-bearing checkpoints must use a future
        # exact-class route once that mapping is reviewed.
        if "architectures" in data:
            raise ValueError("GLM5 text route does not admit declared architectures")
        required = set(cls.__dataclass_fields__)
        missing = required - data.keys()
        if missing:
            raise ValueError(f"incomplete text config: {sorted(missing)}")
        metadata = {
            "transformers_version", "architectures", "output_hidden_states", "return_dict", "dtype",
            "chunk_size_feed_forward", "is_encoder_decoder", "id2label", "label2id", "problem_type",
            "initializer_range", "use_cache", "pad_token_id", "bos_token_id", "eos_token_id",
            "output_router_logits", "router_aux_loss_coef", "_name_or_path", "output_attentions",
            "model_type", "num_key_value_heads", "hidden_act", "qk_rope_head_dim", "attention_bias",
            "attention_dropout", "n_group", "topk_group", "index_kpool_always_select_tail", "head_dim",
            "qk_head_dim",
        }
        unknown = data.keys() - required - metadata
        if unknown:
            raise ValueError(f"unsupported config fields: {sorted(unknown)}")
        for name, expected in {
            "hidden_act": "silu", "qk_rope_head_dim": 0,
            "attention_bias": False, "attention_dropout": 0.0,
            "index_kpool_always_select_tail": True,
        }.items():
            if data.get(name) != expected:
                raise ValueError(f"unsupported {name}: expected {expected!r}")
        if data.get("num_key_value_heads") != data["num_attention_heads"]:
            raise ValueError("unequal attention and KV heads")
        for name in ("output_router_logits", "output_hidden_states", "output_attentions", "is_encoder_decoder"):
            if data.get(name, False) is not False:
                raise ValueError(f"unsupported {name}")
        if data.get("head_dim",0) != 0 or data.get("qk_head_dim",data["qk_nope_head_dim"]) != data["qk_nope_head_dim"]:
            raise ValueError("inconsistent NoPE head metadata")
        if data.get("dtype") not in (None,"float32"):
            raise ValueError("native execution requires FP32 config dtype")
        # Do not accept alternate runtime contracts hidden in checkpoint extras.
        for name in ("quantization_config", "rope_parameters", "rope_scaling", "linear_attn_config",
                     "index_topk_pattern", "index_topk_freq", "index_skip_topk_offset"):
            if data.get(name) is not None:
                raise ValueError(f"unsupported alternate config field {name}")
        values = {name: data[name] for name in required}
        int_fields = [name for name in required if name not in {
            "rms_norm_eps", "hc_eps", "swiglu_limit", "routed_scaling_factor", "linear_lower_bound",
            "norm_topk_prob", "tie_word_embeddings", "layer_types", "mlp_layer_types", "indexer_types",
        }]
        for name in int_fields:
            value = values[name]
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            limits = dict(hidden_size=16384,vocab_size=1048576,num_hidden_layers=128,
                hc_mult=8,hc_sinkhorn_iters=64,intermediate_size=65536,moe_intermediate_size=65536,
                n_routed_experts=4096,n_shared_experts=32,num_experts_per_tok=4096,n_group=4096,
                topk_group=4096,num_attention_heads=256,q_lora_rank=8192,kv_lora_rank=8192,
                qk_nope_head_dim=512,v_head_dim=512,index_n_heads=256,index_head_dim=512,
                index_topk=65536,index_kpool=32,linear_num_heads=256,linear_head_dim=512,
                linear_conv_kernel_dim=32,max_position_embeddings=1048576)
            limit = limits[name]
            if value > limit:
                raise ValueError(f"{name} exceeds bounded native geometry ({limit})")
        if values["num_hidden_layers"] > 128 or values["hc_mult"] > 8 or values["hc_sinkhorn_iters"] > 64:
            raise ValueError("layer/MHC geometry exceeds native bounds")
        if values["linear_conv_kernel_dim"] > 32 or values["index_kpool"] > 32:
            raise ValueError("convolution/pool geometry exceeds native bounds")
        if values["index_topk"] % values["index_kpool"]:
            raise ValueError("index_topk must be divisible by index_kpool")
        if not 1 <= values["num_experts_per_tok"] <= values["n_routed_experts"]:
            raise ValueError("invalid expert selection count")
        if values["n_routed_experts"] < 2:
            raise ValueError("router requires at least two experts per group")
        if (values["n_routed_experts"] % values["n_group"] or
                values["n_routed_experts"] // values["n_group"] < 2 or
                values["topk_group"] > values["n_group"] or
                values["num_experts_per_tok"] >
                values["topk_group"] * (values["n_routed_experts"] // values["n_group"])):
            raise ValueError("invalid grouped expert selection geometry")
        for name in ("rms_norm_eps", "hc_eps", "swiglu_limit", "routed_scaling_factor"):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")
            if value < 1e-12 or value > 1e12:
                raise ValueError(f"{name} exceeds bounded scalar range")
        bound = values["linear_lower_bound"]
        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (float, int)) or not math.isfinite(bound) or not -1e12 <= bound <= -1e-12):
            raise ValueError("linear_lower_bound must be finite negative or None")
        for name in ("norm_topk_prob", "tie_word_embeddings"):
            if type(values[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name, choices in (("layer_types", {"linear_attention", "deepseek_sparse_attention"}),
                              ("mlp_layer_types", {"dense", "sparse"}), ("indexer_types", {"full", "shared"})):
            schedule = values[name]
            if not isinstance(schedule, (list, tuple)) or len(schedule) != values["num_hidden_layers"] or any(v not in choices for v in schedule):
                raise ValueError(f"invalid explicit {name}")
            values[name] = tuple(schedule)
        for i, kind in enumerate(values["layer_types"]):
            if kind == "deepseek_sparse_attention" and values["indexer_types"][i] == "shared":
                if i == 0 or values["layer_types"][i-1] != "deepseek_sparse_attention" or values["indexer_types"][i-1] != "full":
                    raise ValueError("shared DSA requires an immediately preceding full DSA layer")
        return cls(**values)


def tensor_layout(c):
    """Canonical text-model state_dict names. Each layer has 42 private ABI slots."""
    result = {"embed_tokens.weight": (c.vocab_size, c.hidden_size), "norm.weight": (c.hidden_size,)}
    slots = []
    h, r = c.hidden_size, c.hc_mult
    for i in range(c.num_hidden_layers):
        p = f"layers.{i}."
        spec = [None] * 42
        def add(slot, name, shape):
            result[p + name] = shape
            spec[slot] = p + name
        for base, start in (("attn_hc.", 0), ("ffn_hc.", 3)):
            add(start, base + "fn", ((r+2)*r, r*h))
            add(start+1, base + "base", ((r+2)*r,))
            add(start+2, base + "scale", (3,))
        add(6, "input_layernorm.weight", (h,))
        add(7, "post_attention_layernorm.weight", (h,))
        a = "self_attn."
        if c.layer_types[i] == "linear_attention":
            d, n, k = c.linear_head_dim, c.linear_num_heads, c.linear_conv_kernel_dim
            w = d*n
            for slot, name in enumerate(("q_proj", "k_proj", "v_proj"), 8):
                add(slot, a+name+".weight", (w, h))
            add(11, a+"conv1d.weight", (3*w, 1, k))
            add(12, a+"forget_gate.f_a_proj.weight", (d, h))
            add(13, a+"forget_gate.f_b_proj.weight", (w, d))
            add(14, a+"forget_gate.dt_bias", (w,))
            add(15, a+"forget_gate.A_log", (n,))
            add(16, a+"b_proj.weight", (n, h))
            add(17, a+"g_a_proj.weight", (d, h))
            add(18, a+"g_b_proj.weight", (w, d))
            add(19, a+"o_norm.weight", (d,))
            add(20, a+"o_proj.weight", (h, w))
        else:
            n, d, v = c.num_attention_heads, c.qk_nope_head_dim, c.v_head_dim
            for slot, name, shape in (
                (8, "q_a_proj.weight", (c.q_lora_rank, h)), (9, "q_a_layernorm.weight", (c.q_lora_rank,)),
                (10, "q_b_proj.weight", (n*d, c.q_lora_rank)),
                (11, "kv_a_proj_with_mqa.weight", (c.kv_lora_rank, h)), (12, "kv_a_layernorm.weight", (c.kv_lora_rank,)),
                (13, "kv_b_proj.weight", (n*(d+v), c.kv_lora_rank)), (14, "o_proj.weight", (h, n*v)),
            ):
                add(slot, a+name, shape)
            if c.indexer_types[i] == "full":
                d, n = c.index_head_dim, c.index_n_heads
                for slot, name, shape in (
                    (15, "wq_b.weight", (n*d, c.q_lora_rank)), (16, "wk.weight", (d, h)),
                    (17, "k_norm.weight", (d,)), (18, "k_norm.bias", (d,)), (19, "weights_proj.weight", (n, h)),
                    (20, "index_kpool_compress_ape", (c.index_kpool, d)), (21, "index_kpool_compress_gate", (d, h)),
                ):
                    add(slot, a+"indexer."+name, shape)
        if c.mlp_layer_types[i] == "dense":
            for slot, name, shape in ((24, "gate_proj.weight", (c.intermediate_size,h)),
                                      (25, "up_proj.weight", (c.intermediate_size,h)),
                                      (26, "down_proj.weight", (h,c.intermediate_size))):
                add(slot, "mlp."+name, shape)
        else:
            e, m, s = c.n_routed_experts, c.moe_intermediate_size, c.n_shared_experts*c.moe_intermediate_size
            for slot, name, shape in (
                (24, "gate.weight", (e,h)), (25, "gate.e_score_correction_bias", (e,)),
                (26, "experts.gate_up_proj", (e,2*m,h)), (27, "experts.down_proj", (e,h,m)),
                (28, "shared_experts.gate_proj.weight", (s,h)), (29, "shared_experts.up_proj.weight", (s,h)),
                (30, "shared_experts.down_proj.weight", (h,s)),
            ):
                add(slot, "mlp."+name, shape)
        slots.append(tuple(spec))
    return result, tuple(slots)


class IndependentGlm5NextTextDecoder:
    """Buffer/contract owner, not an alternative arithmetic backend."""

    def __init__(self, config, tensors, *, max_tokens=256, max_bytes=512*1024*1024,_omit_experts=False):
        self.config = Glm5NextTextContract.parse(config)
        c = self.config
        if type(max_tokens) is not int or not 1 <= max_tokens <= min(65536, c.max_position_embeddings):
            raise ValueError("max_tokens outside bounded context")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be positive integer")
        self.max_tokens = max_tokens
        self.max_bytes = max_bytes
        layout, self._slots = tensor_layout(c)
        if _omit_experts:
            omitted={name for name in layout if name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
            layout={name:shape for name,shape in layout.items() if name not in omitted}
            self._slots=tuple(tuple(None if name in omitted else name for name in layer) for layer in self._slots)
        normalized = {}
        if not isinstance(tensors, Mapping):
            raise ValueError("tensors must be a state_dict mapping")
        for name, value in tensors.items():
            if not isinstance(name, str):
                raise ValueError("tensor names must be strings")
            key = name
            for prefix in ("model.language_model.", "language_model.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            if key in normalized:
                raise ValueError(f"duplicate canonical tensor {key}")
            normalized[key] = value
        if "lm_head.weight" in normalized:
            layout["lm_head.weight"] = (c.vocab_size, c.hidden_size)
        if set(normalized) != set(layout):
            raise ValueError(f"tensor contract mismatch: missing={sorted(set(layout)-set(normalized))}, extra={sorted(set(normalized)-set(layout))}")
        weight_bytes = sum(math.prod(shape)*4 for shape in layout.values())
        shapes = []
        for kind in c.layer_types:
            if kind == "linear_attention":
                shapes.append(((3*c.linear_num_heads*c.linear_head_dim, c.linear_conv_kernel_dim),
                               (c.linear_num_heads,c.linear_head_dim,c.linear_head_dim)))
            else:
                shapes.append(((max_tokens,c.num_attention_heads,c.qk_nope_head_dim+c.v_head_dim),
                               (max_tokens,2*c.index_head_dim)))
        self._cache_shapes = tuple(shapes)
        cache_bytes = sum(math.prod(s)*4 for pair in shapes for s in pair)
        # Include old and candidate caches, weights and bounded C workspace.
        if weight_bytes + 2*cache_bytes + 8*1024*1024 > max_bytes:
            raise ValueError("resident weight/cache/workspace memory budget exceeded")
        self.weights = {}
        for name, shape in layout.items():
            value = normalized[name]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"expected finite FP32 {name} {shape}")
            owned = np.array(value, dtype=np.float32, order="C", copy=True)
            owned.flags.writeable = False
            self.weights[name] = owned
        if c.tie_word_embeddings and "lm_head.weight" in self.weights and not np.array_equal(self.weights["lm_head.weight"], self.weights["embed_tokens.weight"]):
            raise ValueError("tied LM head differs from embedding")
        self.weights = MappingProxyType(self.weights)
        self._cache = None
        self.position = 0
        self._resident_bytes = weight_bytes + 2*cache_bytes + 8*1024*1024

    @property
    def cache(self):
        """Detached snapshots cannot mutate committed recurrent state."""
        return None if self._cache is None else tuple(tuple(a.copy() for a in pair) for pair in self._cache)

    def reset(self):
        self._cache = None
        self.position = 0

    def forward(self, *args, **kwargs):
        raise NotImplementedError("GLM5-next arithmetic requires NativeGlm5NextTextDecoder and a compiled library")

    def status(self):
        return {"model_type": "glm5_next_text", "position": self.position,
                "max_tokens": self.max_tokens, "native_range_execution": False,
                "output_scope": "text-causal-lm-head" if "lm_head.weight" in self.weights else "text-hidden-only"}
