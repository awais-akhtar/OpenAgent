"""Explicit publisher tensor tables; no shape-based or arbitrary rename inference."""
from collections.abc import Mapping
from dataclasses import dataclass

from .storage import positive
from ..publisher_source_pins import K3_REVISION, V41_REVISION

K3_SOURCE = f"https://huggingface.co/moonshotai/Kimi-K3/blob/{K3_REVISION}/modeling_kimi_linear.py"
K3_INDEX = f"https://huggingface.co/moonshotai/Kimi-K3/blob/{K3_REVISION}/model.safetensors.index.json"
V41_SOURCE = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{V41_REVISION}/inference/model.py"
V41_CONVERT = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/{V41_REVISION}/model.safetensors.index.json"
MX_SOURCE = "https://github.com/vllm-project/compressed-tensors/tree/main/src/compressed_tensors/compressors/mxfp4"
K3_IGNORES = frozenset((
    r"re:.*self_attn.*", r"re:.*shared_experts.*",
    r"re:.*mlp\.(gate|up|gate_up|down)_proj.*", r"re:.*lm_head.*",
    r"re:.*vision_tower.*", r"re:.*mm_projector.*",
))


@dataclass(frozen=True)
class UnsupportedContract:
    contract: str
    path: str
    detail: str


class UnsupportedPublisherContract(ValueError):
    def __init__(self, gaps):
        self.gaps = tuple(gaps)
        super().__init__("; ".join(g.path + ": " + g.detail for g in self.gaps[:8]))


@dataclass(frozen=True)
class MappingContract:
    target: str
    sources: tuple
    shape: tuple
    encoding: str
    evidence: tuple


def geometry(text, key, upper=2**31-1):
    value = positive(text.get(key), "text_config." + key)
    if value > upper:
        raise ValueError("publisher geometry exceeds planning bound: " + key)
    return value


def reject(path, detail):
    raise UnsupportedPublisherContract((UnsupportedContract("configuration", path, detail),))


def k3_quant(config, text):
    q = text.get("quantization_config")
    if config.get("quantization_config") is not None:
        reject("quantization_config", "K3 quantization must be nested in text_config")
    if q is None:
        return False
    if not isinstance(q, Mapping):
        raise ValueError("invalid K3 quantization configuration")
    groups = q.get("config_groups")
    if (q.get("quant_method") != "compressed-tensors" or q.get("format") != "mxfp4-pack-quantized"
            or q.get("quantization_status") != "compressed"
            or not isinstance(groups, Mapping) or set(groups) != {"group_0"}
            or set(q) - {"quant_method", "format", "quantization_status", "config_groups",
                         "global_compression_ratio", "ignore", "kv_cache_scheme"}):
        reject("text_config.quantization_config", "only explicit compressed MXFP4 group_0 is mapped")
    group = groups["group_0"]
    if not isinstance(group, Mapping) or not isinstance(group.get("weights"), Mapping):
        raise ValueError("invalid K3 weight quantization group")
    w = group["weights"]
    expected = {"group_size": 32, "num_bits": 4, "type": "float", "strategy": "group",
                "symmetric": True, "dynamic": False, "scale_dtype": "torch.uint8"}
    if (any(type(w.get(k)) is not type(v) or w.get(k) != v for k, v in expected.items())
            or group.get("format") != "mxfp4-pack-quantized"
            or group.get("targets") != ("Linear",)
            or any(group.get(k) is not None for k in ("input_activations", "output_activations"))
            or any(w.get(k) is not None for k in ("actorder", "block_structure", "zp_dtype"))
            or set(group) - {"format", "targets", "weights", "input_activations", "output_activations"}
            or set(w) - set(expected) - {"actorder", "block_structure", "zp_dtype", "observer", "observer_kwargs"}):
        reject("text_config.quantization_config.config_groups", "unsupported MXFP4 scheme")
    ignores = q.get("ignore")
    if (not isinstance(ignores, tuple) or any(type(v) is not str for v in ignores)
            or len(ignores) != len(K3_IGNORES) or frozenset(ignores) != K3_IGNORES
            or q.get("kv_cache_scheme") is not None):
        reject("text_config.quantization_config.ignore", "exact publisher exclusion policy required; changed regex policies are not inferred")
    return True


def build_contracts(config, *, max_contracts):
    family, text = config.get("model_type"), config.get("text_config")
    identities = {"kimi_k3": ("kimi_linear", "KimiK3ForConditionalGeneration"),
                  "deepseek_v41": ("deepseek_v41_text", "DeepseekV41ForCausalLM")}
    if family not in identities or not isinstance(text, Mapping):
        raise ValueError("exact publisher wrapper and nested text_config required")
    inner, architecture = identities[family]
    if text.get("model_type") != inner or config.get("architectures") != (architecture,):
        raise ValueError("publisher architecture differs from selected exact identity")
    if config.get("tie_word_embeddings") or text.get("tie_word_embeddings"):
        reject("tie_word_embeddings", "tied-weight inventory is not mapped")
    rules = []

    def add(source, target, shape, encoding="float32", evidence=None):
        if len(rules) >= max_contracts:
            raise MemoryError("publisher mapping contract count budget exceeded")
        rules.append(MappingContract(target, (source,) if isinstance(source, str) else source,
                                     shape, encoding, evidence or (
                                         (K3_SOURCE, K3_INDEX) if family == "kimi_k3" else (V41_SOURCE, V41_CONVERT))))

    h, n, vocab = (geometry(text, k, 4096 if k == "num_hidden_layers" else 2**31-1)
                   for k in ("hidden_size", "num_hidden_layers", "vocab_size"))
    if family == "kimi_k3":
        quant = k3_quant(config, text)
        linear = text.get("linear_attn_config")
        if not isinstance(linear, Mapping):
            raise ValueError("explicit K3 hybrid schedule required")
        kda, mla = linear.get("kda_layers"), linear.get("full_attn_layers")
        if (not isinstance(kda, tuple) or not isinstance(mla, tuple)
                or any(type(i) is not int or not 1 <= i <= n for i in kda + mla)
                or len(kda + mla) != n or set(kda + mla) != set(range(1, n+1)) or n not in mla):
            raise ValueError("K3 one-based schedule must partition all layers and end in MLA")
        heads, qr, kr, nope, rope, vd = (geometry(text, k) for k in (
            "num_attention_heads", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
        kh, kd, kernel = (positive(linear.get(k), "linear_attn_config." + k)
                          for k in ("num_heads", "head_dim", "short_conv_kernel_size"))
        dense = text.get("first_k_dense_replace")
        if type(dense) is not int or not 0 <= dense <= n:
            raise ValueError("explicit K3 dense-prefix layer count required")
        if text.get("moe_layer_freq", 1) != 1 or text.get("mla_use_nope") is not True:
            reject("text_config", "only contiguous MoE suffix and MLA nope profile are mapped")
        e, mid, latent, shared, di = (geometry(text, k) for k in (
            "num_experts", "moe_intermediate_size", "routed_expert_hidden_size", "num_shared_experts", "intermediate_size"))
        prefix = "language_model.model."
        add(prefix + "embed_tokens.weight", "k3.text.embedding", (vocab, h))
        add("language_model.lm_head.weight", "k3.text.lm_head", (vocab, h))
        add(prefix + "norm.weight", "k3.text.final_norm", (h,))
        attnres = text.get("attn_res_block_size") is not None
        if attnres:
            positive(text["attn_res_block_size"], "attn_res_block_size")
            add(prefix + "output_attn_res_norm.weight", "k3.text.output_res_norm", (h,))
            add(prefix + "output_attn_res_proj.weight", "k3.text.output_res_query", (1, h))
        for layer in range(n):
            p, t = prefix + f"layers.{layer}.", f"k3.text.layers.{layer}."
            for name in ("input_layernorm", "post_attention_layernorm"):
                add(p + name + ".weight", t + name, (h,))
            if attnres:
                for kind in ("self_attention", "mlp"):
                    add(p + kind + "_res_norm.weight", t + kind + "_res_norm", (h,))
                    add(p + kind + "_res_proj.weight", t + kind + "_res_query", (1, h))
            if layer + 1 in mla:
                shapes = {"q_a_proj.weight": (qr, h), "q_a_layernorm.weight": (qr,),
                          "q_b_proj.weight": (heads*(nope+rope), qr), "kv_a_proj_with_mqa.weight": (kr+rope, h),
                          "kv_a_layernorm.weight": (kr,), "kv_b_proj.weight": (heads*(nope+vd), kr),
                          "o_proj.weight": (h, heads*vd)}
                if text.get("mla_use_output_gate") is True:
                    shapes["g_proj.weight"] = (heads*vd, h)
            else:
                width = kh*kd
                shapes = {name + ".weight": (width, h) for name in ("q_proj", "k_proj", "v_proj")}
                shapes.update({name + ".weight": (width, 1, kernel)
                               for name in ("q_conv1d", "k_conv1d", "v_conv1d")})
                shapes.update({"A_log": (kh,), "dt_bias": (width,), "f_a_proj.weight": (kd, h),
                               "f_b_proj.weight": (width, kd), "b_proj.weight": (kh, h),
                               "o_norm.weight": (kd,), "o_proj.weight": (h, width)})
                if linear.get("use_full_rank_gate") is True:
                    shapes["g_proj.weight"] = (width, h)
                elif linear.get("use_full_rank_gate") is False:
                    shapes.update({"g_a_proj.weight": (kd, h), "g_b_proj.weight": (width, kd)})
                else:
                    raise ValueError("explicit KDA output gate rank required")
            for name, shape in shapes.items():
                add(p + "self_attn." + name, t + "self_attn." + name, shape)
            if layer < dense:
                for proj, shape in (("gate_proj", (di, h)), ("up_proj", (di, h)), ("down_proj", (h, di))):
                    add(p + "mlp." + proj + ".weight", t + "dense." + proj, shape)
                continue
            mp, mt = p + "block_sparse_moe.", t + "moe."
            add(mp + "gate.weight", mt + "router", (e, h))
            add(mp + "gate.e_score_correction_bias", mt + "router_bias", (e,))
            for proj, shape in (("routed_expert_down_proj", (latent, h)), ("routed_expert_up_proj", (h, latent))):
                add(mp + proj + ".weight", mt + proj, shape)
            if text.get("latent_moe_use_norm") is True:
                add(mp + "routed_expert_norm.weight", mt + "latent_norm", (latent,))
            for proj, shape in (("gate_proj", (shared*mid, h)), ("up_proj", (shared*mid, h)), ("down_proj", (h, shared*mid))):
                add(mp + "shared_experts." + proj + ".weight", mt + "shared." + proj, shape)
            for expert in range(e):
                for proj, shape in (("w1", (mid, latent)), ("w3", (mid, latent)), ("w2", (latent, mid))):
                    base = mp + f"experts.{expert}." + proj
                    target = mt + f"experts.{expert}." + {"w1": "gate", "w3": "up", "w2": "down"}[proj]
                    add((base + ".weight_packed", base + ".weight_scale") if quant else base + ".weight",
                        target, shape, "mxfp4-g32" if quant else "float32", (K3_SOURCE, K3_INDEX, MX_SOURCE))
        schedule = tuple("kda" if i in kda else "mla" for i in range(1, n+1))
        if config.get("vision_config") is not None:
            from .k3_vision import vision_geometry, vision_shapes
            for name, shape in vision_shapes(vision_geometry(config)).items():
                add(name, "k3.vision." + name, shape)
    else:
        q = config.get("quantization_config")
        expected = {"quant_method": "fp8", "activation_scheme": "dynamic", "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
        if (not isinstance(q, Mapping) or any(q.get(k) != v for k, v in expected.items())
                or q.get("weight_block_size") != (32, 32) or text.get("quantization_config") is not None):
            reject("quantization_config", "only V4.1 FP8 32x32 UE8M0 with FP4 experts is mapped")
        heads, hd, qr, groups, out_rank, mid, e = (geometry(text, k) for k in (
            "num_attention_heads", "head_dim", "q_lora_rank", "o_groups", "o_lora_rank", "moe_intermediate_size", "n_routed_experts"))
        if heads*hd % groups or text.get("n_shared_experts") != 1:
            raise ValueError("invalid V4.1 grouped output or shared-expert geometry")
        ratios, mtp = text.get("compress_ratios"), text.get("num_nextn_predict_layers")
        if (type(mtp) is not int or mtp < 0 or not isinstance(ratios, tuple) or len(ratios) != n + mtp
                or any(type(r) is not int or r not in (0, 1, 2) for r in ratios)):
            raise ValueError("V4.1 compression schedule must include MTP layer accounting")
        for key in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids"):
            ids = text.get(key)
            if (not isinstance(ids, tuple) or any(type(i) is not int or not 0 <= i < n for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("invalid V4.1 source layer inventory: " + key)
        kv_sources = frozenset(text["kv_source_layer_ids"])
        index_sources = frozenset(text["index_source_layer_ids"])
        engram_layers = text["engram_layer_ids"]
        engram_rows = text.get("engram_num_embeddings")
        hc = geometry(text, "hc_mult")
        ih = geometry(text, "index_n_heads")
        idim = geometry(text, "index_head_dim")
        eh = geometry(text, "engram_n_heads")
        edim = geometry(text, "engram_head_dim")
        order = geometry(text, "engram_max_ngram_size")
        if (order < 2 or not isinstance(engram_rows, tuple)
                or len(engram_rows) != len(engram_layers)
                or any(type(rows) is not int or rows <= 0 for rows in engram_rows)
                or any(ratios[i] == 0 for i in kv_sources | index_sources)
                or not kv_sources <= index_sources):
            raise ValueError("invalid V4.1 compression ownership or Engram geometry")
        engram_sizes = dict(zip(engram_layers, engram_rows, strict=True))
        add("embed.weight", "v41.text.embedding", (vocab, h))
        add("norm.weight", "v41.text.final_norm", (h,))
        add("head.weight", "v41.text.lm_head", (vocab, h))
        if mtp:
            positive(text.get("dspark_block_size"), "dspark_block_size")
            targets = text.get("dspark_target_layer_ids")
            if (not isinstance(targets, tuple) or not targets
                    or any(type(i) is not int or not 0 <= i < n for i in targets)
                    or len(set(targets)) != len(targets) or any(ratios[n:])):
                reject("text_config.dspark_target_layer_ids", "explicit backbone targets and window-only draft stages required")
            draft_experts = geometry(text, "dspark_n_routed_experts")
            markov = geometry(text, "dspark_markov_rank")
        backbone_experts = e
        for layer in range(n + mtp):
            draft = layer >= n
            stage = layer - n
            p = f"mtp.{stage}." if draft else f"layers.{layer}."
            t = f"v41.mtp.{stage}." if draft else f"v41.text.layers.{layer}."
            e = draft_experts if draft else backbone_experts
            if draft and stage == 0:
                add((p + "main_proj.weight", p + "main_proj.scale"), t + "main_proj",
                    (h, h * len(targets)), "fp8-e4m3-g32x32")
                add(p + "main_norm.weight", t + "main_norm", (h,))
            if draft and stage == mtp - 1:
                add(p + "norm.weight", t + "norm", (h,))
                for name in ("embed", "head"):
                    add(p + "markov_head." + name + ".weight", t + "markov_head." + name, (vocab, markov))
                add(p + "confidence_head.proj.weight", t + "confidence_head.proj", (1, h + markov))
            for sublayer in ("attn", "ffn"):
                for suffix, shape in (("fn", ((hc+2)*hc, hc*h)),
                                      ("base", ((hc+2)*hc,)), ("scale", (3,))):
                    name = f"hc_{sublayer}_{suffix}"
                    add(p + name, t + name, shape)
            if layer in kv_sources:
                cp, ct = p + "attn.compressor.", t + "attention.compressor."
                add(cp + "wkv.weight", ct + "wkv", (hd, h))
                add(cp + "norm.weight", ct + "norm", (hd,))
                if ratios[layer] > 1:
                    add(cp + "wgate.weight", ct + "wgate", (hd, h))
            if layer in index_sources:
                ip, it = p + "attn.indexer.", t + "attention.indexer."
                add((ip + "wq_b.weight", ip + "wq_b.scale"), it + "wq_b",
                    (ih*idim, qr), "fp8-e4m3-g32x32")
                add(ip + "weights_proj.weight", it + "weights_proj", (ih, h))
                if layer in kv_sources:
                    add(ip + "wk.weight", it + "wk", (idim, hd))
                    add(ip + "k_norm.weight", it + "k_norm", (idim,))
            if layer in engram_sizes:
                ep, et = p + "engram.", t + "engram."
                add((ep + "embed.weight", ep + "embed.scale"), et + "embedding",
                    (engram_sizes[layer], edim), "fp8-e4m3-row-g32")
                add((ep + "wkv.weight", ep + "wkv.scale"), et + "wkv",
                    ((hc+1)*h, (order-1)*eh*edim), "fp8-e4m3-g32x32")
                for name in ("q_weight", "k_weight"):
                    add(ep + name, et + name, (hc, h))
            for name, shape in (("wq_a", (qr, h)), ("wq_b", (heads*hd, qr)), ("wkv", (hd, h)),
                                ("wo_a", (groups*out_rank, heads*hd//groups)), ("wo_b", (h, groups*out_rank))):
                base = p + "attn." + name
                add((base + ".weight", base + ".scale"), t + "attention." + name,
                    shape, "fp8-e4m3-g32x32")
            for name, shape in (("q_norm.weight", (qr,)), ("kv_norm.weight", (hd,)), ("attn_sink", (heads,))):
                add(p + "attn." + name, t + "attention." + name, shape)
            for name in ("attn_norm", "ffn_norm"):
                add(p + name + ".weight", t + name, (h,))
            add(p + "ffn.gate.weight", t + "moe.router", (e, h))
            add(p + "ffn.gate.bias", t + "moe.router_bias", (e,))
            add(p + "ffn.gate.bias_vl", t + "moe.router_bias_vl", (e,))
            for expert in range(e):
                for proj, shape in (("w1", (mid, h)), ("w3", (mid, h)), ("w2", (h, mid))):
                    base = p + f"ffn.experts.{expert}." + proj
                    add((base + ".weight", base + ".scale"),
                        t + f"moe.experts.{expert}." + {"w1": "gate", "w3": "up", "w2": "down"}[proj],
                        shape, "v41-fp4-g32")
            for proj, shape in (("w1", (mid, h)), ("w3", (mid, h)), ("w2", (h, mid))):
                base = p + "ffn.shared_experts." + proj
                add((base + ".weight", base + ".scale"), t + "moe.shared." + proj,
                    shape, "fp8-e4m3-g32x32")
        from .v41_vision import add_vision_contracts
        add_vision_contracts(config, h, add)
        schedule = ratios
    return family, tuple(rules), schedule
