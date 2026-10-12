"""Publisher attention ownership and arithmetic contracts, without an executor."""
from dataclasses import dataclass
from collections.abc import Mapping

from .contracts import build_contracts, geometry


@dataclass(frozen=True)
class V41LayerProfile:
    layer: int
    compression_ratio: int
    kv_owner: int | None
    index_owner: int | None
    candidate_owner: int | None
    owns_engram: bool
    rotary_theta: float
    rotary_original_context: int
    window_encoding: str = "fp8-e4m3-g32-ue8m0"
    compressed_encoding: str = "fp4-e2m1-g16-e4m3"
    index_encoding: str = "fp4-e2m1-g32-ue8m0"


@dataclass(frozen=True)
class V41EngramProfile:
    layers: tuple[int, ...]
    rows: tuple[int, ...]
    maximum_order: int
    heads_per_order: int
    head_dim: int
    compressed_vocabulary: int
    hash_vocabulary: int
    padding_token: int

    @property
    def projection_input(self):
        return (self.maximum_order-1)*self.heads_per_order*self.head_dim


@dataclass(frozen=True)
class V41PublisherProfile:
    hidden_size: int
    vocabulary: int
    attention_heads: int
    head_dim: int
    rotary_dim: int
    query_rank: int
    output_groups: int
    output_rank: int
    index_heads: int
    index_head_dim: int
    index_topk: int
    window: int
    streams: int
    sinkhorn_iterations: int
    norm_epsilon: float
    sinkhorn_epsilon: float
    rotary_factor: float
    rotary_beta_fast: float
    rotary_beta_slow: float
    candidate_topk_blocks: int
    candidate_block_size: int
    context: int
    layers: tuple[V41LayerProfile, ...]
    expert_count: int
    selected_experts: int
    expert_intermediate: int
    shared_experts: int
    routed_scale: float
    activation_limit: float
    engram: V41EngramProfile
    scoring_function: str = "sqrtsoftplus"
    routing_method: str = "noaux_tc"
    normalize_routed_weights: bool = True

    @property
    def published_checkpoint_supported(self):
        return False

    @property
    def inference_supported(self):
        return False


def _finite_positive(value, name):
    import math

    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("positive finite publisher value required: " + name)
    return float(value)


def v41_publisher_profile(config, *, capacity, max_contracts=1_000_000):
    """Resolve the pinned backbone's shared cache ownership from configuration.

    This profile is not the private composed-equations config. It deliberately
    carries different window, compressed and index cache encodings, and no
    attempt is made to coerce its multi-head geometry into square projections.
    """
    family, _, _ = build_contracts(config, max_contracts=max_contracts)
    if family != "deepseek_v41":
        raise ValueError("V4.1 publisher configuration required")
    text = config["text_config"]
    if (text.get("scoring_func") != "sqrtsoftplus"
            or text.get("topk_method") != "noaux_tc"
            or text.get("norm_topk_prob") is not True
            or text.get("hidden_act") != "silu"):
        raise ValueError("unsupported publisher routing or activation policy")
    experts = geometry(text, "n_routed_experts")
    selected = geometry(text, "num_experts_per_tok")
    if selected > experts:
        raise ValueError("selected expert count exceeds routed inventory")
    context = geometry(text, "max_position_embeddings")
    if type(capacity) is not int or not 0 < capacity <= context:
        raise ValueError("capacity must fit the configured publisher context")
    h = geometry(text, "hidden_size")
    heads = geometry(text, "num_attention_heads")
    hd = geometry(text, "head_dim")
    rd = geometry(text, "qk_rope_head_dim")
    groups = geometry(text, "o_groups")
    ih, idim = geometry(text, "index_n_heads"), geometry(text, "index_head_dim")
    if rd % 2 or rd > min(hd, idim) or heads % groups:
        raise ValueError("invalid publisher rotary or grouped-head geometry")
    if text.get("num_key_value_heads") != 1:
        raise ValueError("the V4.1 profile requires one shared latent KV head")
    rope = text.get("rope_scaling")
    if not isinstance(rope, Mapping) or rope.get("rope_type") != "yarn":
        raise ValueError("explicit publisher YaRN profile required")
    original = geometry(rope, "original_max_position_embeddings")
    theta = _finite_positive(text.get("rope_theta"), "rope_theta")
    compressed_theta = _finite_positive(text.get("compress_rope_theta"), "compress_rope_theta")
    if min(theta, compressed_theta) <= 1:
        raise ValueError("publisher rotary bases must exceed one")
    candidate = text.get("candidate_source_layer_id")
    n = geometry(text, "num_hidden_layers")
    if type(candidate) is not int or not -1 <= candidate < n:
        raise ValueError("invalid publisher candidate source layer")
    kv_sources, index_sources = (frozenset(text[key]) for key in
                                ("kv_source_layer_ids", "index_source_layer_ids"))
    if candidate >= 0 and candidate not in index_sources:
        raise ValueError("candidate source must own an index selection")
    engram = frozenset(text["engram_layer_ids"])
    pad = text.get("engram_pad_token_id")
    if type(pad) is not int or not 0 <= pad < geometry(text, "vocab_size"):
        raise ValueError("Engram padding token outside vocabulary")
    engram_profile = V41EngramProfile(
        layers=text["engram_layer_ids"], rows=text["engram_num_embeddings"],
        maximum_order=geometry(text, "engram_max_ngram_size"),
        heads_per_order=geometry(text, "engram_n_heads"),
        head_dim=geometry(text, "engram_head_dim"),
        compressed_vocabulary=geometry(text, "engram_compressed_vocab_size"),
        hash_vocabulary=geometry(text, "engram_vocab_size"), padding_token=pad,
    )
    ratios = text["compress_ratios"]
    layers, kv_owner, index_owner = [], None, None
    for layer in range(n):
        ratio = ratios[layer]
        if layer in kv_sources:
            kv_owner = layer
        if layer in index_sources:
            index_owner = layer
        if ratio:
            if kv_owner is None or index_owner is None:
                raise ValueError("compressed layer precedes a required cache source")
            if ratios[kv_owner] != ratio or ratios[index_owner] != ratio:
                raise ValueError("shared cache source uses a different compression ratio")
        layers.append(V41LayerProfile(
            layer=layer, compression_ratio=ratio,
            kv_owner=kv_owner if ratio else None,
            index_owner=index_owner if ratio else None,
            candidate_owner=candidate if ratio and candidate >= 0 and layer > candidate else None,
            owns_engram=layer in engram,
            rotary_theta=compressed_theta if ratio else theta,
            rotary_original_context=original if ratio else 0,
        ))
    return V41PublisherProfile(
        hidden_size=h, vocabulary=geometry(text, "vocab_size"),
        attention_heads=heads, head_dim=hd, rotary_dim=rd,
        query_rank=geometry(text, "q_lora_rank"), output_groups=groups,
        output_rank=geometry(text, "o_lora_rank"), index_heads=ih,
        index_head_dim=idim, index_topk=geometry(text, "index_topk"),
        window=geometry(text, "sliding_window"), streams=geometry(text, "hc_mult"),
        sinkhorn_iterations=geometry(text, "hc_sinkhorn_iters"),
        norm_epsilon=_finite_positive(text.get("rms_norm_eps"), "rms_norm_eps"),
        sinkhorn_epsilon=_finite_positive(text.get("hc_eps"), "hc_eps"),
        rotary_factor=_finite_positive(rope.get("factor"), "rope_scaling.factor"),
        rotary_beta_fast=_finite_positive(rope.get("beta_fast"), "rope_scaling.beta_fast"),
        rotary_beta_slow=_finite_positive(rope.get("beta_slow"), "rope_scaling.beta_slow"),
        candidate_topk_blocks=geometry(text, "candidate_topk_blocks"),
        candidate_block_size=geometry(text, "candidate_block_size"),
        context=capacity, layers=tuple(layers),
        expert_count=experts, selected_experts=selected,
        expert_intermediate=geometry(text, "moe_intermediate_size"),
        shared_experts=geometry(text, "n_shared_experts"),
        routed_scale=_finite_positive(text.get("routed_scaling_factor"), "routed_scaling_factor"),
        activation_limit=_finite_positive(text.get("swiglu_limit"), "swiglu_limit"),
        engram=engram_profile,
    )
