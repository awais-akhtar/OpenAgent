"""Explicit handoff to K3Execution's existing publisher-name text schema."""
from dataclasses import dataclass, field
from contextlib import ExitStack, closing
import math
from types import MappingProxyType

import numpy as np

from .contracts import K3_REVISION, UnsupportedContract, UnsupportedPublisherContract
from .storage import positive


_K3_PAYLOAD_ADMISSION = object()


@dataclass(frozen=True)
class K3ExecutionPayload:
    config: object
    tensors: object
    publisher_config: object
    config_sha256: str
    contracts: tuple
    unsupported: tuple
    execution_gaps: tuple
    expert_provider: object = None
    sharded_projections: tuple = ()
    _source_admission: object = field(default=None, repr=False, compare=False)
    _source_root: str | None = field(default=None, repr=False, compare=False)
    _source_files: tuple = field(default=(), repr=False, compare=False)

    @property
    def family(self):
        return "kimi_k3"

    @property
    def inference_supported(self):
        return False

    @property
    def published_checkpoint_supported(self):
        return False


def k3_config(checkpoint, *, capacity, expert_format="fp32",
              projection_epsilon=1e-6, l2_epsilon=1e-6,
              rotary_mode="publisher-none", rotary_theta=10000.0,
              include_vision=False, vision_tower_epsilon=2**-23):
    """Construct the equation executor's text config, not publisher admission.

    Projection/L2 epsilon and rotary policy are explicit runtime choices; they
    are not claimed to be inferred from a publisher numerical kernel contract.
    Vision is explicitly opt-in; no publisher numerical equivalence is promised.
    """
    if checkpoint.report.family != "kimi_k3":
        raise ValueError("K3 execution adapter requires the exact kimi_k3 wrapper")
    positive(capacity, "capacity")
    from ..k3_execution import K3Config

    text = checkpoint.config["text_config"]
    linear = text["linear_attn_config"]
    requirements = {
        "hidden_act": "situ", "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid", "latent_moe_use_norm": True,
        "mla_use_output_gate": True, "mla_use_nope": True,
    }
    if (any(type(text.get(k)) is not type(v) or text.get(k) != v for k, v in requirements.items())
            or linear.get("use_full_rank_gate") is not True
            or linear.get("gate_lower_bound") != -5.0):
        raise UnsupportedPublisherContract((UnsupportedContract(
            "k3_runtime_schema", "text_config", "executor requires Situ, normalized sigmoid routing, latent norm, full-rank gates and gate lower bound -5"),))
    maximum = positive(text.get("max_position_embeddings"), "max_position_embeddings")
    if capacity > maximum:
        raise ValueError("runtime capacity exceeds publisher context configuration")
    if type(include_vision) is not bool:
        raise ValueError("include_vision must be boolean")
    vision = None
    if include_vision:
        from ..k3_execution import K3VisionConfig
        from .k3_vision import vision_geometry
        vision = K3VisionConfig(**vision_geometry(checkpoint.config),
                                tower_epsilon=vision_tower_epsilon)
    return K3Config(
        hidden=text["hidden_size"], vocab=text["vocab_size"],
        layer_types=checkpoint.report.schedule,
        heads=text["num_attention_heads"], nope=text["qk_nope_head_dim"],
        rope=text["qk_rope_head_dim"], value=text["v_head_dim"],
        qrank=text["q_lora_rank"], kvrank=text["kv_lora_rank"],
        kda_heads=linear["num_heads"], kda_dim=linear["head_dim"],
        kernel=linear["short_conv_kernel_size"],
        dense_intermediate=text["intermediate_size"],
        latent=text["routed_expert_hidden_size"],
        expert_intermediate=text["moe_intermediate_size"], experts=text["num_experts"],
        selected=text["num_experts_per_token"], shared=text["num_shared_experts"],
        dense_prefix=text["first_k_dense_replace"],
        residual_block=text["attn_res_block_size"], epsilon=text["rms_norm_eps"],
        situ_beta=text["activation_situ_beta"], situ_linear_beta=text["activation_situ_linear_beta"],
        routed_scale=text["routed_scaling_factor"],
        expert_groups=text["num_expert_group"], selected_groups=text["topk_group"],
        capacity=capacity, expert_format=expert_format, revision=K3_REVISION,
        projection_epsilon=projection_epsilon, l2_epsilon=l2_epsilon,
        rotary_mode=rotary_mode, rotary_theta=rotary_theta, vision=vision,
    )


def _destinations(rule, expert_format):
    if rule.encoding == "float32":
        return ((rule.sources[0], rule.shape, np.float32),)
    if rule.encoding != "mxfp4-g32":
        raise ValueError("non-K3 quantized rule cannot enter K3 execution")
    if expert_format == "fp32":
        source = rule.sources[0]
        if not source.endswith(".weight_packed"):
            raise ValueError("exact K3 packed source spelling required")
        return ((source[:-len(".weight_packed")] + ".weight", rule.shape, np.float32),)
    rows, cols = rule.shape
    return ((rule.sources[0], (rows, cols//2), np.uint8),
            (rule.sources[1], (rows, cols//32), np.uint8))


def load_k3_projection_shard(checkpoint, runtime_name, *, rank, world_size,
                             max_weight_bytes, max_chunk_bytes=1024**2,
                             bank_format='dense'):
    """Load one admitted output-row shard under its exact executor spelling."""
    from dataclasses import replace
    from ..tensor_parallel import load_projection_shard
    if checkpoint.report.family != 'kimi_k3' or type(runtime_name) is not str:
        raise ValueError('exact K3 checkpoint and runtime projection name required')
    if type(bank_format) is not str or bank_format not in ('dense', 'bf16', 'fp8', 'fp4'):
        raise ValueError('explicit K3 projection bank format required')
    allowed = {'q_proj','k_proj','v_proj','q_a_proj','q_b_proj','kv_a_proj_with_mqa','kv_b_proj',
               'o_proj','f_a_proj','f_b_proj','b_proj','g_proj','gate_proj','up_proj','down_proj',
               'w1','w2','w3','lm_head'}
    if not runtime_name.endswith('.weight') or runtime_name.split('.')[-2] not in allowed:
        raise ValueError('unadmitted sharded K3 runtime projection')
    matches = []
    for rule in checkpoint.report.contracts:
        if rule.target.startswith('k3.vision.'):
            continue
        destinations = _destinations(rule,'fp32')
        if len(destinations)==1 and destinations[0][0]==runtime_name:
            matches.append(rule)
    if len(matches)!=1 or len(matches[0].shape)!=2:
        raise ValueError('one exact publisher contract for the runtime projection required')
    shard = load_projection_shard(checkpoint,matches[0].target,rank=rank,world_size=world_size,
        max_weight_bytes=max_weight_bytes,max_chunk_bytes=max_chunk_bytes,
        bank_format=bank_format)
    return replace(shard,name=runtime_name)


def load_k3(checkpoint, *, capacity, max_weight_bytes, expert_format="fp32",
            max_chunk_bytes=1024**2, projection_epsilon=1e-6, l2_epsilon=1e-6,
            rotary_mode="publisher-none", rotary_theta=10000.0,
            include_vision=False, vision_tower_epsilon=2**-23,
            use_expert_provider=False, max_expert_bytes=None, max_mapped_bytes=None,
            sharded_projections=()):
    """Prepare exact runtime names, optionally vision and leased mmap experts.

    Unmapped tensors remain attached as gaps. No library is opened and no
    K3Execution instance is created. Provider mode never materializes the bank.
    """
    positive(max_weight_bytes, "max_weight_bytes")
    positive(max_chunk_bytes, "max_chunk_bytes")
    if (type(sharded_projections) is not tuple or
            any(type(name) is not str for name in sharded_projections) or
            len(set(sharded_projections))!=len(sharded_projections)):
        raise ValueError('explicit unique sharded K3 runtime names required')
    if max_chunk_bytes < 4096:
        raise MemoryError("conversion chunk budget must be at least 4096 bytes")
    if expert_format not in ("fp32", "e2m1-low-first-e8m0-g32"):
        raise ValueError("unsupported explicit K3 executor expert format")
    if type(use_expert_provider) is not bool:
        raise ValueError("use_expert_provider must be boolean")
    if use_expert_provider:
        if expert_format == "fp32":
            raise ValueError("mmap provider requires explicit packed expert format")
        positive(max_expert_bytes, "max_expert_bytes")
    elif max_expert_bytes is not None or max_mapped_bytes is not None:
        raise ValueError("provider budgets require use_expert_provider=True")
    config = k3_config(checkpoint, capacity=capacity, expert_format=expert_format,
                       projection_epsilon=projection_epsilon, l2_epsilon=l2_epsilon,
                       rotary_mode=rotary_mode, rotary_theta=rotary_theta,
                       include_vision=include_vision, vision_tower_epsilon=vision_tower_epsilon)
    all_rules = checkpoint._select(tuple(r.target for r in checkpoint.report.contracts
                                         if include_vision or not r.target.startswith("k3.vision.")))
    expert_rules = tuple(r for r in all_rules if ".block_sparse_moe.experts." in r.sources[0])
    expert_targets = frozenset(r.target for r in expert_rules)
    rules = tuple(r for r in all_rules if not use_expert_provider or r.target not in expert_targets)
    specs = {}
    for rule in rules:
        for name, shape, dtype in _destinations(rule, expert_format):
            if name in specs:
                raise ValueError("duplicate runtime destination: " + name)
            specs[name] = (shape, dtype)
    # The independent worker owns this inventory; require a matching handshake.
    from ..k3_execution import tensor_shapes
    expected = tensor_shapes(config)
    expert_names = {name for r in expert_rules for name, _, _ in _destinations(r, expert_format)}
    expected_provider = {n: s for n, s in expected.items() if n in expert_names} if use_expert_provider else {}
    if use_expert_provider:
        expected = {n: s for n, s in expected.items() if n not in expert_names}
    if set(expected) != set(specs) or any(expected[n] != specs[n][0] for n in expected):
        raise UnsupportedPublisherContract((UnsupportedContract(
            "k3_runtime_schema", "tensor_shapes", "current K3Execution schema differs from mapped contracts"),))
    allowed = {'q_proj','k_proj','v_proj','q_a_proj','q_b_proj','kv_a_proj_with_mqa','kv_b_proj',
               'o_proj','f_a_proj','f_b_proj','b_proj','g_proj','gate_proj','up_proj','down_proj',
               'w1','w2','w3','lm_head'}
    skipped = set(sharded_projections)
    allowed.update({'wqkv','wo','fc0','fc1','0','2','embed_tokens',
                    'routed_expert_down_proj','routed_expert_up_proj'})
    for name in skipped:
        if (name not in specs or len(specs[name][0])!=2 or specs[name][1]!=np.float32 or
                not name.endswith('.weight') or name.split('.')[-2] not in allowed):
            raise ValueError('unadmitted sharded K3 runtime projection: '+name)
    for rule in rules:
        destinations = _destinations(rule,expert_format)
        if any(name in skipped for name,_,_ in destinations) and len(destinations)!=1:
            raise ValueError('sharded K3 projection cannot omit part of a packed destination pair')
    rules = tuple(rule for rule in rules if _destinations(rule,expert_format)[0][0] not in skipped)
    specs = {name:spec for name,spec in specs.items() if name not in skipped}
    required = sum(math.prod(shape)*np.dtype(dtype).itemsize for shape, dtype in specs.values()) + 2*max_chunk_bytes
    if required > max_weight_bytes:
        raise MemoryError(f"K3 execution payload preparation requires {required} weight bytes")
    with ExitStack() as owned:
        provider = None
        if use_expert_provider:
            from .k3_provider import K3MmapExpertProvider
            provider = K3MmapExpertProvider(checkpoint, max_expert_bytes=max_expert_bytes,
                                            max_mapped_bytes=max_mapped_bytes)
            owned.callback(provider.close)
            if provider.tensor_shapes != expected_provider:
                raise UnsupportedPublisherContract((UnsupportedContract(
                    "k3_runtime_schema", "expert_provider", "provider bank differs from K3Execution packed inventory"),))
        arrays = {n: np.empty(shape, dtype) for n, (shape, dtype) in specs.items()}
        checkpoint._metadata.assert_unchanged()
        destinations_by_target = {r.target: _destinations(r, expert_format) for r in rules}
        rules_by_target = {r.target: r for r in rules}
        for packed in (False, True):
            selected = tuple(r.target for r in rules
                             if (r.encoding == "mxfp4-g32" and expert_format != "fp32") == packed)
            if not selected:
                continue
            encoding = "mxfp4-e2m1-g32-ue8m0" if packed else "float32"
            progress = dict.fromkeys(selected, 0)
            with closing(checkpoint.stream(targets=selected, output_encoding=encoding,
                                          max_chunk_bytes=max_chunk_bytes)) as stream:
                for block in stream:
                    if block.target not in progress:
                        raise ValueError("unexpected K3 streamed target: " + block.target)
                    rule = rules_by_target[block.target]
                    data = block.payload
                    if (block.sources != rule.sources or block.shape != rule.shape or
                            block.encoding != encoding or type(block.offset) is not int or
                            block.offset != progress[block.target] or
                            not isinstance(data, np.ndarray)):
                        raise ValueError("K3 stream differs from exact mapped contract")
                    destinations = destinations_by_target[block.target]
                    if packed:
                        if data.dtype != np.uint8 or data.ndim != 2 or data.shape[1] != 17:
                            raise ValueError("invalid K3 packed stream block")
                        count, extent = data.shape[0], math.prod(rule.shape)//32
                    else:
                        if data.dtype != np.float32 or data.ndim != 1:
                            raise ValueError("invalid K3 resident stream block")
                        count, extent = data.size, math.prod(rule.shape)
                    if count <= 0 or block.offset+count > extent:
                        raise ValueError("K3 streamed block exceeds mapped extent")
                    if packed:
                        arrays[destinations[0][0]].reshape(-1, 16)[block.offset:block.offset+count] = data[:, 1:]
                        arrays[destinations[1][0]].reshape(-1)[block.offset:block.offset+count] = data[:, 0]
                    else:
                        arrays[destinations[0][0]].reshape(-1)[block.offset:block.offset+count] = data
                    progress[block.target] += count
            if any(progress[target] != math.prod(rules_by_target[target].shape)//(32 if packed else 1)
                   for target in selected):
                raise ValueError("incomplete K3 streamed payload")
        checkpoint._metadata.assert_unchanged()
        for array in arrays.values():
            array.setflags(write=False)
        gaps = checkpoint.report.execution_gaps + (UnsupportedContract(
            "publisher_exact", "K3Execution", "schema handoff is implemented; BF16/FLA/quantized and vision numerical equivalence remain unestablished"),)
        if not include_vision and checkpoint.config.get("vision_config") is not None:
            gaps += (UnsupportedContract("execution_scope", "vision_config", "vision mapping exists but this preparation explicitly selects text-only execution"),)
        payload = K3ExecutionPayload(config, MappingProxyType(arrays), checkpoint.config,
                                     checkpoint.report.config_sha256, all_rules,
                                     checkpoint.report.unsupported, gaps, provider,
                                     sharded_projections, _K3_PAYLOAD_ADMISSION,
                                     str(checkpoint._metadata.root),
                                     tuple(sorted(checkpoint._metadata.files.items())))
        owned.pop_all()
        return payload


def require_k3_payload_admission(payload):
    """Reject caller-assembled K3 payloads lacking the checkpoint adapter seal."""
    if (type(payload) is not K3ExecutionPayload
            or payload._source_admission is not _K3_PAYLOAD_ADMISSION):
        raise ValueError(
            'K3 execution requires a payload created by the admitted publisher checkpoint adapter')
    return payload
