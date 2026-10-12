"""Source-name V4.1 handoff; does not coerce publisher weights into an executor."""
from dataclasses import dataclass
from contextlib import ExitStack
from types import MappingProxyType

import numpy as np

from .v41_profile import v41_publisher_profile
from .v41_provider import V41EngramRowProvider, V41PackedExpertProvider, V41FP8ProjectionProvider, V41TokenEmbeddingProvider
from .storage import positive


@dataclass(frozen=True)
class V41ExecutionPayload:
    profile: object
    publisher_config: object
    config_sha256: str
    tensors: object
    contracts: tuple
    unsupported: tuple
    execution_gaps: tuple
    family: str = "deepseek_v41"
    engram_provider: object = None
    expert_provider: object = None
    fp8_provider: object = None
    sharded_projections: tuple = ()
    embedding_provider: object = None

    @property
    def inference_supported(self):
        return False

    @property
    def published_checkpoint_supported(self):
        return False


def _shard_rule(rule):
    """Only targets dispatched through ordinary or grouped arithmetic hooks."""
    target = rule.target
    if len(rule.shape) != 2 or not target.startswith(('v41.text.', 'v41.mtp.')):
        return False
    suffix = target.rsplit('.', 1)[-1]
    return (suffix in {'lm_head', 'main_proj', 'wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b',
                       'wgate', 'wk', 'weights_proj', 'router', 'gate', 'up', 'down',
                       'w1', 'w2', 'w3'} or suffix in {'hc_attn_fn', 'hc_ffn_fn'} or
            target.endswith(('markov_head.head', 'confidence_head.proj')))


def load_v41_projection_shard(checkpoint, source_name, *, rank, world_size,
                              max_weight_bytes, max_chunk_bytes=1024**2, axis=0,
                              cancelled=None):
    """Convert one exact source-name projection to an owned output-row shard."""
    from dataclasses import replace
    from ..tensor_parallel import load_projection_shard
    if checkpoint.report.family != 'deepseek_v41' or type(source_name) is not str:
        raise ValueError('exact V4.1 checkpoint and source name required')
    matches = [rule for rule in checkpoint._select(None)
               if rule.sources[0] == source_name and _shard_rule(rule)]
    if len(matches) != 1:
        raise ValueError('one admitted V4.1 projection contract required')
    shard = load_projection_shard(checkpoint, matches[0].target, rank=rank,
        world_size=world_size, max_weight_bytes=max_weight_bytes,
        max_chunk_bytes=max_chunk_bytes, axis=axis, cancelled=cancelled)
    return replace(shard, name=source_name)


def load_v41(checkpoint, *, capacity, max_weight_bytes, max_chunk_bytes=1024**2,
             use_providers=True, max_expert_bytes=64*1024**2,
             max_engram_bytes=4*1024**2, max_engram_rows=4096,
             round_engram_to_bfloat16=True, max_projection_bytes=64*1024**2,
             sharded_projections=()):
    if checkpoint.report.family != "deepseek_v41":
        raise ValueError("V4.1 publisher checkpoint required")
    if type(use_providers) is not bool:
        raise ValueError("use_providers must be boolean")
    positive(max_weight_bytes, "max_weight_bytes")
    positive(max_chunk_bytes, "max_chunk_bytes")
    if max_chunk_bytes < 4096:
        raise MemoryError("conversion chunk budget must be at least 4096 bytes")
    rules = checkpoint._select(None)
    if (type(sharded_projections) is not tuple or
            any(type(name) is not str for name in sharded_projections) or
            len(set(sharded_projections)) != len(sharded_projections)):
        raise ValueError('unique tuple of exact sharded source names required')
    admitted = {rule.sources[0] for rule in rules if _shard_rule(rule)}
    if not set(sharded_projections) <= admitted:
        raise ValueError('unadmitted V4.1 sharded projection selection')
    profile = v41_publisher_profile(checkpoint.config, capacity=capacity,
                                    max_contracts=len(checkpoint.report.contracts))
    # Strict whole-inventory admission precedes payload reads. Quantized pairs
    # become the named logical weight, not fabricated runtime aliases or scales.
    names = [rule.sources[0] for rule in checkpoint.report.contracts]
    if len(set(names)) != len(names):
        raise ValueError("duplicate publisher handoff weight destination")
    engram_provider = expert_provider = fp8_provider = embedding_provider = None
    selected = rules
    with ExitStack() as owned:
        if use_providers:
            embedding_provider=V41TokenEmbeddingProvider(checkpoint,max_output_bytes=max_engram_bytes)
            owned.callback(embedding_provider.close)
            engram_provider = V41EngramRowProvider(checkpoint,
                max_output_bytes=max_engram_bytes, max_rows=max_engram_rows,
                round_to_bfloat16=round_engram_to_bfloat16)
            owned.callback(engram_provider.close)
            expert_provider = V41PackedExpertProvider(checkpoint,
                max_expert_bytes=max_expert_bytes, max_chunk_bytes=max_chunk_bytes)
            owned.callback(expert_provider.close)
            fp8_provider = V41FP8ProjectionProvider(checkpoint,
                max_projection_bytes=max_projection_bytes)
            owned.callback(fp8_provider.close)
            selected = tuple(r for r in rules if r.sources[0] not in embedding_provider.tensor_shapes and r.encoding not in
                             ("fp8-e4m3-row-g32", "v41-fp4-g32")
                             and not (r.encoding == "fp8-e4m3-g32x32"
                                      and not r.sources[0].endswith("wo_a.weight")))
        selected = tuple(rule for rule in selected if rule.sources[0] not in sharded_projections)
        payload = checkpoint.load(max_weight_bytes=max_weight_bytes,
            targets=tuple(r.target for r in selected), max_chunk_bytes=max_chunk_bytes,
            output_encoding="float32")
        if (payload.family != "deepseek_v41" or payload.contracts != selected or
                payload.config_sha256 != checkpoint.report.config_sha256 or
                set(payload.tensors) != {rule.target for rule in selected}):
            raise ValueError("V4.1 resident payload differs from selected publisher contracts")
        for rule in selected:
            array = payload.tensors[rule.target]
            if (not isinstance(array, np.ndarray) or array.shape != rule.shape or
                    array.dtype != np.float32 or not array.flags.c_contiguous or
                    array.flags.writeable):
                raise ValueError("invalid resident V4.1 tensor: " + rule.target)
        checkpoint._metadata.assert_unchanged()
        # Rename dictionary entries only; preserve the admitted resident arrays.
        tensors = MappingProxyType({rule.sources[0]: payload.tensors[rule.target]
                                   for rule in selected})
        result = V41ExecutionPayload(profile, checkpoint.config, payload.config_sha256,
                                     tensors, rules, payload.unsupported,
                                     payload.execution_gaps, engram_provider=engram_provider,
                                     expert_provider=expert_provider, fp8_provider=fp8_provider,
                                     sharded_projections=sharded_projections,
                                     embedding_provider=embedding_provider)
        owned.pop_all()
        return result
