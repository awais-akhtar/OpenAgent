"""Metadata-first publisher conversion service; never an inference factory."""
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from ..mxfp4_expert import decode_code, decode_scale
from .contracts import build_contracts, UnsupportedContract, UnsupportedPublisherContract
from .storage import WIDTHS, positive, snapshot

_FP4 = np.array([decode_code(i) for i in range(16)], dtype=np.float32)


@dataclass(frozen=True)
class MappingReport:
    family: str
    config_sha256: str
    contracts: tuple
    unsupported: tuple
    execution_gaps: tuple
    schedule: tuple

    @property
    def mapping_complete(self):
        return not self.unsupported

    @property
    def inference_supported(self):
        return False

    @property
    def published_checkpoint_supported(self):
        return False


@dataclass(frozen=True)
class MappedTensorBlock:
    """Flat FP32 elements, or flattened 17-byte MXFP4 row groups.

    offset is measured in elements for float32 and in groups for packed output.
    shape is always the complete logical tensor shape, not the block shape.
    """
    target: str
    sources: tuple
    shape: tuple
    offset: int
    encoding: str
    payload: object
    evidence: tuple


@dataclass(frozen=True)
class ExecutionPayload:
    family: str
    config: object
    config_sha256: str
    tensors: object
    contracts: tuple
    unsupported: tuple
    execution_gaps: tuple
    output_encoding: str

    @property
    def inference_supported(self):
        return False

    @property
    def published_checkpoint_supported(self):
        return False


def _admit(rule, tensors):
    missing = tuple(n for n in rule.sources if n not in tensors)
    if missing:
        return UnsupportedContract("missing_tensor", rule.target, "missing exact sources: " + ", ".join(missing))
    entries = tuple(tensors[n] for n in rule.sources)
    if rule.encoding == "float32":
        valid = entries[0].shape == rule.shape and entries[0].dtype in ("F32", "F16", "BF16")
    else:
        rows, cols = rule.shape
        if cols % 32:
            return UnsupportedContract("quantization", rule.target, "input dimension must be divisible by 32; no inferred padding")
        weight, scale = entries
        if rule.encoding in ("mxfp4-g32", "v41-fp4-g32"):
            wd, sd = (("U8",), ("U8",)) if rule.encoding == "mxfp4-g32" else (("I8",), ("F8_E8M0",))
            valid = (weight.dtype in wd and weight.shape == (rows, cols//2)
                     and scale.dtype in sd and scale.shape == (rows, cols//32))
        elif rule.encoding == "fp8-e4m3-row-g32":
            valid = (weight.dtype == "F8_E4M3" and weight.shape == rule.shape
                     and scale.dtype == "F8_E8M0" and scale.shape == (rows, cols//32))
        else:
            valid = (weight.dtype == "F8_E4M3" and weight.shape == rule.shape
                     and scale.dtype == "F8_E8M0" and scale.shape == ((rows+31)//32, cols//32))
    if not valid:
        return UnsupportedContract("tensor_layout", rule.target,
                                   "exact shape/dtype contract mismatch: " + repr(tuple((e.name, e.dtype, e.shape) for e in entries)))
    return None


def _e4m3(raw):
    code = np.frombuffer(raw, dtype=np.uint8)
    exponent = (code >> 3) & 15
    fraction = code & 7
    result = np.ldexp(1 + fraction.astype(np.float32)/8, exponent.astype(np.int16)-7)
    subnormal = exponent == 0
    result[subnormal] = fraction[subnormal].astype(np.float32) * np.float32(2**-9)
    result = np.copysign(result, np.where(code & 128, -1., 1.)).astype(np.float32)
    result[(exponent == 15) & (fraction == 7)] = np.nan
    return result


class PublisherCheckpoint:
    """A local immutable metadata plan with opt-in payload reads.

    open() never reads weight payloads. stream/load reject incomplete inventories
    by default. Passing an explicit target sequence admits ONLY that subset; it
    does not erase other unsupported findings or assert an executable model.
    """

    def __init__(self, metadata, report):
        self._metadata = metadata
        self.report = report
        self._rules = MappingProxyType({r.target: r for r in report.contracts})
        self._blocked = frozenset(g.path for g in report.unsupported
                                  if g.contract in ("missing_tensor", "tensor_layout", "quantization"))

    @classmethod
    def open(cls, directory, *, family, max_metadata_bytes=256*1024**2,
             max_tensors=1_000_000, max_shards=4096):
        if family not in ("kimi_k3", "deepseek_v41"):
            raise ValueError("explicit exact publisher family required")
        metadata = snapshot(directory, max_metadata_bytes=max_metadata_bytes,
                            max_tensors=max_tensors, max_shards=max_shards)
        if metadata.config.get("model_type") != family:
            raise ValueError("publisher family differs from selected identity")
        actual, rules, schedule = build_contracts(metadata.config, max_contracts=max_tensors)
        gaps, claimed = [], set()
        for rule in rules:
            claimed.update(rule.sources)
            gap = _admit(rule, metadata.tensors)
            if gap is not None:
                gaps.append(gap)
        for name in sorted(set(metadata.tensors) - claimed):
            gaps.append(UnsupportedContract("unmapped_tensor", name, "no authoritative exact mapping implemented; tensor is not discarded"))
        execution = [UnsupportedContract("executor", actual, "no matching publisher executor is registered by this service")]
        if actual == "deepseek_v41":
            execution.append(UnsupportedContract("mapping_coverage", actual,
                "mapped backbone, DSpark and vision declarations do not establish publisher numerical or runtime compatibility"))
        else:
            execution.append(UnsupportedContract("publisher_exact", actual, "mapped K3 text/vision payloads do not establish publisher numerical equivalence"))
        return cls(metadata, MappingReport(actual, metadata.config_sha256, rules,
                                          tuple(gaps), tuple(execution), schedule))

    @property
    def config(self):
        return self._metadata.config

    @property
    def tensors(self):
        return self._metadata.tensors

    @property
    def inference_supported(self):
        return False

    @property
    def published_checkpoint_supported(self):
        return False

    def k3_config(self, *, capacity, expert_format="fp32", projection_epsilon=1e-6,
                  l2_epsilon=1e-6, rotary_mode="publisher-none", rotary_theta=10000.0,
                  include_vision=False, vision_tower_epsilon=2**-23):
        from .k3_adapter import k3_config
        return k3_config(self, capacity=capacity, expert_format=expert_format,
                         projection_epsilon=projection_epsilon, l2_epsilon=l2_epsilon,
                         rotary_mode=rotary_mode, rotary_theta=rotary_theta,
                         include_vision=include_vision, vision_tower_epsilon=vision_tower_epsilon)

    def load_k3_execution_payload(self, *, capacity, max_weight_bytes, expert_format="fp32",
                                  max_chunk_bytes=1024**2, projection_epsilon=1e-6,
                                  l2_epsilon=1e-6, rotary_mode="publisher-none", rotary_theta=10000.0,
                                  include_vision=False, vision_tower_epsilon=2**-23,
                                  use_expert_provider=False, max_expert_bytes=None, max_mapped_bytes=None,
                                  sharded_projections=(), source_backend,
                                  source_arithmetic):
        from .k3_adapter import load_k3
        return load_k3(self, capacity=capacity, max_weight_bytes=max_weight_bytes,
                       expert_format=expert_format, max_chunk_bytes=max_chunk_bytes,
                       projection_epsilon=projection_epsilon, l2_epsilon=l2_epsilon,
                       rotary_mode=rotary_mode, rotary_theta=rotary_theta,
                       include_vision=include_vision, vision_tower_epsilon=vision_tower_epsilon,
                       use_expert_provider=use_expert_provider, max_expert_bytes=max_expert_bytes,
                       max_mapped_bytes=max_mapped_bytes, sharded_projections=sharded_projections,
                       source_backend=source_backend, source_arithmetic=source_arithmetic)

    def load_v41_execution_payload(self, *, capacity, max_weight_bytes,
                                   max_chunk_bytes=1024**2, use_providers=True,
                                   max_expert_bytes=64*1024**2,
                                   max_engram_bytes=4*1024**2, max_engram_rows=4096,
                                   round_engram_to_bfloat16=True, max_projection_bytes=64*1024**2,
                                   sharded_projections=()):
        # The public checkpoint API can be used without either the CLI or the
        # execution factory. Gate the exact source inventory at the payload
        # boundary, after metadata planning but before any mapped weight load.
        from ..source_scope import require_checkpoint_inventory_admission
        require_checkpoint_inventory_admission(
            self._metadata.root, family='deepseek-v41', model_type='deepseek_v41',
            backend='native', arithmetic='publisher-bf16-fp8-fp4',
            max_metadata_bytes=max(1, self._metadata.metadata_bytes))
        from .v41_adapter import load_v41
        return load_v41(self, capacity=capacity, max_weight_bytes=max_weight_bytes,
                        max_chunk_bytes=max_chunk_bytes, use_providers=use_providers,
                        max_expert_bytes=max_expert_bytes, max_engram_bytes=max_engram_bytes,
                        max_engram_rows=max_engram_rows,
                        round_engram_to_bfloat16=round_engram_to_bfloat16,
                        max_projection_bytes=max_projection_bytes,
                        sharded_projections=sharded_projections)

    def _select(self, targets):
        if targets is None:
            if self.report.unsupported:
                raise UnsupportedPublisherContract(self.report.unsupported)
            return self.report.contracts
        if isinstance(targets, str):
            raise ValueError("targets must be an explicit sequence, not a string")
        names = tuple(targets)
        if not names or any(type(n) is not str for n in names) or len(set(names)) != len(names):
            raise ValueError("targets must be a nonempty unique sequence of exact target names")
        for name in names:
            if name not in self._rules:
                raise UnsupportedPublisherContract((UnsupportedContract("target", name, "no exact publisher mapping"),))
            if name in self._blocked:
                raise UnsupportedPublisherContract(tuple(g for g in self.report.unsupported if g.path == name))
        return tuple(self._rules[n] for n in names)

    def stream(self, *, targets=None, output_encoding="float32", max_chunk_bytes=1024**2):
        """Yield real mapped blocks; consumers must stage until iteration completes.

        Chunk budget bounds conversion buffers conservatively, not process RSS or
        consumer-retained blocks. No full tensor or shard is read for streaming.
        """
        positive(max_chunk_bytes, "max_chunk_bytes")
        if output_encoding not in ("float32", "mxfp4-e2m1-g32-ue8m0"):
            raise ValueError("unsupported explicit output encoding")
        rules = self._select(targets)
        if output_encoding != "float32" and any(r.encoding not in ("mxfp4-g32", "v41-fp4-g32") for r in rules):
            raise ValueError("packed output requires an explicit FP4-only target selection")
        if max_chunk_bytes < 4096:
            raise MemoryError("conversion chunk budget must be at least 4096 bytes")
        self._metadata.assert_unchanged()
        for rule in rules:
            if rule.encoding == "float32":
                count, tensor = math.prod(rule.shape), self.tensors[rule.sources[0]]
                step = max_chunk_bytes // 32
                width = WIDTHS[tensor.dtype]
                for offset in range(0, count, step):
                    size = min(step, count-offset)
                    raw = self._metadata.read_range(tensor.name, offset*width, size*width)
                    if tensor.dtype == "BF16":
                        payload = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view("<f4")
                    else:
                        payload = np.frombuffer(raw, dtype="<f4" if tensor.dtype == "F32" else "<f2").astype(np.float32)
                    if not np.isfinite(payload).all():
                        raise ValueError("nonfinite publisher tensor: " + tensor.name)
                    payload.setflags(write=False)
                    yield MappedTensorBlock(rule.target, rule.sources, rule.shape, offset,
                                            "float32", payload, rule.evidence)
            else:
                rows, cols = rule.shape
                groups_per_row, total = cols//32, rows*(cols//32)
                per_chunk = max(1, max_chunk_bytes//2048)
                packed = rule.encoding in ("mxfp4-g32", "v41-fp4-g32")
                for first in range(0, total, per_chunk):
                    size = min(per_chunk, total-first)
                    payload = np.empty((size, 17), np.uint8) if output_encoding != "float32" else np.empty(size*32, np.float32)
                    group_bytes = 16 if packed else 32
                    weight_raw = self._metadata.read_range(rule.sources[0], first*group_bytes, size*group_bytes)
                    scale_raw = self._metadata.read_range(rule.sources[1], first, size) if packed else None
                    tile_scales = {}
                    for local in range(size):
                        group = first + local
                        row, col_group = divmod(group, groups_per_row)
                        scale_index = (group if packed or rule.encoding == "fp8-e4m3-row-g32"
                                       else (row//32)*groups_per_row + col_group)
                        if packed:
                            scale_code = scale_raw[local]
                        else:
                            if scale_index not in tile_scales:
                                tile_scales[scale_index] = self._metadata.read_range(rule.sources[1], scale_index, 1)[0]
                            scale_code = tile_scales[scale_index]
                        if scale_code == 255:
                            raise ValueError("nonfinite UE8M0 publisher scale: " + rule.sources[1])
                        scale = decode_scale(scale_code)
                        raw = weight_raw[local*group_bytes:(local+1)*group_bytes]
                        if packed:
                            bytes_ = np.frombuffer(raw, np.uint8)
                            codes = np.empty(32, np.uint8)
                            codes[::2], codes[1::2] = bytes_ & 15, bytes_ >> 4
                            with np.errstate(over="ignore", invalid="ignore"):
                                values = _FP4[codes] * scale
                        else:
                            with np.errstate(over="ignore", invalid="ignore"):
                                values = _e4m3(raw) * scale
                        if not np.isfinite(values).all():
                            raise ValueError("publisher dequantization exceeds finite FP32: " + rule.target)
                        if output_encoding == "float32":
                            payload[local*32:(local+1)*32] = values
                        else:
                            payload[local, 0] = scale_code
                            payload[local, 1:] = bytes_
                    payload.setflags(write=False)
                    yield MappedTensorBlock(rule.target, rule.sources, rule.shape,
                                            first*32 if output_encoding == "float32" else first,
                                            output_encoding, payload, rule.evidence)
        self._metadata.assert_unchanged()

    def read_embedding_rows(self, target, row_ids, *, max_output_bytes,
                            max_rows=4096, round_to_bfloat16=True):
        """Read selected Engram rows without materializing its embedding table.

        Output and temporary-array admission is conservative, not an RSS limit.
        Repeated IDs retain their requested ordering. No rows are returned until
        every read, scale, and file-stamp check has succeeded.
        """
        positive(max_output_bytes, "max_output_bytes")
        positive(max_rows, "max_rows")
        if type(round_to_bfloat16) is not bool:
            raise ValueError("round_to_bfloat16 must be boolean")
        rule, = self._select((target,))
        if rule.encoding != "fp8-e4m3-row-g32":
            raise ValueError("selected target is not a row-scaled Engram table")
        rows, columns = rule.shape
        indices = []
        for index in row_ids:
            if len(indices) >= max_rows:
                raise MemoryError("embedding row count budget exceeded")
            if type(index) is not int or not 0 <= index < rows:
                raise ValueError("embedding row index outside exact table extent")
            indices.append(index)
        required = len(indices)*columns*4 + columns*64 + len(indices)*64
        if required > max_output_bytes:
            raise MemoryError("embedding row output and conversion budget exceeded")
        self._metadata.assert_unchanged()
        result = np.empty((len(indices), columns), np.float32)
        for output_row, source_row in enumerate(indices):
            raw = self._metadata.read_range(rule.sources[0], source_row*columns, columns)
            scale_bytes = self._metadata.read_range(
                rule.sources[1], source_row*(columns//32), columns//32)
            codes = np.frombuffer(scale_bytes, np.uint8)
            if np.any(codes == 255):
                raise ValueError("nonfinite UE8M0 embedding scale")
            scales = np.ldexp(np.ones(codes.size, np.float32), codes.astype(np.int16)-127)
            with np.errstate(over="ignore", invalid="ignore"):
                values = (_e4m3(raw).reshape(-1, 32)*scales[:, None]).reshape(-1)
            if not np.isfinite(values).all():
                raise ValueError("embedding dequantization exceeds finite FP32")
            if round_to_bfloat16:
                bits = values.view(np.uint32)
                rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
                values = rounded.view(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError("embedding conversion exceeds finite BF16")
            result[output_row] = values
        self._metadata.assert_unchanged()
        result.setflags(write=False)
        return result

    def load(self, *, max_weight_bytes, targets=None, output_encoding="float32",
             max_chunk_bytes=1024**2):
        """Materialize a budget-admitted payload, without constructing any decoder."""
        positive(max_weight_bytes, "max_weight_bytes")
        positive(max_chunk_bytes, "max_chunk_bytes")
        if max_chunk_bytes < 4096:
            raise MemoryError("conversion chunk budget must be at least 4096 bytes")
        if output_encoding not in ("float32", "mxfp4-e2m1-g32-ue8m0"):
            raise ValueError("unsupported explicit output encoding")
        rules = self._select(targets)
        if output_encoding != "float32" and any(r.encoding not in ("mxfp4-g32", "v41-fp4-g32") for r in rules):
            raise ValueError("packed output requires FP4-only target selection")
        sizes = {r.target: math.prod(r.shape)*4 if output_encoding == "float32"
                 else math.prod(r.shape)//32*17 for r in rules}
        required = sum(sizes.values()) + 2*max_chunk_bytes
        if required > max_weight_bytes:
            raise MemoryError(f"publisher payload loading requires {required} weight bytes")
        arrays = {r.target: np.empty(r.shape, np.float32) if output_encoding == "float32"
                  else np.empty((math.prod(r.shape)//32, 17), np.uint8) for r in rules}
        for block in self.stream(targets=tuple(r.target for r in rules),
                                 output_encoding=output_encoding, max_chunk_bytes=max_chunk_bytes):
            dest = arrays[block.target].reshape(-1)
            offset = block.offset if output_encoding == "float32" else block.offset*17
            dest[offset:offset+block.payload.size] = block.payload.reshape(-1)
        for array in arrays.values():
            array.setflags(write=False)
        return ExecutionPayload(self.report.family, self.config, self.report.config_sha256,
                                MappingProxyType(arrays), rules, self.report.unsupported,
                                self.report.execution_gaps, output_encoding)


class PublisherCheckpointService:
    """Small coordinator facade, independent of parent-owned execution services."""
    open = staticmethod(PublisherCheckpoint.open)

    @staticmethod
    def load_v41_execution_payload(directory, *, capacity, max_weight_bytes,
                                   max_chunk_bytes=1024**2,
                                   max_metadata_bytes=256*1024**2,
                                   max_tensors=1_000_000, max_shards=4096,
                                   use_providers=True, max_expert_bytes=64*1024**2,
                                   max_engram_bytes=4*1024**2, max_engram_rows=4096,
                                   round_engram_to_bfloat16=True, max_projection_bytes=64*1024**2,
                                   sharded_projections=()):
        checkpoint = PublisherCheckpoint.open(directory, family="deepseek_v41",
            max_metadata_bytes=max_metadata_bytes, max_tensors=max_tensors, max_shards=max_shards)
        return checkpoint.load_v41_execution_payload(capacity=capacity,
            max_weight_bytes=max_weight_bytes, max_chunk_bytes=max_chunk_bytes,
            use_providers=use_providers, max_expert_bytes=max_expert_bytes,
            max_engram_bytes=max_engram_bytes, max_engram_rows=max_engram_rows,
            round_engram_to_bfloat16=round_engram_to_bfloat16,
            max_projection_bytes=max_projection_bytes,
            sharded_projections=sharded_projections)

    @staticmethod
    def load_k3_execution_payload(directory, *, capacity, max_weight_bytes,
                                  expert_format="fp32", max_chunk_bytes=1024**2,
                                  max_metadata_bytes=256*1024**2, max_tensors=1_000_000,
                                  max_shards=4096, projection_epsilon=1e-6, l2_epsilon=1e-6,
                                  rotary_mode="publisher-none", rotary_theta=10000.0,
                                  include_vision=False, vision_tower_epsilon=2**-23,
                                  use_expert_provider=False, max_expert_bytes=None, max_mapped_bytes=None,
                                  sharded_projections=(), source_backend,
                                  source_arithmetic):
        checkpoint = PublisherCheckpoint.open(directory, family="kimi_k3",
            max_metadata_bytes=max_metadata_bytes, max_tensors=max_tensors, max_shards=max_shards)
        return checkpoint.load_k3_execution_payload(capacity=capacity,
            max_weight_bytes=max_weight_bytes, expert_format=expert_format,
            max_chunk_bytes=max_chunk_bytes, projection_epsilon=projection_epsilon,
            l2_epsilon=l2_epsilon, rotary_mode=rotary_mode, rotary_theta=rotary_theta,
            include_vision=include_vision, vision_tower_epsilon=vision_tower_epsilon,
            use_expert_provider=use_expert_provider, max_expert_bytes=max_expert_bytes,
            max_mapped_bytes=max_mapped_bytes, sharded_projections=sharded_projections,
            source_backend=source_backend, source_arithmetic=source_arithmetic)

    @staticmethod
    def load(directory, *, family, max_weight_bytes, targets=None,
             output_encoding="float32", max_chunk_bytes=1024**2,
             max_metadata_bytes=256*1024**2, max_tensors=1_000_000, max_shards=4096):
        checkpoint = PublisherCheckpoint.open(directory, family=family,
            max_metadata_bytes=max_metadata_bytes, max_tensors=max_tensors, max_shards=max_shards)
        return checkpoint.load(max_weight_bytes=max_weight_bytes, targets=targets,
                               output_encoding=output_encoding, max_chunk_bytes=max_chunk_bytes)
