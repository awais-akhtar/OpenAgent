"""A07 — Persisted format → loader → device placement registry.

Maps every admitted persisted expert format to:
  1. The loader module + class that reads that format from disk.
  2. The admitted device placements (native CPU, device/GPU, etc.).
  3. The scale/transform metadata (lattice codes, rotation, block sizes).

Special formats covered:
  - E8 lattice (e8-lattice-caller-v1): native_e8_lattice_expert, CPU-only.
  - IQ3 grouped int3 (iq3-format6-98-v1): iq3_expert, CPU-only (GPU rejected).
  - FP8 block-quantised (fp8-e4m3-b128-f32): native fp8 consumer + device
    conversion path; block size 128, scale dtype float32.
  - BF16 (bf16 / mxfp4-e2m1-g32-ue8m0): publisher-bank native decoder or
    MXFP4 group-quantised expert; device placement admitted for BF16.
  - Float32, F16 dense expert tensors (float32, fp8-e4m3-b128-f32): standard
    loaders for each family.

The registry raises a descriptive error when a (format, placement) pair is
explicitly restricted, so callers cannot silently fall through to a wrong
consumer.

Reading this module never opens a checkpoint, instantiates a decoder,
executes arithmetic, or changes admission policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, FrozenSet, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormatLoaderSpec:
    """Loader specification for one persisted expert format."""
    format_name: str
    loader_module: str
    loader_class: str
    admitted_placements: FrozenSet[str]
    rejected_placements: FrozenSet[str]
    rejection_reasons: Dict[str, str]  # placement → reason

    # Scale / geometry metadata
    block_size: Optional[int] = None        # quantisation group size (Q* formats)
    scale_dtype: Optional[str] = None       # scale tensor dtype (fp8/mxfp4)
    rotation_required: bool = False         # e8/iq3 require column-major rotation
    lattice_codes: bool = False             # e8 format stores packed lattice codes
    placement_loaders: Optional[Mapping[str, Tuple[str, str]]] = None

    def loader_for(self, placement: str) -> Tuple[str, str]:
        """Resolve the loader/consumer that actually owns this placement."""
        return (self.placement_loaders or {}).get(
            placement, (self.loader_module, self.loader_class))

    def require_placement(self, placement: str) -> None:
        """Raise if this placement is rejected for this format."""
        if placement in self.rejected_placements:
            reason = self.rejection_reasons.get(placement, 'placement not admitted')
            raise ValueError(
                f'format {self.format_name!r} rejects placement '
                f'{placement!r}: {reason}')
        if placement not in self.admitted_placements:
            raise ValueError(
                f'format {self.format_name!r} has no entry for placement '
                f'{placement!r}; admitted: {sorted(self.admitted_placements)}')


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _spec(
    fmt: str,
    module: str,
    cls: str,
    admitted: Tuple[str, ...],
    rejected: Tuple[str, ...] = (),
    reject_reasons: Optional[Dict[str, str]] = None,
    block_size: Optional[int] = None,
    scale_dtype: Optional[str] = None,
    rotation: bool = False,
    lattice: bool = False,
    placement_loaders: Optional[Mapping[str, Tuple[str, str]]] = None,
) -> FormatLoaderSpec:
    reasons: Dict[str, str] = reject_reasons or {}
    return FormatLoaderSpec(
        format_name=fmt,
        loader_module=module,
        loader_class=cls,
        admitted_placements=frozenset(admitted),
        rejected_placements=frozenset(rejected),
        rejection_reasons=MappingProxyType(reasons),
        block_size=block_size,
        scale_dtype=scale_dtype,
        rotation_required=rotation,
        lattice_codes=lattice,
        placement_loaders=MappingProxyType(dict(placement_loaders or {})),
    )


# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------

_FORMAT_REGISTRY: Dict[str, FormatLoaderSpec] = {

    # ── Float32 dense (all families) ─────────────────────────────────────
    'float32': _spec(
        'float32',
        'native_checkpoint', 'read_native_text_checkpoint',
        admitted=('native', 'device', 'public-range'),
        scale_dtype='float32',
    ),

    # ── Float16 dense ─────────────────────────────────────────────────────
    'float16': _spec(
        'float16',
        'native_checkpoint', 'read_native_text_checkpoint',
        admitted=('native', 'device', 'public-range'),
        scale_dtype='float16',
    ),

    # ── BF16 publisher bank (DeepSeek V4.1) ───────────────────────────────
    'bf16': _spec(
        'bf16',
        'publisher_checkpoint', 'v41_adapter',
        admitted=('native', 'device', 'public-range'),
        scale_dtype='bfloat16',
    ),

    # ── FP8 publisher bank (DeepSeek V4.1) ────────────────────────────────
    'fp8': _spec(
        'fp8',
        'publisher_checkpoint', 'v41_adapter',
        admitted=('native', 'device', 'public-range'),
        scale_dtype='float8_e4m3fn',
    ),

    # ── FP4 publisher bank (DeepSeek V4.1) ────────────────────────────────
    'fp4': _spec(
        'fp4',
        'publisher_checkpoint', 'v41_adapter',
        admitted=('native', 'device', 'public-range'),
        scale_dtype='float4_e2m1',
    ),

    # ── FP8 block-quantised (fp8-e4m3-b128-f32) ──────────────────────────
    # Block size 128, float32 scales; admitted for native CPU and device GPU.
    # public-range is rejected because the Segment/Edge ABI does not define a
    # packed fp8 projection path.
    'fp8-e4m3-b128-f32': _spec(
        'fp8-e4m3-b128-f32',
        'native_checkpoint', 'read_native_text_checkpoint',
        admitted=('native', 'device'),
        rejected=('public-range',),
        reject_reasons={
            'public-range': (
                'Segment/Edge ABI has no packed FP8 block projection path; '
                'use float32 or fp8 publisher-bank format for public-range')
        },
        block_size=128,
        scale_dtype='float32',
    ),

    # ── E8 lattice (e8-lattice-caller-v1) ────────────────────────────────
    # Column-major rotation required; admits native CPU only.
    # Device placement is rejected: the CUDA FP8 projection does not implement
    # the E8 lattice decode path; use float32 for device execution.
    'e8-lattice-caller-v1': _spec(
        'e8-lattice-caller-v1',
        'native_e8_lattice_expert', 'NativeE8LatticeExpert',
        admitted=('native',),
        rejected=('device', 'public-range'),
        reject_reasons={
            'device': (
                'E8 lattice decode requires column-major rotation and '
                'does not have a device kernel; use float32 for device placement'),
            'public-range': (
                'E8 lattice decode is restricted from Segment/Edge ABI '
                'until a native segment_edge compatible projection is admitted'),
        },
        rotation=True,
        lattice=True,
    ),

    # ── IQ3 grouped int3 (iq3-format6-98-v1) — DeepSeek V4.1 specific ───
    # Column-major rotation required; native CPU only.
    # Device is rejected because no IQ3 CUDA kernel exists in the admitted matrix.
    'iq3-format6-98-v1': _spec(
        'iq3-format6-98-v1',
        'iq3_expert', 'IQ3Expert',
        admitted=('native',),
        rejected=('device', 'public-range', 'distributed'),
        reject_reasons={
            'device': (
                'IQ3 grouped int3 has no admitted CUDA kernel in the current matrix; '
                'dequantise to float32 before device placement'),
            'public-range': (
                'IQ3 format is not exposed through the Segment/Edge ABI'),
            'distributed': (
                'IQ3 distributed expert sharding is not source-reviewed; '
                'use BF16 or FP8 for distributed placement'),
        },
        block_size=32,
        scale_dtype='float32',
        rotation=True,
    ),

    # ── MXFP4 group-quantised (mxfp4-e2m1-g32-ue8m0) ────────────────────
    # Used by OLMoE direct gateway and Kimi K3 packed experts.
    # Group size 32, ue8m0 exponent scales; native and device admitted.
    'mxfp4-e2m1-g32-ue8m0': _spec(
        'mxfp4-e2m1-g32-ue8m0',
        'native_mxfp4_expert', 'NativeMXFP4Expert',
        admitted=('native', 'device', 'public-range'),
        block_size=32,
        scale_dtype='ue8m0',
        placement_loaders={
            'public-range': ('publisher_checkpoint.k3_adapter', 'load_k3'),
        },
    ),

    # ── OLMoE / Mixtral quantised expert formats ─────────────────────────
    'int8-row': _spec(
        'int8-row',
        'native_expert', 'NativeInt8RowExpert',
        admitted=('native', 'public-range'),
        rejected=('device',),
        reject_reasons={
            'device': 'int8-row requires the native leased expert executor; no device kernel',
        },
        scale_dtype='float32',
        placement_loaders={
            'public-range': ('segment_edge_adapters', 'NativeOlmoePublicAdapters'),
        },
    ),
    'int2-row': _spec(
        'int2-row',
        'native_expert', 'NativeInt2RowExpert',
        admitted=('native',),
        rejected=('device', 'public-range'),
        reject_reasons={
            'device': 'int2-row requires native leased executor; no device kernel',
            'public-range': 'int2-row is not projected through Segment/Edge ABI',
        },
        scale_dtype='float32',
    ),
    'int4-row': _spec(
        'int4-row',
        'native_expert', 'NativeInt4RowExpert',
        admitted=('native',),
        rejected=('device', 'public-range'),
        reject_reasons={
            'device': 'int4-row requires native leased executor; no device kernel',
            'public-range': 'int4-row is not projected through Segment/Edge ABI',
        },
        scale_dtype='float32',
    ),
    'int4-grouped-g64': _spec(
        'int4-grouped-g64',
        'native_grouped_int4_expert', 'NativeGroupedInt4Expert',
        admitted=('native',),
        rejected=('device', 'public-range'),
        reject_reasons={
            'device': 'int4-grouped-g64 has no admitted device kernel',
            'public-range': 'int4-grouped-g64 is not projected through Segment/Edge ABI',
        },
        block_size=64,
        scale_dtype='float32',
    ),
    'int3-grouped-g64': _spec(
        'int3-grouped-g64',
        'native_grouped_int4_expert', 'NativeGroupedInt3Expert',
        admitted=('native',),
        rejected=('device', 'public-range'),
        reject_reasons={
            'device': 'int3-grouped-g64 has no admitted device kernel',
            'public-range': 'int3-grouped-g64 is not projected through Segment/Edge ABI',
        },
        block_size=64,
        scale_dtype='float32',
    ),
    # GGUF formats (native expert executor)
    **{fmt: _spec(
        fmt,
        'native_expert', 'NativeGGUFExpert',
        admitted=('native',),
        rejected=('device', 'public-range'),
        reject_reasons={
            'device': f'{fmt} GGUF has no admitted device kernel',
            'public-range': f'{fmt} GGUF is not projected through Segment/Edge ABI',
        },
        scale_dtype='float32',
    ) for fmt in ('q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0', 'q8_1')},

    # ── Restricted non-admitted formats (for tooling visibility) ─────────
    'int4-rans256-g0': _spec(
        'int4-rans256-g0',
        'RESTRICTED', 'RESTRICTED',
        admitted=(),
        rejected=('native', 'device', 'public-range', 'distributed'),
        reject_reasons={p: 'int4-rans256-g0 is not admitted in the Phase 1 matrix; '
                          'it is recorded for audit visibility only'
                        for p in ('native', 'device', 'public-range', 'distributed')},
    ),
    'fp8-e4m3-b128-ue8m0': _spec(
        'fp8-e4m3-b128-ue8m0',
        'RESTRICTED', 'RESTRICTED',
        admitted=(),
        rejected=('native', 'device', 'public-range', 'distributed'),
        reject_reasons={p: 'fp8-e4m3-b128-ue8m0 is not admitted in the Phase 1 matrix; '
                          'it is recorded for audit visibility only'
                        for p in ('native', 'device', 'public-range', 'distributed')},
    ),
}

_REGISTRY: MappingProxyType = MappingProxyType(_FORMAT_REGISTRY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_format_spec(persisted_format: str) -> FormatLoaderSpec:
    """Return the loader spec for a persisted format, or raise."""
    spec = _REGISTRY.get(persisted_format)
    if spec is None:
        raise ValueError(
            f'persisted format {persisted_format!r} is not in the P07 format '
            f'registry; admitted formats: {sorted(_REGISTRY)}')
    return spec


def require_format_placement(persisted_format: str, placement: str) -> FormatLoaderSpec:
    """Return the spec and raise if the placement is not admitted."""
    spec = get_format_spec(persisted_format)
    spec.require_placement(placement)
    return spec


def all_formats() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def admitted_formats_for_placement(placement: str) -> Tuple[str, ...]:
    """Return all formats admitted for a given placement backend."""
    return tuple(
        fmt for fmt, spec in sorted(_REGISTRY.items())
        if placement in spec.admitted_placements
    )


def report_format_placement_matrix() -> str:
    """Human-readable format × placement table for auditing."""
    placements = ('native', 'device', 'public-range', 'distributed')
    header = f"{'Format':<32} " + ' '.join(f'{p:<14}' for p in placements)
    lines = [header, '-' * len(header)]
    for fmt in sorted(_REGISTRY):
        spec = _REGISTRY[fmt]
        row_parts = []
        for p in placements:
            if p in spec.admitted_placements:
                row_parts.append('{:<14}'.format('ADMIT'))
            elif p in spec.rejected_placements:
                row_parts.append('{:<14}'.format('REJECT'))
            else:
                row_parts.append('{:<14}'.format('--'))
        lines.append(f'{fmt:<32} ' + ' '.join(row_parts))
    return '\n'.join(lines)
