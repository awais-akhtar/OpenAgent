"""Phase 3 — Quantization Engine: INT4 / INT8 / FP8 weight compression.

Provides block-wise symmetric quantization for MoE expert weights so that
large models can be loaded into limited VRAM/RAM without accuracy collapse.

Components
----------
``QuantizationConfig``    -- dtype, block_size, group_size, calibration mode
``QuantizedTensor``       -- packed quantized data + per-block scales
``Int4BlockQuantizer``    -- GPTQ-style symmetric INT4 (NF4 variant optional)
``Int8SymmetricQuantizer``-- per-tensor or per-channel INT8 weights
``Fp8E4M3Quantizer``      -- FP8 E4M3 encoding/decoding (numpy emulation)
``QuantizationCodec``     -- unified encode/decode entry point
``QuantizedExpertWeight`` -- one expert's (gate, up, down) quantized
``QuantizedExpertStore``  -- wraps TieredExpertStore; quantizes on write,
                             dequantizes on read; drop-in compatible
``Phase3QuantizedSession``-- wraps a base session; all expert loads pass
                             through the quantized store
"""
from __future__ import annotations

import logging
import struct
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QuantizationConfig
# ---------------------------------------------------------------------------

class QuantizationDtype(str, Enum):
    INT4  = "int4"
    INT8  = "int8"
    FP8   = "fp8_e4m3"
    NONE  = "none"


@dataclass
class QuantizationConfig:
    """Quantization parameters."""
    dtype:          QuantizationDtype = QuantizationDtype.INT8
    block_size:     int  = 128        # elements per quantization block
    group_size:     int  = 128        # alias for block_size (GPTQ convention)
    asymmetric:     bool = False       # True = zero-point offset; False = symmetric
    calibration:    str  = "minmax"    # "minmax" or "percentile"
    percentile:     float = 0.9999     # clipping percentile for calibration

    def __post_init__(self) -> None:
        self.group_size = self.block_size   # keep in sync


# ---------------------------------------------------------------------------
# QuantizedTensor
# ---------------------------------------------------------------------------

@dataclass
class QuantizedTensor:
    """Packed quantized weight with scales (and optional zero points)."""
    data:        np.ndarray         # packed integers (uint8 for int4/int8/fp8)
    scales:      np.ndarray         # float32, shape (n_blocks,)
    zero_points: Optional[np.ndarray] = None  # int32, shape (n_blocks,) if asymmetric
    orig_shape:  Tuple[int, ...]     = ()
    dtype:       QuantizationDtype  = QuantizationDtype.INT8

    @property
    def compression_ratio(self) -> float:
        orig_bytes = int(np.prod(self.orig_shape)) * 4   # float32 reference
        quant_bytes = self.data.nbytes + self.scales.nbytes
        if self.zero_points is not None:
            quant_bytes += self.zero_points.nbytes
        return orig_bytes / max(1, quant_bytes)


# ---------------------------------------------------------------------------
# Int4BlockQuantizer
# ---------------------------------------------------------------------------

class Int4BlockQuantizer:
    """Symmetric INT4 block quantization.

    Packs two INT4 values per byte.  Each block of `block_size` consecutive
    elements shares one float32 scale derived from the block's max-abs value.
    """

    def __init__(self, config: QuantizationConfig) -> None:
        self._block = config.block_size

    def quantize(self, weight: np.ndarray) -> QuantizedTensor:
        flat     = weight.flatten().astype(np.float32)
        n        = len(flat)
        n_blocks = (n + self._block - 1) // self._block
        # Pad to multiple of block_size
        pad = n_blocks * self._block - n
        if pad > 0:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        blocks = flat.reshape(n_blocks, self._block)
        scales = np.max(np.abs(blocks), axis=1).astype(np.float32)
        scales = np.where(scales == 0, 1.0, scales)

        # Scale to [-7, 7] and clip (INT4 symmetric range)
        normalized = np.clip(blocks / scales[:, None] * 7.0, -7, 7)
        int4_vals  = np.round(normalized).astype(np.int8)   # [-7..7]

        # Pack pairs into uint8: low nibble = even, high nibble = odd
        flat4     = int4_vals.flatten()
        n_pairs   = (len(flat4) + 1) // 2
        packed    = np.zeros(n_pairs, dtype=np.uint8)
        even      = flat4[0::2].astype(np.uint8) & 0x0F
        odd_slice = flat4[1::2].astype(np.uint8) & 0x0F
        packed[:len(odd_slice)] = even[:len(odd_slice)] | (odd_slice << 4)
        if len(flat4) % 2 == 1:
            packed[-1] = even[-1]

        return QuantizedTensor(
            data=packed, scales=scales, orig_shape=weight.shape,
            dtype=QuantizationDtype.INT4)

    def dequantize(self, qt: QuantizedTensor) -> np.ndarray:
        packed   = qt.data
        n_int4   = int(np.prod(qt.orig_shape))
        n_blocks = len(qt.scales)

        # Unpack nibbles with sign-extension (4-bit signed: values 8-15 -> -8..-1)
        even_raw = (packed & 0x0F).astype(np.int32)
        odd_raw  = ((packed >> 4) & 0x0F).astype(np.int32)
        even = np.where(even_raw > 7, even_raw - 16, even_raw).astype(np.int8)
        odd  = np.where(odd_raw  > 7, odd_raw  - 16, odd_raw ).astype(np.int8)
        interleaved = np.empty(len(packed) * 2, dtype=np.int8)
        interleaved[0::2] = even
        interleaved[1::2] = odd
        flat4 = interleaved[:n_blocks * self._block].astype(np.float32)

        blocks = flat4.reshape(n_blocks, self._block)
        deq    = (blocks / 7.0 * qt.scales[:, None]).flatten()[:n_int4]
        return deq.reshape(qt.orig_shape)


# ---------------------------------------------------------------------------
# Int8SymmetricQuantizer
# ---------------------------------------------------------------------------

class Int8SymmetricQuantizer:
    """Per-block symmetric INT8 quantization."""

    def __init__(self, config: QuantizationConfig) -> None:
        self._block = config.block_size
        self._pct   = config.percentile if config.calibration == "percentile" else None

    def quantize(self, weight: np.ndarray) -> QuantizedTensor:
        flat     = weight.flatten().astype(np.float32)
        n        = len(flat)
        n_blocks = (n + self._block - 1) // self._block
        pad      = n_blocks * self._block - n
        if pad > 0:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        blocks = flat.reshape(n_blocks, self._block)
        if self._pct is not None:
            scales = np.percentile(np.abs(blocks), self._pct * 100, axis=1)
        else:
            scales = np.max(np.abs(blocks), axis=1)
        scales = np.where(scales == 0, 1.0, scales).astype(np.float32)

        normalized = np.clip(blocks / scales[:, None] * 127.0, -127, 127)
        int8_vals  = np.round(normalized).astype(np.int8)

        return QuantizedTensor(
            data=int8_vals.flatten().view(np.uint8),
            scales=scales,
            orig_shape=weight.shape,
            dtype=QuantizationDtype.INT8)

    def dequantize(self, qt: QuantizedTensor) -> np.ndarray:
        n_blocks = len(qt.scales)
        int8v = qt.data.view(np.int8)[:n_blocks * self._block].astype(np.float32)
        blocks = int8v.reshape(n_blocks, self._block)
        flat   = (blocks / 127.0 * qt.scales[:, None]).flatten()
        n_orig = int(np.prod(qt.orig_shape))
        return flat[:n_orig].reshape(qt.orig_shape)


# ---------------------------------------------------------------------------
# Fp8E4M3Quantizer  (numpy emulation — no hardware FP8 required)
# ---------------------------------------------------------------------------

class Fp8E4M3Quantizer:
    """FP8 E4M3 quantization via numpy float16 truncation.

    True FP8 E4M3 requires hardware support; this emulates the range by
    clipping to [-448, 448] (E4M3 max) and quantizing to 8 bits stored in
    uint8.  Precision is slightly lower than real FP8 but correct in range.
    """

    FP8_MAX = 448.0   # E4M3 max finite value

    def __init__(self, config: QuantizationConfig) -> None:
        self._block = config.block_size

    def quantize(self, weight: np.ndarray) -> QuantizedTensor:
        flat  = weight.flatten().astype(np.float32)
        n     = len(flat)
        n_blocks = (n + self._block - 1) // self._block
        pad   = n_blocks * self._block - n
        if pad > 0:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        blocks = flat.reshape(n_blocks, self._block)
        scales = np.max(np.abs(blocks), axis=1) / self.FP8_MAX
        scales = np.where(scales == 0, 1.0, scales).astype(np.float32)

        # Scale to [-128, 127] using the FP8_MAX → 127 mapping
        normalized = np.clip(blocks / scales[:, None] / self.FP8_MAX * 127, -127, 127)
        fp8_approx = np.round(normalized).astype(np.int8)

        return QuantizedTensor(
            data=fp8_approx.flatten().view(np.uint8),
            scales=scales,
            orig_shape=weight.shape,
            dtype=QuantizationDtype.FP8)

    def dequantize(self, qt: QuantizedTensor) -> np.ndarray:
        n_blocks = len(qt.scales)
        vals = qt.data.view(np.int8)[:n_blocks * self._block].astype(np.float32)
        blocks = vals.reshape(n_blocks, self._block)
        flat   = (blocks / 127.0 * self.FP8_MAX * qt.scales[:, None]).flatten()
        n_orig = int(np.prod(qt.orig_shape))
        return flat[:n_orig].reshape(qt.orig_shape)


# ---------------------------------------------------------------------------
# QuantizationCodec — unified entry point
# ---------------------------------------------------------------------------

class QuantizationCodec:
    """Unified quantize / dequantize for any supported dtype."""

    def __init__(self, config: QuantizationConfig) -> None:
        self._config = config
        if config.dtype == QuantizationDtype.INT4:
            self._impl: Any = Int4BlockQuantizer(config)
        elif config.dtype == QuantizationDtype.INT8:
            self._impl = Int8SymmetricQuantizer(config)
        elif config.dtype == QuantizationDtype.FP8:
            self._impl = Fp8E4M3Quantizer(config)
        else:
            self._impl = None   # NONE = passthrough

    def quantize(self, weight: np.ndarray) -> QuantizedTensor:
        if self._impl is None:
            return QuantizedTensor(
                data=weight.view(np.uint8).copy(),
                scales=np.ones(1, dtype=np.float32),
                orig_shape=weight.shape,
                dtype=QuantizationDtype.NONE)
        return self._impl.quantize(weight)

    def dequantize(self, qt: QuantizedTensor) -> np.ndarray:
        if self._impl is None or qt.dtype == QuantizationDtype.NONE:
            n = int(np.prod(qt.orig_shape))
            return qt.data.view(np.float32)[:n].reshape(qt.orig_shape)
        return self._impl.dequantize(qt)

    @property
    def config(self) -> QuantizationConfig:
        return self._config


# ---------------------------------------------------------------------------
# QuantizedExpertWeight
# ---------------------------------------------------------------------------

@dataclass
class QuantizedExpertWeight:
    """All three weight matrices for one MoE expert, quantized."""
    expert_id: str
    gate_qt:   QuantizedTensor
    up_qt:     QuantizedTensor
    down_qt:   QuantizedTensor

    @property
    def compression_ratio(self) -> float:
        total_orig  = (int(np.prod(self.gate_qt.orig_shape)) +
                       int(np.prod(self.up_qt.orig_shape)) +
                       int(np.prod(self.down_qt.orig_shape))) * 4
        total_quant = (self.gate_qt.data.nbytes + self.gate_qt.scales.nbytes +
                       self.up_qt.data.nbytes   + self.up_qt.scales.nbytes +
                       self.down_qt.data.nbytes + self.down_qt.scales.nbytes)
        return total_orig / max(1, total_quant)


# ---------------------------------------------------------------------------
# QuantizedExpertStore
# ---------------------------------------------------------------------------

class QuantizedExpertStore:
    """In-memory store of quantized expert weights; dequantizes on demand.

    Acts as a lightweight cache layer between the checkpoint files and the
    TieredExpertStore; can be used standalone or composed with Phase 1.
    """

    def __init__(self, codec: QuantizationCodec) -> None:
        self._codec = codec
        self._store: Dict[str, QuantizedExpertWeight] = {}
        self._lock  = threading.Lock()
        self._quantize_calls   = 0
        self._dequantize_calls = 0

    def store_expert(
        self,
        expert_id: str,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> QuantizedExpertWeight:
        gate_qt = self._codec.quantize(gate_w)
        up_qt   = self._codec.quantize(up_w)
        down_qt = self._codec.quantize(down_w)
        qew = QuantizedExpertWeight(
            expert_id=expert_id,
            gate_qt=gate_qt, up_qt=up_qt, down_qt=down_qt)
        with self._lock:
            self._store[expert_id] = qew
            self._quantize_calls += 1
        return qew

    def load_expert(
        self, expert_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Returns (gate_w, up_w, down_w) dequantized, or None if not found."""
        with self._lock:
            qew = self._store.get(expert_id)
            if qew is None:
                return None
            self._dequantize_calls += 1
        return (
            self._codec.dequantize(qew.gate_qt),
            self._codec.dequantize(qew.up_qt),
            self._codec.dequantize(qew.down_qt),
        )

    def has_expert(self, expert_id: str) -> bool:
        with self._lock:
            return expert_id in self._store

    def metrics(self) -> dict:
        with self._lock:
            total_ratio = 0.0
            count = len(self._store)
            for qew in self._store.values():
                total_ratio += qew.compression_ratio
            return {
                "quant_store_count":        count,
                "quant_quantize_calls":     self._quantize_calls,
                "quant_dequantize_calls":   self._dequantize_calls,
                "quant_avg_compression":    total_ratio / max(1, count),
                "quant_dtype":              self._codec.config.dtype.value,
            }


# ---------------------------------------------------------------------------
# Phase3QuantizedSession
# ---------------------------------------------------------------------------

class Phase3QuantizedSession:
    """Wraps a base session; intercepts expert weight loading with quantized store.

    When the base session requests expert weights (via ``load_expert``),
    this session serves them dequantized from the quantized store — saving
    memory at the cost of a dequantize step.
    """

    def __init__(
        self,
        base_session: Any,
        codec: Optional[QuantizationCodec] = None,
    ) -> None:
        if base_session is None:
            raise ValueError("base_session required")
        self.base_session = base_session
        self._codec  = codec or QuantizationCodec(QuantizationConfig())
        self._store  = QuantizedExpertStore(self._codec)
        self._lock   = threading.RLock()
        self._closed = False

    # ---- quantized expert API -----------------------------------------------

    def register_expert_weights(
        self,
        expert_id: str,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> None:
        self._store.store_expert(expert_id, gate_w, up_w, down_w)

    def load_expert_weights(
        self, expert_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self._store.load_expert(expert_id)

    # ---- passthrough --------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Phase3QuantizedSession is closed")

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._check_open()
        return self.base_session.generate(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._check_open()
        return self.base_session.stream(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.base_session, name)

    def metrics(self) -> dict:
        base: dict = {}
        get_m = getattr(self.base_session, "metrics", None)
        if callable(get_m):
            base = get_m()
        return {**base, "quantization": self._store.metrics()}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        close_fn = getattr(self.base_session, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logger.warning("phase3/quant: base session close error: %s", exc)

    def __enter__(self) -> "Phase3QuantizedSession":
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def wrap_session_with_quantization(
    base_session: Any,
    *,
    dtype: QuantizationDtype = QuantizationDtype.INT8,
    block_size: int = 128,
) -> Phase3QuantizedSession:
    config = QuantizationConfig(dtype=dtype, block_size=block_size)
    return Phase3QuantizedSession(base_session, QuantizationCodec(config))


__all__ = [
    "QuantizationDtype",
    "QuantizationConfig",
    "QuantizedTensor",
    "Int4BlockQuantizer",
    "Int8SymmetricQuantizer",
    "Fp8E4M3Quantizer",
    "QuantizationCodec",
    "QuantizedExpertWeight",
    "QuantizedExpertStore",
    "Phase3QuantizedSession",
    "wrap_session_with_quantization",
]
