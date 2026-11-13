"""GGUF reader and quantized tensor kernels for the Qwen3-MoE family.

Only the pieces the Qwen3-MoE decoder needs: a bounds-checked GGUF metadata
and tensor-index reader, and exact dequantization for the ggml types that
Q4_K_M and Q8_0 files use (F32, F16, Q4_K, Q6_K, Q8_0). Every block layout
follows ggml's reference dequantizers; the tests check them against values
computed independently.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_ALIGNMENT_DEFAULT = 32
MAX_GGUF_STRING = 1 << 20
MAX_GGUF_ARRAY = 1 << 26

# ggml tensor types used here: id -> (name, elements per block, bytes per block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    14: ("Q6_K", 256, 210),
    30: ("BF16", 1, 2),
}

_KV_SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_KV_STRING = 8
_KV_ARRAY = 9


class GGUFError(ValueError):
    """The file is not a GGUF file this reader can trust."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int  # relative to the tensor data section
    absolute: int = 0  # byte offset in the file

    @property
    def elements(self) -> int:
        total = 1
        for dim in self.dims:
            total *= dim
        return total

    @property
    def nbytes(self) -> int:
        name, per_block, block_bytes = _ggml_type(self.ggml_type)
        if self.elements % per_block:
            raise GGUFError(f"tensor {self.name} has {self.elements} elements, not a multiple of {per_block}")
        return self.elements // per_block * block_bytes


def _ggml_type(ggml_type: int) -> tuple[str, int, int]:
    try:
        return GGML_TYPES[ggml_type]
    except KeyError:
        raise GGUFError(f"unsupported ggml tensor type {ggml_type}") from None


@dataclass
class GGUFFile:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: dict[str, TensorInfo]
    data_offset: int
    size: int = 0
    _map: mmap.mmap | None = field(default=None, repr=False)

    @classmethod
    def open(cls, path: str | Path) -> "GGUFFile":
        path = Path(path)
        with path.open("rb") as handle:
            reader = _Reader(handle)
            if reader.read(4) != GGUF_MAGIC:
                raise GGUFError("not a GGUF file")
            version = reader.u32()
            if version not in (2, 3):
                raise GGUFError(f"unsupported GGUF version {version}")
            tensor_count = reader.u64()
            kv_count = reader.u64()
            if kv_count > 1 << 20 or tensor_count > 1 << 20:
                raise GGUFError("implausible GGUF header counts")
            metadata: dict[str, Any] = {}
            for _ in range(kv_count):
                key = reader.string()
                metadata[key] = reader.value(reader.u32())
            alignment = int(metadata.get("general.alignment", GGUF_ALIGNMENT_DEFAULT))
            if alignment <= 0 or alignment & (alignment - 1):
                raise GGUFError("general.alignment must be a power of two")
            tensors: dict[str, TensorInfo] = {}
            for _ in range(tensor_count):
                name = reader.string()
                n_dims = reader.u32()
                if not 1 <= n_dims <= 4:
                    raise GGUFError(f"tensor {name} has {n_dims} dimensions")
                dims = tuple(reader.u64() for _ in range(n_dims))
                ggml_type = reader.u32()
                offset = reader.u64()
                if name in tensors:
                    raise GGUFError(f"duplicate tensor {name}")
                tensors[name] = TensorInfo(name, dims, ggml_type, offset)
            position = reader.handle.tell()
            data_offset = (position + alignment - 1) // alignment * alignment
            size = path.stat().st_size
        resolved: dict[str, TensorInfo] = {}
        for name, info in tensors.items():
            absolute = data_offset + info.offset
            if absolute + info.nbytes > size:
                raise GGUFError(f"tensor {name} extends past the end of the file")
            resolved[name] = TensorInfo(info.name, info.dims, info.ggml_type, info.offset, absolute)
        return cls(path=path, version=version, metadata=metadata, tensors=resolved,
                   data_offset=data_offset, size=size)

    def bytes_of(self, name: str) -> memoryview:
        """Raw quantized bytes of one tensor, read through a shared memory map."""
        info = self.tensors[name]
        return self.bytes_at(info.absolute, info.nbytes)

    def bytes_at(self, start: int, count: int) -> memoryview:
        if start < 0 or count < 0 or start + count > self.size:
            raise GGUFError("byte range lies outside the GGUF file")
        return self._mapping()[start:start + count]

    def _mapping(self) -> mmap.mmap:
        if self._map is None:
            with self.path.open("rb") as handle:
                self._map = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        return self._map

    def close(self) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None


class _Reader:
    def __init__(self, handle) -> None:
        self.handle = handle

    def read(self, count: int) -> bytes:
        if count < 0 or count > MAX_GGUF_ARRAY:
            raise GGUFError("implausible GGUF read")
        data = self.handle.read(count)
        if len(data) != count:
            raise GGUFError("truncated GGUF file")
        return data

    def unpack(self, fmt: str, size: int):
        return struct.unpack(fmt, self.read(size))[0]

    def u32(self) -> int:
        return int(self.unpack("<I", 4))

    def u64(self) -> int:
        return int(self.unpack("<Q", 8))

    def string(self) -> str:
        length = self.u64()
        if length > MAX_GGUF_STRING:
            raise GGUFError("GGUF string is too long")
        return self.read(length).decode("utf-8")

    def value(self, value_type: int):
        if value_type in _KV_SCALAR:
            fmt, size = _KV_SCALAR[value_type]
            return self.unpack(fmt, size)
        if value_type == _KV_STRING:
            return self.string()
        if value_type == _KV_ARRAY:
            element_type = self.u32()
            count = self.u64()
            if count > MAX_GGUF_ARRAY:
                raise GGUFError("GGUF array is too long")
            return [self.value(element_type) for _ in range(count)]
        raise GGUFError(f"unsupported GGUF metadata type {value_type}")


# --- dequantization -------------------------------------------------------

def _f16(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view("<f2").astype(np.float32)


def dequantize(raw: memoryview | bytes, ggml_type: int, elements: int) -> np.ndarray:
    """Return the first `elements` values of a tensor as float32, in order."""
    name, per_block, block_bytes = _ggml_type(ggml_type)
    if elements % per_block:
        raise GGUFError("element count is not a whole number of blocks")
    blocks = elements // per_block
    buffer = np.frombuffer(raw, dtype=np.uint8, count=blocks * block_bytes)
    if buffer.size != blocks * block_bytes:
        raise GGUFError("tensor bytes are truncated")
    if ggml_type == 0:
        return np.frombuffer(raw, dtype="<f4", count=elements).astype(np.float32)
    if ggml_type == 1:
        return np.frombuffer(raw, dtype="<f2", count=elements).astype(np.float32)
    if ggml_type == 30:
        bits = np.frombuffer(raw, dtype="<u2", count=elements).astype(np.uint32) << 16
        return bits.view(np.float32)
    blocks_view = buffer.reshape(blocks, block_bytes)
    if ggml_type == 8:
        return _dequant_q8_0(blocks_view).reshape(-1)
    if ggml_type == 12:
        return _dequant_q4_k(blocks_view).reshape(-1)
    if ggml_type == 14:
        return _dequant_q6_k(blocks_view).reshape(-1)
    raise GGUFError(f"no dequantizer for {name}")


def _dequant_q8_0(blocks: np.ndarray) -> np.ndarray:
    scale = _f16(blocks[:, 0:2]).reshape(-1, 1)
    quants = blocks[:, 2:34].view(np.int8).astype(np.float32)
    return scale * quants


def _scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the eight 6-bit scale and minimum pairs of a Q4_K block (ggml get_scale_min_k4)."""
    s = scales.astype(np.uint8)
    sc = np.zeros((s.shape[0], 8), dtype=np.float32)
    mn = np.zeros((s.shape[0], 8), dtype=np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63
            mn[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
            mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, mn


def _dequant_q4_k(blocks: np.ndarray) -> np.ndarray:
    d = _f16(blocks[:, 0:2]).reshape(-1, 1)
    dmin = _f16(blocks[:, 2:4]).reshape(-1, 1)
    sc, mn = _scale_min_k4(blocks[:, 4:16])
    qs = blocks[:, 16:144].reshape(-1, 4, 32)
    low = (qs & 0xF).astype(np.float32)
    high = (qs >> 4).astype(np.float32)
    out = np.empty((blocks.shape[0], 4, 2, 32), dtype=np.float32)
    for chunk in range(4):
        out[:, chunk, 0, :] = d * sc[:, 2 * chunk: 2 * chunk + 1] * low[:, chunk] - dmin * mn[:, 2 * chunk: 2 * chunk + 1]
        out[:, chunk, 1, :] = d * sc[:, 2 * chunk + 1: 2 * chunk + 2] * high[:, chunk] - dmin * mn[:, 2 * chunk + 1: 2 * chunk + 2]
    return out.reshape(blocks.shape[0], 256)


def _dequant_q6_k(blocks: np.ndarray) -> np.ndarray:
    ql = blocks[:, 0:128].reshape(-1, 2, 64)
    qh = blocks[:, 128:192].reshape(-1, 2, 32)
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32).reshape(-1, 2, 8)
    d = _f16(blocks[:, 208:210]).reshape(-1, 1)
    out = np.empty((blocks.shape[0], 2, 128), dtype=np.float32)
    for half in range(2):
        lo = ql[:, half, 0:32].astype(np.int32)
        hi = ql[:, half, 32:64].astype(np.int32)
        high_bits = qh[:, half].astype(np.int32)
        q1 = ((lo & 0xF) | (((high_bits >> 0) & 3) << 4)) - 32
        q2 = ((hi & 0xF) | (((high_bits >> 2) & 3) << 4)) - 32
        q3 = ((lo >> 4) | (((high_bits >> 4) & 3) << 4)) - 32
        q4 = ((hi >> 4) | (((high_bits >> 6) & 3) << 4)) - 32
        for part, q in enumerate((q1, q2, q3, q4)):
            sub = np.repeat(scales[:, half, 2 * part: 2 * part + 2], 16, axis=1)
            out[:, half, 32 * part: 32 * part + 32] = d * sub * q.astype(np.float32)
    return out.reshape(blocks.shape[0], 256)
