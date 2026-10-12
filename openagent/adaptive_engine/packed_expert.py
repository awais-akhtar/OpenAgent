"""Clean-room OpenAgent rowwise INT4 v1; no upstream format compatibility."""

from __future__ import annotations

import numpy as np

from .expert_kernels import (
    ExpertPayloadSource, _finite_array, _finite_matrix, _finite_output,
    _lease, _validate_routes,
    KernelBackendInfo, KernelCapabilities, KernelEvidence,
)

DTYPE = "int4-row"


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must be a positive uint32 integer")
    return value


def _shapes(hidden_size, intermediate_size):
    h = _positive(hidden_size, "hidden_size")
    i = _positive(intermediate_size, "intermediate_size")
    return ((i, h), (i, h), (h, i))


def packed_payload_size(hidden_size: int, intermediate_size: int) -> int:
    return sum(r * ((c + 1) // 2 + 4) for r, c in _shapes(hidden_size, intermediate_size))


def pack_expert(gate, up, down, *, block_rows: int = 32) -> bytes:
    """Pack real matrices using symmetric [-7, 7] quantization, ties to even.

    Inputs and the returned payload are caller-sized; scratch is row-block sized.
    """
    _positive(block_rows, "block_rows")
    matrices = tuple(np.asarray(a) for a in (gate, up, down))
    if matrices[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    i, h = matrices[0].shape
    shapes = _shapes(h, i)
    for a, shape in zip(matrices, shapes):
        if a.shape != shape or a.dtype.kind not in "iuf":
            raise ValueError(f"weights must be real matrices with shape {shape}")
    result = bytearray(packed_payload_size(h, i))
    offset = 0
    scale_offset = sum(r * ((c + 1) // 2) for r, c in shapes)
    for a, (rows, cols) in zip(matrices, shapes):
        stride = (cols + 1) // 2
        packed, scales = pack_matrix(a, block_rows=block_rows)
        result[offset:offset + rows * stride] = packed
        result[scale_offset:scale_offset + rows * 4] = scales.tobytes()
        offset += rows * stride
        scale_offset += rows * 4
    return bytes(result)


def _layout(payload, h, i, block_rows):
    shapes = _shapes(h, i)
    if payload.nbytes != packed_payload_size(h, i):
        raise ValueError("packed payload size mismatch")
    offset = 0
    scale_offset = sum(r * ((c + 1) // 2) for r, c in shapes)
    layout = []
    for rows, cols in shapes:
        stride = (cols + 1) // 2
        # Validate every block before any arithmetic, including unused rows.
        for start in range(0, rows, block_rows):
            count = min(block_rows, rows - start)
            codes = np.frombuffer(payload, np.uint8, count * stride, offset + start * stride).reshape(count, stride)
            scales = np.frombuffer(payload, "<f4", count, scale_offset + start * 4)
            if not np.isfinite(scales).all() or np.any(scales <= 0) or np.any(scales.astype(np.float64) * 7 > np.finfo(np.float32).max):
                raise ValueError("packed scales must be positive, finite, and safely dequantizable")
            if np.any((codes & 15) == 8) or np.any((codes >> 4) == 8):
                raise ValueError("reserved INT4 code -8")
            if cols % 2 and np.any(codes[:, -1] >> 4):
                raise ValueError("nonzero odd-row padding")
        layout.append((rows, cols, offset, scale_offset))
        offset += rows * stride
        scale_offset += rows * 4
    return layout


def _blocks(payload, spec, block_rows):
    rows, cols, offset, scale_offset = spec
    stride = (cols + 1) // 2
    for start in range(0, rows, block_rows):
        count = min(block_rows, rows - start)
        raw = np.frombuffer(payload, np.uint8, count * stride, offset + start * stride).reshape(count, stride)
        codes = np.empty((count, cols), dtype=np.int8)
        codes[:, ::2] = raw & 15
        codes[:, 1::2] = raw[:, :cols // 2] >> 4
        codes[codes >= 8] -= 16
        scales = np.frombuffer(payload, "<f4", count, scale_offset + start * 4)
        yield start, codes.astype(np.float32) * scales[:, None]


def dequantize_expert(payload, *, hidden_size: int, intermediate_size: int, block_rows: int = 32):
    """Return owned FP32 matrices (explicitly allocates all decoded weights)."""
    _positive(block_rows, "block_rows")
    with memoryview(payload) as view:
        if not view.c_contiguous:
            raise ValueError("payload must be contiguous")
        layout = _layout(view, hidden_size, intermediate_size, block_rows)
        matrices = []
        for spec in layout:
            matrix = np.empty(spec[:2], np.float32)
            for start, block in _blocks(view, spec, block_rows):
                matrix[start:start + len(block)] = block
            matrices.append(matrix)
        return tuple(matrices)


class PackedInt4Backend:
    """Leased-byte INT4 executor registered by the shared expert registry."""

    def __init__(self, *, block_rows: int = 32):
        self.block_rows = _positive(block_rows, "block_rows")

    def info(self):
        return KernelBackendInfo(
            name="packed-int4-reference", provider="openagent", available=True,
            priority=-9000,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True),
            evidence=(KernelEvidence("implementation", "Bounded NumPy INT4 decoding", True),),
        )

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        _shapes(hidden_size, intermediate_size)
        if dtype != DTYPE:
            raise ValueError("unsupported packed dtype")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        batch = _finite_matrix(inputs, hidden_size)
        routes = list(route_batch)
        if len(routes) != len(batch):
            raise ValueError("one route list is required per input row")
        routes = [_validate_routes(row, "routes") for row in routes]
        output = np.zeros_like(batch)
        # Process one token/expert at a time to bound activation scratch too.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for row, assignments in enumerate(routes):
                for key, weight in assignments:
                    if not np.isfinite(np.float32(weight)):
                        raise ValueError("route weight must be representable as float32")
                    with _lease(source, key) as payload:
                        layout = _layout(payload, hidden_size, intermediate_size, self.block_rows)
                        projections = []
                        for spec in layout[:2]:
                            projection = np.empty(intermediate_size, np.float32)
                            for start, block in _blocks(payload, spec, self.block_rows):
                                projection[start:start + len(block)] = block @ batch[row]
                            projections.append(projection)
                        gate, up = projections
                        activated = gate * np.exp(-np.logaddexp(0, -gate)) * up
                        _finite_output(activated)
                        for start, block in _blocks(payload, layout[2], self.block_rows):
                            output[row, start:start + len(block)] += np.float32(weight) * (block @ activated)
        return _finite_output(output)


payload_size = packed_payload_size


def decode_expert(payload, hidden, intermediate, *, block_rows=32):
    return dequantize_expert(payload, hidden_size=hidden, intermediate_size=intermediate,
                             block_rows=block_rows)


def pack_matrix(matrix, *, block_rows=32):
    """Pack one projection, returning packed bytes and little-endian FP32 scales."""
    _positive(block_rows, "block_rows")
    a = np.asarray(matrix)
    if a.ndim != 2 or a.dtype.kind not in "iuf":
        raise ValueError("matrix must contain real weights")
    rows, cols = a.shape
    _positive(rows, "rows")
    _positive(cols, "cols")
    stride = (cols + 1) // 2
    packed = bytearray(rows * stride)
    scales = np.empty(rows, dtype="<f4")
    for start in range(0, rows, block_rows):
        end = min(rows, start + block_rows)
        with np.errstate(over="ignore", invalid="ignore"):
            values = a[start:end].astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("weights must be finite and representable as float32")
        maximum = np.max(np.abs(values), axis=1)
        scale = np.maximum(maximum / np.float32(7), np.nextafter(np.float32(0), np.float32(1)))
        scale = np.minimum(scale, np.nextafter(np.float32(float(np.finfo(np.float32).max) / 7), np.float32(0)))
        scale[maximum == 0] = 1
        codes = (np.clip(np.rint(values / scale[:, None]), -7, 7).astype(np.int16) & 15).astype(np.uint8)
        block = codes[:, ::2].copy()
        block[:, :cols // 2] |= codes[:, 1::2] << 4
        packed[start * stride:end * stride] = block.tobytes()
        scales[start:end] = scale
    return bytes(packed), scales


PackedRoutedSwiGLUBackend = PackedInt4Backend
