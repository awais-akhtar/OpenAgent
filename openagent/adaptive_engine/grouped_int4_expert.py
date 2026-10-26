"""Independent INT4 group scaling using public geometry, not reference code.

Geometry: reviewed Colibri docs/FORMATS.md, ordinal 4. OpenAgent explicitly
uses two's-complement nibbles: even columns low, odd columns high. Signed
finite FP32 scales multiply individual codes before projection reduction.
"""

import math
import struct

import numpy as np

from .expert_kernels import (ExpertPayloadSource, KernelBackendInfo, KernelCapabilities,
                             KernelEvidence, _finite_array, _finite_matrix, _lease, _validate_routes)
from .q5_0_expert import _positive, _detach_borrowed_frames

DTYPE = "int4-grouped"
MANIFEST_DTYPE = "int4-grouped-g64"
DEFAULT_GROUP_SIZE = 64


def _shapes(hidden, intermediate):
    _positive(hidden, "hidden_size")
    _positive(intermediate, "intermediate_size")
    return ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))


def _groups(width, group_size):
    return (width + group_size - 1) // group_size


def _layout(hidden, intermediate, group_size):
    _positive(group_size, "group_size")
    shapes = _shapes(hidden, intermediate)
    code_end = sum(rows * ((cols + 1) // 2) for rows, cols in shapes)
    codes, scales = 0, code_end
    layout = []
    for rows, cols in shapes:
        layout.append((rows, cols, codes, scales))
        codes += rows * ((cols + 1) // 2)
        scales += 4 * rows * _groups(cols, group_size)
    return tuple(layout), code_end, scales


def payload_size(hidden_size, intermediate_size, *, group_size=DEFAULT_GROUP_SIZE):
    return _layout(hidden_size, intermediate_size, group_size)[2]


def pack_matrix(matrix, *, group_size=DEFAULT_GROUP_SIZE, row_count=64):
    """Return nibble bytes and little-endian FP32 scales with bounded row tiles.

    Independent max-abs/7, ties-even, [-7,7] policy. Zero groups use scale 1.
    Nonzero underflow scales floor to the smallest FP32 subnormal. Reject
    quantized FP32 weight overflow instead of emitting an unusable payload.
    """
    _positive(group_size, "group_size")
    _positive(row_count, "row_count")
    weights = np.asarray(matrix)
    if weights.ndim != 2 or weights.dtype.kind not in "iuf":
        raise ValueError("weights must be real matrices")
    rows, columns = weights.shape
    _positive(rows, "rows")
    _positive(columns, "columns")
    stride = (columns + 1) // 2
    groups = _groups(columns, group_size)
    codes = bytearray(rows * stride)
    scales = np.empty((rows, groups), dtype="<f4")
    padded_columns = columns + (columns & 1)
    for first in range(0, rows, row_count):
        stop_row = min(rows, first + row_count)
        with np.errstate(over="ignore", invalid="ignore"):
            tile = weights[first:stop_row].astype(np.float32)
        if not np.isfinite(tile).all():
            raise ValueError("weights must be finite and representable as float32")
        quantized = np.zeros((stop_row - first, padded_columns), dtype=np.int8)
        for group in range(groups):
            start, stop = group * group_size, min(columns, (group + 1) * group_size)
            values = tile[:, start:stop]
            maxima = np.max(np.abs(values), axis=1)
            group_scales = np.where(maxima == 0, np.float32(1),
                                    (maxima.astype(np.float64) / 7).astype(np.float32))
            group_scales[(maxima != 0) & (group_scales == 0)] = np.nextafter(np.float32(0), np.float32(1))
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                integers = np.clip(
                    np.rint(values.astype(np.float64) / group_scales.astype(np.float64)[:, None]),
                    -7, 7).astype(np.int8)
                reconstructed = (integers.astype(np.float64) *
                                 group_scales.astype(np.float64)[:, None]).astype(np.float32)
            if not np.isfinite(reconstructed).all():
                raise ValueError("quantized grouped weights must be representable as float32")
            quantized[:, start:stop] = integers
            scales[first:stop_row, group] = group_scales
        unsigned = quantized.astype(np.uint8) & np.uint8(15)
        packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << np.uint8(4))
        codes[first * stride:stop_row * stride] = packed.tobytes()
    return bytes(codes), scales


def pack_expert(gate, up, down, *, group_size=DEFAULT_GROUP_SIZE):
    matrices = [np.asarray(matrix) for matrix in (gate, up, down)]
    if matrices[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    i, h = matrices[0].shape
    if any(matrix.shape != shape for matrix, shape in zip(matrices, _shapes(h, i))):
        raise ValueError("expert projection shape mismatch")
    packed = [pack_matrix(matrix, group_size=group_size) for matrix in matrices]
    return b"".join(pair[0] for pair in packed) + b"".join(pair[1].tobytes() for pair in packed)


def _validate(payload, layout, code_end):
    for offset in range(code_end, payload.nbytes, 4):
        if not math.isfinite(struct.unpack_from("<f", payload, offset)[0]):
            return 1
    for rows, cols, offset, _ in layout:
        if cols % 2:
            stride = (cols + 1) // 2
            for row in range(rows):
                if struct.unpack_from("B", payload, offset + (row + 1) * stride - 1)[0] >> 4:
                    return 3
    return 0


def decode_expert(payload, hidden_size, intermediate_size, *, group_size=DEFAULT_GROUP_SIZE):
    """Explicit full decode, owned FP32 matrices; not kernel scratch."""
    layout, code_end, size = _layout(hidden_size, intermediate_size, group_size)
    with memoryview(payload) as view:
        if not view.c_contiguous or view.nbytes != size:
            raise ValueError("grouped payload must be contiguous with exact size")
        status = _validate(view, layout, code_end)
        if status == 1:
            raise ValueError("group scales must be finite")
        if status:
            raise ValueError("odd-row padding must be zero")
        output = []
        for rows, cols, codes, scales in layout:
            matrix = np.empty((rows, cols), np.float32)
            stride, groups = (cols + 1) // 2, _groups(cols, group_size)
            with np.errstate(over="ignore", invalid="ignore"):
                for row in range(rows):
                    for column in range(cols):
                        byte = struct.unpack_from("B", view, codes + row * stride + column // 2)[0]
                        nibble = (byte >> (4 * (column % 2))) & 15
                        signed = nibble if nibble < 8 else nibble - 16
                        scale = struct.unpack_from("<f", view, scales + 4 * (row * groups + column // group_size))[0]
                        matrix[row, column] = np.float32(np.float64(signed) * scale)
            if not np.isfinite(matrix).all():
                raise ArithmeticError("grouped weight decode overflow")
            output.append(matrix)
    return tuple(output)


def _dot(payload, codes, scales, x, width, group_size):
    total = np.float32(0)
    for group in range(_groups(width, group_size)):
        scale = struct.unpack_from("<f", payload, scales + 4 * group)[0]
        for column in range(group * group_size, min(width, (group + 1) * group_size)):
            byte = struct.unpack_from("B", payload, codes + column // 2)[0]
            code = (byte >> (4 * (column % 2))) & 15
            signed = code if code < 8 else code - 16
            weight = np.float32(signed * scale)
            total = np.float32(total + np.float32(weight * x[column]))
    return total


def _scalar_expert(payload, x, h, i, group_size, coefficient, activation, output):
    layout, end, _ = _layout(h, i, group_size)
    status = _validate(payload, layout, end)
    if status:
        return status
    gate, up, down = layout
    code_stride = (h + 1) // 2
    scale_stride = 4 * _groups(h, group_size)
    for row in range(i):
        g = _dot(payload, gate[2] + row * code_stride, gate[3] + row * scale_stride, x, h, group_size)
        u = _dot(payload, up[2] + row * code_stride, up[3] + row * scale_stride, x, h, group_size)
        if not math.isfinite(g) or not math.isfinite(u):
            return 2
        e = math.exp(-abs(float(g)))
        sigmoid = np.float32((1.0 if g >= 0 else e) / (1.0 + e))
        activation[row] = np.float32(np.float32(g * sigmoid) * u)
        if not math.isfinite(activation[row]):
            return 2
    code_stride, scale_stride = (i + 1) // 2, 4 * _groups(i, group_size)
    for row in range(h):
        value = _dot(payload, down[2] + row * code_stride, down[3] + row * scale_stride,
                     activation, i, group_size)
        if not math.isfinite(value):
            return 2
        output[row] = np.float32(output[row] + np.float32(coefficient * value))
        if not math.isfinite(output[row]):
            return 2
    return 0


def scratch_bytes(hidden_size, intermediate_size, *, group_size=DEFAULT_GROUP_SIZE):
    _layout(hidden_size, intermediate_size, group_size)
    return 4 * intermediate_size


class GroupedInt4ExpertBackend:
    NAME = "grouped-int4-reference"

    def __init__(self, *, group_size=DEFAULT_GROUP_SIZE, max_scratch_bytes=64 * 1024 * 1024):
        self.group_size = _positive(group_size, "group_size")
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def info(self):
        return KernelBackendInfo(name=self.NAME, provider="openagent", available=True, priority=-12000,
            capabilities=KernelCapabilities(frozenset({DTYPE, MANIFEST_DTYPE} if self.group_size == 64 else {DTYPE}), True, True),
            evidence=(KernelEvidence("implementation", "Independent grouped INT4 byte reductions"),))

    def _check_available(self):
        if (type(self.group_size) is not int or self.group_size <= 0 or
                type(self.max_scratch_bytes) is not int or self.max_scratch_bytes <= 0):
            raise RuntimeError("grouped INT4 backend admission state is invalid")

    def _run(self, payload, x, h, i, weight, activation, output):
        return _scalar_expert(payload, x, h, i, self.group_size, weight, activation, output)

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        self._check_available()
        _shapes(hidden_size, intermediate_size)
        x = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, x[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        self._check_available()
        expected = payload_size(hidden_size, intermediate_size, group_size=self.group_size)
        if scratch_bytes(hidden_size, intermediate_size, group_size=self.group_size) > self.max_scratch_bytes:
            raise MemoryError("grouped activation scratch exceeds max_scratch_bytes")
        if dtype != DTYPE and not (dtype == MANIFEST_DTYPE and self.group_size == 64):
            raise ValueError("unsupported grouped dtype")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        batch = np.ascontiguousarray(_finite_matrix(inputs, hidden_size))
        rows = list(route_batch)
        if len(rows) != len(batch):
            raise ValueError("one route list required per input row")
        rows = [_validate_routes(routes, "routes") for routes in rows]
        for routes in rows:
            if any(weight > float(np.finfo(np.float32).max) for _, weight in routes):
                raise ValueError("route weights must be representable as float32")
        output = np.zeros_like(batch)
        if not len(batch):
            return output
        activation = np.empty(intermediate_size, np.float32)
        for row, routes in enumerate(rows):
            for key, weight in routes:
                failure = None
                with _lease(source, key) as payload:
                    if payload.nbytes != expected:
                        raise ValueError("grouped payload size mismatch")
                    try:
                        with np.errstate(over="ignore", invalid="ignore"):
                            status = self._run(payload, batch[row], hidden_size, intermediate_size,
                                               np.float32(weight), activation, output[row])
                    except BaseException as error:
                        _detach_borrowed_frames(error)
                        failure = error
                if failure is not None:
                    raise failure from None
                if status == 1:
                    raise ValueError("group scales must be finite")
                if status == 3:
                    raise ValueError("odd-row padding must be zero")
                if status:
                    raise ArithmeticError("grouped computation produced non-finite output")
        return output
