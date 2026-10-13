"""OpenAgent INT2 row and INT3 group-64 payloads with explicit bit geometry.

These are owned formats, not GGUF, GPTQ, AWQ or an upstream checkpoint ABI.
Codes are little-endian bit streams within each row; scales follow all codes.
"""

import math

import numpy as np

from .expert_kernels import (
    ExpertPayloadSource, KernelBackendInfo, KernelCapabilities, KernelEvidence,
    _finite_array, _finite_matrix, _lease, _validate_routes,
)
from .q5_0_expert import _detach_borrowed_frames
from .q8_0_expert import _positive

INT2 = "int2-row"
INT3 = "int3-grouped-g64"


def parameters(dtype):
    if dtype == INT2:
        return 2, 0
    if dtype == INT3:
        return 3, 64
    raise ValueError("unsupported low-bit payload dtype")


def _shapes(hidden, intermediate):
    _positive(hidden, "hidden_size")
    _positive(intermediate, "intermediate_size")
    return ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))


def payload_size(hidden_size, intermediate_size, *, dtype):
    bits, group = parameters(dtype)
    return sum(rows * ((columns * bits + 7) // 8)
               + rows * ((columns + group - 1) // group if group else 1) * 4
               for rows, columns in _shapes(hidden_size, intermediate_size))


def pack_matrix(matrix, *, dtype):
    """Return packed row codes and FP32 scales using independent symmetric RNE."""
    bits, group = parameters(dtype)
    values = np.asarray(matrix)
    if values.ndim != 2 or values.dtype.kind not in "iuf":
        raise ValueError("weights must be real matrices")
    rows, columns = values.shape
    _positive(rows, "rows")
    _positive(columns, "columns")
    group = group or columns
    groups = (columns + group - 1) // group
    stride = (columns * bits + 7) // 8
    output = bytearray(rows * stride)
    scales = np.empty((rows, groups), dtype="<f4")
    maximum_code = (1 << (bits - 1)) - 1
    for row in range(rows):
        for index in range(groups):
            start, end = index * group, min(columns, (index + 1) * group)
            with np.errstate(over="ignore", invalid="ignore"):
                block = values[row, start:end].astype(np.float32)
            if not np.isfinite(block).all():
                raise ValueError("weights must be finite float32")
            maximum = float(np.max(np.abs(block)))
            delta = np.float32(maximum / maximum_code) if maximum else np.float32(1)
            if maximum and delta == 0:
                delta = np.nextafter(np.float32(0), np.float32(1))
            codes = np.clip(np.rint(block.astype(np.float64) / float(delta)),
                            -maximum_code, maximum_code).astype(np.int8)
            with np.errstate(over="ignore"):
                if not np.isfinite((codes.astype(np.float64) * float(delta)).astype(np.float32)).all():
                    raise ValueError("quantized weights exceed float32 range")
            scales[row, index] = delta
            for column, code in enumerate(codes, start):
                address, shift = divmod(column * bits, 8)
                encoded = int(code) & ((1 << bits) - 1)
                output[row * stride + address] |= (encoded << shift) & 255
                if shift + bits > 8:
                    output[row * stride + address + 1] |= encoded >> (8 - shift)
    return bytes(output), scales


def pack_expert(gate, up, down, *, dtype):
    values = [np.asarray(matrix) for matrix in (gate, up, down)]
    if values[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    intermediate, hidden = values[0].shape
    if any(matrix.shape != shape for matrix, shape in zip(values, _shapes(hidden, intermediate))):
        raise ValueError("expert projection shape mismatch")
    packed = [pack_matrix(matrix, dtype=dtype) for matrix in values]
    return b"".join(codes for codes, _ in packed) + b"".join(scales.tobytes() for _, scales in packed)


def _code(raw, offset, column, bits):
    address, shift = divmod(column * bits, 8)
    value = int(raw[offset + address]) >> shift
    if shift + bits > 8:
        value |= int(raw[offset + address + 1]) << (8 - shift)
    value &= (1 << bits) - 1
    return value if value < (1 << (bits - 1)) else value - (1 << bits)


def _read_f32(raw, offset):
    word = 0
    for byte in range(4):
        word |= int(raw[offset + byte]) << (8 * byte)
    exponent, fraction = (word >> 23) & 255, word & 0x7fffff
    if exponent == 255:
        return np.float32(np.nan)
    magnitude = math.ldexp(float(fraction), -149) if exponent == 0 else math.ldexp(
        float(0x800000 + fraction), exponent - 150)
    return np.float32(-magnitude if word & 0x80000000 else magnitude)


def _dot(raw, codes, scales, vector, columns, bits, group):
    total = np.float32(0)
    for index in range((columns + group - 1) // group):
        delta = _read_f32(raw, scales + index * 4)
        for column in range(index * group, min(columns, (index + 1) * group)):
            weight = np.float32(np.float32(_code(raw, codes, column, bits)) * delta)
            total = np.float32(total + np.float32(weight * vector[column]))
    return total


def _kernel(raw, vector, hidden, intermediate, bits, group, coefficient, activation, output):
    gate_stride, down_stride = (hidden * bits + 7) // 8, (intermediate * bits + 7) // 8
    gate_group, down_group = (group or hidden), (group or intermediate)
    gate_groups = (hidden + gate_group - 1) // gate_group
    down_groups = (intermediate + down_group - 1) // down_group
    up_offset = intermediate * gate_stride
    down_offset = 2 * up_offset
    scale_offset = down_offset + hidden * down_stride
    up_scales = scale_offset + intermediate * gate_groups * 4
    down_scales = up_scales + intermediate * gate_groups * 4
    for address in range(scale_offset, raw.size, 4):
        if not math.isfinite(_read_f32(raw, address)):
            return 1
    gate_unused = (hidden * bits) % 8
    down_unused = (intermediate * bits) % 8
    if gate_unused:
        for row in range(2 * intermediate):
            if int(raw[(row + 1) * gate_stride - 1]) >> gate_unused:
                return 3
    if down_unused:
        for row in range(hidden):
            if int(raw[down_offset + (row + 1) * down_stride - 1]) >> down_unused:
                return 3
    for row in range(intermediate):
        gate = _dot(raw, row * gate_stride, scale_offset + row * gate_groups * 4,
                    vector, hidden, bits, gate_group)
        up = _dot(raw, up_offset + row * gate_stride, up_scales + row * gate_groups * 4,
                  vector, hidden, bits, gate_group)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        exponential = math.exp(-abs(float(gate)))
        sigmoid = np.float32((1.0 if gate >= 0 else exponential) / (1.0 + exponential))
        activation[row] = np.float32(np.float32(gate * sigmoid) * up)
        if not math.isfinite(activation[row]):
            return 2
    for row in range(hidden):
        value = _dot(raw, down_offset + row * down_stride, down_scales + row * down_groups * 4,
                     activation, intermediate, bits, down_group)
        output[row] = np.float32(output[row] + np.float32(value * coefficient))
        if not math.isfinite(value) or not math.isfinite(output[row]):
            return 2
    return 0


class LowBitExpertBackend:
    """Fixed-geometry leased CPU execution, without decoded weight matrices."""

    def __init__(self, dtype=INT2, *, max_scratch_bytes=64 * 1024 * 1024):
        self.dtype = dtype
        self.bits, self.group = parameters(dtype)
        self.NAME = dtype + "-reference"
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def info(self):
        return KernelBackendInfo(name=self.NAME, provider="openagent", available=True, priority=-12000,
            capabilities=KernelCapabilities(frozenset({self.dtype}), True, True),
            evidence=(KernelEvidence("implementation", "Direct low-bit row reductions"),))

    def _check_available(self):
        if (self.dtype not in (INT2, INT3) or type(self.bits) is not int or
                self.bits not in (2, 3) or (self.group is not None and
                (type(self.group) is not int or self.group <= 0)) or
                type(self.max_scratch_bytes) is not int or self.max_scratch_bytes <= 0):
            raise RuntimeError("low-bit backend admission state is invalid")

    def _run(self, raw, *arguments):
        return _kernel(raw, *arguments)

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=None):
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=None):
        self._check_available()
        dtype = self.dtype if dtype is None else dtype
        if dtype != self.dtype:
            raise ValueError("selected low-bit backend has another format")
        length = payload_size(hidden_size, intermediate_size, dtype=dtype)
        if intermediate_size * 4 > self.max_scratch_bytes:
            raise MemoryError("activation exceeds scratch budget")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        vectors = np.ascontiguousarray(_finite_matrix(inputs, hidden_size))
        routing = list(route_batch)
        if len(routing) != len(vectors):
            raise ValueError("one route list required per input row")
        routing = [_validate_routes(row, "routes") for row in routing]
        if any(weight > float(np.finfo(np.float32).max) for routes in routing for _, weight in routes):
            raise ValueError("route weight must be representable as float32")
        output = np.zeros_like(vectors)
        activation = np.empty(intermediate_size, np.float32)
        for row, routes in enumerate(routing):
            for key, weight in routes:
                failure = None
                with _lease(source, key) as payload:
                    if payload.nbytes != length:
                        raise ValueError("low-bit payload size mismatch")
                    raw = np.frombuffer(payload, np.uint8)
                    try:
                        with np.errstate(over="ignore", invalid="ignore"):
                            status = self._run(raw, vectors[row], hidden_size, intermediate_size,
                                               self.bits, self.group, np.float32(weight), activation, output[row])
                    except BaseException as error:
                        _detach_borrowed_frames(error)
                        failure = error
                    finally:
                        del raw
                if failure is not None:
                    raise failure from None
                if status == 1:
                    raise ValueError("scales must be finite")
                if status == 3:
                    raise ValueError("row padding must be zero")
                if status:
                    raise ArithmeticError("low-bit computation produced non-finite output")
        return output
