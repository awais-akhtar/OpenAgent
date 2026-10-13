"""Independent Q5_0 byte contract, quantization and scalar expert execution.

Public field metadata: ggml-org/ggml src/ggml-common.h, block_q5_0.
No upstream quantizer or execution implementation is used.
"""

import math
import struct
import traceback

import numpy as np

from .expert_kernels import (ExpertPayloadSource, KernelBackendInfo,
                             KernelCapabilities, KernelEvidence, _finite_array,
                             _finite_matrix, _lease, _validate_routes)

DTYPE = "q5_0"
BLOCK_WIDTH = 32
BLOCK_BYTES = 22
_BLOCK = struct.Struct("<eI16B")


def _positive(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _shapes(hidden_size, intermediate_size):
    h = _positive(hidden_size, "hidden_size")
    i = _positive(intermediate_size, "intermediate_size")
    if h % BLOCK_WIDTH or i % BLOCK_WIDTH:
        raise ValueError("Q5_0 row widths must be multiples of 32")
    return ((i, h), (i, h), (h, i))


def payload_size(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return hidden_size * intermediate_size * 3 // BLOCK_WIDTH * BLOCK_BYTES


def pack_matrix(matrix, *, block_count=64):
    """Positive max-abs/15 half delta; ties-even centered codes in [-16,15].

    Nonzero half underflow is floored to 2^-24. Zero blocks store zero delta
    and centered-zero codes. This policy is not upstream quantizer parity.
    """
    _positive(block_count, "block_count")
    weights = np.asarray(matrix)
    if weights.ndim != 2 or weights.dtype.kind not in "iuf":
        raise ValueError("weights must be real matrices")
    rows, width = weights.shape
    _positive(rows, "rows")
    _positive(width, "columns")
    if width % BLOCK_WIDTH:
        raise ValueError("Q5_0 row widths must be multiples of 32")
    result = bytearray(rows * width // BLOCK_WIDTH * BLOCK_BYTES)
    cursor = 0
    for row in weights:
        for start in range(0, width, BLOCK_WIDTH * block_count):
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.asarray(row[start:start + BLOCK_WIDTH * block_count],
                                    dtype=np.float32).reshape(-1, BLOCK_WIDTH)
            if not np.isfinite(values).all():
                raise ValueError("weights must be finite and representable as float32")
            for block in values:
                peak = float(np.max(np.abs(block)))
                with np.errstate(over="ignore", under="ignore"):
                    delta = np.float16(peak / 15.0)
                if not np.isfinite(delta):
                    raise ValueError("Q5_0 delta exceeds binary16 range")
                if peak > 0 and delta == 0:
                    delta = np.float16(2 ** -24)
                quants = np.clip(np.rint(block.astype(np.float64) / (float(delta) or 1.0)),
                                 -16, 15).astype(np.int16) + 16
                high_bits = sum((int(code) >> 4) << lane for lane, code in enumerate(quants))
                nibbles = [(int(quants[lane]) & 15) | ((int(quants[lane + 16]) & 15) << 4)
                           for lane in range(16)]
                _BLOCK.pack_into(result, cursor, float(delta), high_bits, *nibbles)
                cursor += BLOCK_BYTES
    return bytes(result)


def pack_expert(gate, up, down, *, block_count=64):
    matrices = tuple(np.asarray(matrix) for matrix in (gate, up, down))
    if matrices[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    i, h = matrices[0].shape
    if tuple(matrix.shape for matrix in matrices) != _shapes(h, i):
        raise ValueError("expert projection shape mismatch")
    return b"".join(pack_matrix(matrix, block_count=block_count) for matrix in matrices)


def decode_expert(payload, hidden_size, intermediate_size):
    """Explicit full-weight decode. Owned matrices are not execution scratch."""
    shapes = _shapes(hidden_size, intermediate_size)
    with memoryview(payload) as view:
        if not view.c_contiguous or view.nbytes != payload_size(hidden_size, intermediate_size):
            raise ValueError("Q5_0 payload must be contiguous with exact size")
        output = []
        address = 0
        for shape in shapes:
            matrix = np.empty(shape, np.float32)
            flat = matrix.reshape(-1)
            for begin in range(0, flat.size, BLOCK_WIDTH):
                delta, high, *packed = _BLOCK.unpack_from(view, address)
                if not math.isfinite(delta):
                    raise ValueError("Q5_0 deltas must be finite")
                for lane in range(BLOCK_WIDTH):
                    low = (packed[lane] & 15) if lane < 16 else (packed[lane - 16] >> 4)
                    code = low + 16 * ((high >> lane) & 1) - 16
                    flat[begin + lane] = code * delta
                address += BLOCK_BYTES
            output.append(matrix)
    return tuple(output)


def _dot(payload, address, vector, columns):
    """Scalar direct-byte projection with one fixed-size unpacked block tuple."""
    total = np.float32(0)
    for start in range(0, columns, BLOCK_WIDTH):
        delta, high, *packed = _BLOCK.unpack_from(payload, address)
        for lane in range(BLOCK_WIDTH):
            nibble = (packed[lane] % 16) if lane < 16 else (packed[lane - 16] // 16)
            code = nibble + 16 * ((high >> lane) & 1) - 16
            weight = np.float32(code * delta)
            total = np.float32(total + np.float32(weight * vector[start + lane]))
        address += BLOCK_BYTES
    return total


def _scalar_expert(payload, vector, h, i, route_weight, activation, output):
    for address in range(0, payload.nbytes, BLOCK_BYTES):
        if not math.isfinite(struct.unpack_from("<e", payload, address)[0]):
            return 1
    row_bytes = h // BLOCK_WIDTH * BLOCK_BYTES
    extent = row_bytes * i
    for row in range(i):
        gate = _dot(payload, row * row_bytes, vector, h)
        up = _dot(payload, extent + row * row_bytes, vector, h)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        if gate < 0:
            exponential = math.exp(float(gate))
            sigmoid = exponential / (1.0 + exponential)
        else:
            sigmoid = 1.0 / (1.0 + math.exp(-float(gate)))
        activation[row] = np.float32(np.float32(gate * np.float32(sigmoid)) * up)
        if not math.isfinite(activation[row]):
            return 2
    down_bytes = i // BLOCK_WIDTH * BLOCK_BYTES
    for row in range(h):
        down = _dot(payload, 2 * extent + row * down_bytes, activation, i)
        if not math.isfinite(down):
            return 2
        output[row] = np.float32(output[row] + np.float32(route_weight * down))
        if not math.isfinite(output[row]):
            return 2
    return 0


def scratch_bytes(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return intermediate_size * np.dtype(np.float32).itemsize


def _detach_borrowed_frames(error):
    queue = [error]
    visited = set()
    while queue:
        current = queue.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        queue.extend(item for item in (current.__cause__, current.__context__) if item is not None)
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


class Q5_0ExpertBackend:
    """Explicit scalar baseline over leased bytes; no decoded weight arrays."""

    NAME = "q5_0-reference"

    def __init__(self, *, max_scratch_bytes=64 * 1024 * 1024):
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def info(self):
        return KernelBackendInfo(
            name=self.NAME, provider="openagent", available=True, priority=-12000,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True),
            evidence=(KernelEvidence("implementation", "Independent Q5_0 scalar byte reductions"),),
        )

    def _check_available(self):
        if (type(self.max_scratch_bytes) is not int or self.max_scratch_bytes <= 0 or
                BLOCK_WIDTH != 32 or BLOCK_BYTES != 22):
            raise RuntimeError("Q5_0 backend admission state is invalid")

    def _run_expert(self, payload, vector, h, i, weight, activation, output):
        return _scalar_expert(payload, vector, h, i, weight, activation, output)

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        self._check_available()
        _shapes(hidden_size, intermediate_size)
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        self._check_available()
        size = payload_size(hidden_size, intermediate_size)
        needed = scratch_bytes(hidden_size, intermediate_size)
        if needed > self.max_scratch_bytes:
            raise MemoryError("Q5_0 activation scratch exceeds max_scratch_bytes")
        if dtype != DTYPE:
            raise ValueError("unsupported Q5_0 dtype")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        batch = np.ascontiguousarray(_finite_matrix(inputs, hidden_size))
        route_rows = list(route_batch)
        if len(route_rows) != len(batch):
            raise ValueError("one route list required per input row")
        route_rows = [_validate_routes(row, "routes") for row in route_rows]
        for routes in route_rows:
            if any(weight > float(np.finfo(np.float32).max) for _, weight in routes):
                raise ValueError("route weights must be representable as float32")
        result = np.zeros_like(batch)
        if not len(batch):
            return result
        activation = np.empty(intermediate_size, np.float32)
        for row, routes in enumerate(route_rows):
            for key, weight in routes:
                failure = None
                with _lease(source, key) as payload:
                    if payload.nbytes != size:
                        raise ValueError("Q5_0 payload size mismatch")
                    try:
                        with np.errstate(over="ignore", invalid="ignore"):
                            status = self._run_expert(payload, batch[row], hidden_size,
                                                      intermediate_size, np.float32(weight),
                                                      activation, result[row])
                    except BaseException as error:
                        _detach_borrowed_frames(error)
                        failure = error
                if failure is not None:
                    raise failure from None
                if status == 1:
                    raise ValueError("Q5_0 deltas must be finite")
                if status != 0:
                    raise ArithmeticError("Q5_0 computation produced non-finite output")
        return result
