"""Independent little-endian Q8_0 expert payloads and bounded leased execution.

Metadata reference (not quantizer/kernel implementation):
https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
Q8_0: 32 signed byte codes following one IEEE binary16 scale (34 bytes).
Decoded weights are float32(scale) * float32(code), including -128, signed
and zero scales. Nonfinite scales are rejected. Rows must be multiples of 32;
no implicit padding. Expert payloads concatenate row-major gate, up, down,
with shapes [i,h], [i,h], [h,i], no header or inter-matrix alignment.
Dimensions and integrity hashes are trusted external metadata.

The optional packer uses max(abs(block))/127 rounded to binary16, then
ties-to-even codes in [-127,127]. Zero blocks use scale zero. Nonzero scales
are floored to the smallest binary16 subnormal; scale overflow is rejected.
This policy is independently specified, NOT upstream quantizer byte parity.

Execution keeps one expert leased at a time, never caches payloads or decoded
matrices, and decodes at most block_count*32 weights at once. Weight scratch
is O(block_count*32), activation scratch O(i+h), excluding caller input/output,
route metadata, the borrowed payload and NumPy/BLAS internal workspace.
Packing returns caller-sized bytes; decode_expert deliberately allocates full
matrices. The shared registry and sparse manifest support explicit q8_0
payloads; model checkpoint CLI and snapshot adapters are not extended. No GGUF reader, native
kernel, upstream execution parity, quality or performance claim is made.
"""

from __future__ import annotations

import numpy as np

from .expert_kernels import (
    ExpertPayloadSource, _finite_array, _finite_matrix, _finite_output,
    _lease, _validate_routes, KernelBackendInfo, KernelCapabilities, KernelEvidence,
)

DTYPE = "q8_0"
BLOCK_WIDTH = 32
BLOCK_BYTES = 34
_RECORD = np.dtype([("scale", "<f2"), ("codes", "i1", (32,))], align=False)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shapes(hidden, intermediate):
    _positive(hidden, "hidden_size")
    _positive(intermediate, "intermediate_size")
    if hidden % BLOCK_WIDTH or intermediate % BLOCK_WIDTH:
        raise ValueError("Q8_0 projection row widths must be multiples of 32")
    return ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))


def payload_size(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return 3 * hidden_size * intermediate_size // BLOCK_WIDTH * BLOCK_BYTES


def pack_matrix(matrix, *, block_count=64):
    """Return Q8_0 row-major bytes with bounded conversion scratch."""
    _positive(block_count, "block_count")
    a = np.asarray(matrix)
    if a.ndim != 2 or a.dtype.kind not in "iuf":
        raise ValueError("weights must be real matrices")
    rows, cols = a.shape
    _positive(rows, "rows")
    _positive(cols, "columns")
    if cols % BLOCK_WIDTH:
        raise ValueError("Q8_0 row widths must be multiples of 32")
    result = bytearray(rows * cols // BLOCK_WIDTH * BLOCK_BYTES)
    offset = 0
    for row in a:
        for start in range(0, cols, block_count * BLOCK_WIDTH):
            with np.errstate(over="ignore", invalid="ignore"):
                values = row[start:start + block_count * BLOCK_WIDTH].astype(np.float32).reshape(-1, 32)
            if not np.isfinite(values).all():
                raise ValueError("weights must be finite and representable as float32")
            maxima = np.max(np.abs(values), axis=1)
            with np.errstate(over="ignore", under="ignore"):
                scales = (maxima / np.float32(127)).astype("<f2")
            if not np.isfinite(scales).all():
                raise ValueError("Q8_0 scale exceeds binary16 range")
            scales[(maxima != 0) & (scales == 0)] = np.nextafter(np.float16(0), np.float16(1))
            divisor = scales.astype(np.float32)
            divisor[divisor == 0] = 1
            records = np.empty(len(values), dtype=_RECORD)
            records["scale"] = scales
            records["codes"] = np.clip(np.rint(values / divisor[:, None]), -127, 127).astype(np.int8)
            data = records.tobytes()
            result[offset:offset + len(data)] = data
            offset += len(data)
    return bytes(result)


def pack_expert(gate, up, down, *, block_count=64):
    matrices = tuple(np.asarray(a) for a in (gate, up, down))
    if matrices[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    intermediate, hidden = matrices[0].shape
    shapes = _shapes(hidden, intermediate)
    if any(a.shape != shape for a, shape in zip(matrices, shapes)):
        raise ValueError("expert projection shape mismatch")
    return b"".join(pack_matrix(a, block_count=block_count) for a in matrices)


def _validate_payload(payload, hidden, intermediate, block_count):
    if payload.nbytes != payload_size(hidden, intermediate):
        raise ValueError("Q8_0 payload size mismatch")
    for index in range(0, payload.nbytes // BLOCK_BYTES, block_count):
        count = min(block_count, payload.nbytes // BLOCK_BYTES - index)
        records = np.frombuffer(payload, _RECORD, count, index * BLOCK_BYTES)
        valid = bool(np.isfinite(records["scale"]).all())
        # Exception tracebacks retain locals; drop borrowed arrays before raising.
        del records
        if not valid:
            raise ValueError("Q8_0 scales must be finite")


def _decoded_blocks(payload, offset, count, block_count):
    for index in range(0, count, block_count):
        length = min(block_count, count - index)
        records = np.frombuffer(payload, _RECORD, length, offset + index * BLOCK_BYTES)
        values = records["codes"].astype(np.float32)
        values *= records["scale"].astype(np.float32)[:, None]
        del records
        yield index * BLOCK_WIDTH, values.reshape(-1)
        del values


def decode_expert(payload, hidden_size, intermediate_size, *, block_count=64):
    """Explicit full-matrix decode convenience API; returned arrays own data."""
    _positive(block_count, "block_count")
    shapes = _shapes(hidden_size, intermediate_size)
    with memoryview(payload) as view:
        if not view.c_contiguous:
            raise ValueError("payload must be contiguous")
        _validate_payload(view, hidden_size, intermediate_size, block_count)
        output = []
        offset = 0
        for rows, cols in shapes:
            matrix = np.empty((rows, cols), np.float32)
            flat = matrix.reshape(-1)
            count = rows * cols // BLOCK_WIDTH
            for start, values in _decoded_blocks(view, offset, count, block_count):
                flat[start:start + len(values)] = values
                del values
            output.append(matrix)
            offset += count * BLOCK_BYTES
    return tuple(output)


class Q8_0ExpertBackend:
    """FP32 SwiGLU executor for explicitly selected Q8_0 expert byte leases."""

    NAME = "q8_0-numpy"

    def __init__(self, *, block_count=64):
        self.block_count = _positive(block_count, "block_count")

    def info(self):
        return KernelBackendInfo(
            name=self.NAME, provider="openagent", available=True, priority=-11_000,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True),
            evidence=(KernelEvidence("implementation", "Block-bounded NumPy Q8_0 decoding", True),),
        )

    def _project(self, payload, offset, rows, cols, vector):
        output = np.empty(rows, np.float32)
        blocks_per_row = cols // BLOCK_WIDTH
        for row in range(rows):
            total = np.float32(0)
            for start, values in _decoded_blocks(
                payload, offset + row * blocks_per_row * BLOCK_BYTES,
                blocks_per_row, self.block_count,
            ):
                total += values @ vector[start:start + len(values)]
                del values
            output[row] = total
        return _finite_output(output)

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        _shapes(hidden_size, intermediate_size)
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        _shapes(hidden_size, intermediate_size)
        if dtype != DTYPE:
            raise ValueError("unsupported Q8_0 dtype")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        batch = _finite_matrix(inputs, hidden_size)
        routes = list(route_batch)
        if len(routes) != len(batch):
            raise ValueError("one route list is required per input row")
        routes = [_validate_routes(row, "routes") for row in routes]
        for assignments in routes:
            for _, weight in assignments:
                if weight > float(np.finfo(np.float32).max):
                    raise ValueError("route weight must be representable as float32")
        output = np.zeros_like(batch)
        matrix_bytes = hidden_size * intermediate_size // BLOCK_WIDTH * BLOCK_BYTES
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for row, assignments in enumerate(routes):
                for key, weight in assignments:
                    with _lease(source, key) as payload:
                        _validate_payload(payload, hidden_size, intermediate_size, self.block_count)
                        gate = self._project(payload, 0, intermediate_size, hidden_size, batch[row])
                        up = self._project(payload, matrix_bytes, intermediate_size, hidden_size, batch[row])
                        activated = _finite_output(gate * np.exp(-np.logaddexp(0, -gate)) * up)
                        down = self._project(payload, 2 * matrix_bytes, hidden_size, intermediate_size, activated)
                        output[row] += np.float32(weight) * down
                        _finite_output(output[row])
        return output
