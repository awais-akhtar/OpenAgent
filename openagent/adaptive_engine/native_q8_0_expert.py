"""Explicitly registered Numba Q8_0 CPU kernel, direct leased-byte operands.

No float16 arrays, decoded weight matrices, scale arrays, JIT disk cache or
automatic registration. Binary16 is decoded with integer fields and ldexp.
Scratch is one caller-bounded FP32 intermediate vector reused for every route;
input/output and Python routing metadata are accounted for separately.
"""

from __future__ import annotations

import math
import traceback

import numpy as np

from .expert_kernels import (
    ExpertPayloadSource, KernelBackendInfo, KernelCapabilities, KernelEvidence,
    _finite_array, _finite_matrix, _lease, _validate_routes,
)
from .q8_0_expert import DTYPE, _positive, _shapes, payload_size

try:
    import numba as _numba
except Exception as exc:
    _numba = None
    _unavailable = f"Numba could not be loaded: {type(exc).__name__}: {exc}"
else:
    _unavailable = "Numba JIT is disabled" if _numba.config.DISABLE_JIT else ""


def _half_bits(bits):
    exponent = (bits >> 10) & 31
    fraction = bits & 1023
    if exponent == 31:
        return np.float32(np.nan)
    if exponent == 0:
        value = np.float32(math.ldexp(float(fraction), -24))
    else:
        value = np.float32(math.ldexp(float(1024 + fraction), int(exponent) - 25))
    return -value if bits & 32768 else value


def _dot(raw, offset, vector, width):
    total = np.float32(0)
    for block in range(width // 32):
        base = offset + block * 34
        scale = _half_bits(int(raw[base]) | (int(raw[base + 1]) << 8))
        for column in range(32):
            code = int(raw[base + 2 + column])
            if code >= 128:
                code -= 256
            # Scale before multiplication/reduction; do not sum unscaled codes.
            weight = np.float32(np.float32(code) * scale)
            product = np.float32(weight * vector[block * 32 + column])
            total = np.float32(total + product)
    return total


def _loop(raw, vector, route_weight, hidden, intermediate, activation, output):
    # Validate all scales before arithmetic, including zero-weight routes.
    for base in range(0, raw.size, 34):
        bits = int(raw[base]) | (int(raw[base + 1]) << 8)
        if ((bits >> 10) & 31) == 31:
            return 1
    gate_stride = hidden // 32 * 34
    matrix_bytes = intermediate * gate_stride
    for row in range(intermediate):
        gate = _dot(raw, row * gate_stride, vector, hidden)
        up = _dot(raw, matrix_bytes + row * gate_stride, vector, hidden)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        if gate >= 0:
            sigmoid = np.float32(1.0 / (1.0 + math.exp(-float(gate))))
        else:
            exponential = math.exp(float(gate))
            sigmoid = np.float32(exponential / (1.0 + exponential))
        value = np.float32(np.float32(gate * sigmoid) * up)
        if not math.isfinite(value):
            return 2
        activation[row] = value
    down_stride = intermediate // 32 * 34
    for row in range(hidden):
        down = _dot(raw, 2 * matrix_bytes + row * down_stride, activation, intermediate)
        if not math.isfinite(down):
            return 2
        value = np.float32(output[row] + np.float32(route_weight * down))
        if not math.isfinite(value):
            return 2
        output[row] = value
    return 0


if _numba is not None and not _unavailable:
    _half_bits = _numba.njit(inline="always", fastmath=False)(_half_bits)
    _dot = _numba.njit(inline="always", fastmath=False)(_dot)
    _compiled_loop = _numba.njit(nogil=True, fastmath=False, cache=False)(_loop)
else:
    _compiled_loop = None


def scratch_bytes(hidden_size, intermediate_size):
    """Exact explicit numerical scratch, excluding inputs/output and leases."""
    _shapes(hidden_size, intermediate_size)
    return intermediate_size * np.dtype(np.float32).itemsize


def _clear_failure_tracebacks(error):
    """Drop borrowed argument frames throughout both explicit and implicit chains."""
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


class NativeQ8_0ExpertBackend:
    """Optional CPU JIT backend; register an instance explicitly to select it."""

    NAME = "q8_0-numba"

    def __init__(self, *, max_scratch_bytes=64 * 1024 * 1024):
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def info(self):
        available = _compiled_loop is not None
        return KernelBackendInfo(
            name=self.NAME, provider="numba", available=available, priority=100,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True, compiled=available),
            evidence=(KernelEvidence("implementation", "Direct Q8_0 byte reductions; fastmath disabled"),),
            unavailable_reason="" if available else (_unavailable or "Q8_0 JIT unavailable"),
        )

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        _shapes(hidden_size, intermediate_size)
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        if _compiled_loop is None:
            raise RuntimeError(_unavailable or "Q8_0 JIT unavailable")
        needed = scratch_bytes(hidden_size, intermediate_size)
        if needed > self.max_scratch_bytes:
            raise MemoryError("Q8_0 activation scratch exceeds max_scratch_bytes")
        if dtype != DTYPE:
            raise ValueError("unsupported Q8_0 dtype")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        batch = np.ascontiguousarray(_finite_matrix(inputs, hidden_size))
        route_rows = list(route_batch)
        if len(route_rows) != len(batch):
            raise ValueError("one route list is required per input row")
        route_rows = [_validate_routes(row, "routes") for row in route_rows]
        for assignments in route_rows:
            for _, weight in assignments:
                if weight > float(np.finfo(np.float32).max):
                    raise ValueError("route weight must be representable as float32")
        output = np.zeros_like(batch)
        if not len(batch):
            return output
        activation = np.empty(intermediate_size, np.float32)
        expected = payload_size(hidden_size, intermediate_size)
        for row, assignments in enumerate(route_rows):
            for key, weight in assignments:
                failure = None
                with _lease(source, key) as payload:
                    if payload.nbytes != expected:
                        raise ValueError("Q8_0 payload size mismatch")
                    raw = np.frombuffer(payload, np.uint8)
                    try:
                        status = _compiled_loop(raw, batch[row], np.float32(weight), hidden_size,
                                                intermediate_size, activation, output[row])
                    except BaseException as error:
                        # Preserve the original type without its borrowed argument frames.
                        _clear_failure_tracebacks(error)
                        failure = error
                    finally:
                        del raw
                # Raise outside both except and lease: no implicit borrowed context survives.
                if failure is not None:
                    raise failure from None
                if status == 1:
                    raise ValueError("Q8_0 scales must be finite")
                if status != 0:
                    raise ArithmeticError("Q8_0 computation produced non-finite output")
        return output
