"""Direct-byte Q4_0 CPU reduction with optional Numba compilation.

No decoded weights, scale arrays, disk JIT cache or automatic registration.
The native backend is unavailable without active Numba; no silent fallback.
"""

import math
import traceback

import numpy as np

from .expert_kernels import (ExpertPayloadSource, KernelBackendInfo,
                             KernelCapabilities, KernelEvidence, _finite_array,
                             _finite_matrix, _lease, _validate_routes)
from .q4_0_expert import DTYPE, _positive, _shapes, payload_size


def _delta(raw, address):
    word = int(raw[address]) + 256 * int(raw[address + 1])
    power = (word >> 10) & 31
    mantissa = word & 1023
    if power == 31:
        return np.float32(np.nan)
    magnitude = math.ldexp(float(mantissa if power == 0 else mantissa + 1024),
                           -24 if power == 0 else power - 25)
    return np.float32(-magnitude if word & 32768 else magnitude)


def _projection(raw, address, vector, width):
    accumulator = np.float32(0)
    for block in range(width // 32):
        base = address + 18 * block
        delta = _delta(raw, base)
        for lane in range(32):
            packed = int(raw[base + 2 + lane % 16])
            code = (packed & 15) if lane < 16 else (packed >> 4)
            weight = np.float32((code - 8) * delta)
            accumulator = np.float32(accumulator + np.float32(weight * vector[block * 32 + lane]))
    return accumulator


def _evaluate(raw, x, h, i, mix, activation, output):
    for address in range(0, len(raw), 18):
        if not math.isfinite(_delta(raw, address)):
            return 1
    stride = h // 32 * 18
    extent = stride * i
    for neuron in range(i):
        gate = _projection(raw, neuron * stride, x, h)
        up = _projection(raw, extent + neuron * stride, x, h)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        decay = math.exp(-abs(float(gate)))
        sigmoid = np.float32((1 if gate >= 0 else decay) / (1 + decay))
        activation[neuron] = np.float32(np.float32(gate * sigmoid) * up)
        if not math.isfinite(activation[neuron]):
            return 2
    for neuron in range(h):
        value = _projection(raw, 2 * extent + neuron * (i // 32 * 18), activation, i)
        if not math.isfinite(value):
            return 2
        output[neuron] = np.float32(output[neuron] + np.float32(mix * value))
        if not math.isfinite(output[neuron]):
            return 2
    return 0


_compiled = False
_reason = ""
_compiled_evaluate = None
try:
    import numba
    if not numba.config.DISABLE_JIT:
        numba.extending.register_jitable(inline="always")(_delta)
        numba.extending.register_jitable(_projection)
        _compiled_evaluate = numba.njit(nogil=True, cache=False, fastmath=False)(_evaluate)
        _compiled = True
    else:
        _reason = "Numba JIT disabled"
except Exception as error:
    _reason = f"Numba unavailable ({type(error).__name__})"


def scratch_bytes(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return 4 * intermediate_size


class _DirectBackend:
    NAME = "q4_0-reference"
    compiled = False

    def __init__(self, *, max_scratch_bytes=64 * 1024 * 1024):
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")

    def info(self):
        return KernelBackendInfo(
            name=self.NAME, provider="numba" if self.compiled else "openagent",
            available=not self.compiled or _compiled,
            priority=100 if self.compiled else -12000,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True,
                                            compiled=self.compiled and _compiled),
            evidence=(KernelEvidence("implementation", "Direct-byte Q4_0 reductions"),),
            unavailable_reason=_reason if self.compiled and not _compiled else "",
        )

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        if self.compiled and not _compiled:
            raise RuntimeError(_reason or "Q4_0 JIT unavailable")
        required = scratch_bytes(hidden_size, intermediate_size)
        if required > self.max_scratch_bytes:
            raise MemoryError("Q4_0 activation scratch exceeds limit")
        if dtype != DTYPE:
            raise ValueError("unsupported Q4_0 dtype")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        x = np.ascontiguousarray(_finite_matrix(inputs, hidden_size))
        assignments = list(route_batch)
        if len(assignments) != len(x):
            raise ValueError("one route list required per input row")
        assignments = [_validate_routes(row, "routes") for row in assignments]
        if any(weight > float(np.finfo(np.float32).max) for row in assignments for _, weight in row):
            raise ValueError("route weight must be representable as float32")
        result = np.zeros_like(x)
        activation = np.empty(intermediate_size, np.float32)
        for row, routes in enumerate(assignments):
            for key, weight in routes:
                failure = None
                with _lease(source, key) as payload:
                    if payload.nbytes != payload_size(hidden_size, intermediate_size):
                        raise ValueError("Q4_0 payload size mismatch")
                    raw = np.frombuffer(payload, np.uint8)
                    try:
                        with np.errstate(over="ignore", invalid="ignore"):
                            evaluator = _compiled_evaluate if self.compiled else _evaluate
                            status = evaluator(raw, x[row], hidden_size, intermediate_size,
                                               np.float32(weight), activation, result[row])
                    except BaseException as error:
                        # Compiler exceptions can retain borrowed buffers in chained frames.
                        pending, seen = [error], set()
                        while pending:
                            item = pending.pop()
                            if id(item) in seen:
                                continue
                            seen.add(id(item))
                            pending.extend(e for e in (item.__cause__, item.__context__) if e is not None)
                            traceback.clear_frames(item.__traceback__)
                            item.__traceback__ = item.__cause__ = item.__context__ = None
                        failure = error
                    finally:
                        del raw
                if failure is not None:
                    raise failure from None
                if status == 1:
                    raise ValueError("Q4_0 deltas must be finite")
                if status != 0:
                    raise ArithmeticError("Q4_0 computation produced non-finite output")
        return result


class NativeQ4_0ExpertBackend(_DirectBackend):
    """Explicitly selected compiled backend, unavailable without active Numba."""

    NAME = "q4_0-numba"
    compiled = True
