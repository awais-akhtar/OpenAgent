"""Explicit compiled streamed IQ3 SwiGLU expert execution; no fallback."""
import math
import numpy as np

from .iq3_lattice import DTYPE, IQ3Grid, _signs, rotation_blocks, payload_size
from .expert_kernels import (ExpertPayloadSource, KernelBackendInfo, KernelCapabilities,
                            KernelEvidence, _finite_matrix, _finite_array, _validate_routes, _lease)
from .q5_0_expert import _detach_borrowed_frames


def _rotate(vector, signs):
    result = vector.copy()
    offset = 0
    remaining = len(result)
    while remaining:
        width = remaining & -remaining
        while width > 32768:
            width //= 2
        for lane in range(width):
            result[offset+lane] *= signs[offset+lane]
        stride = 1
        while stride < width:
            for first in range(0, width, stride*2):
                for lane in range(stride):
                    left = result[offset+first+lane]
                    right = result[offset+first+stride+lane]
                    result[offset+first+lane] = left+right
                    result[offset+first+stride+lane] = left-right
            stride *= 2
        divisor = np.float32(math.sqrt(width))
        for lane in range(width):
            result[offset+lane] /= divisor
        remaining -= width
        offset += width
    return result


def _dot(raw, offset, row, columns, vector, grid):
    total = 0.0
    for block in range(columns//256):
        base = offset+(row*(columns//256)+block)*98
        half = int(raw[base+96]) | (int(raw[base+97]) << 8)
        exponent, mantissa = (half >> 10) & 31, half & 1023
        if half & 32768 or exponent == 31:
            return math.nan
        scale = mantissa*2.0**-24 if exponent == 0 else (1024+mantissa)*2.0**(exponent-25)
        for sub in range(8):
            word = 0
            for byte in range(4):
                word |= int(raw[base+64+sub*4+byte]) << (8*byte)
            factor = scale*(0.5+(word >> 28))*0.25
            for octet in range(4):
                sign_word = (word >> (7*octet)) & 127
                parity = 0
                for bit in range(7):
                    parity ^= (sign_word >> bit) & 1
                for lane in range(8):
                    index = sub*32+octet*8+lane
                    magnitude = int(grid[int(raw[base+index//4]), index % 4])
                    negative = ((sign_word >> lane) & 1) if lane < 7 else parity
                    weight = factor*magnitude
                    total += (-weight if negative else weight)*float(vector[block*256+index])
    return total


def _kernel(raw, vector, hidden, intermediate, coefficient, output, grid, hsigns, msigns):
    rotated = _rotate(vector, hsigns)
    activation = np.empty(intermediate, dtype=np.float32)
    matrix_bytes = intermediate*(hidden//256)*98
    for row in range(intermediate):
        gate = np.float32(_dot(raw, 0, row, hidden, rotated, grid))
        up = np.float32(_dot(raw, matrix_bytes, row, hidden, rotated, grid))
        if not math.isfinite(gate) or not math.isfinite(up):
            return 1
        sigmoid = 1.0/(1.0+math.exp(-gate)) if gate >= 0 else math.exp(gate)/(1.0+math.exp(gate))
        activation[row] = gate*sigmoid*up
        if not math.isfinite(activation[row]):
            return 2
    rotated_mid = _rotate(activation, msigns)
    for row in range(hidden):
        value = _dot(raw, matrix_bytes*2, row, intermediate, rotated_mid, grid)
        result = output[row]+value*coefficient
        if not math.isfinite(result):
            return 2
        output[row] = result
        if not math.isfinite(output[row]):
            return 2
    return 0


_compiled, _reason = None, ''
try:
    import numba
    if numba.config.DISABLE_JIT:
        _reason = 'Numba JIT disabled'
    else:
        numba.extending.register_jitable(_rotate)
        numba.extending.register_jitable(_dot)
        _compiled = numba.njit(cache=False, nogil=True, fastmath=False)(_kernel)
except Exception as error:
    _reason = str(error)


class NativeIQ3ExpertBackend:
    def __init__(self, contract, *, max_scratch_bytes=64*1024**2):
        if type(contract) is not IQ3Grid or type(max_scratch_bytes) is not int or max_scratch_bytes <= 0:
            raise ValueError('admitted IQ3 grid and positive scratch budget required')
        self.contract, self.max_scratch_bytes = contract, max_scratch_bytes
        self.NAME = DTYPE+'-'+contract.identity+'-numba'

    def info(self):
        ready = _compiled is not None
        return KernelBackendInfo(name=self.NAME, provider='numba', available=ready, priority=100,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True, compiled=ready),
            evidence=(KernelEvidence('contract', self.contract.identity),),
            unavailable_reason='' if ready else (_reason or 'IQ3 JIT unavailable'))

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        vector = _finite_array(inputs, (hidden_size,), 'inputs')
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        if _compiled is None:
            raise RuntimeError(_reason or 'IQ3 JIT unavailable')
        if dtype != DTYPE or not isinstance(source, ExpertPayloadSource):
            raise ValueError('IQ3 dtype and leased expert payload source required')
        length = 2*payload_size(intermediate_size, hidden_size)+payload_size(hidden_size, intermediate_size)
        observed = _finite_matrix(inputs, hidden_size)
        if observed.size*16+(hidden_size+intermediate_size)*24+4096 > self.max_scratch_bytes:
            raise MemoryError('IQ3 expert workspace exceeds admission')
        vectors = np.ascontiguousarray(observed, dtype=np.float32)
        routes = [_validate_routes(row, 'routes') for row in route_batch]
        if len(routes) != len(vectors):
            raise ValueError('one IQ3 route list required per input row')
        hsigns = np.concatenate(tuple(_signs(width) for width in rotation_blocks(hidden_size)))
        msigns = np.concatenate(tuple(_signs(width) for width in rotation_blocks(intermediate_size)))
        output = np.zeros_like(vectors)
        for row, selected in enumerate(routes):
            for key, coefficient in selected:
                failure = None
                with _lease(source, key) as payload:
                    if payload.nbytes != length:
                        raise ValueError('IQ3 expert payload extent mismatch')
                    raw = np.frombuffer(payload, dtype=np.uint8)
                    try:
                        status = _compiled(raw, vectors[row], hidden_size, intermediate_size,
                            float(coefficient), output[row], self.contract.grid, hsigns, msigns)
                    except BaseException as error:
                        _detach_borrowed_frames(error)
                        failure = error
                    finally:
                        del raw
                if failure is not None:
                    raise failure from None
                if status:
                    raise ArithmeticError('invalid IQ3 scale or nonfinite expert computation')
        return output
