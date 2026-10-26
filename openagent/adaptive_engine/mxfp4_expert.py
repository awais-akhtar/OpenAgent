"""Owned row-group MXFP4 E2M1/UE8M0 bytes, independent public numeric equations."""
from __future__ import annotations

import math
import traceback
import numpy as np
from .expert_kernels import (ExpertPayloadSource, KernelBackendInfo, KernelCapabilities,
                             KernelEvidence, _finite_array, _finite_matrix, _lease, _validate_routes)

DTYPE = "mxfp4-e2m1-g32-ue8m0"
SCALE_ENCODING = "ue8m0"
BLOCK_WIDTH = 32
BLOCK_BYTES = 17


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _encoding(value):
    if value != SCALE_ENCODING:
        raise ValueError("unsupported scale encoding; expected explicit ue8m0")


def _shapes(hidden, intermediate):
    _positive(hidden, "hidden_size")
    _positive(intermediate, "intermediate_size")
    return ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))


def _size(rows, cols):
    return rows * ((cols+31)//32) * 17


def matrix_payload_size(rows, columns):
    _positive(rows, "rows")
    _positive(columns, "columns")
    return _size(rows, columns)


def payload_size(hidden_size, intermediate_size):
    return sum(_size(*shape) for shape in _shapes(hidden_size, intermediate_size))


def scratch_bytes(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return intermediate_size * 4


def _code(bits):
    exponent, fraction = (bits >> 1) & 3, bits & 1
    value = fraction * .5 if exponent == 0 else math.ldexp(1 + fraction * .5, exponent-1)
    return np.float32(-value if bits & 8 else value)


def _scale(bits):
    return np.float32(np.nan) if bits == 255 else np.float32(math.ldexp(1., bits-127))


def decode_code(code):
    if isinstance(code, bool) or not isinstance(code, (int,np.integer)) or not 0 <= code < 16:
        raise ValueError("E2M1 code must be a nibble")
    return _code(int(code))


def decode_scale(code):
    if isinstance(code, bool) or not isinstance(code, (int,np.integer)) or not 0 <= code < 256:
        raise ValueError("UE8M0 code must be a byte")
    return _scale(int(code))


_LEVELS = np.array([0,.5,1,1.5,2,3,4,6],np.float64)


def pack_matrix(matrix, *, scale_encoding=SCALE_ENCODING, row_count=64):
    """Bounded row-tile conversion; power-of-two ceiling scale and RNE codes."""
    _encoding(scale_encoding)
    _positive(row_count, "row_count")
    a = np.asarray(matrix)
    if a.ndim != 2 or a.dtype.kind not in "iuf":
        raise ValueError("weights must be a real matrix")
    rows, cols = a.shape
    groups = (cols + 31) // 32
    result = bytearray(matrix_payload_size(rows, cols))
    padded_columns = groups * 32
    for first in range(0, rows, row_count):
        stop = min(rows, first + row_count)
        with np.errstate(over="ignore", invalid="ignore"):
            tile = a[first:stop].astype(np.float32)
        if not np.isfinite(tile).all():
            raise ValueError("weights must be finite float32")
        padded = np.zeros((stop-first, padded_columns), np.float32)
        padded[:, :cols] = tile
        grouped = padded.reshape(stop-first, groups, 32)
        maxima = np.max(np.abs(grouped), axis=2).astype(np.float64)
        exponents = np.zeros(maxima.shape, np.int32)
        nonzero = maxima != 0
        with np.errstate(divide="ignore", invalid="ignore"):
            exponents[nonzero] = np.clip(np.ceil(np.log2(maxima[nonzero] / 6.0)), -127, 127).astype(np.int32)
        scales = np.ldexp(np.ones(exponents.shape, np.float64), exponents)
        magnitude = np.abs(grouped.astype(np.float64)) / scales[:, :, None]
        high = np.minimum(np.searchsorted(_LEVELS, magnitude.reshape(-1)), 7).reshape(magnitude.shape)
        low = np.maximum(high - 1, 0)
        distance_low, distance_high = magnitude - _LEVELS[low], _LEVELS[high] - magnitude
        codes = np.where((distance_low < distance_high) |
                         ((distance_low == distance_high) & ((low & 1) == 0)), low, high).astype(np.uint8)
        if np.any(_LEVELS[codes] * scales[:, :, None] > float(np.finfo(np.float32).max)):
            raise ValueError("MXFP4 quantized weight exceeds float32 range")
        codes |= np.signbit(grouped).astype(np.uint8) * np.uint8(8)
        records = np.empty((stop-first, groups, BLOCK_BYTES), np.uint8)
        records[:, :, 0] = (exponents + 127).astype(np.uint8)
        records[:, :, 1:] = codes[:, :, 0::2] | (codes[:, :, 1::2] << np.uint8(4))
        start = first * groups * BLOCK_BYTES
        result[start:start + records.nbytes] = records.tobytes()
    return bytes(result)


def pack_expert(gate, up, down, *, scale_encoding=SCALE_ENCODING):
    _encoding(scale_encoding)
    arrays = tuple(np.asarray(a) for a in (gate,up,down))
    if arrays[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    intermediate, hidden = arrays[0].shape
    if any(a.shape != s for a,s in zip(arrays,_shapes(hidden,intermediate))):
        raise ValueError("expert projection shape mismatch")
    return b"".join(pack_matrix(a,scale_encoding=scale_encoding) for a in arrays)


def _valid(raw, offset, rows, cols):
    groups = (cols+31)//32
    for row in range(rows):
        for group in range(groups):
            base = offset+(row*groups+group)*17
            scale_code = int(raw[base])
            if scale_code == 255:
                return 1
            for index in range(32):
                byte = int(raw[base+1+index//2])
                code = (byte >> (4*(index%2))) & 15
                if group*32+index >= cols:
                    if code != 0:
                        return 1
                elif not math.isfinite(np.float32(_code(code) * _scale(scale_code))):
                    return 2
    return 0


def _dot(raw, offset, row, rows, cols, vector):
    total = np.float32(0)
    groups = (cols+31)//32
    for column in range(cols):
        base = offset+(row*groups+column//32)*17
        byte = int(raw[base+1+(column%32)//2])
        code = (byte >> (4*(column%2))) & 15
        weight = np.float32(_code(code) * _scale(int(raw[base])))
        total = np.float32(total + np.float32(weight * vector[column]))
    return total


def _loop(raw, vector, route_weight, hidden, intermediate, activation, output):
    stride = _size(intermediate,hidden)
    for projection in range(3):
        rows = intermediate if projection < 2 else hidden
        cols = hidden if projection < 2 else intermediate
        status = _valid(raw,projection*stride,rows,cols)
        if status:
            return status
    for row in range(intermediate):
        gate = _dot(raw,0,row,intermediate,hidden,vector)
        up = _dot(raw,stride,row,intermediate,hidden,vector)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        if gate >= 0:
            sigmoid = np.float32(1 / (1+math.exp(-float(gate))))
        else:
            exponential = math.exp(float(gate))
            sigmoid = np.float32(exponential/(1+exponential))
        activation[row] = np.float32(np.float32(gate*sigmoid)*up)
        if not math.isfinite(activation[row]):
            return 2
    for row in range(hidden):
        down = _dot(raw,2*stride,row,hidden,intermediate,activation)
        value = np.float32(output[row]+np.float32(route_weight*down))
        if not math.isfinite(down) or not math.isfinite(value):
            return 2
        output[row] = value
    return 0


def _clear_failure(error):
    pending,seen = [error],set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(e for e in (current.__cause__,current.__context__) if e is not None)
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = current.__cause__ = current.__context__ = None


class MXFP4ExpertBackend:
    NAME = "mxfp4-e2m1-g32-ue8m0-reference"

    def __init__(self, *, max_scratch_bytes=64*1024*1024, scale_encoding=SCALE_ENCODING):
        _encoding(scale_encoding)
        self.max_scratch_bytes = _positive(max_scratch_bytes,"max_scratch_bytes")

    def _kernel(self):
        return _loop

    def info(self):
        return KernelBackendInfo(name=self.NAME,provider="openagent",available=True,priority=-11000,
            capabilities=KernelCapabilities(frozenset({DTYPE}),True,True),
            evidence=(KernelEvidence("implementation","Scalar direct-byte E2M1/UE8M0 CPU reference"),))

    def execute(self,source,inputs,routes,*,hidden_size,intermediate_size,dtype=DTYPE):
        vector = _finite_array(inputs,(hidden_size,),"inputs")
        return self.execute_batch(source,vector[None],[routes],hidden_size=hidden_size,
                                  intermediate_size=intermediate_size,dtype=dtype)[0]

    def execute_batch(self,source,inputs,route_batch,*,hidden_size,intermediate_size,dtype=DTYPE):
        kernel = self._kernel()
        if dtype != DTYPE:
            raise ValueError("unsupported MXFP4 dtype")
        if scratch_bytes(hidden_size,intermediate_size) > self.max_scratch_bytes:
            raise MemoryError("MXFP4 activation scratch exceeds max_scratch_bytes")
        if not isinstance(source,ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        batch = np.ascontiguousarray(_finite_matrix(inputs,hidden_size))
        routes = list(route_batch)
        if len(routes) != len(batch):
            raise ValueError("one route list is required per input row")
        routes = [_validate_routes(row,"routes") for row in routes]
        for assignments in routes:
            for _,weight in assignments:
                if weight > float(np.finfo(np.float32).max):
                    raise ValueError("route weight must be representable as float32")
        output = np.zeros_like(batch)
        activation = np.empty(intermediate_size,np.float32)
        expected = payload_size(hidden_size,intermediate_size)
        with np.errstate(over="ignore",invalid="ignore",under="ignore"):
            for row,assignments in enumerate(routes):
                for key,weight in assignments:
                    failure = None
                    with _lease(source,key) as payload:
                        if payload.nbytes != expected:
                            raise ValueError("MXFP4 payload size mismatch")
                        raw = np.frombuffer(payload,np.uint8)
                        try:
                            status = kernel(raw,batch[row],np.float32(weight),hidden_size,
                                            intermediate_size,activation,output[row])
                        except BaseException as error:
                            _clear_failure(error)
                            failure = error
                        finally:
                            del raw
                    if failure is not None:
                        raise failure from None
                    if status == 1:
                        raise ValueError("MXFP4 scale must be finite and padding must be zero")
                    if status != 0:
                        raise ArithmeticError("MXFP4 computation or decoded weight is non-finite")
        return output
