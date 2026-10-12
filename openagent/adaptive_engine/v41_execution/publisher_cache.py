"""Immutable packed publisher cache rows; decoding is transient FP32/BF16.

FP4 stores the earlier element in the low nibble. Each group has one scale
byte: unsigned E4M3 for g16, exponent plus 127 (UE8M0) for g32. FP8 values
are signed E4M3 finite codes. This is a cache format, not a weight provider.
Numerical parity with the publisher implementation has not been established.
"""
from dataclasses import dataclass
import math

import numpy as np


_FORMATS = {
    'fp8-e4m3-g32-ue8m0': (32, 8),
    'fp4-e2m1-g16-e4m3': (16, 4),
    'fp4-e2m1-g32-ue8m0': (32, 4),
}
_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP8 = tuple(math.ldexp(8.0 + (code & 7), (code >> 3) - 10)
             if code >> 3 else math.ldexp(float(code & 7), -9)
             for code in range(127))


def packed_nbytes(size, encoding):
    if encoding not in _FORMATS:
        raise ValueError('unsupported publisher cache encoding')
    group, bits = _FORMATS[encoding]
    if type(size) is not int or size <= 0 or size % group:
        raise ValueError('publisher cache row must contain complete scale groups')
    return size * bits // 8 + size // group


def _nearest(value, levels):
    # Even-code ties reproduce the finite E4M3 / E2M1 rounding contract.
    return min(range(len(levels)), key=lambda code: (abs(value-levels[code]), code & 1))


@dataclass(frozen=True)
class PackedPublisherRow:
    encoding: str
    size: int
    values: bytes
    scales: bytes

    def __post_init__(self):
        total = packed_nbytes(self.size, self.encoding)
        group, bits = _FORMATS[self.encoding]
        if (type(self.values) is not bytes or type(self.scales) is not bytes or
                len(self.values) != self.size * bits // 8 or
                len(self.scales) != self.size // group or
                len(self.values) + len(self.scales) != total):
            raise ValueError('invalid packed publisher cache buffers')
        if bits == 8 and any((code & 127) == 127 for code in self.values):
            raise ValueError('nonfinite E4M3 cache code')
        if self.encoding == 'fp4-e2m1-g16-e4m3':
            if any(code == 0 or code >= 127 for code in self.scales):
                raise ValueError('invalid positive E4M3 cache scale')
        elif any(code == 255 for code in self.scales):
            raise ValueError('nonfinite UE8M0 cache scale')

    @property
    def nbytes(self):
        return len(self.values) + len(self.scales)

    def decode(self):
        group, bits = _FORMATS[self.encoding]
        codes = np.frombuffer(self.values, dtype=np.uint8)
        if bits == 4:
            expanded = np.empty(self.size, dtype=np.uint8)
            expanded[::2], expanded[1::2] = codes & 15, codes >> 4
            codes = expanded
        levels = _FP4 if bits == 4 else _FP8
        mask, sign = (7, 8) if bits == 4 else (127, 128)
        output = np.empty(self.size, dtype=np.float32)
        for block, scale_code in enumerate(self.scales):
            scale = (_FP8[scale_code] if self.encoding == 'fp4-e2m1-g16-e4m3'
                     else math.ldexp(1.0, scale_code-127))
            for j in range(block*group, (block+1)*group):
                code = int(codes[j])
                output[j] = math.copysign(levels[code & mask] * scale,
                                          -1.0 if code & sign else 1.0)
        # Attention sees the same BF16 activation boundary as cache roundtrip.
        words = output.view(np.uint32)
        words += np.uint32(0x7fff) + ((words >> 16) & 1)
        words &= np.uint32(0xffff0000)
        if not np.isfinite(output).all():
            raise ValueError('packed publisher cache decode overflow')
        return output


def pack_row(row, encoding):
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    packed_nbytes(int(row.size), encoding)
    if not np.isfinite(row).all():
        raise ValueError('finite publisher cache row required')
    group, bits = _FORMATS[encoding]
    levels = _FP4 if bits == 4 else _FP8
    codes, scales = bytearray(), bytearray()
    for start in range(0, row.size, group):
        block = row[start:start+group]
        maximum = float(np.max(np.abs(block)))
        if encoding == 'fp4-e2m1-g16-e4m3':
            raw = float(np.float32(np.float32(max(maximum, 6*2**-9))/np.float32(6)))
            scale_code = _nearest(raw, _FP8)
            scale = _FP8[scale_code]
        else:
            floor = 6*2**-126 if bits == 4 else float(np.float32(1e-4))
            raw = float(np.float32(np.float32(max(maximum, floor))/np.float32(6 if bits == 4 else 448)))
            exponent = math.ceil(math.log2(raw))
            scale_code, scale = exponent+127, math.ldexp(1.0, exponent)
            if not 0 <= scale_code < 255:
                raise ValueError('publisher cache scale outside UE8M0 extent')
        if scale <= 0:
            raise ValueError('publisher cache scale underflow')
        scales.append(scale_code)
        for value in block:
            normalized = float(np.float32(value/np.float32(scale)))
            code = _nearest(abs(normalized), levels)
            codes.append(code | ((8 if bits == 4 else 128) if np.signbit(value) else 0))
    values = bytes(codes) if bits == 8 else bytes(
        codes[i] | (codes[i+1] << 4) for i in range(0, len(codes), 2))
    return PackedPublisherRow(encoding, int(row.size), values, bytes(scales))
