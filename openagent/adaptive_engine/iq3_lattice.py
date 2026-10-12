"""Independent decoder for the reference format-6 98-byte IQ3 container.

The caller supplies the doubled-magnitude grid and its expected SHA-256.
Decoded weights remain in the container's rotated coordinate system; applying
the model's rotation and registering an expert backend are separate operations.
"""
from dataclasses import dataclass
import hashlib
import math
import struct
import os
import tempfile
import stat
from pathlib import Path

import numpy as np

DTYPE = 'iq3-format6-98-v1'
BLOCK_WEIGHTS = 256
BLOCK_BYTES = 98
ROTATION = 'xorshift64star-417-lowbit-fwht32768-v1'


def rotation_blocks(columns):
    if type(columns) is not int or columns <= 0:
        raise ValueError('positive IQ3 rotation width required')
    remaining, blocks = columns, []
    while remaining:
        width = remaining & -remaining
        while width > 32768:
            width //= 2
        blocks.append(width)
        remaining -= width
    return tuple(blocks)


def _signs(width):
    """Regenerate sign bits with explicitly wrapped unsigned arithmetic."""
    mask = (1 << 64)-1
    state = 417+width
    result = np.empty(width, dtype=np.float32)
    for start in range(0, width, 8):
        state ^= state >> 12
        state ^= (state << 25) & mask
        state ^= state >> 27
        state &= mask
        byte = ((state*2685821657736338717) & mask) >> 56
        for bit in range(min(8, width-start)):
            result[start+bit] = -1.0 if (byte >> bit) & 1 else 1.0
    return result


def rotate_iq3(values, *, max_bytes, inverse=False):
    """Rotate rows for storage/input or invert them, without a dense transform."""
    observed = np.asarray(values)
    if (observed.dtype.kind != 'f' or observed.ndim not in (1, 2) or not observed.size or
            not np.isfinite(observed).all() or type(inverse) is not bool):
        raise ValueError('finite IQ3 vector or row matrix required')
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('positive IQ3 rotation budget required')
    columns = observed.shape[-1]
    if observed.size*24+min(columns, 32768)*4+4096 > max_bytes:
        raise MemoryError('IQ3 rotation workspace exceeds admission')
    output = np.array(observed, dtype=np.float32, order='C', copy=True)
    rows = output.reshape(-1, columns)
    start = 0
    for width in rotation_blocks(columns):
        block = rows[:, start:start+width]
        signs = _signs(width)
        if not inverse:
            block *= signs
        stride = 1
        while stride < width:
            for lane in range(0, width, stride*2):
                left = block[:, lane:lane+stride].copy()
                right = block[:, lane+stride:lane+stride*2].copy()
                block[:, lane:lane+stride] = left+right
                block[:, lane+stride:lane+stride*2] = left-right
            stride *= 2
        block /= np.float32(math.sqrt(width))
        if inverse:
            block *= signs
        start += width
    if not np.isfinite(output).all():
        raise ValueError('nonfinite IQ3 rotated values')
    return output


@dataclass(frozen=True, init=False)
class IQ3Grid:
    grid: object
    identity: str

    def __init__(self, grid, *, expected_sha256):
        observed = np.asarray(grid)
        if (observed.shape != (256, 4) or observed.dtype != np.uint8 or
                np.any(observed == 0) or np.any(observed % 2) or np.any(observed > 62)):
            raise ValueError('exact doubled-magnitude uint8 IQ3 grid required')
        if (type(expected_sha256) is not str or len(expected_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in expected_sha256)):
            raise ValueError('explicit lowercase IQ3 grid SHA-256 required')
        owned = np.array(observed, dtype=np.uint8, order='C', copy=True)
        digest = hashlib.sha256(owned.tobytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError('IQ3 grid identity mismatch')
        # Immutable bytes backing prevents callers from re-enabling writes.
        owned = np.frombuffer(owned.tobytes(), dtype=np.uint8).reshape(256, 4)
        object.__setattr__(self, 'grid', owned)
        object.__setattr__(self, 'identity', digest)

    def metadata(self):
        return dict(format=DTYPE, grid_sha256=self.identity, rotation=ROTATION)

    def write(self, destination):
        """Publish the exact admitted grid atomically beside model artifacts."""
        path = Path(destination)
        descriptor, temporary = tempfile.mkstemp(prefix='.iq3-grid-', dir=path.parent)
        try:
            try:
                output = os.fdopen(descriptor, 'wb')
            except BaseException:
                os.close(descriptor)
                raise
            with output:
                output.write(b'OAIQ3G01')
                output.write(bytes.fromhex(self.identity))
                output.write(self.grid.tobytes())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            if os.name != 'nt':
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @classmethod
    def read(cls, source, *, expected_sha256):
        """Read a fixed-size grid and check both stored and caller identities."""
        path = Path(source)
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != 1064:
                raise ValueError('fixed-size regular IQ3 grid file required')
            raw = stream.read(1065)
            after = os.fstat(stream.fileno())
        def stamp(info):
            return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if stamp(before) != stamp(after) or stamp(before) != stamp(path.stat()):
            raise ValueError('IQ3 grid file changed during admission')
        if len(raw) != 1064 or raw[:8] != b'OAIQ3G01':
            raise ValueError('invalid IQ3 grid file extent or version')
        if raw[8:40].hex() != expected_sha256:
            raise ValueError('stored IQ3 grid identity differs from caller admission')
        return cls(np.frombuffer(raw[40:], dtype=np.uint8).reshape(256, 4),
                   expected_sha256=expected_sha256)


def validate_contract_metadata(value):
    if (not isinstance(value, dict) and not hasattr(value, 'keys')):
        raise ValueError('IQ3 contract metadata required')
    if (set(value) != {'format', 'grid_sha256', 'rotation'} or value['format'] != DTYPE or
            value['rotation'] != ROTATION or type(value['grid_sha256']) is not str or
            len(value['grid_sha256']) != 64 or
            any(c not in '0123456789abcdef' for c in value['grid_sha256'])):
        raise ValueError('exact IQ3 format/grid/rotation contract required')


def payload_size(rows, cols):
    if (type(rows) is not int or type(cols) is not int or rows <= 0 or
            cols <= 0 or cols % BLOCK_WEIGHTS):
        raise ValueError('positive IQ3 matrix with 256-aligned columns required')
    return rows*(cols//BLOCK_WEIGHTS)*BLOCK_BYTES


def decode_iq3(payload, *, rows, cols, grid, max_bytes):
    """Decode indices, sign parity, subscales and FP16 super-scales exactly."""
    if type(grid) is not IQ3Grid:
        raise ValueError('admitted IQ3 grid contract required')
    size = payload_size(rows, cols)
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('positive IQ3 decode admission budget required')
    if size+rows*cols*4+4096 > max_bytes:
        raise MemoryError('IQ3 decode buffers exceed admission')
    if isinstance(payload, np.ndarray):
        if payload.dtype != np.uint8 or not payload.flags.c_contiguous:
            raise ValueError('contiguous uint8 IQ3 payload required')
        raw = memoryview(payload).cast('B')
    elif isinstance(payload, (bytes, bytearray, memoryview)):
        raw = memoryview(payload).cast('B')
    else:
        raise ValueError('binary IQ3 payload required')
    if len(raw) != size:
        raise ValueError('IQ3 payload extent mismatch')
    output = np.empty((rows, cols), dtype=np.float32)
    blocks = cols//BLOCK_WEIGHTS
    for row in range(rows):
        for block in range(blocks):
            offset = (row*blocks+block)*BLOCK_BYTES
            scale = struct.unpack_from('<e', raw, offset+96)[0]
            if not math.isfinite(scale) or raw[offset+97] & 128:
                raise ValueError('invalid IQ3 FP16 super-scale')
            for sub in range(8):
                word = struct.unpack_from('<I', raw, offset+64+sub*4)[0]
                factor = scale*(0.5+(word >> 28))*0.25
                for octet in range(4):
                    signs = (word >> (octet*7)) & 127
                    parity = signs.bit_count() & 1
                    for lane in range(8):
                        index = sub*32+octet*8+lane
                        magnitude = int(grid.grid[raw[offset+index//4], index % 4])
                        negative = ((signs >> lane) & 1) if lane < 7 else parity
                        value = factor*magnitude
                        output[row, block*256+index] = -value if negative else value
    if not np.isfinite(output).all():
        raise ValueError('nonfinite decoded IQ3 matrix')
    return output


def encode_iq3(weights, *, grid, max_bytes, rotate=True):
    """Pack independent nearest-grid IQ3 weights with parity-correct signs.

    The RMS/20 super-scale is rounded to its stored FP16 value before the
    sixteen subscale choices are searched. Ties retain the lowest grid/code.
    Search arithmetic uses FP64; numerical agreement with other converters
    is not claimed. ``rotate=False`` requires already rotated weight rows.
    """
    if type(grid) is not IQ3Grid or type(rotate) is not bool:
        raise ValueError('admitted IQ3 grid and explicit rotation policy required')
    observed = np.asarray(weights)
    if (observed.ndim != 2 or observed.dtype.kind != 'f' or not observed.size or
            not np.isfinite(observed).all()):
        raise ValueError('finite IQ3 weight matrix required')
    rows, cols = observed.shape
    size = payload_size(rows, cols)
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('positive IQ3 encoder budget required')
    admission = size+observed.size*32+min(cols, 32768)*4+256*1024
    if admission > max_bytes:
        raise MemoryError('IQ3 encoder buffers exceed admission')
    values = (rotate_iq3(observed, max_bytes=max_bytes-size) if rotate else
              np.array(observed, dtype=np.float32, order='C', copy=True))
    if not np.isfinite(values).all():
        raise ValueError('IQ3 weights exceed FP32 representation')
    output = np.zeros((rows, cols//256*98), dtype=np.uint8)
    table = grid.grid.astype(np.float64)
    norms = np.sum(table*table, axis=1)
    for row in range(rows):
        for block in range(cols//256):
            data = values[row, block*256:(block+1)*256].astype(np.float64)
            magnitude = np.abs(data)
            negative = data < 0
            for octet in range(32):
                start = octet*8
                if int(np.count_nonzero(negative[start:start+8])) % 2:
                    lane = start+int(np.argmin(magnitude[start:start+8]))
                    negative[lane] = not negative[lane]
            anchor = math.sqrt(float(np.mean(magnitude*magnitude)))/20.0+1e-12
            try:
                raw_scale = struct.pack('<e', anchor)
            except (OverflowError, struct.error) as error:
                raise ValueError('IQ3 super-scale exceeds finite FP16 range') from error
            scale = struct.unpack('<e', raw_scale)[0]
            base = block*98
            output[row, base+96:base+98] = np.frombuffer(raw_scale, dtype=np.uint8)
            for sub in range(8):
                vectors = magnitude[sub*32:(sub+1)*32].reshape(8, 4)
                products = vectors @ table.T
                best_error, best_indices, best_code = math.inf, None, 0
                for code in range(16):
                    factor = scale*(0.5+code)*0.25
                    scores = factor*factor*norms[None, :]-2*factor*products
                    indices = np.argmin(scores, axis=1)
                    residual = vectors-factor*table[indices]
                    error = float(np.sum(residual*residual))
                    if error < best_error:
                        best_error, best_indices, best_code = error, indices.copy(), code
                output[row, base+sub*8:base+(sub+1)*8] = best_indices
                word = best_code << 28
                for octet in range(4):
                    for lane in range(7):
                        if negative[sub*32+octet*8+lane]:
                            word |= 1 << (octet*7+lane)
                output[row, base+64+sub*4:base+68+sub*4] = np.frombuffer(
                    struct.pack('<I', word), dtype=np.uint8)
    return output
