"""Bounded active-prefix native Mixtral cache transport, without token replay."""
import hashlib
import hmac
import json
import struct
from contextlib import contextmanager

import numpy as np

from .segment_snapshot import _read, _write, _pairs, _chunks

_MAGIC = b'OAMXKV01'
_HEADER_LIMIT = 16384
_CHUNK = 1024 * 1024


def _check(model, limit):
    from .native_mixtral_decoder import NativeMixtralDecoder
    if type(model) is not NativeMixtralDecoder:
        raise TypeError('exact NativeMixtralDecoder required')
    if type(limit) is not int or limit < 44:
        raise ValueError('explicit positive snapshot byte budget required')
    model._check_open()
    if model._expert_provider is not None and model._expert_provider.identity != model._expert_identity:
        raise ValueError('Mixtral expert execution identity changed')


@contextmanager
def _access(model, limit):
    from .native_mixtral_decoder import NativeMixtralDecoder
    if type(model) is not NativeMixtralDecoder:
        raise TypeError('exact NativeMixtralDecoder required')
    with model._lock:
        _check(model, limit)
        model._busy = True
        try:
            yield
        finally:
            model._busy = False


def _canonical(header):
    return json.dumps(header, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('ascii')


def write_mixtral_cache(model, output, *, max_bytes):
    """Stream only committed slots while holding the model lock.

    The caller owns the stream and atomic publication of any destination file.
    A failed write may leave a partial stream, never a successful checkpoint.
    """
    with _access(model, max_bytes):
        batch = 0 if model._cache is None else model._cache[0][0].shape[0]
        position = model._position
        config = model.config
        shape = (batch, position, config['num_key_value_heads'], config['head_dim'])
        arrays = [] if model._cache is None else [
            array[:, :position] for layer in model._cache for array in layer]
        mask = model._mask
        if (batch == 0 and (position or mask is not None)) or (
                batch and (mask is None or mask.shape != (batch, position))):
            raise ValueError('inconsistent committed Mixtral state')
        header = dict(version=1, identity=model.identity, context=model.max_context,
                      batch=batch, position=position, shape=list(shape),
                      arrays=len(arrays), mask_bytes=batch * position)
        encoded = _canonical(header)
        extent = 44 + len(encoded) + sum(a.nbytes for a in arrays) + batch * position
        if len(encoded) > _HEADER_LIMIT or extent > max_bytes:
            raise MemoryError('Mixtral checkpoint exceeds wire admission')
        current = 0 if model._cache is None else sum(a.nbytes for layer in model._cache for a in layer)
        model._admit(model._weight_bytes + current + model._scratch_bytes + _CHUNK * 3)
        digest = hashlib.sha256()
        for data in (_MAGIC, struct.pack('<I', len(encoded)), encoded):
            digest.update(data)
            _write(output, data)
        for array in arrays:
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError('invalid committed Mixtral cache geometry')
            for data in _chunks(array):
                digest.update(data)
                _write(output, data)
        if mask is not None:
            if mask.dtype != np.uint8 or np.any(mask > 1):
                raise ValueError('invalid committed Mixtral padding mask')
            flat = mask.reshape(-1)
            for start in range(0, flat.size, _CHUNK):
                data = flat[start:start + _CHUNK].tobytes()
                digest.update(data)
                _write(output, data)
        _write(output, digest.digest())
        return extent


def restore_mixtral_cache(model, source, *, max_bytes):
    """Stage a bounded full-capacity cache; publish after checksum and EOF.

    Explicit rotary positions are already incorporated into stored keys. The
    physical position and padding mask preserve the decoder continuation rules.
    Restoring across a different execution identity is intentionally rejected.
    """
    with _access(model, max_bytes):
        prefix = _read(source, 12)
        if prefix[:8] != _MAGIC:
            raise ValueError('invalid Mixtral checkpoint magic')
        length = struct.unpack('<I', prefix[8:])[0]
        if not 0 < length <= min(_HEADER_LIMIT, max_bytes - 44):
            raise ValueError('Mixtral checkpoint header exceeds admission')
        encoded = _read(source, length)
        try:
            header = json.loads(encoded, object_pairs_hook=_pairs,
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ValueError('invalid Mixtral checkpoint header') from error
        fields = {'version', 'identity', 'context', 'batch', 'position',
                  'shape', 'arrays', 'mask_bytes'}
        if type(header) is not dict or set(header) != fields:
            raise ValueError('invalid Mixtral checkpoint fields')
        for name in ('version', 'context', 'batch', 'position', 'arrays', 'mask_bytes'):
            if type(header[name]) is not int:
                raise ValueError('Mixtral checkpoint geometry must use integers')
        if (header['version'] != 1 or header['identity'] != model.identity or
                header['context'] != model.max_context):
            raise ValueError('Mixtral checkpoint execution identity differs')
        batch, position = header['batch'], header['position']
        if not 0 <= batch <= 16 or not 0 <= position <= model.max_context or (not batch and position):
            raise ValueError('Mixtral checkpoint batch or position exceeds admission')
        config = model.config
        shape = (batch, position, config['num_key_value_heads'], config['head_dim'])
        count = 2 * config['num_hidden_layers'] if batch else 0
        if (type(header['shape']) is not list or
                any(type(d) is not int for d in header['shape']) or
                header['shape'] != list(shape) or header['arrays'] != count or
                header['mask_bytes'] != batch * position):
            raise ValueError('Mixtral checkpoint payload geometry differs')
        elements = batch * position * config['num_key_value_heads'] * config['head_dim']
        extent = 44 + length + count * elements * 4 + batch * position
        full_shape = (batch, model.max_context, *shape[2:])
        full_bytes = count * batch * model.max_context * shape[2] * shape[3] * 4
        if extent > max_bytes or full_bytes + batch * position + _CHUNK * 3 > max_bytes:
            raise MemoryError('Mixtral restore wire or staging exceeds admission')
        current = 0 if model._cache is None else sum(a.nbytes for layer in model._cache for a in layer)
        model._admit(model._weight_bytes + current + full_bytes + batch * position
                     + model._scratch_bytes + _CHUNK * 3)
        digest = hashlib.sha256(prefix + encoded)
        arrays = []
        mask=array=target=values=data=cache=None
        committed=False
        try:
            for _ in range(count):
                array = np.zeros(full_shape, dtype=np.float32)
                # Per-batch active prefixes are contiguous; unused slots stay zero.
                for row in range(batch):
                    target = array[row, :position].reshape(-1)
                    for start in range(0, target.size, _CHUNK // 4):
                        size = min(_CHUNK // 4, target.size - start)
                        data = _read(source, size * 4)
                        digest.update(data)
                        values = np.frombuffer(data, dtype='<f4')
                        if not np.isfinite(values).all():
                            raise ValueError('nonfinite Mixtral checkpoint cache')
                        target[start:start + size] = values
                arrays.append(array)
                array=target=values=data=None
            mask = None if not batch else np.empty((batch, position), dtype=np.uint8)
            if mask is not None:
                target = mask.reshape(-1)
                for start in range(0, target.size, _CHUNK):
                    size = min(_CHUNK, target.size - start)
                    data = _read(source, size)
                    digest.update(data)
                    values = np.frombuffer(data, dtype=np.uint8)
                    if np.any(values > 1):
                        raise ValueError('invalid Mixtral checkpoint padding mask')
                    target[start:start + size] = values
            if not hmac.compare_digest(bytes(_read(source, 32)), digest.digest()):
                raise ValueError('Mixtral checkpoint checksum mismatch')
            if source.read(1) != b'':
                raise ValueError('trailing Mixtral checkpoint bytes')
            if mask is not None and not np.all(mask) and model._run_inputs is None:
                raise ValueError('loaded Mixtral native ABI cannot continue masked state')
            cache = None if not batch else [tuple(arrays[i:i + 2]) for i in range(0, count, 2)]
            model._cache, model._position, model._mask = cache, position, mask
            committed=True
            return position
        finally:
            if not committed:
                arrays.clear();mask=None;cache=None
            array=target=values=data=None
