"""Bounded native MLA/indexer continuation transport without token replay."""
import hashlib
import hmac
import json
import struct

import numpy as np

from .segment_snapshot import _pairs, _read, _write

MAGIC = b'OAGDSA01'
CHUNK = 262144


def _header(model, batch, position):
    c = model.config
    widths = [[c.kv_lora_rank, c.qk_rope_head_dim,
               c.index_head_dim if kind == 'full' else 0] for kind in c.indexer_types]
    return dict(identity=model.identity, batch=batch, position=position,
                widths=widths, mask_shape=[batch, position] if batch else [])


def _extent(header, length):
    return 44 + length + header['batch'] * header['position'] * (
        1 + 4 * sum(sum(widths) for widths in header['widths']))


def _finite(array):
    flat = array.reshape(-1)
    for offset in range(0, flat.size, CHUNK):
        if not np.isfinite(flat[offset:offset + CHUNK]).all():
            raise ValueError('nonfinite native GLM continuation')


def write_cache(model, output, *, max_bytes):
    with model._lock:
        model._check_open()
        if type(max_bytes) is not int or max_bytes < 44:
            raise ValueError('explicit native GLM cache transport budget required')
        batch = 0 if model.mask is None else model.mask.shape[0]
        header = _header(model, batch, model.position)
        encoded = json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        extent = _extent(header, len(encoded))
        if len(encoded) > 65536 or extent > max_bytes:
            raise MemoryError('native GLM cache output exceeds admission')
        model._admit_snapshot(3 * 1024**2)
        model._busy = True
        try:
            model._cancel()
            digest = hashlib.sha256()
            prefix = MAGIC + struct.pack('<I', len(encoded)) + encoded
            digest.update(prefix)
            _write(output, prefix)
            if batch:
                if len(model.cache) != len(header['widths']):
                    raise ValueError('native GLM cache layer inventory differs')
                for state, widths in zip(model.cache, header['widths']):
                    if state is None or len(state) != 3:
                        raise ValueError('native GLM physical continuation missing')
                    for array, width in zip(state, widths):
                        if not width:
                            if array is not None:
                                raise ValueError('shared GLM indexer has unexpected state')
                            continue
                        if (array is None or array.dtype != np.float32 or
                                array.shape != (batch, model.config.max_position_embeddings, width)):
                            raise ValueError('native GLM physical geometry differs')
                        for row in array[:, :model.position]:
                            flat = row.reshape(-1)
                            for begin in range(0, flat.size, CHUNK):
                                model._cancel()
                                host = np.ascontiguousarray(flat[begin:begin + CHUNK], dtype='<f4')
                                _finite(host)
                                raw = memoryview(host).cast('B')
                                digest.update(raw)
                                _write(output, raw)
                if model.mask.shape != (batch, model.position) or model.mask.dtype != np.uint8:
                    raise ValueError('native GLM prefix mask geometry differs')
                flat = model.mask.reshape(-1)
                for begin in range(0, flat.size, CHUNK):
                    model._cancel()
                    chunk = flat[begin:begin + CHUNK]
                    if np.any(chunk > 1):
                        raise ValueError('native GLM prefix mask must be binary')
                    raw = memoryview(np.ascontiguousarray(chunk)).cast('B')
                    digest.update(raw)
                    _write(output, raw)
            model._cancel()
            _write(output, digest.digest())
            return extent
        finally:
            model._busy = False


def restore_cache(model, source, *, max_bytes):
    with model._lock:
        model._check_open()
        if type(max_bytes) is not int or max_bytes < 44:
            raise ValueError('explicit native GLM cache transport budget required')
        model._admit_snapshot(3 * 1024**2)
        model._busy = True
        pending=[];state=[];mask=None;array=None;flat=row=raw=None;committed=False
        try:
            model._cancel()
            prefix = _read(source, 12)
            if prefix[:8] != MAGIC:
                raise ValueError('invalid native GLM cache magic')
            length = struct.unpack('<I', prefix[8:])[0]
            if not 0 < length <= min(65536, max_bytes - 44):
                raise ValueError('native GLM cache header exceeds admission')
            encoded = _read(source, length)
            header = json.loads(encoded, object_pairs_hook=_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if type(header) is not dict or set(header) != {'identity', 'batch', 'position', 'widths', 'mask_shape'}:
                raise ValueError('invalid native GLM continuation fields')
            batch, position = header['batch'], header['position']
            if (type(batch) is not int or not 0 <= batch <= 128 or type(position) is not int
                    or not 0 <= position <= model.config.max_position_embeddings
                    or type(header['widths']) is not list or any(type(row) is not list
                        or len(row)!=3 or any(type(width) is not int or width<0 for width in row)
                        for row in header['widths'])
                    or type(header['mask_shape']) is not list or any(type(value) is not int
                        or value<0 for value in header['mask_shape'])
                    or bool(batch) != bool(position) or header != _header(model, batch, position)):
                raise ValueError('native GLM cache identity or geometry differs')
            if _extent(header, length) > max_bytes:
                raise MemoryError('native GLM cache payload exceeds admission')
            required = batch * model.config.max_position_embeddings * 4 * sum(
                sum(widths) for widths in header['widths']) + batch * position
            model._admit_snapshot(required + 3 * 1024**2)
            digest = hashlib.sha256(prefix + encoded)
            for widths in header['widths']:
                if not batch:
                    pending.append(None)
                    continue
                state = []
                for width in widths:
                    if not width:
                        state.append(None)
                        continue
                    array = np.zeros((batch, model.config.max_position_embeddings, width), np.float32)
                    for row in array[:, :position]:
                        flat = row.reshape(-1)
                        for begin in range(0, flat.size, CHUNK):
                            model._cancel()
                            end = min(flat.size, begin + CHUNK)
                            raw = _read(source, (end - begin) * 4)
                            digest.update(raw)
                            flat[begin:end] = np.frombuffer(raw, '<f4')
                            _finite(flat[begin:end])
                    state.append(array)
                pending.append(tuple(state))
            mask = None
            if batch:
                mask = np.empty((batch, position), np.uint8)
                flat = mask.reshape(-1)
                for begin in range(0, flat.size, CHUNK):
                    model._cancel()
                    end = min(flat.size, begin + CHUNK)
                    raw = _read(source, end - begin)
                    digest.update(raw)
                    flat[begin:end] = np.frombuffer(raw, np.uint8)
                    if np.any(flat[begin:end] > 1):
                        raise ValueError('native GLM prefix mask must be binary')
            if not hmac.compare_digest(_read(source, 32), digest.digest()) or source.read(1):
                raise ValueError('native GLM cache checksum or trailing extent differs')
            model._cancel()
            model.cache, model.mask, model.position = pending, mask, position
            committed=True
        finally:
            if not committed:
                pending.clear();state.clear();mask=None
            array=flat=row=raw=None
            model._busy = False


__all__ = ['write_cache', 'restore_cache']
