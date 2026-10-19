"""Bounded local K3Snapshot persistence; checksums detect corruption, not forgery."""
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile

import numpy as np

from .cache import MLALatentPages
from .executor import K3Execution, K3Snapshot
from ..session_recovery import _path as _safe_path
from ..native_library_identity import _stamp
from ..segment_snapshot import _write
from ..generation_checkpoint_files import _remove_staging
from ..owned_file_descriptor import close_after_failure,fdopen_owned


_MAGIC = b'OAK3S001'
_FLA_MAGIC = b'OAK3S002'
_EXACT_MAGIC = b'OAK3S003'
_PREFIX = struct.Struct('<8sIQ')
_HEADER_LIMIT = 65536
_ARRAY_LIMIT = 65536
_CHUNK_BYTES = 1024**2


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError('positive checkpoint limit required: ' + name)
    return value


def _limits(max_payload_bytes, max_header_bytes, max_arrays, chunk_bytes):
    if not 4<=_positive(chunk_bytes, 'chunk')<=8*1024**2:
        raise ValueError('checkpoint chunk must be between one FP32 element and 8 MiB')
    return (_positive(max_payload_bytes, 'payload'),
            min(_positive(max_header_bytes, 'header'), _HEADER_LIMIT),
            min(_positive(max_arrays, 'arrays'), _ARRAY_LIMIT),
            chunk_bytes)


def _model(model):
    if type(model) is not K3Execution:
        raise ValueError('exact K3Execution required')
    if model._closed or model._busy:
        raise RuntimeError('K3 executor unavailable for checkpoint')


def _path(value, *, existing):
    path = _safe_path(value).resolve(strict=existing)
    if path.drive.startswith('\\\\') or not path.parent.is_dir():
        raise ValueError('local checkpoint path with an existing parent required')
    return path


def _layout(model, position, maximum, array_limit):
    c = model.config
    if type(position) is not int or not 0 <= position <= c.capacity:
        raise ValueError('invalid K3 checkpoint position')
    mla = {i for i, kind in enumerate(c.layer_types) if kind == 'mla'}
    kda = {i for i, kind in enumerate(c.layer_types) if kind == 'kda'}
    if set(model._mla) != mla or set(model._kda) != kda:
        raise ValueError('live K3 cache inventory differs from configuration')
    records, size = [], 0

    def add(name, shape, dtype='<f4'):
        nonlocal size
        if len(records) >= array_limit:
            raise MemoryError('K3 checkpoint array inventory exceeds admission')
        count = math.prod(shape) * np.dtype(dtype).itemsize
        if count > maximum - size:
            raise MemoryError('K3 checkpoint payload exceeds admission')
        records.append(dict(name=name, dtype=dtype, shape=list(shape), nbytes=count))
        size += count

    for layer in sorted(mla):
        cache = model._mla[layer]
        if (type(cache) is not MLALatentPages or cache.rank != c.kvrank or cache.rope != c.rope
                or type(cache.page_tokens) is not int or cache.page_tokens <= 0):
            raise ValueError('unsupported K3 MLA cache geometry')
        pages = (position + cache.page_tokens - 1) // cache.page_tokens
        if pages > array_limit - len(records):
            raise MemoryError('K3 checkpoint page inventory exceeds admission')
        for page in range(pages):
            rows = min(cache.page_tokens, position - page * cache.page_tokens)
            add(f'mla.{layer}.page.{page}', (rows, c.kvrank + c.rope))
    for layer in sorted(kda):
        if model._fla_policy is not None:
            if position:
                add(f'kda.{layer}.state', (1, c.kda_heads, c.kda_dim, c.kda_dim))
                for name in ('q', 'k', 'v'):
                    add(f'kda.{layer}.history.{name}',
                        (1, c.kda_heads*c.kda_dim, c.kernel), '<u2')
            continue
        add(f'kda.{layer}.state', (c.kda_heads, c.kda_dim, c.kda_dim))
        for name in ('q', 'k', 'v'):
            add(f'kda.{layer}.history.{name}', (c.kernel - 1, c.kda_heads, c.kda_dim))
    return records, size


def _indexed(items, expected, name):
    if type(items) is not tuple:
        raise ValueError('K3 snapshot inventory must be tuple: ' + name)
    result = {}
    for item in items:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError('invalid K3 snapshot layer entry: ' + name)
        layer, value = item
        if type(layer) is not int or layer not in expected or layer in result:
            raise ValueError('duplicate/foreign K3 snapshot layer: ' + name)
        result[layer] = value
    if set(result) != expected:
        raise ValueError('missing K3 snapshot layers: ' + name)
    return result


def _arrays(model, snapshot, records):
    if (type(snapshot) is not K3Snapshot or snapshot.identity != model._identity
            or type(snapshot.position) is not int):
        raise ValueError('K3 snapshot identity/position mismatch')
    mla = _indexed(snapshot.mla, set(model._mla), 'MLA')
    kda = _indexed(snapshot.kda, set(model._kda), 'KDA')
    arrays = {}
    for layer, pages in mla.items():
        if type(pages) is not tuple:
            raise ValueError('K3 snapshot latent pages must be tuple')
        for page, array in enumerate(pages):
            arrays[f'mla.{layer}.page.{page}'] = array
    for layer, item in kda.items():
        if model._fla_policy is not None:
            from .fla_mixer import FLAMixerSnapshot
            mixer = model._kda[layer]
            if (type(item) is not FLAMixerSnapshot or item.identity != mixer.identity
                    or item.position != snapshot.position or type(item.convolution) is not tuple
                    or len(item.convolution) != 3):
                raise ValueError('FLA snapshot identity/position mismatch')
            if not snapshot.position:
                if item.recurrent is not None or item.convolution != (None, None, None):
                    raise ValueError('empty FLA cache contains occupied tensors')
                continue
            torch = mixer._torch
            tensors = (item.recurrent, *item.convolution)
            names = (f'kda.{layer}.state',
                     *(f'kda.{layer}.history.{name}' for name in ('q', 'k', 'v')))
            for index, (name, tensor) in enumerate(zip(names, tensors)):
                expected_dtype = torch.float32 if index == 0 else torch.bfloat16
                if (not isinstance(tensor, torch.Tensor) or tensor.dtype != expected_dtype
                        or tensor.device != mixer.device or not bool(torch.isfinite(tensor).all())):
                    raise ValueError('invalid raw FLA cache tensor')
                cpu = tensor.detach().contiguous().cpu()
                arrays[name] = (cpu.numpy().copy() if index == 0 else
                               cpu.view(torch.int16).numpy().view(np.uint16).copy())
            continue
        if type(item) is not tuple or len(item) != 3:
            raise ValueError('K3 snapshot needs KDA position/state/history')
        position, state, histories = item
        if (type(position) is not int or position != snapshot.position
                or type(histories) is not tuple or len(histories) != 3):
            raise ValueError('invalid K3 snapshot KDA position/history')
        arrays[f'kda.{layer}.state'] = state
        for name, array in zip(('q', 'k', 'v'), histories):
            arrays[f'kda.{layer}.history.{name}'] = array
    if set(arrays) != {record['name'] for record in records}:
        raise ValueError('K3 snapshot array inventory differs from complete layout')
    for record in records:
        array = arrays[record['name']]
        if (type(array) is not np.ndarray or array.dtype != np.dtype(record['dtype']).newbyteorder('=')
                or array.shape != tuple(record['shape']) or not array.flags.c_contiguous
                or array.nbytes != record['nbytes'] or not _finite(array, record['dtype'])):
            raise ValueError('invalid K3 snapshot array: ' + record['name'])
    return arrays


def _finite(array, dtype):
    if dtype == '<u2':
        return not np.any((array & np.uint16(0x7f80)) == np.uint16(0x7f80))
    return bool(np.isfinite(array).all())


def _config_digest(model):
    # Executor identity already includes weights/provider identity and config.
    # A separate config digest makes schema mismatches explicit in the format.
    from dataclasses import asdict
    raw = json.dumps(asdict(model.config), sort_keys=True, separators=(',', ':'),
                     ensure_ascii=True, allow_nan=False).encode('ascii')
    return hashlib.sha256(raw).hexdigest()


def save_k3_checkpoint(model, destination, *, max_payload_bytes,
                       max_header_bytes=_HEADER_LIMIT, max_arrays=_ARRAY_LIMIT,
                       chunk_bytes=_CHUNK_BYTES):
    """Save one locked committed snapshot with fsync and same-directory replace.

    Limits bound detached array bytes, descriptor count and metadata bytes.
    Weights, prompt tokens and provider resources are not snapshot members.
    """
    if type(model) is not K3Execution:
        raise ValueError('exact K3Execution required')
    maximum, header_limit, array_limit, chunk = _limits(
        max_payload_bytes, max_header_bytes, max_arrays, chunk_bytes)
    path = _path(destination, existing=False)
    with model._operation():
        records, size = _layout(model, model._position, maximum, array_limit)
        if model._admit(0)+2*min(chunk,maximum)>model.max_bytes:
            raise MemoryError('K3 snapshot write staging exceeds model admission')
        snapshot = model._snapshot_locked(cancelled=model._check_cancelled)
        arrays = _arrays(model, snapshot, records)
        format_version = 3 if model._exact_evidence is not None else 2 if model._fla_policy is not None else 1
        metadata=dict(version=format_version, identity=snapshot.identity,
            config_sha256=_config_digest(model), position=snapshot.position, records=records)
        if format_version==3:
            metadata.update(exact_evidence=snapshot.exact_evidence,media_case=snapshot.media_case,
                bf16_recurrence=model._fla_policy is not None)
        header = json.dumps(metadata,
            sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
        if len(header) > header_limit:
            raise MemoryError('K3 checkpoint metadata exceeds admission')
        prefix = _PREFIX.pack(_EXACT_MAGIC if format_version==3 else _FLA_MAGIC if format_version == 2 else _MAGIC, len(header), size)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                                             prefix='.k3-snapshot-', delete=False) as output:
                temporary = Path(output.name)
                digest = hashlib.sha256(prefix + header)
                _write(output,prefix)
                _write(output,header)
                written = 0
                for record in records:
                    dtype = np.dtype(record['dtype'])
                    elements = max(1, min(chunk, maximum) // dtype.itemsize)
                    flat = arrays[record['name']].reshape(-1)
                    for start in range(0, flat.size, elements):
                        model._check_cancelled()
                        data = flat[start:start + elements].astype(dtype, copy=False).tobytes()
                        _write(output,data)
                        digest.update(data)
                        written += len(data)
                if written != size:
                    raise ValueError('K3 checkpoint payload extent mismatch')
                model._check_cancelled();_write(output,digest.digest())
                output.flush()
                os.fsync(output.fileno())
            model._check_cancelled();_safe_path(path)
            os.replace(temporary, path)
            temporary = None
            if os.name != 'nt':
                descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            _remove_staging(temporary,sys.exc_info()[1])


def _unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('duplicate K3 checkpoint metadata field')
        result[name] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite K3 checkpoint metadata constant')


def _validate_records(actual, expected):
    if type(actual) is not list or len(actual) != len(expected):
        raise ValueError('K3 checkpoint missing/extra array records')
    for observed, required in zip(actual, expected):
        if (type(observed) is not dict or set(observed) != set(required)
                or type(observed['shape']) is not list
                or any(type(n) is not int or n < 0 for n in observed['shape'])
                or type(observed['nbytes']) is not int
                or type(observed['name']) is not str or type(observed['dtype']) is not str
                or observed != required):
            raise ValueError('K3 checkpoint array dtype/shape/order mismatch')


def _snapshot(model, position, arrays,exact_evidence=None,media_case=None):
    mla, kda = [], []
    for layer, kind in enumerate(model.config.layer_types):
        if kind == 'mla':
            page_tokens = model._mla[layer].page_tokens
            count = (position + page_tokens - 1) // page_tokens
            mla.append((layer, tuple(arrays[f'mla.{layer}.page.{i}'] for i in range(count))))
        else:
            if model._fla_policy is not None:
                from .fla_mixer import FLAMixerSnapshot
                mixer = model._kda[layer]
                if not position:
                    item = FLAMixerSnapshot(mixer.identity, 0, None, (None, None, None))
                else:
                    torch = mixer._torch
                    recurrent = torch.from_numpy(arrays[f'kda.{layer}.state'].copy()).to(mixer.device)
                    history = tuple(torch.from_numpy(
                        arrays[f'kda.{layer}.history.{name}'].view(np.int16).copy()
                    ).view(torch.bfloat16).to(mixer.device) for name in ('q', 'k', 'v'))
                    item = FLAMixerSnapshot(mixer.identity, position, recurrent, history)
                kda.append((layer, item))
                continue
            kda.append((layer, (position, arrays[f'kda.{layer}.state'],
                tuple(arrays[f'kda.{layer}.history.{name}'] for name in ('q', 'k', 'v')))))
    return K3Snapshot(model._identity, position, tuple(mla), tuple(kda),exact_evidence,media_case)


def load_k3_checkpoint(model, source, *, max_payload_bytes,
                       max_header_bytes=_HEADER_LIMIT, max_arrays=_ARRAY_LIMIT,
                       chunk_bytes=_CHUNK_BYTES):
    """Validate the entire bounded file before transactional executor restore.

    SHA-256 covers header, layout and payload. No pickle or array-code loading.
    The executor also applies its own resident/restore staging admission budget.
    """
    if type(model) is not K3Execution:
        raise ValueError('exact K3Execution required')
    maximum, header_limit, array_limit, chunk = _limits(
        max_payload_bytes, max_header_bytes, max_arrays, chunk_bytes)
    path = _path(source, existing=True)
    with model._operation():
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0)
                             | getattr(os, 'O_NONBLOCK', 0)|getattr(os,'O_NOFOLLOW',0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('regular K3 checkpoint file required')
            if _stamp(info)!=_stamp(_safe_path(path).stat()):
                raise ValueError('K3 checkpoint replaced during opening')
            if info.st_size > _PREFIX.size + header_limit + maximum + 32:
                raise MemoryError('K3 checkpoint file exceeds admission')
        except BaseException as primary:
            close_after_failure(descriptor,primary,owner='K3 checkpoint input')
            raise
        stream=fdopen_owned(descriptor,'rb',owner='K3 checkpoint input')
        with stream:
            def read_exact(size):
                model._check_cancelled()
                data = stream.read(size)
                if len(data) != size:
                    raise ValueError('truncated K3 checkpoint')
                return data

            prefix = read_exact(_PREFIX.size)
            magic, header_size, payload_size = _PREFIX.unpack(prefix)
            format_version = 3 if model._exact_evidence is not None else 2 if model._fla_policy is not None else 1
            if magic != (_EXACT_MAGIC if format_version==3 else _FLA_MAGIC if format_version == 2 else _MAGIC):
                raise ValueError('unsupported K3 checkpoint format')
            if not 1 <= header_size <= header_limit or payload_size > maximum:
                raise MemoryError('K3 checkpoint header/payload exceeds admission')
            if info.st_size != _PREFIX.size + header_size + payload_size + 32:
                raise ValueError('K3 checkpoint file extent mismatch')
            raw_header = read_exact(header_size)
            try:
                header = json.loads(raw_header.decode('ascii'), object_pairs_hook=_unique_fields,
                                    parse_constant=_reject_constant)
            except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
                raise ValueError('invalid K3 checkpoint metadata') from error
            fields={'version', 'identity', 'config_sha256', 'position', 'records'}
            if format_version==3:fields.update(('exact_evidence','media_case','bf16_recurrence'))
            if (type(header) is not dict or set(header) != fields
                    or type(header['version']) is not int or header['version'] != format_version
                    or header['identity'] != model._identity
                    or header['config_sha256'] != _config_digest(model)):
                raise ValueError('K3 checkpoint version/model identity mismatch')
            if format_version==3:
                if (header['exact_evidence']!=model._exact_evidence.identity or
                        type(header['bf16_recurrence']) is not bool or
                        header['bf16_recurrence']!=(model._fla_policy is not None)):
                    raise ValueError('K3 checkpoint exact policy or recurrence precision differs')
                model._exact_evidence.admit_snapshot(header['media_case'],header['position'])
            records, size = _layout(model, header['position'], maximum, array_limit)
            _validate_records(header['records'], records)
            if size != payload_size:
                raise ValueError('K3 checkpoint payload layout mismatch')
            required=model._admit(max(0,header['position']-model._position))
            if required+2*size+min(chunk,maximum)>model.max_bytes:
                raise MemoryError('K3 complete checkpoint restoration staging exceeds model admission')
            digest = hashlib.sha256(prefix + raw_header)
            arrays = {}
            for record in records:
                dtype = np.dtype(record['dtype'])
                array = np.empty(tuple(record['shape']), dtype=dtype)
                view = memoryview(array.reshape(-1)).cast('B')
                offset = 0
                while offset < len(view):
                    model._check_cancelled()
                    end = min(len(view), offset + chunk)
                    count = stream.readinto(view[offset:end])
                    if not count:
                        raise ValueError('truncated K3 checkpoint array')
                    digest.update(view[offset:offset + count])
                    offset += count
                # A native-endian array is required by the observed restore ABI.
                array = array.astype(dtype.newbyteorder('='), copy=False)
                if not _finite(array, record['dtype']):
                    raise ValueError('nonfinite K3 checkpoint array')
                arrays[record['name']] = array
            checksum = read_exact(32)
            model._check_cancelled()
            trailing = stream.read(1)
            if not hmac.compare_digest(checksum, digest.digest()) or trailing:
                raise ValueError('K3 checkpoint checksum/trailing data mismatch')
            if _stamp(os.fstat(stream.fileno()))!=_stamp(info) or _stamp(_safe_path(path).stat())!=_stamp(info):
                raise ValueError('K3 checkpoint changed during restoration')
            model._admit(max(0, header['position']-model._position))
            snapshot = _snapshot(model, header['position'], arrays,
                header.get('exact_evidence'),header.get('media_case'))
            _arrays(model, snapshot, records)
            model._check_cancelled()
            model._restore_locked(snapshot, cancelled=model._check_cancelled)
            return snapshot.position
