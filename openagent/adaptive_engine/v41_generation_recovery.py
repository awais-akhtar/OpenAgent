"""Bounded publisher V4.1 generation frames, including compound DSpark state."""
import copy
import hashlib
import hmac
import json
import os
import math
import struct
import tempfile
import threading
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np

from .segment_snapshot import _read, _write, _pairs

_PREFIX = struct.Struct('<8sIQQ')
_MAGIC = b'OAV4GN01'
_FIELDS = {'schema', 'identity', 'tokenizer', 'position', 'prompt_ids',
           'generated_ids', 'temperature', 'rng', 'stops', 'dspark',
           'proposal_rng', 'accepted', 'proposed', 'stop_prefixes'}


@dataclass(frozen=True)
class V41GenerationBoundary:
    identity: str
    tokenizer_sha256: str
    position: int
    prompt_ids: tuple
    generated_ids: tuple
    temperature: float
    stop_token_ids: tuple
    dspark: bool
    accepted: int
    proposed: int
    rng: dict = None
    proposal_rng: dict = None
    stop_prefixes: tuple = ()


def tokenizer_identity(path, *, max_bytes=256*1024**2):
    from .session_recovery import _path
    path=_path(path)
    before = path.stat()
    if type(max_bytes) is not int or max_bytes<=0:
        raise ValueError('positive tokenizer identity extent admission required')
    if not 0<before.st_size<=max_bytes:
        raise MemoryError('tokenizer identity file exceeds admission')
    def stamp(info):return info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns
    digest = hashlib.sha256()
    with path.open('rb') as source:
        if stamp(os.fstat(source.fileno()))!=stamp(before):
            raise ValueError('tokenizer changed during generation opening')
        count=0
        for data in iter(lambda: source.read(65536), b''):
            count+=len(data)
            if count>before.st_size:
                raise ValueError('tokenizer grew during generation identity read')
            digest.update(data)
        if count!=before.st_size:
            raise ValueError('tokenizer truncated during generation identity read')
        if stamp(os.fstat(source.fileno()))!=stamp(before):
            raise ValueError('tokenizer changed during generation identity read')
    after = path.stat()
    if stamp(before)!=stamp(after):
        raise ValueError('tokenizer changed during generation admission')
    return digest.hexdigest()


def admit(header, logits, *, model, tokenizer, max_bytes, host_bytes):
    if (type(max_bytes) is not int or max_bytes <= 128 or type(host_bytes) is not int
            or host_bytes <= 0 or type(header) is not dict or set(header) != _FIELDS
            or header['schema'] != 'openagent.v41-generation-boundary.v2'
            or header['identity'] != model.identity or header['tokenizer'] != tokenizer
            or type(header['position']) is not int or not 0 < header['position'] <= model.profile.context
            or type(header['dspark']) is not bool):
        raise ValueError('matching bounded V4.1 generation boundary required')
    for key in ('prompt_ids', 'generated_ids', 'stops'):
        ids = header[key]
        if (type(ids) is not list or len(ids) > model.profile.context
                or any(type(token) is not int or not 0 <= token < model.profile.vocabulary for token in ids)):
            raise ValueError('bounded V4.1 vocabulary history required')
    prefixes = header['stop_prefixes']
    if (type(prefixes) is not list or len(prefixes) > 64 or
            any(type(prefix) is not list or not prefix or len(prefix) > 128 or
                any(type(token) is not int or not 0 <= token < model.profile.vocabulary
                    for token in prefix) for prefix in prefixes) or
            len({tuple(prefix) for prefix in prefixes}) != len(prefixes)):
        raise ValueError('bounded V4.1 stop-prefix inventory required')
    if (not header['prompt_ids'] or len(header['prompt_ids']) + len(header['generated_ids']) != header['position']
            or len(set(header['stops'])) != len(header['stops'])):
        raise ValueError('V4.1 generation history differs from cache position')
    if (type(header['temperature']) not in (int, float) or not math.isfinite(header['temperature'])
            or header['temperature'] < 0):
        raise ValueError('finite V4.1 sampling temperature required')
    for name in ('accepted', 'proposed'):
        if type(header[name]) is not int or header[name] < 0:
            raise ValueError('nonnegative speculative counters required')
    if (header['accepted'] > header['proposed'] or
            (not header['dspark'] and (header['accepted'] or header['proposed']))):
        raise ValueError('speculative counters differ from generation policy')
    if type(header['rng']) is not dict or header['rng'].get('bit_generator') != 'PCG64':
        raise ValueError('complete original PCG64 state required')
    random = np.random.PCG64()
    random.state = copy.deepcopy(header['rng'])
    proposal_rng = header['proposal_rng']
    if header['dspark']:
        if type(proposal_rng) is not dict or proposal_rng.get('bit_generator') != 'PCG64':
            raise ValueError('complete DSpark proposal PCG64 state required')
        proposal_random = np.random.PCG64()
        proposal_random.state = copy.deepcopy(proposal_rng)
    elif proposal_rng is not None:
        raise ValueError('non-DSpark boundary cannot carry proposal RNG state')
    row = np.asarray(logits)
    if row.shape != (model.profile.vocabulary,) or row.dtype.kind != 'f' or not np.isfinite(row).all():
        raise ValueError('complete finite V4.1 next-token logits required')
    raw = json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
    if len(raw) > min(1024**2, max_bytes // 2) or len(raw)*32 + row.size*32 + 65536 > host_bytes:
        raise MemoryError('V4.1 generation metadata exceeds host admission')
    return raw


def _publish_locked(sink, header, logits, owner, *, model, tokenizer, max_bytes, host_bytes, cancelled=None):
    """A synchronous single-use writer keeps live state inside its request lease."""
    frozen = copy.deepcopy(header)
    raw = admit(frozen, logits, model=model, tokenizer=tokenizer,
                max_bytes=max_bytes, host_bytes=host_bytes)
    if (model.length!=frozen['position'] or tuple(model.snapshot().tokens)!=
            tuple(frozen['prompt_ids']+frozen['generated_ids'])):
        raise ValueError('V4.1 envelope differs from live cache continuation')
    if frozen['dspark'] and (tuple(owner.history)!=tuple(frozen['prompt_ids']+frozen['generated_ids'])
            or not np.array_equal(owner.next_logits,logits)
            or owner.rng.bit_generator.state!=frozen['rng']
            or getattr(getattr(owner.proposer, 'rng', None), 'bit_generator', None) is None
            or owner.proposer.rng.bit_generator.state!=frozen['proposal_rng']):
        raise ValueError('V4.1 envelope differs from committed speculative verifier state')
    row = np.asarray(logits, dtype='<f8').tobytes()
    active, consumed, written = True, False, False
    thread = threading.get_ident()
    def check():
        if cancelled is not None and cancelled():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('V4.1 checkpoint publication cancelled')
    def write_frame(output):
        nonlocal consumed, written
        if not active or consumed or threading.get_ident() != thread:
            raise RuntimeError('V4.1 checkpoint writer expired or already consumed')
        consumed = True
        check()
        overhead = _PREFIX.size + len(raw) + len(row) + 32
        if overhead >= max_bytes:
            raise MemoryError('V4.1 generation envelope exhausted cache admission')
        with tempfile.TemporaryFile('w+b') as cache:
            owner.write_cache(cache, max_bytes=max_bytes-overhead)
            size = cache.tell()
            if not 0 < size <= max_bytes-overhead:
                raise ValueError('V4.1 cache writer exceeded declared extent')
            prefix = _PREFIX.pack(_MAGIC, len(raw), len(row), size)
            digest = hashlib.sha256()
            for data in (prefix, raw, row):
                check(); digest.update(data); _write(output, data)
            cache.seek(0)
            remaining = size
            while remaining:
                check()
                data = _read(cache, min(65536, remaining))
                digest.update(data); _write(output, data); remaining -= len(data)
            check(); _write(output, digest.digest())
            written = True
            return overhead + size
    try:
        boundary=V41GenerationBoundary(frozen['identity'],frozen['tokenizer'],frozen['position'],
            tuple(frozen['prompt_ids']),tuple(frozen['generated_ids']),frozen['temperature'],
            tuple(frozen['stops']),frozen['dspark'],frozen['accepted'],frozen['proposed'],
            copy.deepcopy(frozen['rng']),copy.deepcopy(frozen['proposal_rng']),
            tuple(tuple(prefix) for prefix in frozen['stop_prefixes']))
        sink(boundary, write_frame)
        if not written:
            raise RuntimeError('checkpoint sink did not consume the complete V4.1 frame')
    finally:
        active = False


def publish(sink, header, logits, owner, *, model, tokenizer, max_bytes, host_bytes, cancelled=None):
    """Serialize boundary capture and frame writing against every physical owner.

    Serving callers already hold their request lease, but this codec is also a
    direct API.  It therefore takes the same reentrant owner locks as the
    compound cache transport instead of relying on a caller-held lock.
    """
    if cancelled is not None and cancelled():
        from .execution_service import ExecutionCancelled
        raise ExecutionCancelled('V4.1 checkpoint publication cancelled')
    locks = []
    if hasattr(owner, 'proposer'):
        proposer = owner.proposer
        target = getattr(proposer, 'backbone', None)
        draft = getattr(proposer, 'draft', None)
        if target is None or getattr(target, 'model', None) is not model or draft is None:
            raise ValueError('complete owned DSpark lock chain required for V4.1 publication')
        locks.extend((getattr(owner, '_lock', None), getattr(proposer, '_lock', None),
                      getattr(target, '_lock', None), getattr(model, '_lock', None),
                      getattr(draft, '_lock', None)))
    elif owner is model:
        locks.append(getattr(model, '_lock', None))
    else:
        raise ValueError('V4.1 publication requires the model or its owned DSpark decoder')
    if any(lock is None or not callable(getattr(lock, '__enter__', None))
           or not callable(getattr(lock, '__exit__', None)) for lock in locks):
        raise ValueError('reentrant V4.1 physical-state owner locks required')
    if len({id(lock) for lock in locks}) != len(locks):
        raise ValueError('distinct V4.1 physical-state locks required')
    with ExitStack() as acquired:
        for lock in locks:
            acquired.enter_context(lock)
        if cancelled is not None and cancelled():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('V4.1 checkpoint publication cancelled')
        return _publish_locked(sink, header, logits, owner, model=model,
            tokenizer=tokenizer, max_bytes=max_bytes, host_bytes=host_bytes,
            cancelled=cancelled)


def read(source, *, model, tokenizer, max_bytes, host_bytes, cancelled=None):
    """Checksum and admit the entire envelope before exposing cache staging."""
    if type(max_bytes) is not int or max_bytes <= 128:
        raise ValueError('positive complete V4.1 recovery admission required')
    def take(count):
        if cancelled is not None and cancelled():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('V4.1 checkpoint recovery cancelled')
        return _read(source, count)
    prefix = take(_PREFIX.size)
    magic, header_bytes, logit_bytes, cache_bytes = _PREFIX.unpack(prefix)
    if (magic != _MAGIC or not 0 < header_bytes <= min(1024**2, max_bytes//2)
            or logit_bytes != model.profile.vocabulary*8 or not cache_bytes
            or _PREFIX.size + header_bytes + logit_bytes + cache_bytes + 32 > max_bytes):
        raise ValueError('V4.1 generation frame geometry differs')
    if header_bytes*32 + logit_bytes*4 + 65536 > host_bytes:
        raise MemoryError('V4.1 recovery metadata exceeds host admission')
    raw = take(header_bytes)
    header = json.loads(raw, object_pairs_hook=_pairs,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    data = take(logit_bytes)
    logits = np.frombuffer(data, dtype='<f8').copy()
    admit(header, logits, model=model, tokenizer=tokenizer, max_bytes=max_bytes, host_bytes=host_bytes)
    cache = tempfile.TemporaryFile('w+b')
    try:
        digest = hashlib.sha256(prefix + raw + data)
        remaining = cache_bytes
        while remaining:
            data = take(min(65536, remaining))
            digest.update(data); _write(cache, data); remaining -= len(data)
        if not hmac.compare_digest(take(32), digest.digest()) or _read(source, 1):
            raise ValueError('V4.1 generation checksum or EOF differs')
        cache.seek(0)
        return header, logits, cache, cache_bytes
    except BaseException:
        cache.close()
        raise


__all__ = ['V41GenerationBoundary', 'publish', 'read', 'admit', 'tokenizer_identity']
