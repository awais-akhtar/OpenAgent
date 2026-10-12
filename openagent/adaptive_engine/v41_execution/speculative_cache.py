"""Durable compound DSpark state; checksums detect corruption, not authenticity."""
import copy
import hashlib
import hmac
import json
import struct
import tempfile

import numpy as np

from .mtp_cache import _budget, _read, _write, _rng, _unique

_PREFIX = struct.Struct('<8sI')
_MAGIC = b'OASPC001'


def transport_speculative_cache(decoder, stream, *, max_bytes, restoring=False):
    from .dspark import DSparkBackbone, DSparkProposer
    _budget(max_bytes)
    target, proposer = decoder.target, decoder.proposer
    if (type(target) is not DSparkBackbone or type(proposer) is not DSparkProposer or
            proposer.backbone is not target):
        raise ValueError('durable speculative cache requires the owned DSpark composition')
    draft = proposer.draft
    with decoder._lock, proposer._lock, target._lock, target.model._lock, draft._lock:
        if decoder._busy or decoder._quarantined:
            raise RuntimeError('speculative decoder is unavailable')
        target._check_transport()
        draft._check_transport()
        policy = [decoder.vocabulary, decoder.context, decoder.temperature,
                  decoder.draft_length, proposer.temperature]
        identity = [target.model.identity, draft.provider.identity]
        expected_rng = decoder.rng.bit_generator.state['bit_generator']
        expected_proposal_rng = proposer.rng.bit_generator.state['bit_generator']
        decoder._busy = True
        try:
            with tempfile.TemporaryFile() as target_file, tempfile.TemporaryFile() as draft_file:
                if not restoring:
                    if decoder.next_logits is None or not decoder.history:
                        raise RuntimeError('prefilled speculative generation required for persistence')
                    if (type(decoder.history) is not tuple or len(decoder.history) > decoder.context or
                            any(type(token) is not int or not 0 <= token < decoder.vocabulary
                                for token in decoder.history) or
                            np.asarray(decoder.next_logits).shape != (decoder.vocabulary,) or
                            not np.isfinite(decoder.next_logits).all()):
                        raise ValueError('invalid speculative verifier state')
                    if target.history != decoder.history:
                        raise ValueError('speculative history differs from target cache')
                    decoder._check_target_state(decoder.history, decoder.next_logits)
                    target.write_cache(target_file, max_bytes=max_bytes)
                    remaining = max_bytes-target_file.tell()
                    if remaining <= _PREFIX.size+32:
                        raise MemoryError('target cache leaves no admitted draft staging budget')
                    draft.write_cache(draft_file, max_bytes=remaining)
                    sizes = [target_file.tell(), draft_file.tell()]
                    states = [copy.deepcopy(decoder.rng.bit_generator.state),
                              copy.deepcopy(proposer.rng.bit_generator.state)]
                    _rng(states[0], expected_rng)
                    _rng(states[1], expected_proposal_rng)
                    raw = json.dumps(dict(version=1, identity=identity, policy=policy,
                        count=len(decoder.history), sizes=sizes, rng=states), sort_keys=True,
                        separators=(',', ':'), allow_nan=False).encode('utf-8')
                    total = _PREFIX.size+len(raw)+sum(sizes)+len(decoder.history)*8+decoder.vocabulary*8+32
                    if len(raw) > 65536 or total > max_bytes:
                        raise MemoryError('compound speculative frame exceeds admission')
                    prefix = _PREFIX.pack(_MAGIC, len(raw))
                    digest = hashlib.sha256(prefix+raw)
                    target._cache_transport = draft._cache_transport = target.model._busy = True
                    try:
                        _write(stream, prefix)
                        _write(stream, raw)
                        for source, size in zip((target_file, draft_file), sizes):
                            source.seek(0)
                            while size:
                                data = _read(source, min(65536, size))
                                digest.update(data)
                                _write(stream, data)
                                size -= len(data)
                        for start in range(0, len(decoder.history), 8192):
                            data = np.asarray(decoder.history[start:start+8192], dtype='<u8').tobytes()
                            digest.update(data)
                            _write(stream, data)
                        data = np.asarray(decoder.next_logits, dtype='<f8').tobytes()
                        digest.update(data)
                        _write(stream, data)
                        _write(stream, digest.digest())
                    finally:
                        target._cache_transport = draft._cache_transport = target.model._busy = False
                    return len(decoder.history)
                target._cache_transport = draft._cache_transport = target.model._busy = True
                try:
                    prefix = _read(stream, _PREFIX.size)
                    magic, size = _PREFIX.unpack(prefix)
                    if magic != _MAGIC or not 0 < size <= 65536 or _PREFIX.size+size+32 > max_bytes:
                        raise ValueError('invalid compound speculative cache header')
                    raw = _read(stream, size)
                    header = json.loads(raw, object_pairs_hook=_unique,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite metadata')))
                    if (type(header) is not dict or set(header) !=
                            {'version', 'identity', 'policy', 'count', 'sizes', 'rng'} or
                            type(header['version']) is not int or header['version'] != 1 or
                            header['identity'] != identity or
                            type(header['policy']) is not list or len(header['policy']) != 5 or
                            any(type(header['policy'][i]) is not int for i in (0, 1, 3)) or
                            any(type(header['policy'][i]) is not float for i in (2, 4)) or
                            header['policy'] != policy or
                            type(header['count']) is not int or not 0 < header['count'] <= decoder.context or
                            type(header['sizes']) is not list or len(header['sizes']) != 2 or
                            any(type(v) is not int or v <= 0 for v in header['sizes']) or
                            type(header['rng']) is not list or len(header['rng']) != 2):
                        raise ValueError('compound speculative cache policy or inventory mismatch')
                    count, sizes = header['count'], header['sizes']
                    if _PREFIX.size+size+sum(sizes)+count*8+decoder.vocabulary*8+32 > max_bytes:
                        raise MemoryError('compound speculative frame exceeds admission')
                    _rng(header['rng'][0], expected_rng)
                    _rng(header['rng'][1], expected_proposal_rng)
                    digest = hashlib.sha256(prefix+raw)
                    for output, remaining in zip((target_file, draft_file), sizes):
                        while remaining:
                            data = _read(stream, min(65536, remaining))
                            digest.update(data)
                            _write(output, data)
                            remaining -= len(data)
                    data = _read(stream, count*8)
                    digest.update(data)
                    ids = np.frombuffer(data, dtype='<u8')
                    if np.any(ids >= decoder.vocabulary):
                        raise ValueError('invalid speculative cache token')
                    history = tuple(int(token) for token in ids)
                    data = _read(stream, decoder.vocabulary*8)
                    digest.update(data)
                    logits = np.frombuffer(data, dtype='<f8').copy()
                    if not np.isfinite(logits).all():
                        raise ValueError('nonfinite speculative verifier logits')
                    if not hmac.compare_digest(_read(stream, 32), digest.digest()) or stream.read(1) != b'':
                        raise ValueError('compound speculative cache checksum or trailing bytes mismatch')
                finally:
                    target._cache_transport = draft._cache_transport = target.model._busy = False
                old_target, old_draft = target.snapshot(), proposer.snapshot()
                old_rng = copy.deepcopy(decoder.rng.bit_generator.state)
                try:
                    target_file.seek(0)
                    draft_file.seek(0)
                    target.restore_cache(target_file, max_bytes=sizes[0])
                    draft.restore_cache(draft_file, max_bytes=sizes[1])
                    if target.history != history:
                        raise ValueError('speculative history differs from recovered target tokens')
                    decoder._check_target_state(history, logits)
                    decoder.rng.bit_generator.state = copy.deepcopy(header['rng'][0])
                    proposer.rng.bit_generator.state = copy.deepcopy(header['rng'][1])
                except BaseException as failure:
                    decoder._rollback(old_target, old_draft, old_rng, failure)
                    raise
                decoder.history, decoder.next_logits = history, logits
                return count
        finally:
            decoder._busy = False
