"""Bounded token-replay checkpoints; these do not serialize native caches."""
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import struct
import tempfile


_PREFIX = struct.Struct('<8sI')
_MAGIC = b'OAREPLAY'
_HEADER_LIMIT = 8192


@dataclass(frozen=True)
class ReplayCheckpoint:
    identity: str
    vocabulary: int
    context: int
    token_ids: tuple[int, ...]

    def validate(self, *, max_tokens):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('positive replay token budget required')
        if (type(self.identity) is not str or not self.identity or
                len(self.identity.encode('utf-8')) > 4096):
            raise ValueError('bounded model identity required')
        for value in (self.vocabulary, self.context):
            if type(value) is not int or not 0 < value <= 2**31-1:
                raise ValueError('invalid replay model geometry')
        if (type(self.token_ids) is not tuple or
                len(self.token_ids) > min(self.context, max_tokens) or
                any(type(token) is not int or not 0 <= token < self.vocabulary
                    for token in self.token_ids)):
            raise ValueError('invalid replay token history')
        return self


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate replay metadata field')
        result[key] = value
    return result


def _path(value):
    path = Path(value).absolute()
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('replay checkpoint paths cannot traverse links or reparse points')
    return path


def save_replay_checkpoint(checkpoint, destination, *, max_tokens):
    """Atomically replace one local file after bounded, checksummed serialization.

    The containing directory must already exist. SHA-256 detects corruption,
    not malicious modification by someone able to rewrite the entire file.
    """
    if type(checkpoint) is not ReplayCheckpoint:
        raise TypeError('ReplayCheckpoint required')
    checkpoint.validate(max_tokens=max_tokens)
    path = _path(destination)
    if path.exists() and not path.is_file():
        raise ValueError('regular replay checkpoint destination required')
    header = json.dumps(dict(version=1, identity=checkpoint.identity,
        vocabulary=checkpoint.vocabulary, context=checkpoint.context,
        count=len(checkpoint.token_ids)), ensure_ascii=True,
        separators=(',', ':'), sort_keys=True).encode('ascii')
    if len(header) > _HEADER_LIMIT:
        raise ValueError('replay metadata exceeds admission')
    prefix = _PREFIX.pack(_MAGIC, len(header))
    descriptor, temporary = tempfile.mkstemp(prefix='.replay-', dir=path.parent)
    try:
        try:
            stream = os.fdopen(descriptor, 'wb')
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            digest = hashlib.sha256(prefix+header)
            stream.write(prefix)
            stream.write(header)
            for offset in range(0, len(checkpoint.token_ids), 4096):
                tokens = checkpoint.token_ids[offset:offset+4096]
                data = struct.pack('<'+'I'*len(tokens), *tokens)
                digest.update(data)
                stream.write(data)
            stream.write(digest.digest())
            stream.flush()
            os.fsync(stream.fileno())
        _path(path)
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


def load_replay_checkpoint(source, *, max_tokens):
    """Read only bounded metadata and uint32 tokens; no pickle or executable data."""
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('positive replay token budget required')
    path = _path(source)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0)
                         | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('regular replay checkpoint source required')
        if info.st_size > _PREFIX.size+_HEADER_LIMIT+max_tokens*4+32:
            raise MemoryError('replay checkpoint exceeds admission')
        stream = os.fdopen(descriptor, 'rb')
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        def read_exact(size):
            data = stream.read(size)
            if len(data) != size:
                raise ValueError('truncated replay checkpoint')
            return data

        prefix = read_exact(_PREFIX.size)
        magic, size = _PREFIX.unpack(prefix)
        if magic != _MAGIC or not 0 < size <= _HEADER_LIMIT:
            raise ValueError('invalid replay checkpoint prefix')
        raw_header = read_exact(size)
        try:
            header = json.loads(raw_header.decode('ascii'), object_pairs_hook=_pairs,
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError('invalid replay checkpoint metadata') from error
        if (type(header) is not dict or set(header) != {
                'version', 'identity', 'vocabulary', 'context', 'count'} or
                type(header['version']) is not int or header['version'] != 1 or
                type(header['count']) is not int or not 0 <= header['count'] <= max_tokens):
            raise ValueError('invalid replay checkpoint schema')
        ReplayCheckpoint(header['identity'], header['vocabulary'], header['context'],
                         ()).validate(max_tokens=max_tokens)
        count = header['count']
        if count > header['context'] or info.st_size != _PREFIX.size+size+count*4+32:
            raise ValueError('replay checkpoint extent mismatch')
        digest = hashlib.sha256(prefix+raw_header)
        tokens = []
        for offset in range(0, count, 4096):
            data = read_exact(min(4096, count-offset)*4)
            digest.update(data)
            tokens.extend(token[0] for token in struct.iter_unpack('<I', data))
        checksum = read_exact(32)
        trailing = stream.read(1)
        if not hmac.compare_digest(digest.digest(), checksum) or trailing:
            raise ValueError('replay checkpoint checksum or trailing data mismatch')
        return ReplayCheckpoint(header['identity'], header['vocabulary'], header['context'],
                                tuple(tokens)).validate(max_tokens=max_tokens)
