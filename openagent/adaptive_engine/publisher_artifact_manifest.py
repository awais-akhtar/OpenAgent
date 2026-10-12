"""Local publisher provenance hashes; no inference or platform certification.

APIs: create_manifest(root, *, budgets, publisher, revision, tokenizer_files,
library_files, config_file='config.json', index_file=None, shard_files=None,
cancel_event=None), verify_manifest(root, manifest, *, budgets, config_file,
index_file, shard_files, tokenizer_files, library_files, cancel_event=None),
save_manifest(manifest, destination, *, budgets, cancel_event=None).
Index references select shards when shard_files is omitted. Without an index,
explicit shard_files are required. Paths are canonical root-relative POSIX paths.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import uuid

SCHEMA = 'openagent.publisher-artifacts.v1'
KINDS = {'config', 'index', 'weights', 'tokenizer', 'dependency'}


class ManifestCancelled(RuntimeError):
    """Cooperative cancellation leaves an existing saved manifest untouched."""


@dataclass(frozen=True)
class ManifestBudgets:
    max_files: int
    max_file_bytes: int
    max_total_bytes: int
    max_json_bytes: int
    max_index_entries: int
    chunk_bytes: int = 1024 * 1024

    def __post_init__(self):
        for value in vars(self).values():
            if type(value) is not int or value <= 0:
                raise ValueError('manifest budgets must be positive integers')
        if self.chunk_bytes > 8 * 1024 * 1024:
            raise ValueError('hash chunks must not exceed 8 MiB')


def _cancel(event):
    if event is not None and event.is_set():
        raise ManifestCancelled('publisher manifest operation cancelled')


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _json(raw):
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    def floating(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('nonfinite JSON number')
        return parsed
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject, parse_float=floating)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('invalid or excessively nested JSON') from exc


def _relative(value):
    if (type(value) is not str or not value or '\\' in value or ':' in value or
            '\0' in value or value.startswith('/') or
            any(part in ('', '.', '..') or part.endswith(('.', ' ')) for part in value.split('/'))):
        raise ValueError('canonical local relative path required')
    if PurePosixPath(value).as_posix() != value:
        raise ValueError('noncanonical relative path')
    return value


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _safe_path(path):
    # Reject symlinks and Windows junction/reparse components, including ancestors.
    path = Path(os.path.abspath(path))
    for component in reversed((path, *path.parents)):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('symlink/reparse paths are not publisher artifacts')
    return path


@contextmanager
def _open(path):
    path = _safe_path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('artifact must be a regular file')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as source:
        stamp = _stamp(os.fstat(source.fileno()))
        if stamp != _stamp(before):
            raise ValueError('artifact changed while opening')
        yield source, stamp
        if (_stamp(os.fstat(source.fileno())) != stamp or
                _stamp(_safe_path(path).lstat()) != stamp):
            raise ValueError('artifact mutated or replaced while reading: ' + str(path))


def _hash(root, name, kind, budgets, event):
    _cancel(event)
    digest = hashlib.sha256()
    raw = bytearray() if kind in ('config', 'index') else None
    with _open(root / name) as (source, stamp):
        if stamp[3] <= 0 or stamp[3] > budgets.max_file_bytes:
            raise ValueError('artifact file byte budget exceeded or empty: ' + name)
        if raw is not None and stamp[3] > budgets.max_json_bytes:
            raise ValueError('structured JSON byte budget exceeded')
        count = 0
        while True:
            _cancel(event)
            block = source.read(min(budgets.chunk_bytes, stamp[3] - count + 1))
            if not block:
                break
            count += len(block)
            if count > stamp[3]:
                raise ValueError('artifact grew while hashing')
            digest.update(block)
            if raw is not None:
                raw.extend(block)
        if count != stamp[3]:
            raise ValueError('artifact truncated while hashing')
    _cancel(event)
    document = _json(raw) if raw is not None else None
    if raw is not None and type(document) is not dict:
        raise ValueError('checkpoint config/index must be JSON objects')
    return dict(kind=kind, path=name, bytes=count, sha256=digest.hexdigest()), document, stamp


def _index(document, budgets):
    if set(document) - {'metadata', 'weight_map'}:
        raise ValueError('unknown checkpoint index fields')
    mapping = document.get('weight_map')
    if type(mapping) is not dict or not 0 < len(mapping) <= budgets.max_index_entries:
        raise ValueError('nonempty bounded index weight_map required')
    if any(type(name) is not str or not name or '\0' in name for name in mapping):
        raise ValueError('invalid index tensor name')
    shards = {_relative(value) for value in mapping.values()}
    metadata = document.get('metadata', {})
    if type(metadata) is not dict or set(metadata) - {'total_size'}:
        raise ValueError('unsupported index metadata fields')
    if 'total_size' in metadata and (type(metadata['total_size']) is not int or metadata['total_size'] <= 0):
        raise ValueError('index total_size must be a positive integer')
    return shards, len(mapping)


def create_manifest(root, *, budgets, publisher, revision, tokenizer_files,
                    library_files, config_file='config.json', index_file=None,
                    shard_files=None, cancel_event=None):
    """Hash declared complete local files; budgets are required, never inferred."""
    _cancel(cancel_event)
    if not isinstance(budgets, ManifestBudgets):
        raise TypeError('ManifestBudgets required')
    if any(type(value) is not str or not value.strip() or '\0' in value
           for value in (publisher, revision)):
        raise ValueError('explicit nonempty publisher and revision required')
    root = _safe_path(root)
    if not root.is_dir():
        raise ValueError('local checkpoint root must be a directory')
    rows, stamps, names, used = [], {}, set(), 0

    def add(name, kind):
        nonlocal used
        _cancel(cancel_event)
        name = _relative(name)
        key = os.path.normcase(name)
        if key in names or len(names) >= budgets.max_files:
            raise ValueError('duplicate artifact or file count budget exceeded')
        path = _safe_path(root / name)
        info = path.lstat()
        if used + info.st_size > budgets.max_total_bytes:
            raise ValueError('total artifact byte budget exceeded')
        row, document, stamp = _hash(root, name, kind, budgets, cancel_event)
        used += row['bytes']
        if used > budgets.max_total_bytes:
            raise ValueError('total artifact byte budget exceeded')
        rows.append(row)
        stamps[name] = stamp
        names.add(key)
        return document

    add(config_file, 'config')
    index_entries = 0
    if index_file is not None:
        index_shards, index_entries = _index(add(index_file, 'index'), budgets)
        if shard_files is None:
            shard_files = sorted(index_shards)
        elif type(shard_files) not in (list, tuple) or set(shard_files) != index_shards:
            raise ValueError('declared shards differ from structured index references')
    for values, kind in ((shard_files, 'weights'), (tokenizer_files, 'tokenizer'),
                         (library_files, 'dependency')):
        if type(values) not in (list, tuple) or not values or len(values) > budgets.max_files:
            raise ValueError('explicit bounded nonempty artifact lists required: ' + kind)
        for name in values:
            add(name, kind)
    for name, stamp in stamps.items():
        _cancel(cancel_event)
        if _stamp(_safe_path(root / name).lstat()) != stamp:
            raise ValueError('artifact changed during inventory hashing: ' + name)
    manifest = dict(schema=SCHEMA, publisher=publisher, revision=revision, certification=False,
                    files=sorted(rows, key=lambda row: row['path']), file_count=len(rows),
                    total_bytes=used, index_entries=index_entries)
    _validate(manifest, budgets)
    _encode(manifest, budgets, cancel_event)
    return manifest


def _validate(manifest, budgets):
    if type(manifest) is not dict or set(manifest) != {
            'schema', 'publisher', 'revision', 'certification', 'files',
            'file_count', 'total_bytes', 'index_entries'}:
        raise ValueError('strict publisher manifest fields required')
    if manifest['schema'] != SCHEMA or manifest['certification'] is not False:
        raise ValueError('unsupported manifest schema or certification claim')
    for key in ('publisher', 'revision'):
        if type(manifest[key]) is not str or not manifest[key].strip() or '\0' in manifest[key]:
            raise ValueError('invalid publisher provenance metadata')
    rows = manifest['files']
    if type(rows) is not list or not 0 < len(rows) <= budgets.max_files:
        raise ValueError('manifest file count budget exceeded')
    counts = {kind: 0 for kind in KINDS}
    names, total = set(), 0
    for row in rows:
        if type(row) is not dict or set(row) != {'kind', 'path', 'bytes', 'sha256'}:
            raise ValueError('strict artifact row fields required')
        if type(row['kind']) is not str or row['kind'] not in KINDS:
            raise ValueError('unknown artifact kind')
        name = _relative(row['path'])
        key = os.path.normcase(name)
        if key in names:
            raise ValueError('duplicate manifest artifact path')
        names.add(key)
        counts[row['kind']] += 1
        if type(row['bytes']) is not int or not 0 < row['bytes'] <= budgets.max_file_bytes:
            raise ValueError('invalid artifact byte count')
        if type(row['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('invalid artifact SHA256')
        total += row['bytes']
    if (counts['config'] != 1 or counts['index'] > 1 or
            any(not counts[kind] for kind in ('weights', 'tokenizer', 'dependency'))):
        raise ValueError('complete config/weights/tokenizer/library inventory required')
    if (type(manifest['file_count']) is not int or manifest['file_count'] != len(rows) or
            type(manifest['total_bytes']) is not int or manifest['total_bytes'] != total or
            total > budgets.max_total_bytes or type(manifest['index_entries']) is not int or
            not 0 <= manifest['index_entries'] <= budgets.max_index_entries or
            bool(counts['index']) != bool(manifest['index_entries'])):
        raise ValueError('manifest totals or index counts differ')
    if rows != sorted(rows, key=lambda row: row['path']):
        raise ValueError('manifest inventory must be canonically ordered')


def _encode(manifest, budgets, event):
    encoded = bytearray()
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    for piece in encoder.iterencode(manifest):
        _cancel(event)
        raw = piece.encode('utf-8')
        if len(encoded) + len(raw) > budgets.max_json_bytes:
            raise ValueError('manifest JSON byte budget exceeded')
        encoded.extend(raw)
    return encoded


def verify_manifest(root, manifest, *, budgets, config_file, index_file,
                    shard_files, tokenizer_files, library_files, cancel_event=None):
    """Verify exact caller-requested inventory and bytes; mismatch raises ValueError.

    All inventory arguments are required. index_file=None explicitly requests
    an unindexed checkpoint. A manifest cannot nominate its own trusted subset.
    """
    _cancel(cancel_event)
    if not isinstance(budgets, ManifestBudgets):
        raise TypeError('ManifestBudgets required')
    if isinstance(manifest, (str, Path)):
        with _open(manifest) as (source, stamp):
            if stamp[3] > budgets.max_json_bytes:
                raise ValueError('manifest JSON byte budget exceeded')
            raw = bytearray()
            while True:
                _cancel(cancel_event)
                block = source.read(min(budgets.chunk_bytes, budgets.max_json_bytes - len(raw) + 1))
                if not block:
                    break
                raw.extend(block)
                if len(raw) > budgets.max_json_bytes:
                    raise ValueError('manifest JSON byte budget exceeded')
        manifest = _json(raw)
    _validate(manifest, budgets)
    _encode(manifest, budgets, cancel_event)
    requested = [('config', _relative(config_file))]
    if index_file is not None:
        requested.append(('index', _relative(index_file)))
    for values, kind in ((shard_files, 'weights'), (tokenizer_files, 'tokenizer'),
                         (library_files, 'dependency')):
        if type(values) not in (list, tuple) or not values or len(values) > budgets.max_files:
            raise ValueError('complete explicit requested inventory required: ' + kind)
        requested.extend((kind, _relative(name)) for name in values)
    if len(requested) > budgets.max_files or len({os.path.normcase(name) for _, name in requested}) != len(requested):
        raise ValueError('requested inventory duplicate or file count budget exceeded')
    declared = {(row['kind'], row['path']) for row in manifest['files']}
    if set(requested) != declared:
        raise ValueError('manifest file set differs from exact requested checkpoint/library inventory')
    actual = create_manifest(root, budgets=budgets, publisher=manifest['publisher'],
        revision=manifest['revision'], config_file=config_file, index_file=index_file,
        shard_files=shard_files, tokenizer_files=tokenizer_files,
        library_files=library_files, cancel_event=cancel_event)
    if actual != manifest:
        raise ValueError('publisher artifact digest/size/index inventory mismatch')
    _cancel(cancel_event)
    return dict(verified=True, certification=False, schema=SCHEMA,
                file_count=actual['file_count'], total_bytes=actual['total_bytes'],
                publisher=actual['publisher'], revision=actual['revision'])


def save_manifest(manifest, destination, *, budgets, cancel_event=None):
    """Atomically save strict metadata; saving does not verify artifact contents."""
    _cancel(cancel_event)
    if not isinstance(budgets, ManifestBudgets):
        raise TypeError('ManifestBudgets required')
    _validate(manifest, budgets)
    encoded = _encode(manifest, budgets, cancel_event)
    destination = Path(os.path.abspath(destination))
    _safe_path(destination.parent)
    if destination.exists() or destination.is_symlink():
        _safe_path(destination)
        if not destination.is_file():
            raise ValueError('manifest destination must be a regular file')
    temporary = destination.with_name(destination.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as output:
            for offset in range(0, len(encoded), budgets.chunk_bytes):
                _cancel(cancel_event)
                output.write(encoded[offset:offset + budgets.chunk_bytes])
            output.flush()
            os.fsync(output.fileno())
        _cancel(cancel_event)
        _safe_path(destination.parent)
        os.replace(temporary, destination)
        if os.name != 'nt':
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(destination), bytes=len(encoded),
                sha256=hashlib.sha256(encoded).hexdigest(), certification=False)
