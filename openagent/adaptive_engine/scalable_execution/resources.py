"""Admission and immutable packed-region leases; model math is backend-owned."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import mmap
import os
from pathlib import Path
import threading
import sys
from ..session_recovery import _path as _safe_path
from ..native_library_identity import _stamp as _file_stamp
from ..execution_service import ExecutionCancelled


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class Cancelled(ExecutionCancelled):
    """Cancelled before the durable commit point."""


def check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise Cancelled("execution cancelled")


class Budget:
    """Nonblocking, thread-safe reservations for one memory pool."""

    def __init__(self, capacity_bytes):
        self.capacity_bytes = positive(capacity_bytes, "capacity_bytes")
        self.used_bytes = 0
        self._lock = threading.Lock()

    @contextmanager
    def reserve(self, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError("reservation must be a nonnegative integer")
        with self._lock:
            if amount > self.capacity_bytes - self.used_bytes:
                raise MemoryError("memory pool admission rejected")
            self.used_bytes += amount
        try:
            yield
        finally:
            with self._lock:
                self.used_bytes -= amount


class LeasePool:
    """Keep escaped Python buffer exports charged until they can be released."""

    def __init__(self, budget):
        self.budget = budget
        self._pending = []
        self._lock = threading.RLock()
        self._active = 0
        self._closed = False

    @staticmethod
    def _release(buffer):
        if isinstance(buffer, mmap.mmap):
            buffer.close()
        else:
            buffer.clear()

    def reap(self):
        with self._lock:
            retained = []
            failure = None
            for buffer, reservation in self._pending:
                try:
                    if buffer is not None:self._release(buffer)
                except BufferError:
                    retained.append((buffer, reservation))
                except BaseException as error:
                    retained.append((buffer, reservation))
                    if failure is None:
                        failure = error
                else:
                    try:reservation.__exit__(None, None, None)
                    except BaseException as error:
                        retained.append((None,reservation))
                        if failure is None:failure=error
            self._pending = retained
            if failure is not None:
                raise failure

    @contextmanager
    def acquire(self, amount, factory):
        self.reap()
        with self._lock:
            if self._closed:
                raise RuntimeError('lease pool is closed')
            self._active += 1
        reservation = self.budget.reserve(amount)
        try:
            reservation.__enter__()
        except BaseException:
            with self._lock:
                self._active -= 1
            raise
        buffer = None
        try:
            buffer = factory()
            yield buffer
        finally:
            primary = sys.exc_info()[1]
            try:
                if buffer is not None:
                    try:
                        self._release(buffer)
                    except BaseException as failure:
                        with self._lock:
                            self._pending.append((buffer, reservation))
                        if primary is not None:
                            primary.add_note('buffer disposal requires retry; reservation retained: '+str(failure))
                        else:
                            raise RuntimeError('buffer disposal requires retry; reservation retained') from failure
                    else:
                        buffer=None
                if buffer is None:
                    try:reservation.__exit__(None, None, None)
                    except BaseException as failure:
                        with self._lock:self._pending.append((None,reservation))
                        if primary is None:raise
                        primary.add_note('lease reservation release requires retry: '+repr(failure))
            finally:
                with self._lock:
                    self._active -= 1

    def close(self):
        self.reap()
        with self._lock:
            if self._active:
                raise RuntimeError('cannot close an active lease pool')
            if self._pending:
                raise RuntimeError("release escaped buffer views before closing")
            self._closed = True


@dataclass(frozen=True)
class CacheLayout:
    """Explicit serialized per-step payload, not an inferred attention layout.

    Recurrent payloads are per-step state snapshots, not a rolling KV tensor.
    Custom adapters must include their ordering/version in layout_identity.
    """

    kind: str
    bytes_per_token: int
    layout_identity: str
    variable_size: bool = False

    def __post_init__(self):
        if self.kind not in {"dense-kv", "latent-kv", "recurrent-state", "custom"}:
            raise ValueError("unsupported cache layout strategy")
        positive(self.bytes_per_token, "bytes_per_token")
        if type(self.variable_size) is not bool:
            raise ValueError("variable_size must be boolean")
        if not self.layout_identity:
            raise ValueError("cache serialization identity required")

    @classmethod
    def dense_kv(cls, kv_heads, head_dim, scalar_bytes, *, layout_identity):
        for name, value in (("kv_heads", kv_heads), ("head_dim", head_dim),
                            ("scalar_bytes", scalar_bytes)):
            positive(value, name)
        return cls("dense-kv", 2 * kv_heads * head_dim * scalar_bytes, layout_identity)


@dataclass(frozen=True)
class Geometry:
    """Exact backend cache layout and supported dispatch geometry."""

    identity: str
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    cache_scalar_bytes: int
    max_context_tokens: int
    max_batch_tokens: int
    packed_formats: frozenset[str]
    cache_layouts: tuple[CacheLayout, ...]

    def __post_init__(self):
        if not self.identity or not self.packed_formats:
            raise ValueError("backend identity and supported formats required")
        for name in ("hidden_size", "intermediate_size", "layers", "attention_heads",
                     "kv_heads", "head_dim", "cache_scalar_bytes", "max_context_tokens",
                     "max_batch_tokens"):
            positive(getattr(self, name), name)
        if len(self.cache_layouts) != self.layers or any(
                not isinstance(layout, CacheLayout) for layout in self.cache_layouts):
            raise ValueError("one explicit cache layout per layer is required")
        if self.max_batch_tokens > self.max_context_tokens:
            raise ValueError("batch limit exceeds context limit")

    @property
    def bytes_per_token(self):
        return sum(layout.bytes_per_token for layout in self.cache_layouts)

    @property
    def attention_projection_size(self):
        return self.attention_heads * self.head_dim

    @property
    def variable_cache_size(self):
        return any(layout.variable_size for layout in self.cache_layouts)

    def admit_cache_payload(self, tokens, size):
        maximum = tokens * self.bytes_per_token
        if not 0 < size <= maximum or (not self.variable_cache_size and size != maximum):
            raise ValueError("cache page bytes do not match backend geometry")

    def admit(self, *, tokens, position, packed_format):
        positive(tokens, "tokens")
        if type(position) is not int or position < 0:
            raise ValueError("position must be nonnegative")
        if tokens > self.max_batch_tokens or position + tokens > self.max_context_tokens:
            raise ValueError("backend token geometry exceeded")
        if packed_format not in self.packed_formats:
            raise ValueError("unsupported backend packed format")


@dataclass(frozen=True)
class PackedRegion:
    path: str
    offset: int
    size: int
    sha256: str
    packed_format: str
    storage: str = ""


class PackedStore:
    """Region-sized mmap, charged alignment, SHA-256 on every lease.

    Files must stay immutable. Callbacks must release all derived views before
    returning; retaining a native pointer beyond a lease is unsupported.
    """

    def __init__(self, root, regions, *, mapped_budget, storage_roots=None):
        self.roots = {"": _safe_path(root).resolve(strict=True)}
        for name, path in (storage_roots or {}).items():
            if not name or "/" in name or "\\" in name:
                raise ValueError("invalid storage root name")
            self.roots[name] = _safe_path(path).resolve(strict=True)
        if any(not path.is_dir() for path in self.roots.values()):
            raise ValueError("storage roots must be directories")
        self.regions = dict(regions)
        self.budget = mapped_budget
        self._pool = LeasePool(mapped_budget)
        self._lock = threading.RLock()
        self._active = 0
        self._closed = False
        self._signatures = {}
        for key, region in self.regions.items():
            if not key or type(region.offset) is not int or region.offset < 0:
                raise ValueError("invalid packed region")
            positive(region.size, "region size")
            if len(region.sha256) != 64 or any(c not in "0123456789abcdef" for c in region.sha256):
                raise ValueError("every packed region requires SHA-256")
            if not region.packed_format:
                raise ValueError("packed format required")
            self._path(region)

    @classmethod
    def from_sparse_manifest(cls, root, manifest, **options):
        manifest.validate()
        regions = {f"{layer.id}/{key}": PackedRegion(
            value.path, value.offset, value.size, value.sha256, layer.dtype, value.storage)
            for layer in manifest.layers for key, value in layer.regions.items()}
        return cls(root, regions, **options)

    def _path(self, region):
        root = self.roots[region.storage]
        path = _safe_path(root / region.path).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("packed shard escapes declared root")
        stat = path.stat()
        signature = _file_stamp(stat)
        if self._signatures.setdefault(path, signature) != signature:
            raise ValueError("immutable packed shard changed")
        if region.offset + region.size > stat.st_size:
            raise ValueError("packed region exceeds shard")
        return path

    @contextmanager
    def acquire(self, key, *, cancelled=None):
        check_cancel(cancelled)
        with self._lock:
            if self._closed:
                raise RuntimeError("packed store closed")
            region = self.regions[key]
            path = self._path(region)
            self._active += 1
        start = region.offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        padding = region.offset - start
        mapped_bytes = ((padding + region.size + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        try:
            with path.open("rb") as source:
                stat = os.fstat(source.fileno())
                if _file_stamp(stat) != self._signatures[path]:
                    raise ValueError("packed shard replaced during open")
                with self._pool.acquire(mapped_bytes, lambda: mmap.mmap(
                        source.fileno(), padding + region.size,
                        access=mmap.ACCESS_READ, offset=start)) as mapping:
                    with memoryview(mapping)[padding:padding + region.size] as payload:
                        digest = hashlib.sha256()
                        for offset in range(0, region.size, 1024 * 1024):
                            check_cancel(cancelled)
                            with payload[offset:offset + 1024 * 1024] as chunk:
                                digest.update(chunk)
                        if digest.hexdigest() != region.sha256:
                            raise ValueError("packed region checksum mismatch")
                        check_cancel(cancelled)
                        if _file_stamp(os.fstat(source.fileno()))!=self._signatures[path] or self._path(region)!=path:
                            raise ValueError('packed shard changed during checksum admission')
                        yield payload
                        if _file_stamp(os.fstat(source.fileno()))!=self._signatures[path] or self._path(region)!=path:
                            raise ValueError('packed shard changed during execution')
        finally:
            with self._lock:
                self._active -= 1

    def close(self):
        with self._lock:
            if self._active:
                raise RuntimeError("cannot close packed store with live leases")
            self._pool.close()
            self._closed = True
