"""Bounded disk-backed weights for explicitly routed CPU expert workloads."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import mmap
from pathlib import Path
import stat
import threading
import time
from typing import Iterable, Iterator

from .expert_kernels import ExpertKernelRegistry


DEFAULT_EXPERT_KERNELS = ExpertKernelRegistry()


@dataclass(frozen=True)
class ExpertRegion:
    """An immutable byte range containing one backend-defined expert payload."""

    path: str
    offset: int
    size: int
    sha256: str = ""
    storage: str = ""


@dataclass(frozen=True)
class ResolvedExpertRegion:
    """Validated local range that a trusted execution backend may reopen."""

    path: str
    offset: int
    size: int
    sha256: str = ""


@dataclass
class _Entry:
    payload: bytearray | memoryview
    leases: int = 0
    prefetched: bool = False


class DiskExpertStore:
    """LRU payload cache with leases and bounded, non-evicting read-ahead.

    The byte limit covers resident payloads, including in-flight reads. Model
    activations, Python overhead and the OS file cache are outside that limit.
    Files must remain immutable while the store is open. Consumers must not
    retain views after their lease ends.
    """

    def __init__(
        self,
        root: str | Path,
        regions: dict[str, ExpertRegion],
        *,
        capacity_bytes: int,
        io_mode: str = "buffered",
        storage_roots: dict[str, str | Path] | None = None,
        prefetch_workers: int = 1,
        prefetch_depth: int = 2,
    ) -> None:
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("expert root must be a directory")
        self.storage_roots = {"": self.root}
        for name, value in (storage_roots or {}).items():
            if not name or "/" in name or "\\" in name:
                raise ValueError("storage root names must be simple non-empty identifiers")
            resolved = Path(value).resolve(strict=True)
            if not resolved.is_dir():
                raise ValueError("each expert storage root must be a directory")
            self.storage_roots[name] = resolved
        self.capacity_bytes = capacity_bytes
        if io_mode not in {"buffered", "mmap", "direct", "uring"}:
            raise ValueError("io_mode must be buffered, mmap, direct or uring")
        self.io_mode = io_mode
        if (type(prefetch_workers) is not int or type(prefetch_depth) is not int or
                prefetch_workers <= 0 or prefetch_depth < prefetch_workers):
            raise ValueError("prefetch_depth must be at least the positive worker count")
        self.prefetch_workers = prefetch_workers
        self.prefetch_depth = prefetch_depth
        self.regions = dict(regions)
        self._file_signatures: dict[Path, tuple[int, int]] = {}
        self._verified: set[str] = set()
        self._maps: dict[Path, mmap.mmap] = {}
        for key, region in self.regions.items():
            if (type(key) is not str or not key or type(region) is not ExpertRegion or
                    type(region.storage) is not str or
                    type(region.offset) is not int or type(region.size) is not int or
                    region.offset < 0 or region.size <= 0 or region.size > self.capacity_bytes):
                raise ValueError("invalid expert region or region exceeds capacity")
            if region.sha256 and (len(region.sha256) != 64 or any(c not in "0123456789abcdef" for c in region.sha256)):
                raise ValueError("sha256 must be a lowercase hexadecimal digest")
            self._path(region)
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=self.prefetch_workers, thread_name_prefix="expert-read")
        self._pending: dict[str, Future] = {}
        self._reservations: dict[str, int] = {}
        self._pinned: set[str] = set()
        self._closed = False
        self._stats = dict(hits=0, misses=0, bytes_read=0, mapped_bytes=0, evictions=0,
                           prefetch_hits=0, prefetch_errors=0, prefetch_dropped=0,
                           demand_waits=0, read_seconds=0.0)

    def _path(self, region: ExpertRegion) -> Path:
        if (type(region.path) is not str or not region.path or
                Path(region.path).is_absolute()):
            raise ValueError("expert file path must be a relative non-empty string")
        storage_root = self.storage_roots.get(region.storage)
        if storage_root is None:
            raise ValueError(f"unknown expert storage root: {region.storage}")
        candidate = storage_root / region.path
        probe = candidate.absolute()
        while True:
            try:
                info = probe.lstat()
            except FileNotFoundError:
                pass
            else:
                if (stat.S_ISLNK(info.st_mode) or
                        getattr(info, 'st_file_attributes', 0) & 0x400):
                    raise ValueError("expert file paths cannot traverse links or reparse points")
            if probe == storage_root or probe.parent == probe:
                break
            probe = probe.parent
        path = candidate.resolve(strict=True)
        if not path.is_relative_to(storage_root) or not path.is_file():
            raise ValueError("expert file must remain inside its declared storage root")
        file_stat = path.stat()
        signature = (file_stat.st_size, file_stat.st_mtime_ns)
        original = self._file_signatures.setdefault(path, signature)
        if signature != original:
            raise ValueError("expert file changed while the store was open")
        if region.offset + region.size > file_stat.st_size:
            raise ValueError("expert range exceeds file size")
        return path

    def describe_region(self, key: str) -> ResolvedExpertRegion:
        """Return a containment-checked descriptor without loading its bytes."""

        with self._lock:
            self._check_open()
            try:
                region = self.regions[key]
            except KeyError:
                raise KeyError(key) from None
            path = self._path(region)
            return ResolvedExpertRegion(
                path=str(path),
                offset=region.offset,
                size=region.size,
                sha256=region.sha256,
            )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("expert store is closed")

    def _used(self) -> int:
        return sum(len(entry.payload) for entry in self._entries.values())

    def _make_room(self, size: int, *, prefetch: bool) -> bool:
        needed = self._used() + sum(self._reservations.values()) + size - self.capacity_bytes
        if needed > 0:
            if prefetch:
                return False
            candidates = [
                name for name, entry in self._entries.items()
                if not entry.leases and name not in self._pinned
            ]
            if sum(len(self._entries[name].payload) for name in candidates) < needed:
                raise MemoryError("leased or pinned experts occupy the cache; release or unpin them before loading another expert")
            for name in candidates:
                entry = self._entries.pop(name)
                needed -= len(entry.payload)
                self._release(entry)
                self._stats["evictions"] += 1
                if needed <= 0:
                    break
        return True

    def _read_payload(self, key: str, cancelled=None) -> bytearray | memoryview:
        region = self.regions[key]
        path = self._path(region)
        started = time.perf_counter()
        bytes_read = 0
        payload: bytearray | memoryview | None = None
        try:
            if self.io_mode == "mmap":
                with self._lock:
                    mapping = self._maps.get(path)
                    if mapping is None:
                        with path.open("rb") as source:
                            mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
                        self._maps[path] = mapping
                payload: bytearray | memoryview = memoryview(mapping)[region.offset:region.offset + region.size].toreadonly()
                bytes_read = region.size
            elif self.io_mode in {"direct", "uring"}:
                if self.io_mode == "uring":
                    from .uring_io import read_region
                else:
                    from .expert_io import read_region

                payload = bytearray(read_region(path, region.offset, region.size))
                bytes_read = len(payload)
                if bytes_read != region.size:
                    raise OSError("expert file ended before the declared region")
            else:
                payload = bytearray(region.size)
                with path.open("rb", buffering=0) as source:
                    source.seek(region.offset)
                    with memoryview(payload) as view:
                        offset = 0
                        while offset < region.size:
                            if cancelled is not None and cancelled():
                                raise InterruptedError("expert payload read cancelled")
                            count = source.readinto(view[offset:])
                            if not count:
                                raise OSError("expert file ended before the declared region")
                            offset += count
                            bytes_read += count
            if cancelled is not None and cancelled():
                raise InterruptedError("expert payload read cancelled")
            if region.sha256 and key not in self._verified:
                if hashlib.sha256(payload).hexdigest() != region.sha256:
                    raise ValueError("expert checksum mismatch")
                with self._lock:
                    self._verified.add(key)
        except BaseException as primary:
            # A failed checksum, short read, or cancellation must not leave a
            # partially materialized payload (or mmap view) live until GC.
            try:
                if isinstance(payload, memoryview):
                    payload.release()
            except BaseException as cleanup:
                primary.add_note("failed expert payload cleanup also failed: " + repr(cleanup))
            payload = None
            raise
        finally:
            with self._lock:
                counter = "mapped_bytes" if self.io_mode == "mmap" else "bytes_read"
                self._stats[counter] += bytes_read
                self._stats["read_seconds"] += time.perf_counter() - started
        assert payload is not None
        return payload

    def _load(self, key: str, *, prefetch: bool = False, cancelled=None) -> _Entry | None:
        region = self.regions[key]
        if not self._make_room(region.size, prefetch=prefetch):
            return None
        payload = self._read_payload(key, cancelled)
        try:
            entry = _Entry(payload, prefetched=prefetch)
            self._entries[key] = entry
            return entry
        except BaseException as primary:
            try:
                self._release_payload(payload)
            except BaseException as cleanup:
                primary.add_note("failed expert cache admission cleanup also failed: " + repr(cleanup))
            raise

    @contextmanager
    def acquire(self, key: str, *, cancelled=None) -> Iterator[memoryview]:
        if cancelled is not None and not callable(cancelled):
            raise TypeError("expert lease cancellation must be callable")
        pending = None
        with self._lock:
            self._check_open()
            if key not in self.regions:
                raise KeyError(key)
            pending = self._pending.get(key)
            if pending is not None and not pending.done():
                self._stats["demand_waits"] += 1
        if pending is not None:
            if cancelled is not None and cancelled():
                raise InterruptedError("expert lease acquisition cancelled")
            pending.result()
        if cancelled is not None and cancelled():
            raise InterruptedError("expert lease acquisition cancelled")
        with self._lock:
            self._check_open()
            if cancelled is not None and cancelled():
                raise InterruptedError("expert lease acquisition cancelled")
            entry = self._entries.get(key)
            if entry is None:
                self._stats["misses"] += 1
                entry = self._load(key, cancelled=cancelled)
            else:
                self._stats["hits"] += 1
                if entry.prefetched:
                    self._stats["prefetch_hits"] += 1
                    entry.prefetched = False
            entry.leases += 1
            self._entries.move_to_end(key)
            view = memoryview(entry.payload).toreadonly()
        try:
            yield view
        finally:
            view.release()
            with self._lock:
                entry.leases -= 1

    def prefetch(self, key: str, *, headroom_bytes: int = 0) -> Future | None:
        """Queue a bounded advisory read; never evict a demand-loaded expert."""
        with self._lock:
            self._check_open()
            if headroom_bytes < 0:
                raise ValueError("headroom_bytes cannot be negative")
            if key not in self.regions:
                raise KeyError(key)
            if key in self._entries:
                return None
            size = self.regions[key].size
            if self._used() + sum(self._reservations.values()) + size + headroom_bytes > self.capacity_bytes:
                self._stats["prefetch_dropped"] += 1
                skipped = Future()
                skipped.set_result(False)
                return skipped
            self._pending = {name: job for name, job in self._pending.items() if not job.done()}
            if key in self._pending:
                return self._pending[key]
            if len(self._pending) >= self.prefetch_depth:
                return None
            self._reservations[key] = size
            try:
                job = self._pool.submit(self._prefetch, key)
            except Exception:
                self._reservations.pop(key, None)
                raise
            self._pending[key] = job
            return job

    def prefetch_many(self, keys: Iterable[str], *, headroom_bytes: int = 0) -> list[Future]:
        """Queue distinct advisory reads up to the store's bounded queue."""
        jobs: list[Future] = []
        for key in dict.fromkeys(keys):
            job = self.prefetch(key, headroom_bytes=headroom_bytes)
            if job is not None:
                jobs.append(job)
        return jobs

    def _prefetch(self, key: str) -> bool:
        payload: bytearray | memoryview | None = None
        try:
            payload = self._read_payload(key)
        except (OSError, ValueError):
            with self._lock:
                self._reservations.pop(key, None)
                self._stats["prefetch_errors"] += 1
            return False
        except BaseException:
            # Memory/admission failures must not strand the reservation while
            # the worker future carries the original failure to the caller.
            with self._lock:
                self._reservations.pop(key, None)
                self._stats["prefetch_errors"] += 1
            raise
        try:
            with self._lock:
                self._reservations.pop(key, None)
                if self._closed or key in self._entries:
                    self._release_payload(payload)
                    return False
                if not self._make_room(len(payload), prefetch=True):
                    self._stats["prefetch_dropped"] += 1
                    self._release_payload(payload)
                    return False
                self._entries[key] = _Entry(payload, prefetched=True)
                # Ownership has transferred to the cache entry.
                payload = None
                return True
        except BaseException as primary:
            if payload is not None:
                try:
                    self._release_payload(payload)
                except BaseException as cleanup:
                    primary.add_note("failed expert prefetch cleanup also failed: " + repr(cleanup))
            raise

    def pin(self, keys: Iterable[str], *, load: bool = False) -> None:
        """Protect selected experts from eviction, optionally warming them first."""
        selected = tuple(dict.fromkeys(keys))
        with self._lock:
            self._check_open()
            unknown = [key for key in selected if key not in self.regions]
            if unknown:
                raise KeyError(unknown[0])
            total = sum(self.regions[key].size for key in selected)
            if total > self.capacity_bytes:
                raise MemoryError("pinned experts exceed cache capacity")
        if load:
            for key in selected:
                with self.acquire(key):
                    pass
        with self._lock:
            self._pinned.update(selected)

    def unpin(self, keys: Iterable[str]) -> None:
        with self._lock:
            for key in keys:
                self._pinned.discard(key)

    def evict(self, key: str) -> bool:
        """Evict one unleased, unpinned entry."""
        with self._lock:
            self._check_open()
            entry = self._entries.get(key)
            if entry is None:
                return False
            if entry.leases or key in self._pinned:
                return False
            self._entries.pop(key)
            self._release(entry)
            self._stats["evictions"] += 1
            return True

    @staticmethod
    def _release(entry: _Entry) -> None:
        DiskExpertStore._release_payload(entry.payload)

    @staticmethod
    def _release_payload(payload: bytearray | memoryview) -> None:
        if isinstance(payload, memoryview):
            payload.release()

    def status(self) -> dict:
        with self._lock:
            requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / requests if requests else 0.0
            read_mib_s = (
                self._stats["bytes_read"] / (1024 * 1024) / self._stats["read_seconds"]
                if self._stats["read_seconds"] > 0 else 0.0
            )
            return {**self._stats, "hit_rate": hit_rate, "read_mib_s": read_mib_s,
                    "capacity_bytes": self.capacity_bytes,
                    "resident_bytes": self._used(), "resident_experts": len(self._entries),
                    "checksums_verified": len(self._verified),
                    "active_leases": sum(e.leases for e in self._entries.values()),
                    "resident_keys": list(self._entries), "pinned_keys": sorted(self._pinned),
                    "pending_reads": sum(not job.done() for job in self._pending.values()),
                    "reserved_prefetch_bytes": sum(self._reservations.values()),
                    "mapped_files": len(self._maps),
                    "storage_roots": sorted(self.storage_roots),
                    "prefetch_workers": self.prefetch_workers,
                    "prefetch_depth": self.prefetch_depth,
                    "io_mode": self.io_mode, "closed": self._closed}

    def close(self) -> None:
        with self._lock:
            if any(entry.leases for entry in self._entries.values()):
                raise RuntimeError("release all expert leases before closing")
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            for entry in self._entries.values():
                self._release(entry)
            self._entries.clear()
            self._reservations.clear()
            self._pinned.clear()
            for mapping in self._maps.values():
                mapping.close()
            self._maps.clear()

    def __enter__(self) -> DiskExpertStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def routed_swiglu(store: DiskExpertStore, inputs, routes, *, hidden_size: int,
                  intermediate_size: int, dtype: str = "float32",
                  kernels: ExpertKernelRegistry | None = None,
                  preferred_backend: str | None = None, allow_fallback: bool = True):
    """Execute one CPU MoE FFN using unchanged caller-supplied routing weights.

    Each region stores row-major gate [I,H], up [I,H], down [H,I], in that
    order. Only little-endian FP32/FP16 payloads are supported. This is an FFN
    primitive, not a tokenizer, attention implementation or model decoder.
    """
    routes = list(routes)
    if routes and routes[0][0] in store.regions:
        store.prefetch_many(
            (key for key, _ in routes[1:]),
            headroom_bytes=store.regions[routes[0][0]].size,
        )
    registry = kernels or DEFAULT_EXPERT_KERNELS
    backend = registry.resolve(dtype=dtype, preferred=preferred_backend, allow_fallback=allow_fallback)
    return backend.execute(
        store,
        inputs,
        routes,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
    )


def routed_swiglu_batch(store: DiskExpertStore, inputs, route_batch, *, hidden_size: int,
                        intermediate_size: int, dtype: str = "float32",
                        kernels: ExpertKernelRegistry | None = None,
                        preferred_backend: str | None = None, allow_fallback: bool = True):
    """Execute a token batch expert-by-expert while preserving every route.

    Grouping avoids loading the same expert once per token. It does not merge,
    normalize, reorder or otherwise alter caller-supplied routing weights.
    """
    routes = [list(items) for items in route_batch]
    expert_ids = list(dict.fromkeys(key for items in routes for key, _ in items))
    if expert_ids and expert_ids[0] in store.regions:
        store.prefetch_many(
            expert_ids[1:],
            headroom_bytes=store.regions[expert_ids[0]].size,
        )
    registry = kernels or DEFAULT_EXPERT_KERNELS
    backend = registry.resolve(dtype=dtype, batch=True, preferred=preferred_backend,
                               allow_fallback=allow_fallback)
    return backend.execute_batch(
        store,
        inputs,
        routes,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
    )
