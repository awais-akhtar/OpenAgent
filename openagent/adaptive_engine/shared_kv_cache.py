"""Phase 2B: Shared KV cache across sessions on the same checkpoint.

When multiple sessions process prompts that share a common prefix (system
prompt, few-shot examples, RAG context) they each compute the KV cache for
that prefix independently — wasting memory and compute.

Phase 2B provides a **process-wide prefix KV cache** keyed by
``(checkpoint_id, prefix_token_hash)``.  Sessions that share a prefix reuse
the cached KV tensors for it and start generation from there.

Architecture
------------
``KVBlock``              -- one contiguous chunk of KV tensors (one layer)
``PrefixKVEntry``        -- cached KV for a complete prefix (all layers)
``SharedKVCacheShard``   -- one checkpoint's prefix KV pool
``SharedKVCacheRegistry``-- process-wide registry
``SharedKVLease``        -- per-session handle
``open_shared_kv_cache`` -- high-level entry point

The KV tensors are stored as numpy arrays (model-agnostic).  Production
sessions that use native tensors should convert at the boundary.

Memory limits
-------------
The cache is bounded by ``max_entries`` and ``max_bytes``.  LRU eviction
runs when either limit is exceeded.  Pinned entries are never evicted.
The byte budget counts stored tensor payloads, not Python metadata or temporary
conversion buffers. Complete-prefix lookup requires an explicit expected_layers
inventory; caches without it retain partial blocks but never advertise reuse.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

_DEFAULT_MAX_ENTRIES     = 128    # max cached prefixes per checkpoint
_DEFAULT_MAX_BYTES       = 4 << 30  # 4 GB total KV memory per checkpoint
_MIN_PREFIX_LEN          = 8     # don't cache very short prefixes


# ---------------------------------------------------------------------------
# KVBlock — one layer's KV tensors for a prefix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KVBlock:
    """K and V tensors for one transformer layer, for a fixed prefix."""
    layer_id:  int
    key:       np.ndarray    # (seq_len, num_heads, head_dim) or (1, ...)
    value:     np.ndarray
    dtype:     str = "fp16"

    @property
    def nbytes(self) -> int:
        return self.key.nbytes + self.value.nbytes


def _block_view(block):
    # Read-only data alone does not freeze ndarray shape/stride metadata.
    # Give readers independent views, never the shard's own array objects.
    return KVBlock(block.layer_id, block.key.view(), block.value.view(), block.dtype)


# ---------------------------------------------------------------------------
# PrefixKVEntry — all layers' KV for one prefix
# ---------------------------------------------------------------------------

class PrefixKVEntry:
    """Cached KV state for a complete prefix across all layers."""

    def __init__(
        self,
        prefix_hash: str,
        prefix_len:  int,
        token_ids:   List[int],
        expected_layers: Optional[int] = None,
    ) -> None:
        self._prefix_hash = prefix_hash
        self._prefix_len = prefix_len
        self._token_ids = tuple(token_ids)
        self._expected_layers = expected_layers
        self._blocks: Dict[int, KVBlock] = {}
        self._managed = False
        self.created_at   = time.monotonic()
        self.last_access  = time.monotonic()
        self.access_count = 0
        self._pin_count = 0
        self._lock = threading.Lock()

    @property
    def prefix_hash(self):
        return self._prefix_hash

    @property
    def prefix_len(self):
        return self._prefix_len

    @property
    def token_ids(self):
        return self._token_ids

    @property
    def expected_layers(self):
        return self._expected_layers

    @property
    def pin_count(self):
        with self._lock:
            return self._pin_count

    def add_block(self, block: KVBlock) -> None:
        with self._lock:
            if self._managed:
                raise RuntimeError('managed KV blocks must be written through their shard')
            self._blocks[block.layer_id] = block

    @property
    def blocks(self):
        with self._lock:
            return MappingProxyType({layer: _block_view(block) for layer, block in self._blocks.items()})

    def get_block(self, layer_id: int) -> Optional[KVBlock]:
        with self._lock:
            b = self._blocks.get(layer_id)
            if b:
                self.last_access  = time.monotonic()
                self.access_count += 1
            return _block_view(b) if b is not None else None

    @property
    def is_complete(self) -> bool:
        """True only for an explicitly declared, complete layer inventory."""
        with self._lock:
            return (self.expected_layers is not None and
                    len(self._blocks) == self.expected_layers and
                    all(0 <= layer < self.expected_layers for layer in self._blocks))

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(b.nbytes for b in self._blocks.values())

    def pin(self) -> None:
        with self._lock:
            if self._managed:
                raise RuntimeError('managed KV pins must be owned through a lease or shard')
            self._pin_count += 1

    def unpin(self) -> None:
        with self._lock:
            if self._managed:
                raise RuntimeError('managed KV pins must be owned through a lease or shard')
            if self._pin_count > 0:
                self._pin_count -= 1


# ---------------------------------------------------------------------------
# SharedKVCacheShard — one checkpoint's KV pool
# ---------------------------------------------------------------------------

def _hash_prefix(token_ids: List[int]) -> str:
    h = hashlib.sha256()
    for t in token_ids:
        h.update(t.to_bytes(4, "little"))
    return h.hexdigest()


def _tokens(token_ids, *, allow_empty=False):
    if type(token_ids) not in (list, tuple) or (not token_ids and not allow_empty) or any(
            type(token) is not int or not 0 <= token < 2**31 for token in token_ids):
        raise ValueError('nonempty signed-32-bit token sequence required')
    return tuple(token_ids)


def _cache_contract(checkpoint_id, max_entries, max_bytes, expected_layers):
    if type(checkpoint_id) is not str or not checkpoint_id or len(checkpoint_id) > 4096:
        raise ValueError('bounded nonempty checkpoint identity required')
    if type(max_entries) is not int or max_entries <= 0 or type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('positive integer shared KV budgets required')
    if expected_layers is not None and (type(expected_layers) is not int or not 0 < expected_layers < 2**31):
        raise ValueError('positive signed-32-bit KV layer inventory required')


class SharedKVCacheShard:
    """Thread-safe prefix KV cache for one checkpoint."""

    def __init__(
        self,
        checkpoint_id: str,
        max_entries:   int = _DEFAULT_MAX_ENTRIES,
        max_bytes:     int = _DEFAULT_MAX_BYTES,
        expected_layers: Optional[int] = None,
    ) -> None:
        _cache_contract(checkpoint_id, max_entries, max_bytes, expected_layers)
        self.checkpoint_id = checkpoint_id
        self.max_entries   = max_entries
        self.max_bytes     = max_bytes
        self.expected_layers = expected_layers
        self._entries: Dict[str, PrefixKVEntry] = {}
        self._lock     = threading.RLock()
        self._lease_count = 0
        self._hits    = 0
        self._misses  = 0
        self._evictions = 0
        self._total_bytes_cached = 0

    # ---- lease --------------------------------------------------------------

    def acquire(self) -> None:
        with self._lock:
            self._lease_count += 1

    def release(self) -> int:
        with self._lock:
            self._lease_count = max(0, self._lease_count - 1)
            return self._lease_count

    @property
    def lease_count(self) -> int:
        with self._lock:
            return self._lease_count

    # ---- prefix matching ----------------------------------------------------

    def find_prefix(self, token_ids: List[int]) -> Optional[PrefixKVEntry]:
        """Return the longest cached prefix of token_ids, or None."""
        token_ids = _tokens(token_ids, allow_empty=True)
        if len(token_ids) < _MIN_PREFIX_LEN:
            return None
        with self._lock:
            lengths = {entry.prefix_len for entry in self._entries.values()
                       if _MIN_PREFIX_LEN <= entry.prefix_len <= len(token_ids) and entry.is_complete}
            best = None
            if lengths:
                digest = hashlib.sha256()
                # Hash each token once and only as far as the longest cached
                # candidate. The old descending rehash loop was quadratic in
                # the entire request length, even for a tiny cached prefix.
                for length, token in enumerate(token_ids[:max(lengths)], 1):
                    digest.update(token.to_bytes(4, 'little'))
                    if length in lengths:
                        entry = self._entries.get(digest.hexdigest())
                        if entry is not None and entry.is_complete and entry.token_ids == token_ids[:length]:
                            best = entry
            if best is not None:
                self._hits += 1
                best.last_access = time.monotonic()
                best.access_count += 1
                return best
            self._misses += 1
        return None

    def get_or_create(self, token_ids: List[int]) -> PrefixKVEntry:
        """Return an existing entry or create a new placeholder."""
        token_ids = _tokens(token_ids)
        key = _hash_prefix(token_ids)
        with self._lock:
            if key in self._entries:
                return self._entries[key]
            victims = self._room_for(0, True)
            entry = PrefixKVEntry(
                prefix_hash=key,
                prefix_len=len(token_ids),
                token_ids=token_ids,
                expected_layers=self.expected_layers,
            )
            entry._managed = True
            self._evict(victims)
            self._entries[key] = entry
            return entry

    def store_block(
        self,
        token_ids: List[int],
        layer_id:  int,
        key_tensor:   np.ndarray,
        value_tensor: np.ndarray,
        dtype: str = "fp16",
    ) -> PrefixKVEntry:
        """Store a KV block for a prefix's layer."""
        token_ids = _tokens(token_ids)
        prefix_hash = _hash_prefix(token_ids)
        if type(layer_id) is not int or layer_id < 0 or (
                self.expected_layers is not None and layer_id >= self.expected_layers):
            raise ValueError('KV layer is outside the declared inventory')
        if type(dtype) is not str or dtype not in ('fp16', 'fp32'):
            raise ValueError('shared KV storage supports explicit fp16 or fp32 only')
        if (not isinstance(key_tensor, np.ndarray) or not isinstance(value_tensor, np.ndarray) or
                key_tensor.shape != value_tensor.shape or key_tensor.dtype.kind != 'f' or
                value_tensor.dtype.kind != 'f' or not key_tensor.size or
                key_tensor.ndim not in (3, 4) or
                (key_tensor.ndim == 4 and key_tensor.shape[0] != 1) or
                key_tensor.shape[-3] != len(token_ids)):
            raise ValueError('matching finite sequence/head/dimension KV tensors required')
        target_dtype = np.dtype(np.float16 if dtype == 'fp16' else np.float32)
        block_bytes = key_tensor.size * target_dtype.itemsize * 2
        if block_bytes > self.max_bytes:
            raise MemoryError('KV block exceeds the shared cache byte budget')
        if not np.isfinite(key_tensor).all() or not np.isfinite(value_tensor).all():
            raise ValueError('shared KV tensors must be finite')
        with self._lock:
            entry = self._entries.get(prefix_hash)
            if entry is not None and entry.pin_count:
                raise RuntimeError('cannot mutate a pinned KV prefix')
            previous = entry.get_block(layer_id) if entry is not None else None
            delta = block_bytes - (previous.nbytes if previous is not None else 0)
            victims = self._room_for(delta, entry is None, protected=prefix_hash)
            # Immutable bytes detach borrowed arrays and prevent callers from
            # re-enabling write access on cached numpy storage.
            with np.errstate(over='ignore', invalid='ignore'):
                key = np.frombuffer(key_tensor.astype(target_dtype).tobytes(), dtype=target_dtype).reshape(key_tensor.shape)
                value = np.frombuffer(value_tensor.astype(target_dtype).tobytes(), dtype=target_dtype).reshape(value_tensor.shape)
            if not np.isfinite(key).all() or not np.isfinite(value).all():
                raise ValueError('KV conversion produced nonfinite storage')
            if entry is None:
                entry = PrefixKVEntry(prefix_hash, len(token_ids), token_ids, self.expected_layers)
                entry._managed = True
            self._evict(victims)
            with entry._lock:
                entry._blocks[layer_id] = KVBlock(layer_id, key, value, dtype)
            self._entries[prefix_hash] = entry
            self._total_bytes_cached += delta
            return entry

    def _room_for(self, extra_bytes, new_entry, protected=None):
        count = len(self._entries) + int(new_entry)
        size = self._total_bytes_cached + extra_bytes
        victims = []
        for _, key in sorted((entry.last_access, key) for key, entry in self._entries.items()
                             if not entry.pin_count and key != protected):
            if count <= self.max_entries and size <= self.max_bytes:
                break
            victims.append(key)
            count -= 1
            size -= self._entries[key].nbytes
        if count > self.max_entries or size > self.max_bytes:
            raise MemoryError('shared KV capacity is exhausted by pinned/protected prefixes')
        return victims

    def _evict(self, victims):
        for key in victims:
            entry = self._entries.pop(key)
            self._total_bytes_cached -= entry.nbytes
            self._evictions += 1

    def pin(self, prefix_hash: str) -> bool:
        with self._lock:
            e = self._entries.get(prefix_hash)
            if e:
                with e._lock:
                    e._pin_count += 1
                return True
            return False

    def unpin(self, prefix_hash: str) -> None:
        with self._lock:
            e = self._entries.get(prefix_hash)
            if e:
                with e._lock:
                    if e._pin_count > 0:
                        e._pin_count -= 1

    def metrics(self) -> dict:
        with self._lock:
            return {
                "kv_checkpoint_id":    self.checkpoint_id[:16] + "...",
                "kv_entries":          len(self._entries),
                "kv_bytes_cached":     self._total_bytes_cached,
                "kv_hits":             self._hits,
                "kv_misses":           self._misses,
                "kv_evictions":        self._evictions,
                "kv_lease_count":      self._lease_count,
            }


# ---------------------------------------------------------------------------
# SharedKVCacheRegistry
# ---------------------------------------------------------------------------

class SharedKVCacheRegistry:
    """Process-wide KV cache registry."""

    def __init__(self) -> None:
        self._shards: Dict[str, SharedKVCacheShard] = {}
        self._lock = threading.Lock()

    def open_cache(
        self,
        checkpoint_id: str,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_bytes:   int = _DEFAULT_MAX_BYTES,
        expected_layers: Optional[int] = None,
    ) -> "SharedKVLease":
        _cache_contract(checkpoint_id, max_entries, max_bytes, expected_layers)
        with self._lock:
            shard = self._shards.get(checkpoint_id)
            if shard is None:
                shard = SharedKVCacheShard(checkpoint_id, max_entries, max_bytes, expected_layers)
                self._shards[checkpoint_id] = shard
                logger.info("phase2: new shared KV cache for %s...", checkpoint_id[:16])
            elif (shard.max_entries, shard.max_bytes, shard.expected_layers) != (
                    max_entries, max_bytes, expected_layers):
                raise ValueError('shared KV budgets or layer inventory differ from the existing checkpoint cache')
            shard.acquire()
            try:
                return SharedKVLease(shard, self)
            except BaseException:
                if shard.release() == 0:
                    self._shards.pop(checkpoint_id)
                raise

    def _release(self, shard: SharedKVCacheShard) -> None:
        with self._lock:
            if self._shards.get(shard.checkpoint_id) is not shard:
                raise ValueError('shared KV release belongs to a foreign or retired shard')
            remaining = shard.release()
            if remaining == 0:
                self._shards.pop(shard.checkpoint_id, None)
                with shard._lock:
                    shard._entries.clear()
                    shard._total_bytes_cached = 0
        if remaining == 0:
            logger.info("phase2: shared KV cache destroyed: %s...", shard.checkpoint_id[:16])

    def metrics(self) -> dict:
        with self._lock:
            return {
                "kv_caches": len(self._shards),
                "shards": [s.metrics() for s in self._shards.values()],
            }


_kv_registry = SharedKVCacheRegistry()


# ---------------------------------------------------------------------------
# SharedKVLease
# ---------------------------------------------------------------------------

class SharedKVLease:
    """Per-session KV cache handle."""

    def __init__(self, shard: SharedKVCacheShard, registry: SharedKVCacheRegistry) -> None:
        self._shard    = shard
        self._registry = registry
        self._closed   = False
        self._lock     = threading.RLock()
        self._pinned: Set[str] = set()

    # ---- API ----------------------------------------------------------------

    def find_prefix(self, token_ids: List[int]) -> Optional[PrefixKVEntry]:
        with self._lock:
            self._check_open()
            return self._shard.find_prefix(token_ids)

    def store_block(
        self,
        token_ids: List[int],
        layer_id: int,
        key_tensor: np.ndarray,
        value_tensor: np.ndarray,
        dtype: str = "fp16",
    ) -> PrefixKVEntry:
        with self._lock:
            self._check_open()
            return self._shard.store_block(token_ids, layer_id, key_tensor, value_tensor, dtype)

    def get_or_create(self, token_ids: List[int]) -> PrefixKVEntry:
        with self._lock:
            self._check_open()
            return self._shard.get_or_create(token_ids)

    def pin(self, prefix_hash: str) -> None:
        with self._lock:
            self._check_open()
            if prefix_hash in self._pinned:
                return
            if not self._shard.pin(prefix_hash):
                raise KeyError('cannot pin a missing KV prefix')
            self._pinned.add(prefix_hash)

    def unpin(self, prefix_hash: str) -> None:
        with self._lock:
            self._check_open()
            if prefix_hash not in self._pinned:
                return
            self._shard.unpin(prefix_hash)
            self._pinned.discard(prefix_hash)

    def _check_open(self):
        if self._closed:
            raise RuntimeError('shared KV lease is closed')

    def metrics(self) -> dict:
        return self._shard.metrics()

    # ---- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for ph in list(self._pinned):
                self._shard.unpin(ph)
            self._pinned.clear()
            self._registry._release(self._shard)
            self._closed = True

    def __enter__(self) -> "SharedKVLease":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def open_shared_kv_cache(
    checkpoint_id: str,
    *,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_bytes:   int = _DEFAULT_MAX_BYTES,
    expected_layers: Optional[int] = None,
    registry: Optional[SharedKVCacheRegistry] = None,
) -> SharedKVLease:
    """Open (or join) the shared KV cache for a checkpoint.

    Returns a ``SharedKVLease`` that must be closed when the session ends.
    Without expected_layers, partial storage is allowed but prefix reuse is
    disabled. The declaration is structural metadata, not runtime evidence.
    """
    reg = registry or _kv_registry
    return reg.open_cache(checkpoint_id, max_entries=max_entries, max_bytes=max_bytes,
                          expected_layers=expected_layers)


def kv_registry_metrics() -> dict:
    return _kv_registry.metrics()


__all__ = [
    "KVBlock",
    "PrefixKVEntry",
    "SharedKVCacheShard",
    "SharedKVCacheRegistry",
    "SharedKVLease",
    "open_shared_kv_cache",
    "kv_registry_metrics",
]
