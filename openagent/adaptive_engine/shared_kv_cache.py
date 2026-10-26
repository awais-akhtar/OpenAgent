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
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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

@dataclass
class KVBlock:
    """K and V tensors for one transformer layer, for a fixed prefix."""
    layer_id:  int
    key:       np.ndarray    # (seq_len, num_heads, head_dim) or (1, ...)
    value:     np.ndarray
    dtype:     str = "fp16"

    @property
    def nbytes(self) -> int:
        return self.key.nbytes + self.value.nbytes


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
    ) -> None:
        self.prefix_hash = prefix_hash
        self.prefix_len  = prefix_len
        self.token_ids   = list(token_ids)
        self.blocks: Dict[int, KVBlock] = {}  # layer_id → KVBlock
        self.created_at   = time.monotonic()
        self.last_access  = time.monotonic()
        self.access_count = 0
        self.pin_count    = 0
        self._lock = threading.Lock()

    def add_block(self, block: KVBlock) -> None:
        with self._lock:
            self.blocks[block.layer_id] = block

    def get_block(self, layer_id: int) -> Optional[KVBlock]:
        with self._lock:
            b = self.blocks.get(layer_id)
            if b:
                self.last_access  = time.monotonic()
                self.access_count += 1
            return b

    @property
    def is_complete(self) -> bool:
        """True once at least one block has been stored (partial is useful)."""
        with self._lock:
            return bool(self.blocks)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(b.nbytes for b in self.blocks.values())

    def pin(self) -> None:
        with self._lock:
            self.pin_count += 1

    def unpin(self) -> None:
        with self._lock:
            if self.pin_count > 0:
                self.pin_count -= 1


# ---------------------------------------------------------------------------
# SharedKVCacheShard — one checkpoint's KV pool
# ---------------------------------------------------------------------------

def _hash_prefix(token_ids: List[int]) -> str:
    h = hashlib.sha256()
    for t in token_ids:
        h.update(t.to_bytes(4, "little"))
    return h.hexdigest()


class SharedKVCacheShard:
    """Thread-safe prefix KV cache for one checkpoint."""

    def __init__(
        self,
        checkpoint_id: str,
        max_entries:   int = _DEFAULT_MAX_ENTRIES,
        max_bytes:     int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.max_entries   = max_entries
        self.max_bytes     = max_bytes
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
        if len(token_ids) < _MIN_PREFIX_LEN:
            return None
        # Try exact match first, then progressively shorter
        for length in range(len(token_ids), _MIN_PREFIX_LEN - 1, -1):
            key = _hash_prefix(token_ids[:length])
            with self._lock:
                entry = self._entries.get(key)
            if entry is not None and entry.is_complete:
                with self._lock:
                    self._hits += 1
                entry.last_access  = time.monotonic()
                entry.access_count += 1
                return entry
        with self._lock:
            self._misses += 1
        return None

    def get_or_create(self, token_ids: List[int]) -> PrefixKVEntry:
        """Return an existing entry or create a new placeholder."""
        key = _hash_prefix(token_ids)
        with self._lock:
            if key in self._entries:
                return self._entries[key]
            self._maybe_evict()
            entry = PrefixKVEntry(
                prefix_hash=key,
                prefix_len=len(token_ids),
                token_ids=token_ids,
            )
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
        entry = self.get_or_create(token_ids)
        block = KVBlock(
            layer_id=layer_id,
            key=key_tensor.astype(np.float16 if dtype == "fp16" else np.float32, copy=False),
            value=value_tensor.astype(np.float16 if dtype == "fp16" else np.float32, copy=False),
            dtype=dtype,
        )
        entry.add_block(block)
        with self._lock:
            self._total_bytes_cached += block.nbytes
        return entry

    def _maybe_evict(self) -> None:
        """LRU evict when over limit (called under _lock)."""
        while (len(self._entries) >= self.max_entries or
               self._total_bytes_cached >= self.max_bytes):
            candidates = [
                (e.last_access, k)
                for k, e in self._entries.items()
                if e.pin_count == 0
            ]
            if not candidates:
                break
            _, evict_key = min(candidates)
            evicted = self._entries.pop(evict_key)
            self._total_bytes_cached = max(0, self._total_bytes_cached - evicted.nbytes)
            self._evictions += 1

    def pin(self, prefix_hash: str) -> None:
        with self._lock:
            e = self._entries.get(prefix_hash)
            if e:
                e.pin()

    def unpin(self, prefix_hash: str) -> None:
        with self._lock:
            e = self._entries.get(prefix_hash)
            if e:
                e.unpin()

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
    ) -> "SharedKVLease":
        with self._lock:
            shard = self._shards.get(checkpoint_id)
            if shard is None:
                shard = SharedKVCacheShard(checkpoint_id, max_entries, max_bytes)
                self._shards[checkpoint_id] = shard
                logger.info("phase2: new shared KV cache for %s...", checkpoint_id[:16])
            shard.acquire()
        return SharedKVLease(shard, self)

    def _release(self, shard: SharedKVCacheShard) -> None:
        remaining = shard.release()
        if remaining == 0:
            with self._lock:
                self._shards.pop(shard.checkpoint_id, None)
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
        self._lock     = threading.Lock()
        self._pinned: Set[str] = set()

    # ---- API ----------------------------------------------------------------

    def find_prefix(self, token_ids: List[int]) -> Optional[PrefixKVEntry]:
        return self._shard.find_prefix(token_ids)

    def store_block(
        self,
        token_ids: List[int],
        layer_id: int,
        key_tensor: np.ndarray,
        value_tensor: np.ndarray,
        dtype: str = "fp16",
    ) -> PrefixKVEntry:
        return self._shard.store_block(token_ids, layer_id, key_tensor, value_tensor, dtype)

    def get_or_create(self, token_ids: List[int]) -> PrefixKVEntry:
        return self._shard.get_or_create(token_ids)

    def pin(self, prefix_hash: str) -> None:
        self._shard.pin(prefix_hash)
        with self._lock:
            self._pinned.add(prefix_hash)

    def unpin(self, prefix_hash: str) -> None:
        self._shard.unpin(prefix_hash)
        with self._lock:
            self._pinned.discard(prefix_hash)

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
            self._closed = True
        self._registry._release(self._shard)

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
    registry: Optional[SharedKVCacheRegistry] = None,
) -> SharedKVLease:
    """Open (or join) the shared KV cache for a checkpoint.

    Returns a ``SharedKVLease`` that must be closed when the session ends.
    """
    reg = registry or _kv_registry
    return reg.open_cache(checkpoint_id, max_entries=max_entries, max_bytes=max_bytes)


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
