"""Phase 2A: Shared expert store across concurrent sessions.

In Phase 1 each session owns its own TieredExpertStore.  When two sessions
run the same checkpoint concurrently they both load the same expert weights
from disk — doubling I/O and RAM usage.

Phase 2A provides a **process-wide shared expert store** keyed by checkpoint
identity.  All sessions that open the same checkpoint share:

  - One warm RAM tier (pinned hot experts)
  - One VRAM tier (GPU experts, if available)
  - One set of prefetch workers
  - One routing-history learning cache

Sessions hold a **lease** on the store; the store is destroyed when the last
lease is released.

Architecture
------------
``SharedExpertEntry``      -- one cached expert (bytes + metadata)
``SharedExpertStoreShard`` -- one checkpoint's shared expert pool
``SharedExpertStoreRegistry`` -- process-wide registry (keyed by checkpoint id)
``SharedExpertLease``      -- per-session handle; returned to registry on close
``open_shared_expert_store`` -- high-level entry point
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RAM_EXPERTS   = 256   # per checkpoint
_DEFAULT_MAX_VRAM_EXPERTS  = 64    # per checkpoint (GPU)
_DEFAULT_PREFETCH_WORKERS  = 2
_LEASE_CLEANUP_GRACE_S     = 5.0   # seconds to wait before destroying idle store


# ---------------------------------------------------------------------------
# SharedExpertEntry
# ---------------------------------------------------------------------------

@dataclass
class SharedExpertEntry:
    """One expert's weights held in the shared store."""
    expert_id:    str
    layer_id:     int
    weight_bytes: bytes          # raw bytes (gate+up+down or fused)
    dtype:        str  = "fp16"
    tier:         str  = "ram"   # "ram" | "vram"
    pin_count:    int  = 0       # sessions pinning this expert
    last_access:  float = field(default_factory=time.monotonic)
    access_count: int   = 0


# ---------------------------------------------------------------------------
# SharedExpertStoreShard — one checkpoint's pool
# ---------------------------------------------------------------------------

class SharedExpertStoreShard:
    """Thread-safe expert cache for one checkpoint.

    All concurrent sessions sharing this checkpoint get/put experts here.
    Reference-counted via ``lease_count``; destroyed when count reaches 0.
    """

    def __init__(
        self,
        checkpoint_id: str,
        checkpoint_dir: Path,
        max_ram_experts:  int = _DEFAULT_MAX_RAM_EXPERTS,
        max_vram_experts: int = _DEFAULT_MAX_VRAM_EXPERTS,
    ) -> None:
        self.checkpoint_id  = checkpoint_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_ram_experts  = max_ram_experts
        self.max_vram_experts = max_vram_experts

        self._ram:  Dict[str, SharedExpertEntry] = {}
        self._vram: Dict[str, SharedExpertEntry] = {}
        self._lock  = threading.RLock()
        self._lease_count = 0
        self._created_at  = time.monotonic()

        # Routing heat: expert_id → fire count (all sessions contribute)
        self._heat: Dict[str, int] = {}
        # Prefetch queue: (layer_id, expert_id) tuples
        self._prefetch_queue: List[Tuple[int, str]] = []
        self._prefetch_lock  = threading.Lock()

        # Metrics
        self._hits   = 0
        self._misses = 0
        self._evictions = 0

    # ---- lease management ---------------------------------------------------

    def acquire(self) -> None:
        with self._lock:
            self._lease_count += 1

    def release(self) -> int:
        """Decrement lease count; return new count."""
        with self._lock:
            self._lease_count = max(0, self._lease_count - 1)
            return self._lease_count

    @property
    def lease_count(self) -> int:
        with self._lock:
            return self._lease_count

    # ---- expert get/put -----------------------------------------------------

    def get(self, expert_id: str) -> Optional[SharedExpertEntry]:
        with self._lock:
            entry = self._vram.get(expert_id) or self._ram.get(expert_id)
            if entry is not None:
                entry.last_access = time.monotonic()
                entry.access_count += 1
                self._hits += 1
            else:
                self._misses += 1
            return entry

    def put(
        self,
        expert_id: str,
        layer_id: int,
        weight_bytes: bytes,
        dtype: str = "fp16",
        tier: str = "ram",
    ) -> SharedExpertEntry:
        with self._lock:
            # Check if already cached
            existing = self._ram.get(expert_id) or self._vram.get(expert_id)
            if existing is not None:
                return existing

            entry = SharedExpertEntry(
                expert_id=expert_id,
                layer_id=layer_id,
                weight_bytes=weight_bytes,
                dtype=dtype,
                tier=tier,
            )

            if tier == "vram":
                if len(self._vram) >= self.max_vram_experts:
                    self._evict_lru(self._vram)
                self._vram[expert_id] = entry
            else:
                if len(self._ram) >= self.max_ram_experts:
                    self._evict_lru(self._ram)
                self._ram[expert_id] = entry

            return entry

    def _evict_lru(self, store: Dict[str, SharedExpertEntry]) -> None:
        """Evict the least-recently-used unpinned expert."""
        candidates = [(e.last_access, k) for k, e in store.items() if e.pin_count == 0]
        if not candidates:
            return
        _, evict_key = min(candidates)
        del store[evict_key]
        self._evictions += 1

    def pin(self, expert_id: str) -> None:
        with self._lock:
            for store in (self._vram, self._ram):
                e = store.get(expert_id)
                if e:
                    e.pin_count += 1
                    return

    def unpin(self, expert_id: str) -> None:
        with self._lock:
            for store in (self._vram, self._ram):
                e = store.get(expert_id)
                if e and e.pin_count > 0:
                    e.pin_count -= 1
                    return

    # ---- routing heat -------------------------------------------------------

    def record_routing(self, expert_ids: List[str]) -> None:
        with self._lock:
            for eid in expert_ids:
                self._heat[eid] = self._heat.get(eid, 0) + 1

    def hot_experts(self, top_k: int = 32) -> List[str]:
        with self._lock:
            return sorted(self._heat, key=lambda k: -self._heat[k])[:top_k]

    # ---- prefetch queue -----------------------------------------------------

    def submit_prefetch(self, layer_id: int, expert_ids: List[str]) -> None:
        with self._prefetch_lock:
            for eid in expert_ids:
                self._prefetch_queue.append((layer_id, eid))

    def drain_prefetch_queue(self, max_items: int = 16) -> List[Tuple[int, str]]:
        with self._prefetch_lock:
            items = self._prefetch_queue[:max_items]
            self._prefetch_queue = self._prefetch_queue[max_items:]
            return items

    # ---- metrics ------------------------------------------------------------

    def metrics(self) -> dict:
        with self._lock:
            return {
                "checkpoint_id": self.checkpoint_id[:16] + "...",
                "ram_experts":   len(self._ram),
                "vram_experts":  len(self._vram),
                "lease_count":   self._lease_count,
                "hits":          self._hits,
                "misses":        self._misses,
                "evictions":     self._evictions,
                "heat_tracked":  len(self._heat),
                "prefetch_queue_len": len(self._prefetch_queue),
            }


# ---------------------------------------------------------------------------
# SharedExpertStoreRegistry — process-wide singleton
# ---------------------------------------------------------------------------

class SharedExpertStoreRegistry:
    """Process-wide registry mapping checkpoint_id → SharedExpertStoreShard.

    Thread-safe.  Call ``open_store(checkpoint_id, ...)`` to get a lease.
    Call ``lease.close()`` when the session ends.
    """

    def __init__(self) -> None:
        self._stores: Dict[str, SharedExpertStoreShard] = {}
        self._lock = threading.Lock()

    def open_store(
        self,
        checkpoint_id: str,
        checkpoint_dir: Path,
        max_ram_experts:  int = _DEFAULT_MAX_RAM_EXPERTS,
        max_vram_experts: int = _DEFAULT_MAX_VRAM_EXPERTS,
    ) -> "SharedExpertLease":
        with self._lock:
            shard = self._stores.get(checkpoint_id)
            if shard is None:
                shard = SharedExpertStoreShard(
                    checkpoint_id=checkpoint_id,
                    checkpoint_dir=checkpoint_dir,
                    max_ram_experts=max_ram_experts,
                    max_vram_experts=max_vram_experts,
                )
                self._stores[checkpoint_id] = shard
                logger.info("phase2: new shared expert store for %s...", checkpoint_id[:16])
            shard.acquire()
        return SharedExpertLease(shard, self)

    def _release(self, shard: SharedExpertStoreShard) -> None:
        remaining = shard.release()
        if remaining == 0:
            with self._lock:
                self._stores.pop(shard.checkpoint_id, None)
            logger.info("phase2: shared expert store destroyed (no more leases): %s...",
                        shard.checkpoint_id[:16])

    def active_stores(self) -> int:
        with self._lock:
            return len(self._stores)

    def metrics(self) -> dict:
        with self._lock:
            return {
                "shared_stores": len(self._stores),
                "shards": [s.metrics() for s in self._stores.values()],
            }


# Process-wide singleton
_registry = SharedExpertStoreRegistry()


# ---------------------------------------------------------------------------
# SharedExpertLease — per-session handle
# ---------------------------------------------------------------------------

class SharedExpertLease:
    """Per-session handle to a SharedExpertStoreShard.

    Implements the same ``get`` / ``put`` / ``submit_prefetch`` / ``pin`` /
    ``unpin`` interface as ``TieredExpertStore`` so Phase 1 adapters can use
    it as a drop-in replacement.
    """

    def __init__(self, shard: SharedExpertStoreShard, registry: SharedExpertStoreRegistry) -> None:
        self._shard    = shard
        self._registry = registry
        self._closed   = False
        self._lock     = threading.Lock()
        # Track experts this lease pinned so we can unpin on close
        self._pinned: Set[str] = set()

    # ---- Phase 1 store interface ----------------------------------------

    def get(self, expert_id: str) -> Optional[SharedExpertEntry]:
        return self._shard.get(expert_id)

    def put(self, expert_id: str, layer_id: int, weight_bytes: bytes,
            dtype: str = "fp16", tier: str = "ram") -> SharedExpertEntry:
        return self._shard.put(expert_id, layer_id, weight_bytes, dtype, tier)

    def pin(self, expert_id: str) -> None:
        self._shard.pin(expert_id)
        with self._lock:
            self._pinned.add(expert_id)

    def unpin(self, expert_id: str) -> None:
        self._shard.unpin(expert_id)
        with self._lock:
            self._pinned.discard(expert_id)

    def submit_prefetch(self, layer_id: int, expert_ids: List[str]) -> None:
        self._shard.submit_prefetch(layer_id, expert_ids)

    def promote_expert(self, expert_id: str, tier: str) -> None:
        """Phase 1D placement hint — update tier on cached entry."""
        entry = self._shard.get(expert_id)
        if entry is not None:
            entry.tier = tier

    def record_routing(self, expert_ids: List[str]) -> None:
        self._shard.record_routing(expert_ids)

    def hot_experts(self, top_k: int = 32) -> List[str]:
        return self._shard.hot_experts(top_k)

    def metrics(self) -> dict:
        return self._shard.metrics()

    # ---- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            # Unpin everything this lease pinned
            for eid in list(self._pinned):
                self._shard.unpin(eid)
            self._pinned.clear()
            self._closed = True
        self._registry._release(self._shard)

    def __enter__(self) -> "SharedExpertLease":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def open_shared_expert_store(
    checkpoint_id: str,
    checkpoint_dir: Path,
    *,
    max_ram_experts:  int = _DEFAULT_MAX_RAM_EXPERTS,
    max_vram_experts: int = _DEFAULT_MAX_VRAM_EXPERTS,
    registry: Optional[SharedExpertStoreRegistry] = None,
) -> SharedExpertLease:
    """Open (or join) the shared expert store for a checkpoint.

    Returns a ``SharedExpertLease`` that must be closed when the session ends.
    Multiple calls with the same ``checkpoint_id`` return leases to the same
    underlying shard — expert weights are shared, not duplicated.
    """
    reg = registry or _registry
    return reg.open_store(
        checkpoint_id=checkpoint_id,
        checkpoint_dir=Path(checkpoint_dir),
        max_ram_experts=max_ram_experts,
        max_vram_experts=max_vram_experts,
    )


def registry_metrics() -> dict:
    """Return metrics for the process-wide shared expert store registry."""
    return _registry.metrics()


__all__ = [
    "SharedExpertEntry",
    "SharedExpertStoreShard",
    "SharedExpertStoreRegistry",
    "SharedExpertLease",
    "open_shared_expert_store",
    "registry_metrics",
]
