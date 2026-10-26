"""Phase 2A: Shared expert store across concurrent sessions.

In Phase 1 each session owns its own TieredExpertStore.  When two sessions
run the same checkpoint concurrently they both load the same expert weights
from disk — doubling I/O and RAM usage.

Phase 2A provides a **process-wide shared expert store** keyed by checkpoint
identity.  All sessions that open the same checkpoint share:

  - One warm RAM tier (pinned hot experts)
  - Logical tier indices over immutable host weight bytes (not device allocation)
  - One prefetch request queue (execution requires an external consumer)
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

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

@dataclass(frozen=True)
class SharedExpertEntry:
    """One expert's weights held in the shared store."""
    expert_id:    str
    layer_id:     int
    weight_bytes: bytes          # raw bytes (gate+up+down or fused)
    dtype:        str  = "fp16"
    tier:         str  = "ram"   # Logical bucket; never proof of accelerator placement.
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
        if type(checkpoint_id) is not str or not checkpoint_id or len(checkpoint_id) > 4096:
            raise ValueError('bounded nonempty checkpoint identity required')
        if (type(max_ram_experts) is not int or max_ram_experts < 0 or
                type(max_vram_experts) is not int or max_vram_experts < 0):
            raise ValueError('nonnegative integer expert tier capacities required')
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
                object.__setattr__(entry, 'last_access', time.monotonic())
                object.__setattr__(entry, 'access_count', entry.access_count + 1)
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
        if type(expert_id) is not str or not expert_id or type(layer_id) is not int or layer_id < 0:
            raise ValueError('exact expert identity and nonnegative layer required')
        if type(dtype) is not str or not dtype or tier not in ('ram', 'vram'):
            raise ValueError('explicit expert dtype and valid logical cache tier required')
        if not isinstance(weight_bytes, (bytes, bytearray, memoryview)) or not weight_bytes:
            raise ValueError('nonempty expert weight bytes required')
        weight_bytes = bytes(weight_bytes)
        with self._lock:
            # Check if already cached
            existing = self._ram.get(expert_id) or self._vram.get(expert_id)
            if existing is not None:
                if (existing.layer_id, existing.dtype, existing.weight_bytes, existing.tier) != (
                        layer_id, dtype, weight_bytes, tier):
                    raise ValueError('expert identity conflicts with already cached weights or metadata')
                return existing

            entry = SharedExpertEntry(
                expert_id=expert_id,
                layer_id=layer_id,
                weight_bytes=weight_bytes,
                dtype=dtype,
                tier=tier,
            )

            if tier == "vram":
                if self.max_vram_experts == 0:
                    raise MemoryError('logical VRAM expert tier is disabled')
                if len(self._vram) >= self.max_vram_experts:
                    self._evict_lru(self._vram)
                self._vram[expert_id] = entry
            else:
                if self.max_ram_experts == 0:
                    raise MemoryError('RAM expert tier is disabled')
                if len(self._ram) >= self.max_ram_experts:
                    self._evict_lru(self._ram)
                self._ram[expert_id] = entry

            return entry

    def _evict_lru(self, store: Dict[str, SharedExpertEntry]) -> None:
        """Evict the least-recently-used unpinned expert."""
        candidates = [(e.last_access, k) for k, e in store.items() if e.pin_count == 0]
        if not candidates:
            raise MemoryError('expert tier capacity is exhausted by pinned entries')
        _, evict_key = min(candidates)
        del store[evict_key]
        self._evictions += 1

    def pin(self, expert_id: str) -> bool:
        with self._lock:
            for store in (self._vram, self._ram):
                e = store.get(expert_id)
                if e:
                    object.__setattr__(e, 'pin_count', e.pin_count + 1)
                    return True
            return False

    def unpin(self, expert_id: str) -> None:
        with self._lock:
            for store in (self._vram, self._ram):
                e = store.get(expert_id)
                if e and e.pin_count > 0:
                    object.__setattr__(e, 'pin_count', e.pin_count - 1)
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
        if type(checkpoint_id) is not str or not checkpoint_id or len(checkpoint_id) > 4096:
            raise ValueError('bounded nonempty checkpoint identity required')
        if (type(max_ram_experts) is not int or max_ram_experts < 0 or
                type(max_vram_experts) is not int or max_vram_experts < 0):
            raise ValueError('nonnegative integer expert tier capacities required')
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
            elif (shard.checkpoint_dir.resolve(), shard.max_ram_experts, shard.max_vram_experts) != (
                    Path(checkpoint_dir).resolve(), max_ram_experts, max_vram_experts):
                raise ValueError('shared expert checkpoint root or capacities differ from existing store')
            shard.acquire()
            try:
                return SharedExpertLease(shard, self)
            except BaseException:
                if shard.release() == 0:
                    self._stores.pop(checkpoint_id)
                raise

    def _release(self, shard: SharedExpertStoreShard) -> None:
        with self._lock:
            if self._stores.get(shard.checkpoint_id) is not shard:
                raise ValueError('shared expert release belongs to a foreign or retired shard')
            remaining = shard.release()
            if remaining == 0:
                self._stores.pop(shard.checkpoint_id, None)
                with shard._lock, shard._prefetch_lock:
                    shard._ram.clear()
                    shard._vram.clear()
                    shard._heat.clear()
                    shard._prefetch_queue.clear()
        if remaining == 0:
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
        self._lock     = threading.RLock()
        # Track experts this lease pinned so we can unpin on close
        self._pinned: Set[str] = set()

    # ---- Phase 1 store interface ----------------------------------------

    def get(self, expert_id: str) -> Optional[SharedExpertEntry]:
        with self._lock:
            self._check_open()
            return self._shard.get(expert_id)

    def put(self, expert_id: str, layer_id: int, weight_bytes: bytes,
            dtype: str = "fp16", tier: str = "ram") -> SharedExpertEntry:
        with self._lock:
            self._check_open()
            return self._shard.put(expert_id, layer_id, weight_bytes, dtype, tier)

    def pin(self, expert_id: str) -> None:
        with self._lock:
            self._check_open()
            if expert_id in self._pinned:
                return
            if not self._shard.pin(expert_id):
                raise KeyError('cannot pin a missing expert')
            self._pinned.add(expert_id)

    def unpin(self, expert_id: str) -> None:
        with self._lock:
            self._check_open()
            if expert_id not in self._pinned:
                return
            self._shard.unpin(expert_id)
            self._pinned.discard(expert_id)

    def submit_prefetch(self, layer_id: int, expert_ids: List[str]) -> None:
        with self._lock:
            self._check_open()
            self._shard.submit_prefetch(layer_id, expert_ids)

    def promote_expert(self, expert_id: str, tier: str) -> None:
        """No device placement can be certified by relabeling host weight bytes."""
        with self._lock:
            self._check_open()
            entry = self._shard.get(expert_id)
            if entry is not None and entry.tier == tier:
                return
            raise ValueError('expert promotion requires a physical transfer/placement adapter')

    def record_routing(self, expert_ids: List[str]) -> None:
        with self._lock:
            self._check_open()
            self._shard.record_routing(expert_ids)

    def hot_experts(self, top_k: int = 32) -> List[str]:
        with self._lock:
            self._check_open()
            return self._shard.hot_experts(top_k)

    def _check_open(self):
        if self._closed:
            raise RuntimeError('shared expert lease is closed')

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
            self._registry._release(self._shard)
            self._closed = True

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
