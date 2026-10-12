"""Bounded residency tiers for sparse expert payloads.

This module manages where expert bytes are resident.  It deliberately does not
execute expert kernels: an accelerator adapter only owns upload and release of
opaque payload handles.  Model adapters remain responsible for computation.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Protocol, runtime_checkable


class ResidencyTier(str, Enum):
    STORAGE = "storage"
    MAPPED_HOST = "mapped_host"
    HOT_HOST = "hot_host"
    ACCELERATOR = "accelerator"


class PromotionPhase(str, Enum):
    IDLE = "idle"
    QUEUED = "queued"
    PROMOTING = "promoting"
    RESIDENT = "resident"
    FAILED = "failed"


@dataclass(frozen=True)
class BackendCapability:
    """Facts reported by an accelerator adapter, without inferred support."""

    available: bool
    backend: str
    device: str = ""
    supports_upload: bool = False
    executes_experts: bool = False
    simulated: bool = False
    reason: str = ""


@runtime_checkable
class AcceleratorBackend(Protocol):
    """Minimal adapter for accelerator residency, not kernel execution."""

    def capability(self) -> BackendCapability: ...

    def upload(self, expert_id: str, payload: memoryview) -> Any: ...

    def release(self, handle: Any) -> None: ...


@dataclass(frozen=True)
class PromotionResult:
    expert_id: str
    requested_tier: ResidencyTier
    resident_tier: ResidencyTier
    success: bool
    fallback: bool
    error: str = ""


@dataclass(frozen=True)
class PromotionSnapshot:
    expert_id: str
    tier: ResidencyTier
    phase: PromotionPhase
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    fallback_tier: ResidencyTier = ResidencyTier.STORAGE


@dataclass(frozen=True)
class ExpertLease:
    """A leased resident copy.  Accelerator payloads are opaque handles."""

    expert_id: str
    tier: ResidencyTier
    payload: Any
    size_bytes: int


@dataclass
class _ResidentCopy:
    payload: Any
    size: int
    leases: int = 0
    pinned: bool = False
    touched_at: float = field(default_factory=time.monotonic)


@dataclass
class _Expert:
    expert_id: str
    size: int
    source: bytes | bytearray | memoryview | Callable[[], bytes | bytearray | memoryview]
    copies: dict[ResidencyTier, _ResidentCopy] = field(default_factory=dict)


class InMemoryAccelerator:
    """Deterministic test adapter.  It stores bytes in RAM and runs no kernels."""

    def __init__(
        self,
        *,
        available: bool = True,
        fail_uploads: set[str] | None = None,
        device: str = "in-memory-test-device",
    ) -> None:
        self.available = bool(available)
        self.fail_uploads = set(fail_uploads or ())
        self.device = device
        self.payloads: dict[str, bytes] = {}
        self.released: list[str] = []

    def capability(self) -> BackendCapability:
        return BackendCapability(
            available=self.available,
            backend="in-memory",
            device=self.device,
            supports_upload=self.available,
            executes_experts=False,
            simulated=True,
            reason="test residency adapter; no accelerator kernels are executed",
        )

    def upload(self, expert_id: str, payload: memoryview) -> str:
        if not self.available:
            raise RuntimeError("in-memory accelerator is unavailable")
        if expert_id in self.fail_uploads:
            raise RuntimeError(f"configured upload failure for {expert_id}")
        handle = f"resident:{expert_id}"
        self.payloads[handle] = bytes(payload)
        return handle

    def release(self, handle: Any) -> None:
        key = str(handle)
        self.payloads.pop(key, None)
        self.released.append(key)


class ExpertTierManager:
    """Manage byte-bounded expert residency with lease-safe LRU eviction."""

    _HOST_TIERS = (ResidencyTier.HOT_HOST, ResidencyTier.MAPPED_HOST)
    _PREFERENCE = (
        ResidencyTier.ACCELERATOR,
        ResidencyTier.HOT_HOST,
        ResidencyTier.MAPPED_HOST,
        ResidencyTier.STORAGE,
    )

    def __init__(
        self,
        *,
        mapped_host_bytes: int,
        hot_host_bytes: int,
        accelerator_bytes: int = 0,
        accelerator: AcceleratorBackend | None = None,
        promotion_workers: int = 1,
    ) -> None:
        budgets = {
            ResidencyTier.MAPPED_HOST: int(mapped_host_bytes),
            ResidencyTier.HOT_HOST: int(hot_host_bytes),
            ResidencyTier.ACCELERATOR: int(accelerator_bytes),
        }
        if any(value < 0 for value in budgets.values()):
            raise ValueError("tier byte budgets cannot be negative")
        if promotion_workers <= 0:
            raise ValueError("promotion_workers must be positive")
        self._budgets = budgets
        self._used = Counter({tier: 0 for tier in budgets})
        self._reserved = Counter({tier: 0 for tier in budgets})
        self._peak = Counter({tier: 0 for tier in budgets})
        self._experts: dict[str, _Expert] = {}
        self._lru = {tier: OrderedDict() for tier in budgets}
        self._accelerator = accelerator
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=promotion_workers, thread_name_prefix="expert-promote")
        self._pending: dict[tuple[str, ResidencyTier], Future[PromotionResult]] = {}
        self._states: dict[tuple[str, ResidencyTier], PromotionSnapshot] = {}
        self._closed = False
        self._telemetry = Counter(
            promotion_attempts=0,
            promotion_completed=0,
            promotion_failures=0,
            fallback_count=0,
            evictions=0,
            lease_acquisitions=0,
            active_leases=0,
            bytes_promoted=0,
            accelerator_uploads=0,
            accelerator_upload_failures=0,
        )
        self._latency_seconds = 0.0

    def register(
        self,
        expert_id: str,
        source: bytes | bytearray | memoryview | Callable[[], bytes | bytearray | memoryview],
        *,
        size_bytes: int | None = None,
    ) -> None:
        if not expert_id:
            raise ValueError("expert_id is required")
        if callable(source):
            if size_bytes is None or size_bytes <= 0:
                raise ValueError("callable sources require a positive size_bytes")
            size = int(size_bytes)
        else:
            size = len(source)
            if size <= 0 or (size_bytes is not None and int(size_bytes) != size):
                raise ValueError("source size must be positive and match size_bytes")
        with self._lock:
            self._check_open()
            if expert_id in self._experts:
                raise ValueError(f"expert already registered: {expert_id}")
            self._experts[expert_id] = _Expert(expert_id, size, source)

    def capability(self) -> dict[str, Mapping[str, Any]]:
        if self._accelerator is None:
            accelerator = BackendCapability(
                available=False,
                backend="none",
                reason="no accelerator residency adapter configured",
            )
        else:
            try:
                accelerator = self._accelerator.capability()
            except Exception as exc:
                accelerator = BackendCapability(
                    available=False,
                    backend=type(self._accelerator).__name__,
                    reason=f"capability probe failed: {exc}",
                )
        return {
            ResidencyTier.STORAGE.value: {
                "available": True,
                "bounded": False,
                "description": "registered immutable source",
            },
            ResidencyTier.MAPPED_HOST.value: {
                "available": self._budgets[ResidencyTier.MAPPED_HOST] > 0,
                "budget_bytes": self._budgets[ResidencyTier.MAPPED_HOST],
                "description": "host view tier; actual file mapping belongs to the source adapter",
            },
            ResidencyTier.HOT_HOST.value: {
                "available": self._budgets[ResidencyTier.HOT_HOST] > 0,
                "budget_bytes": self._budgets[ResidencyTier.HOT_HOST],
                "description": "copied hot host bytes, optionally pinned against eviction",
            },
            ResidencyTier.ACCELERATOR.value: {
                **accelerator.__dict__,
                "budget_bytes": self._budgets[ResidencyTier.ACCELERATOR],
            },
        }

    def promote(
        self,
        expert_id: str,
        tier: ResidencyTier | str,
        *,
        asynchronous: bool = False,
    ) -> PromotionResult | Future[PromotionResult]:
        target = ResidencyTier(tier)
        if target is ResidencyTier.STORAGE:
            return PromotionResult(expert_id, target, ResidencyTier.STORAGE, True, False)
        pending: Future[PromotionResult] | None = None
        with self._lock:
            self._check_open()
            self._expert(expert_id)
            key = (expert_id, target)
            existing = self._pending.get(key)
            if existing is not None and not existing.done():
                pending = existing
            elif target in self._experts[expert_id].copies:
                self._touch(expert_id, target)
                return PromotionResult(expert_id, target, target, True, False)
            elif asynchronous:
                self._set_state(expert_id, target, PromotionPhase.QUEUED)
                future = self._pool.submit(self._promote, expert_id, target)
                self._pending[key] = future
                future.add_done_callback(lambda _job, pending_key=key: self._clear_pending(pending_key))
                return future
        if pending is not None:
            return pending if asynchronous else pending.result()
        return self._promote(expert_id, target)

    def promotion_state(self, expert_id: str, tier: ResidencyTier | str) -> PromotionSnapshot:
        target = ResidencyTier(tier)
        with self._lock:
            self._expert(expert_id)
            return self._states.get(
                (expert_id, target),
                PromotionSnapshot(expert_id, target, PromotionPhase.IDLE),
            )

    @contextmanager
    def acquire(
        self,
        expert_id: str,
        *,
        preferred: ResidencyTier | str = ResidencyTier.HOT_HOST,
        promote: bool = True,
    ) -> Iterator[ExpertLease]:
        requested = ResidencyTier(preferred)
        if promote and requested is not ResidencyTier.STORAGE:
            result = self.promote(expert_id, requested)
            if isinstance(result, Future):
                result.result()
        with self._lock:
            self._check_open()
            expert = self._expert(expert_id)
            tier = self._best_tier(expert, requested)
            if tier is ResidencyTier.STORAGE:
                payload: Any = self._materialize(expert)
                copy = None
            else:
                copy = expert.copies[tier]
                copy.leases += 1
                copy.touched_at = time.monotonic()
                self._touch(expert_id, tier)
                payload = copy.payload
            self._telemetry["lease_acquisitions"] += 1
            self._telemetry["active_leases"] += 1
            lease = ExpertLease(expert_id, tier, payload, expert.size)
        try:
            yield lease
        finally:
            with self._lock:
                if copy is not None:
                    copy.leases -= 1
                elif isinstance(payload, memoryview):
                    payload.release()
                self._telemetry["active_leases"] -= 1

    def pin_hot(self, expert_id: str, *, load: bool = True) -> bool:
        if load:
            result = self.promote(expert_id, ResidencyTier.HOT_HOST)
            if isinstance(result, Future):
                result = result.result()
            if not result.success:
                return False
        with self._lock:
            expert = self._expert(expert_id)
            copy = expert.copies.get(ResidencyTier.HOT_HOST)
            if copy is None:
                return False
            copy.pinned = True
            return True

    def unpin_hot(self, expert_id: str) -> bool:
        with self._lock:
            copy = self._expert(expert_id).copies.get(ResidencyTier.HOT_HOST)
            if copy is None:
                return False
            copy.pinned = False
            return True

    def evict(self, expert_id: str, tier: ResidencyTier | str) -> bool:
        target = ResidencyTier(tier)
        if target is ResidencyTier.STORAGE:
            return False
        with self._lock:
            copy = self._expert(expert_id).copies.get(target)
            if copy is None or copy.leases or copy.pinned:
                return False
            self._drop_copy(expert_id, target, count_eviction=True)
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            tiers = {}
            for tier, budget in self._budgets.items():
                resident = [
                    expert_id
                    for expert_id, expert in self._experts.items()
                    if tier in expert.copies
                ]
                tiers[tier.value] = {
                    "budget_bytes": budget,
                    "used_bytes": self._used[tier],
                    "reserved_bytes": self._reserved[tier],
                    "available_bytes": max(0, budget - self._used[tier] - self._reserved[tier]),
                    "peak_bytes": self._peak[tier],
                    "resident_experts": resident,
                    "resident_count": len(resident),
                    "leased_count": sum(
                        expert.copies[tier].leases
                        for expert in self._experts.values()
                        if tier in expert.copies
                    ),
                    "pinned_count": sum(
                        int(expert.copies[tier].pinned)
                        for expert in self._experts.values()
                        if tier in expert.copies
                    ),
                }
            states = {
                f"{expert_id}:{tier.value}": {
                    "phase": snapshot.phase.value,
                    "started_at": snapshot.started_at,
                    "finished_at": snapshot.finished_at,
                    "error": snapshot.error,
                    "fallback_tier": snapshot.fallback_tier.value,
                }
                for (expert_id, tier), snapshot in self._states.items()
            }
            return {
                "registered_experts": len(self._experts),
                "registered_bytes": sum(expert.size for expert in self._experts.values()),
                "tiers": tiers,
                "capabilities": self.capability(),
                "promotions": states,
                "telemetry": {
                    **dict(self._telemetry),
                    "promotion_seconds": self._latency_seconds,
                    "pending_promotions": sum(not future.done() for future in self._pending.values()),
                },
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._telemetry["active_leases"]:
                raise RuntimeError("cannot close the tier manager while expert leases are active")
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            for expert_id, expert in list(self._experts.items()):
                for tier in list(expert.copies):
                    self._drop_copy(expert_id, tier, count_eviction=False)

    def __enter__(self) -> "ExpertTierManager":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _promote(self, expert_id: str, tier: ResidencyTier) -> PromotionResult:
        started = time.monotonic()
        with self._lock:
            self._check_open()
            expert = self._expert(expert_id)
            if tier in expert.copies:
                return PromotionResult(expert_id, tier, tier, True, False)
            self._telemetry["promotion_attempts"] += 1
            self._set_state(expert_id, tier, PromotionPhase.PROMOTING, started=started)
            try:
                self._reserve(tier, expert.size)
            except (MemoryError, RuntimeError) as exc:
                return self._promotion_failed(expert, tier, started, exc)

        payload: Any = None
        source: memoryview | None = None
        try:
            source = self._materialize(expert)
            if tier is ResidencyTier.MAPPED_HOST:
                payload = source
                source = None
            elif tier is ResidencyTier.HOT_HOST:
                payload = bytes(source)
            elif tier is ResidencyTier.ACCELERATOR:
                capability = self._accelerator_capability()
                if not capability.available or not capability.supports_upload:
                    raise RuntimeError(capability.reason or "accelerator upload is unavailable")
                assert self._accelerator is not None
                payload = self._accelerator.upload(expert_id, source)
                with self._lock:
                    self._telemetry["accelerator_uploads"] += 1
            else:
                raise ValueError(f"unsupported promotion tier: {tier.value}")
        except Exception as exc:
            with self._lock:
                self._reserved[tier] -= expert.size
                if tier is ResidencyTier.ACCELERATOR:
                    self._telemetry["accelerator_upload_failures"] += 1
                return self._promotion_failed(expert, tier, started, exc)
        finally:
            if source is not None:
                source.release()

        with self._lock:
            self._reserved[tier] -= expert.size
            if self._closed:
                if tier is ResidencyTier.ACCELERATOR and self._accelerator is not None:
                    self._accelerator.release(payload)
                return self._promotion_failed(expert, tier, started, RuntimeError("tier manager closed during promotion"))
            expert.copies[tier] = _ResidentCopy(payload, expert.size)
            self._lru[tier][expert_id] = None
            self._used[tier] += expert.size
            self._peak[tier] = max(self._peak[tier], self._used[tier])
            self._telemetry["promotion_completed"] += 1
            self._telemetry["bytes_promoted"] += expert.size
            self._latency_seconds += time.monotonic() - started
            self._set_state(expert_id, tier, PromotionPhase.RESIDENT, started=started, finished=time.time())
            return PromotionResult(expert_id, tier, tier, True, False)

    def _promotion_failed(
        self,
        expert: _Expert,
        tier: ResidencyTier,
        started: float,
        exc: Exception,
    ) -> PromotionResult:
        fallback = self._best_tier(expert, ResidencyTier.ACCELERATOR)
        self._telemetry["promotion_failures"] += 1
        self._telemetry["fallback_count"] += 1
        self._latency_seconds += time.monotonic() - started
        error = str(exc) or type(exc).__name__
        self._set_state(
            expert.expert_id,
            tier,
            PromotionPhase.FAILED,
            started=started,
            finished=time.time(),
            error=error,
            fallback=fallback,
        )
        return PromotionResult(expert.expert_id, tier, fallback, False, True, error)

    def _reserve(self, tier: ResidencyTier, size: int) -> None:
        budget = self._budgets[tier]
        if size > budget:
            raise MemoryError(f"expert requires {size} bytes but {tier.value} budget is {budget}")
        needed = self._used[tier] + self._reserved[tier] + size - budget
        if needed > 0:
            for expert_id in list(self._lru[tier]):
                copy = self._experts[expert_id].copies[tier]
                if copy.leases or copy.pinned:
                    continue
                released = copy.size
                self._drop_copy(expert_id, tier, count_eviction=True)
                needed -= released
                if needed <= 0:
                    break
        if needed > 0:
            raise MemoryError(f"leased or pinned experts exhaust the {tier.value} budget")
        self._reserved[tier] += size

    def _drop_copy(self, expert_id: str, tier: ResidencyTier, *, count_eviction: bool) -> None:
        expert = self._experts[expert_id]
        copy = expert.copies.pop(tier)
        self._lru[tier].pop(expert_id, None)
        self._used[tier] -= copy.size
        if tier is ResidencyTier.ACCELERATOR and self._accelerator is not None:
            self._accelerator.release(copy.payload)
        if isinstance(copy.payload, memoryview):
            copy.payload.release()
        if count_eviction:
            self._telemetry["evictions"] += 1

    def _materialize(self, expert: _Expert) -> memoryview:
        source = expert.source() if callable(expert.source) else expert.source
        try:
            view = memoryview(source).toreadonly()
        except TypeError as exc:
            raise TypeError("expert source must return a bytes-like object") from exc
        if len(view) != expert.size:
            view.release()
            raise ValueError("expert source size changed after registration")
        return view

    def _accelerator_capability(self) -> BackendCapability:
        if self._accelerator is None:
            return BackendCapability(False, "none", reason="no accelerator residency adapter configured")
        try:
            return self._accelerator.capability()
        except Exception as exc:
            return BackendCapability(False, type(self._accelerator).__name__, reason=f"capability probe failed: {exc}")

    def _best_tier(self, expert: _Expert, requested: ResidencyTier) -> ResidencyTier:
        if requested is ResidencyTier.STORAGE:
            return ResidencyTier.STORAGE
        maximum = self._PREFERENCE.index(requested)
        allowed = self._PREFERENCE[maximum:]
        for tier in allowed:
            if tier is ResidencyTier.STORAGE or tier in expert.copies:
                return tier
        return ResidencyTier.STORAGE

    def _touch(self, expert_id: str, tier: ResidencyTier) -> None:
        lru = self._lru[tier]
        if expert_id in lru:
            lru.move_to_end(expert_id)

    def _expert(self, expert_id: str) -> _Expert:
        try:
            return self._experts[expert_id]
        except KeyError as exc:
            raise KeyError(f"unknown expert: {expert_id}") from exc

    def _set_state(
        self,
        expert_id: str,
        tier: ResidencyTier,
        phase: PromotionPhase,
        *,
        started: float = 0.0,
        finished: float = 0.0,
        error: str = "",
        fallback: ResidencyTier = ResidencyTier.STORAGE,
    ) -> None:
        self._states[(expert_id, tier)] = PromotionSnapshot(
            expert_id,
            tier,
            phase,
            started_at=started,
            finished_at=finished,
            error=error,
            fallback_tier=fallback,
        )

    def _clear_pending(self, key: tuple[str, ResidencyTier]) -> None:
        with self._lock:
            self._pending.pop(key, None)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("expert tier manager is closed")

