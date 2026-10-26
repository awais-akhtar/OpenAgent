"""Phase 1B execution facade over the real sparse-expert runtime.

Phase 1B speedups (batch-union expert loading, asynchronous read-ahead,
learned hot-expert pinning and a persistent routing-history cache) are
implemented by the production expert path, not by this module:

* ``expert_store.routed_swiglu_batch`` leases each distinct expert once per
  token batch (batch-union).
* ``expert_store.DiskExpertStore`` owns bounded asynchronous read-ahead
  workers, demand waits, leases and pinning.
* ``expert_runtime.SparseExpertRuntime`` predicts the next experts from the
  learned route model, prefetches them, rebalances the hot set into a byte
  budget and saves/loads fingerprint-bound route profiles.

This facade binds those mechanisms to one universal session:

1. It stores per-checkpoint state in an isolated directory derived from the
   exact checkpoint identity and route, under the engine's runtime root.
2. It reloads the learned route profile for each runtime on open and saves it
   on close (the learning cache).
3. When given an explicit byte budget, it pins the learned hot set after each
   committed request (just-in-time hot pinning).
4. It reports metrics computed only from the runtime's own counters.  When a
   value was not observed it is reported as ``None``, never estimated.

It starts no threads.  ``close`` saves state, closes the base session exactly
once, raises every cleanup failure, and stays retryable after a failure.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

from .failure_reporting import add_failure_note, failure_detail
from .expert_runtime import SparseExpertRuntime
from .expert_store import DiskExpertStore


PHASE1B_STATE_FORMAT = "openagent.phase1b-checkpoint-state.v1"


def phase1b_state_directory(cache_root: str | Path, base_session: Any) -> Path:
    """Return the isolated Phase 1B state directory for one admitted checkpoint.

    The key covers the resolved checkpoint directory, the exact config digest,
    the six-field route and, when the owner exposes one, the owner's artifact
    digest.  Two checkpoints of one family therefore never share state, and
    nothing is written inside the checkpoint itself.
    """
    manifest = getattr(base_session, "manifest", None)
    if manifest is None or not getattr(manifest, "config_sha256", ""):
        raise ValueError("Phase 1B requires an admitted universal session with checkpoint identity")
    owner = getattr(base_session, "owner", None)
    artifact_identity = getattr(owner, "checkpoint_identity", None)
    digest = hashlib.sha256()
    for part in (PHASE1B_STATE_FORMAT, manifest.model_identity, *manifest.route.as_tuple(),
                 artifact_identity if type(artifact_identity) is str else ""):
        digest.update(part.encode("utf-8") + b"\0")
    root = Path(cache_root).resolve()
    checkpoint_root = Path(manifest.directory).resolve()
    if root == checkpoint_root or checkpoint_root in root.parents:
        raise ValueError("Phase 1B state must not be stored inside the checkpoint directory")
    return root / digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if not denominator else numerator / denominator


class Phase1BOptimizedSession:
    """Bind Phase 1B expert-path controls and real telemetry to one session."""

    def __init__(self, base_session: Any, *, cache_root: str | Path,
                 hot_budget_bytes: int | None = None) -> None:
        if base_session is None:
            raise ValueError("base_session is required")
        if hot_budget_bytes is not None and (type(hot_budget_bytes) is not int or hot_budget_bytes < 0):
            raise ValueError("hot_budget_bytes must be a non-negative integer or None")
        self.base_session = base_session
        self.state_directory = phase1b_state_directory(cache_root, base_session)
        self.hot_budget_bytes = hot_budget_bytes
        self._lock = threading.RLock()
        self._closed = False
        self._base_closed = False
        self._profiles_saved = False
        owner = getattr(base_session, "owner", None)
        sources = getattr(owner, "expert_telemetry_sources", None)
        found = tuple(sources()) if callable(sources) else ()
        for source in found:
            if not isinstance(source, (SparseExpertRuntime, DiskExpertStore)):
                raise TypeError("expert telemetry source must be a SparseExpertRuntime or DiskExpertStore")
        self._runtimes = tuple(source for source in found if isinstance(source, SparseExpertRuntime))
        self._stores = tuple(source for source in found if isinstance(source, DiskExpertStore))
        self.state_directory.mkdir(parents=True, exist_ok=True)
        self.profiles_loaded = 0
        for runtime in self._runtimes:
            path = self._profile_path(runtime)
            if path.is_file():
                runtime.load_route_profile(path)
                self.profiles_loaded += 1

    # -- identity --------------------------------------------------------
    @property
    def manifest(self) -> Any:
        return self.base_session.manifest

    @property
    def expert_streaming(self) -> bool:
        return bool(self._runtimes or self._stores)

    def _profile_path(self, runtime: SparseExpertRuntime) -> Path:
        key = hashlib.sha256((runtime.manifest.model_id + "\0" + runtime.manifest.fingerprint)
                             .encode("utf-8")).hexdigest()[:32]
        return self.state_directory / f"route-profile-{key}.json"

    # -- requests ----------------------------------------------------------
    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Phase 1B session is closed")

    def _after_commit(self) -> None:
        if not self.hot_budget_bytes:
            return
        for runtime in self._runtimes:
            budget = min(self.hot_budget_bytes, runtime.store.capacity_bytes)
            runtime.rebalance_hot(budget_bytes=budget, load=False)

    def generate(self, *args: Any, cancelled: Callable[[], bool] | None = None, **kwargs: Any) -> Any:
        with self._lock:
            self._check_open()
            result = self.base_session.generate(*args, cancelled=cancelled, **kwargs)
            self._after_commit()
            return result

    def stream(self, *args: Any, cancelled: Callable[[], bool] | None = None, **kwargs: Any) -> Iterator[Any]:
        with self._lock:
            self._check_open()
        yield from self.base_session.stream(*args, cancelled=cancelled, **kwargs)
        with self._lock:
            self._after_commit()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.base_session, name)

    # -- telemetry -------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        """Phase 1B metrics computed only from runtime counters."""
        expert_calls = unique_loads = batched_calls = batched_unique = 0
        hits = misses = submitted = prefetch_hits = demand_waits = bytes_read = 0
        read_seconds = 0.0
        pinned = 0
        stores = [runtime.store for runtime in self._runtimes] + list(self._stores)
        for runtime in self._runtimes:
            compute = runtime.status()["compute"]
            expert_calls += compute["expert_calls"]
            unique_loads += compute["unique_expert_loads"]
            batched_calls += compute["batched_expert_calls"]
            batched_unique += compute["batched_unique_expert_loads"]
        seen: set[int] = set()
        for store in stores:
            if id(store) in seen:
                continue
            seen.add(id(store))
            status = store.status()
            hits += status["hits"]
            misses += status["misses"]
            submitted += status["prefetch_submitted"]
            prefetch_hits += status["prefetch_hits"]
            demand_waits += status["demand_waits"]
            bytes_read += status["bytes_read"]
            read_seconds += status["read_seconds"]
            pinned += len(status["pinned_keys"])
        dedup = _ratio(unique_loads, expert_calls)
        batch_dedup = _ratio(batched_unique, batched_calls)
        return {
            "expert_streaming": self.expert_streaming,
            "runtimes": len(self._runtimes),
            "stores": len(seen),
            "expert_route_requests": expert_calls,
            "unique_expert_loads": unique_loads,
            "expert_load_deduplication": None if dedup is None else 1.0 - dedup,
            "batched_expert_route_requests": batched_calls,
            "batched_unique_expert_loads": batched_unique,
            "batch_union_reduction": None if batch_dedup is None else 1.0 - batch_dedup,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": _ratio(hits, hits + misses),
            "async_reads_submitted": submitted,
            "async_read_hits": prefetch_hits,
            "async_read_hit_rate": _ratio(prefetch_hits, submitted),
            "demand_waits": demand_waits,
            "bytes_read": bytes_read,
            "read_mib_s": (bytes_read / (1024 * 1024) / read_seconds) if read_seconds > 0 else None,
            "pinned_experts": pinned,
            "route_profiles_loaded": self.profiles_loaded,
            "state_directory": str(self.state_directory),
        }

    # -- lifecycle -------------------------------------------------------------
    def close(self) -> None:
        """Save learned routes, then close the base session once.

        Every failure is raised (the first as primary, the rest as notes).
        Steps that already succeeded are not repeated on retry.
        """
        with self._lock:
            if self._closed:
                return
            primary: BaseException | None = None

            def record(error: BaseException, label: str) -> None:
                nonlocal primary
                if primary is None:
                    primary = error
                else:
                    add_failure_note(primary, label + ": " + failure_detail(error))

            if not self._profiles_saved:
                saved = True
                for runtime in self._runtimes:
                    try:
                        runtime.save_route_profile(self._profile_path(runtime))
                    except BaseException as error:
                        saved = False
                        record(error, "additional route-profile save failure")
                self._profiles_saved = saved
            if primary is None and not self._base_closed:
                try:
                    self.base_session.close()
                    self._base_closed = True
                except BaseException as error:
                    record(error, "base session close failure")
            if primary is not None:
                raise primary
            self._closed = True

    def status(self) -> dict[str, Any]:
        base = self.base_session.status()
        with self._lock:
            return {**base, "phase1b": {"closed": self._closed, "base_closed": self._base_closed,
                                        "profiles_saved": self._profiles_saved,
                                        "hot_budget_bytes": self.hot_budget_bytes}}

    def __enter__(self) -> "Phase1BOptimizedSession":
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def wrap_universal_session_with_phase1b(base_session: Any, *, cache_root: str | Path,
                                        hot_budget_bytes: int | None = None) -> Phase1BOptimizedSession:
    """Bind Phase 1B controls and telemetry to an admitted universal session."""
    return Phase1BOptimizedSession(base_session, cache_root=cache_root,
                                   hot_budget_bytes=hot_budget_bytes)


__all__ = ["PHASE1B_STATE_FORMAT", "Phase1BOptimizedSession", "phase1b_state_directory",
           "wrap_universal_session_with_phase1b"]
