"""Load Balancer — routes inference requests across multiple service instances.

Distributes ``BatchRequest`` (or raw token generation calls) across multiple
``Phase2MultiSessionService`` instances running on the same machine or cluster.

Components
----------
``ServiceInstanceState``  -- health, latency EMA, in-flight request count
``LoadBalancingStrategy`` -- ROUND_ROBIN / LEAST_LOADED / LATENCY_AWARE / RANDOM
``HealthCheckResult``     -- probe outcome with latency measurement
``ServiceRegistry``       -- thread-safe registry of backend instances
``LoadBalancer``          -- primary routing class; selects an instance per request
``LoadBalancerProxy``     -- context manager that tracks a borrow; auto-returns
``Phase3LoadBalancedService`` -- top-level facade matching Phase2 session interface
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

from .failure_reporting import add_failure_note, failure_detail

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoadBalancingStrategy
# ---------------------------------------------------------------------------

class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN    = "round_robin"
    LEAST_LOADED   = "least_loaded"
    LATENCY_AWARE  = "latency_aware"
    RANDOM         = "random"


# ---------------------------------------------------------------------------
# ServiceInstanceState
# ---------------------------------------------------------------------------

@dataclass
class ServiceInstanceState:
    """Live health and performance state for one backend instance."""
    instance_id:    str
    service:        Any                # Phase2MultiSessionService or duck-type
    healthy:        bool  = True
    draining:       bool  = False
    in_flight:      int   = 0
    probes_in_flight: int = 0
    latency_ema_ms: float = 0.0        # exponential moving average of latency
    total_requests: int   = 0
    total_errors:   int   = 0
    last_health_ts: float = field(default_factory=time.monotonic)
    ema_alpha:      float = 0.1        # EMA smoothing factor

    def record_latency(self, latency_ms: float) -> None:
        if self.latency_ema_ms == 0.0:
            self.latency_ema_ms = latency_ms
        else:
            self.latency_ema_ms = (
                self.ema_alpha * latency_ms +
                (1.0 - self.ema_alpha) * self.latency_ema_ms
            )
        self.total_requests += 1

    def record_error(self) -> None:
        self.total_errors += 1

    @property
    def error_rate(self) -> float:
        return self.total_errors / max(1, self.total_requests)

    @property
    def load_score(self) -> float:
        """Lower is better: balances in-flight count with latency."""
        return self.in_flight + self.latency_ema_ms / 1000.0


# ---------------------------------------------------------------------------
# HealthCheckResult
# ---------------------------------------------------------------------------

@dataclass
class HealthCheckResult:
    instance_id: str
    healthy:     bool
    latency_ms:  float
    error:       Optional[str] = None


# ---------------------------------------------------------------------------
# ServiceRegistry
# ---------------------------------------------------------------------------

class ServiceRegistry:
    """Thread-safe registry of backend service instances."""

    def __init__(self) -> None:
        self._instances: Dict[str, ServiceInstanceState] = {}
        self._lock = threading.RLock()

    def register(self, instance_id: str, service: Any) -> None:
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("nonempty backend instance id required")
        with self._lock:
            if instance_id in self._instances:
                raise ValueError(f"backend instance already registered: {instance_id}")
            self._instances[instance_id] = ServiceInstanceState(
                instance_id=instance_id, service=service)
        logger.info("lb: registered instance %s", instance_id)

    def deregister(self, instance_id: str) -> None:
        with self._lock:
            state = self._instances.get(instance_id)
            if state is None:
                return
            if state.in_flight or state.probes_in_flight:
                raise RuntimeError("cannot deregister backend with active leases or probes")
            self._instances.pop(instance_id)
        logger.info("lb: deregistered instance %s", instance_id)

    def begin_drain(self, instance_id: str) -> Optional[ServiceInstanceState]:
        with self._lock:
            state = self._instances.get(instance_id)
            if state is not None:
                state.draining = True
            return state

    def healthy_instances(self) -> List[ServiceInstanceState]:
        with self._lock:
            return [s for s in self._instances.values() if s.healthy and not s.draining]

    def all_instances(self) -> List[ServiceInstanceState]:
        with self._lock:
            return list(self._instances.values())

    def get(self, instance_id: str) -> Optional[ServiceInstanceState]:
        with self._lock:
            return self._instances.get(instance_id)

    def probe(
        self,
        instance_id: str,
        timeout_s: float = 2.0,
    ) -> HealthCheckResult:
        """Send a lightweight health probe to an instance."""
        with self._lock:
            state = self._instances.get(instance_id)
            if state is None:
                return HealthCheckResult(instance_id, False, 0.0, "not registered")
            if state.draining:
                return HealthCheckResult(instance_id, False, 0.0, "backend is draining")
            state.probes_in_flight += 1
        t0 = time.monotonic()
        try:
            metrics_fn = getattr(state.service, "metrics", None)
            if callable(metrics_fn):
                metrics_fn()
            latency_ms = (time.monotonic() - t0) * 1000.0
            with self._lock:
                if not state.draining:
                    state.healthy = True
                    state.last_health_ts = time.monotonic()
                    state.record_latency(latency_ms)
            return HealthCheckResult(instance_id, True, latency_ms)
        except Exception as exc:
            with self._lock:
                state.healthy = False
                state.record_error()
            return HealthCheckResult(instance_id, False, 0.0, str(exc))
        finally:
            with self._lock:
                state.probes_in_flight = max(0, state.probes_in_flight - 1)


# ---------------------------------------------------------------------------
# LoadBalancer
# ---------------------------------------------------------------------------

class LoadBalancer:
    """Selects a backend instance for each incoming request.

    Args:
        registry:    ``ServiceRegistry`` to select from.
        strategy:    Routing strategy.
        health_interval_s: How often to background-probe all instances.
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
        health_interval_s: float = 10.0,
    ) -> None:
        self._registry = registry
        self._strategy = strategy
        self._rr_index = 0
        self._lock     = threading.Lock()

        # Background health checker
        self._stop_ev      = threading.Event()
        self._health_thread = threading.Thread(
            target=self._health_loop,
            daemon=True,
            name="lb-health-check",
        )
        self._health_interval_s = health_interval_s
        self._health_thread.start()

        # Telemetry
        self._total_routed   = 0
        self._no_instance    = 0
        self._strategy_calls: Dict[str, int] = {s.value: 0 for s in LoadBalancingStrategy}

    # ---- routing ------------------------------------------------------------

    def select(self) -> Optional[ServiceInstanceState]:
        """Choose an instance using the configured strategy."""
        instances = self._registry.healthy_instances()
        if not instances:
            with self._lock:
                self._no_instance += 1
            return None

        with self._lock:
            self._total_routed += 1
            self._strategy_calls[self._strategy.value] += 1

            if self._strategy == LoadBalancingStrategy.ROUND_ROBIN:
                state = instances[self._rr_index % len(instances)]
                self._rr_index += 1
                return state

            if self._strategy == LoadBalancingStrategy.RANDOM:
                return random.choice(instances)

            if self._strategy == LoadBalancingStrategy.LEAST_LOADED:
                return min(instances, key=lambda s: s.in_flight)

            if self._strategy == LoadBalancingStrategy.LATENCY_AWARE:
                return min(instances, key=lambda s: s.load_score)

            return instances[0]

    def borrow(self) -> "LoadBalancerProxy":
        """RAII borrow of an instance; records in-flight on acquire, releases on done."""
        state = self.select()
        return LoadBalancerProxy(state)

    # ---- health loop --------------------------------------------------------

    def _health_loop(self) -> None:
        while not self._stop_ev.wait(timeout=self._health_interval_s):
            for inst in self._registry.all_instances():
                try:
                    self._registry.probe(inst.instance_id)
                except Exception as exc:
                    logger.debug("lb: health probe error for %s: %s",
                                 inst.instance_id, exc)

    def stop(self) -> None:
        self._stop_ev.set()
        self._health_thread.join(timeout=5.0)
        if self._health_thread.is_alive():
            raise TimeoutError("load-balancer health checker did not stop before deadline")

    # ---- telemetry ----------------------------------------------------------

    def metrics(self) -> dict:
        with self._lock:
            instances = self._registry.all_instances()
            return {
                "lb_strategy":        self._strategy.value,
                "lb_total_routed":    self._total_routed,
                "lb_no_instance":     self._no_instance,
                "lb_healthy_count":   len(self._registry.healthy_instances()),
                "lb_total_count":     len(instances),
                "lb_strategy_calls":  dict(self._strategy_calls),
                "lb_instances": [
                    {
                        "id":          s.instance_id,
                        "healthy":     s.healthy,
                        "in_flight":   s.in_flight,
                        "latency_ema_ms": round(s.latency_ema_ms, 2),
                        "error_rate":  round(s.error_rate, 4),
                    }
                    for s in instances
                ],
            }


# ---------------------------------------------------------------------------
# LoadBalancerProxy — RAII in-flight tracker
# ---------------------------------------------------------------------------

class LoadBalancerProxy:
    """Context manager that increments in_flight on enter, decrements on exit."""

    def __init__(self, state: Optional[ServiceInstanceState]) -> None:
        self._state = state
        self._lock = threading.Lock()
        self._done = False
        if state is not None:
            state.in_flight += 1

    @property
    def service(self) -> Optional[Any]:
        return self._state.service if self._state else None

    @property
    def available(self) -> bool:
        return self._state is not None

    def done(self, latency_ms: float = 0.0, error: bool = False) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
            if self._state is None:
                return
            self._state.in_flight = max(0, self._state.in_flight - 1)
            if latency_ms > 0:
                self._state.record_latency(latency_ms)
            if error:
                self._state.record_error()

    def __enter__(self) -> "LoadBalancerProxy":
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        self.done(error=exc_type is not None)


class _LoadBalancedSession:
    """Keep a backend lease alive for exactly the lifetime of its session."""

    def __init__(self, session: Any, proxy: LoadBalancerProxy,
                 on_close_failure, on_close_success) -> None:
        self._session = session
        self._proxy = proxy
        self._on_close_failure = on_close_failure
        self._on_close_success = on_close_success
        self._lock = threading.Lock()
        self._closed = False
        self._close_failed = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._session, name)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            close_fn = getattr(self._session, "close", None)
            try:
                if callable(close_fn):
                    close_fn()
            except BaseException:
                self._close_failed = True
                self._on_close_failure(self)
                raise
            self._closed = True
            self._proxy.done(error=self._close_failed)
            self._on_close_success(self)

    def __enter__(self) -> "_LoadBalancedSession":
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Phase3LoadBalancedService — high-level facade
# ---------------------------------------------------------------------------

class Phase3LoadBalancedService:
    """Routes ``open_session`` calls across registered backend instances.

    Callers use this exactly like a single ``Phase2MultiSessionService``:
    call ``open_session()`` to get a handle; the balancer picks which backend
    serves it.
    """

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
        health_interval_s: float = 10.0,
    ) -> None:
        self._registry = ServiceRegistry()
        self._balancer = LoadBalancer(
            self._registry, strategy, health_interval_s)
        self._closed = False
        self._lock   = threading.Lock()
        self._close_lock = threading.Lock()
        self._health_stopped = False
        self._closed_backends: set[str] = set()
        self._retired_sessions: Dict[int, _LoadBalancedSession] = {}

    def register_backend(self, instance_id: str, service: Any) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Phase3LoadBalancedService is closed")
            self._registry.register(instance_id, service)

    def deregister_backend(self, instance_id: str, timeout: float = 5.0) -> None:
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or
                timeout < 0):
            raise ValueError("finite nonnegative backend drain timeout required")
        with self._close_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Phase3LoadBalancedService is closed")
                state = self._registry.begin_drain(instance_id)
            if state is None:
                return

            deadline = time.monotonic() + timeout
            while state.in_flight or state.probes_in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"backend {instance_id} drain timed out with active leases or probes"
                    )
                time.sleep(min(0.01, remaining))

            close_fn = getattr(state.service, "close", None)
            if callable(close_fn):
                close_fn()
            self._registry.deregister(instance_id)

    def open_session(self, **kwargs: Any) -> Any:
        """Open a session on the best available backend instance."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Phase3LoadBalancedService is closed")
            # Reserve the lease inside the close gate so shutdown cannot
            # observe zero in-flight and race a new session open.
            proxy = self._balancer.borrow()
        if not proxy.available or proxy.service is None:
            raise RuntimeError("No healthy backend available")
        try:
            open_fn = getattr(proxy.service, "open_session", None)
            if not callable(open_fn):
                raise RuntimeError("Backend service has no open_session method")
            session = open_fn(**kwargs)
        except BaseException:
            proxy.done(error=True)
            raise
        return _LoadBalancedSession(
            session, proxy, self._retain_failed_session, self._forget_retired_session
        )

    def _retain_failed_session(self, session: _LoadBalancedSession) -> None:
        with self._lock:
            self._retired_sessions[id(session)] = session

    def _forget_retired_session(self, session: _LoadBalancedSession) -> None:
        with self._lock:
            self._retired_sessions.pop(id(session), None)

    def borrow(self) -> LoadBalancerProxy:
        with self._lock:
            if self._closed:
                raise RuntimeError("Phase3LoadBalancedService is closed")
            return self._balancer.borrow()

    def probe_all(self) -> List[HealthCheckResult]:
        results = []
        for inst in self._registry.all_instances():
            results.append(self._registry.probe(inst.instance_id))
        return results

    def metrics(self) -> dict:
        return self._balancer.metrics()

    def close(self, timeout: float = 5.0) -> None:
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or
                timeout < 0):
            raise ValueError("finite nonnegative load-balancer close timeout required")
        with self._close_lock:
            with self._lock:
                self._closed = True
            if not self._health_stopped:
                self._balancer.stop()
                self._health_stopped = True

            # Existing sessions own leases until their close() completes.
            # Never tear down a backend while one of those callers still owns
            # it; timeout leaves the facade closed to new work but retryable.
            deadline = time.monotonic() + timeout
            while True:
                with self._lock:
                    retired = tuple(self._retired_sessions.values())
                cleanup_failure: BaseException | None = None
                for session in retired:
                    try:
                        session.close()
                    except BaseException as exc:
                        if cleanup_failure is None:
                            cleanup_failure = exc
                        else:
                            add_failure_note(
                                cleanup_failure,
                                "retired session close failed: " + failure_detail(exc),
                            )
                if cleanup_failure is not None:
                    add_failure_note(
                        cleanup_failure,
                        "backend services retained because session cleanup failed; retry close()",
                    )
                    raise cleanup_failure
                active = sum(
                    state.in_flight for state in self._registry.all_instances()
                )
                if active == 0:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"load balancer shutdown timed out with {active} active session lease(s)"
                    )
                time.sleep(min(0.01, remaining))

            primary: BaseException | None = None
            for inst in self._registry.all_instances():
                if inst.instance_id in self._closed_backends:
                    continue
                close_fn = getattr(inst.service, "close", None)
                if not callable(close_fn):
                    self._closed_backends.add(inst.instance_id)
                    continue
                try:
                    close_fn()
                    self._closed_backends.add(inst.instance_id)
                except BaseException as exc:
                    if primary is None:
                        primary = exc
                    else:
                        add_failure_note(
                            primary,
                            f"backend {inst.instance_id} close failed: {failure_detail(exc)}",
                        )
            if primary is not None:
                raise primary

    def __enter__(self) -> "Phase3LoadBalancedService":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "LoadBalancingStrategy",
    "ServiceInstanceState",
    "HealthCheckResult",
    "ServiceRegistry",
    "LoadBalancer",
    "LoadBalancerProxy",
    "Phase3LoadBalancedService",
]
