"""Inference Service — top-level Phase 3 service integrating all components.

``InferenceService`` is the single entry point for production deployment.
It wires together:

1. ``Phase2MultiSessionService``  — shared experts + KV + continuous batching
2. ``Phase3LoadBalancedService``  — routes across multiple Phase 2 backends
3. ``Phase3TelemetryCollector``   — scrapes metrics from every component
4. ``ServingEngine``              — OpenAI-compatible request processing
5. Optional ``Phase3QuantizedSession`` — quantized weight loading
6. Optional ``Phase3TensorParallelSession`` — TP expert dispatch
7. Optional ``Phase3SpecTreeSession``   — tree speculative decoding

Usage::

    service = InferenceService.build(
        checkpoint_id="mixtral-8x7b-v0.1",
        checkpoint_dir=Path("/models/mixtral"),
        forward_fn=my_forward,
    )
    service.start()

    # OpenAI-style call
    response = service.complete({"model": "mixtral", "prompt": "Hello!"})

    # Direct session
    with service.open_session() as sess:
        tokens = sess.generate(context_ids=[1, 2, 3], max_new_tokens=64)

    service.stop()
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .load_balancer import (
    LoadBalancingStrategy,
    Phase3LoadBalancedService,
)
from .telemetry_exporter import Phase3TelemetryCollector, TelemetryRegistry
from .serving_engine import (
    ServingEngine,
    CompletionRequest,
    ChatCompletionRequest,
    CompletionResponse,
    ChatCompletionResponse,
    StreamChunk,
)
from .multi_session_service import Phase2MultiSessionService, create_multi_session_service
from .continuous_batching import SchedulerConfig
from .memory_pressure_manager import MemoryThresholds
from .request_queue import Phase2RequestQueue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InferenceServiceConfig
# ---------------------------------------------------------------------------

@dataclass
class InferenceServiceConfig:
    """All configuration for an InferenceService instance."""

    # Model identity
    model_name:      str   = "openagent"
    checkpoint_id:   str   = "default"
    checkpoint_dir:  Path  = field(default_factory=lambda: Path("."))

    # Phase 2 batching
    scheduler_config: Optional[SchedulerConfig] = None

    # Memory management
    memory_thresholds: Optional[MemoryThresholds] = None

    # Request queue
    max_queue_depth:    int   = 512
    global_rate_rps:    float = 1000.0
    per_session_rate:   float = 50.0

    # Load balancer
    lb_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED

    # Telemetry
    telemetry_interval_s: float = 5.0

    # Feature flags
    enable_quantization: bool = False
    enable_tensor_parallel: bool = False
    enable_spec_tree: bool = False
    tp_world_size: int = 1
    quantization_dtype: str = "int8"

    # Session pool
    min_sessions: int = 2
    max_sessions: int = 16


# ---------------------------------------------------------------------------
# InferenceService
# ---------------------------------------------------------------------------

class InferenceService:
    """Production inference service integrating all Phase 1/2/3 components.

    Lifecycle:
        1. ``build()`` or ``__init__()`` — construct with config
        2. ``start()``                   — start all background threads
        3. Use via ``complete()``, ``chat_complete()``, or ``open_session()``
        4. ``stop()``                    — graceful shutdown
    """

    def __init__(
        self,
        config: InferenceServiceConfig,
        forward_fn: Callable[[Any], Any],
        *,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self._config     = config
        self._forward_fn = forward_fn
        self._tokenizer  = tokenizer
        self._started    = False
        self._stopped    = False
        self._stopping   = False
        self._lock       = threading.Lock()
        self._stop_lock  = threading.Lock()
        self._telemetry_stopped = False
        self._balancer_stopped = False

        # Build request queue (optional; used before routing to Phase 2)
        self._request_queue = Phase2RequestQueue(
            max_size=config.max_queue_depth,
            global_rate=config.global_rate_rps,
            per_session_rate=config.per_session_rate,
        )

        # Build Phase 2 service (single backend; add more via load balancer)
        self._phase2 = Phase2MultiSessionService(
            checkpoint_id=config.checkpoint_id,
            checkpoint_dir=config.checkpoint_dir,
            forward_fn=forward_fn,
            config=config.scheduler_config,
            memory_thresholds=config.memory_thresholds,
            request_queue=self._request_queue,
        )

        # Load balancer (single backend by default; callers can add more)
        self._balancer = Phase3LoadBalancedService(
            strategy=config.lb_strategy,
            health_interval_s=10.0,
        )
        self._balancer.register_backend("primary", self._phase2)

        # Telemetry collector
        self._telemetry = Phase3TelemetryCollector(
            registry=TelemetryRegistry(),
            scrape_prefix="openagent",
        )
        self._telemetry.add_source("phase2", self._phase2.metrics)
        self._telemetry.add_source("queue", self._phase2.request_queue_metrics)
        self._telemetry.add_source("lb", self._balancer.metrics)

        # Serving engine
        self._serving = ServingEngine(
            service=self._balancer,
            tokenizer=tokenizer,
            model_name=config.model_name,
        )
        self._telemetry.add_source("serving", self._serving.metrics)

        logger.info("InferenceService configured: model=%s", config.model_name)

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> "InferenceService":
        with self._lock:
            if self._stopping or self._stopped:
                raise RuntimeError("InferenceService cannot start during or after shutdown")
            if self._started:
                return self
            self._started = True
        self._telemetry.start(interval_s=self._config.telemetry_interval_s)
        logger.info("InferenceService started: %s", self._config.model_name)
        return self

    def stop(self) -> None:
        with self._stop_lock:
            with self._lock:
                if self._stopped:
                    return
                self._stopping = True
            logger.info("InferenceService stopping: %s", self._config.model_name)

            # Telemetry and load-balancer health probes can call backend
            # metrics. Do not close model stores until both observers stop.
            if not self._telemetry_stopped:
                self._telemetry.stop()
                self._telemetry_stopped = True
            if not self._balancer_stopped:
                self._balancer.close()
                self._balancer_stopped = True

            with self._lock:
                self._stopped = True
            logger.info("InferenceService stopped")

    def __enter__(self) -> "InferenceService":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ---- session interface --------------------------------------------------

    def open_session(self, session_id: Optional[str] = None) -> Any:
        """Open a Phase 2 session whose balancer lease lasts until close()."""
        sid = session_id or str(uuid.uuid4())[:8]
        return self._balancer.open_session(session_id=sid)

    # ---- OpenAI-compatible API ----------------------------------------------

    def complete(self, request_dict: dict) -> dict:
        """Process /v1/completions and return response dict."""
        req  = CompletionRequest.from_dict(request_dict)
        resp = self._serving.process_completion(req)
        return resp.to_dict()

    def chat_complete(self, request_dict: dict) -> dict:
        """Process /v1/chat/completions and return response dict."""
        req  = ChatCompletionRequest.from_dict(request_dict)
        resp = self._serving.process_chat_completion(req)
        return resp.to_dict()

    def stream_complete(self, request_dict: dict) -> Iterator[str]:
        """Stream /v1/completions as SSE lines."""
        req = CompletionRequest.from_dict(request_dict)
        req.stream = True
        source = self._serving.stream_completion(req)
        primary: BaseException | None = None
        try:
            for chunk in source:
                yield chunk.to_sse_line()
            yield StreamChunk.done_line()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup = ServingEngine._close_stream_source(source, primary)
            if cleanup is not None:
                raise cleanup

    def stream_chat(self, request_dict: dict) -> Iterator[str]:
        """Stream /v1/chat/completions as SSE lines."""
        req = ChatCompletionRequest.from_dict(request_dict)
        req.stream = True
        source = self._serving.stream_chat_completion(req)
        primary: BaseException | None = None
        try:
            for chunk in source:
                yield chunk.to_sse_line()
            yield StreamChunk.done_line()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup = ServingEngine._close_stream_source(source, primary)
            if cleanup is not None:
                raise cleanup

    # ---- backend management -------------------------------------------------

    def add_backend(self, instance_id: str, service: Any) -> None:
        """Register an additional Phase 2 backend for load balancing."""
        self._balancer.register_backend(instance_id, service)
        self._telemetry.add_source(f"backend_{instance_id}",
                                   getattr(service, "metrics", lambda: {}))

    def remove_backend(self, instance_id: str) -> None:
        self._balancer.deregister_backend(instance_id)
        self._telemetry.remove_source(f"backend_{instance_id}")

    # ---- telemetry ----------------------------------------------------------

    def metrics(self) -> dict:
        return {
            "phase2":   self._phase2.metrics(),
            "lb":       self._balancer.metrics(),
            "serving":  self._serving.metrics(),
            "queue":    self._phase2.request_queue_metrics(),
            "telemetry": self._telemetry.metrics(),
        }

    def prometheus_metrics(self) -> str:
        """Render all metrics in Prometheus text format."""
        self._telemetry.scrape_once()
        return self._telemetry.prometheus_text()

    # ---- factory ------------------------------------------------------------

    @classmethod
    def build(
        cls,
        checkpoint_id: str,
        checkpoint_dir: Path,
        forward_fn: Callable[[Any], Any],
        *,
        model_name: str = "openagent",
        tokenizer: Optional[Any] = None,
        config: Optional[InferenceServiceConfig] = None,
    ) -> "InferenceService":
        """Convenience factory: create and start an InferenceService."""
        cfg = config or InferenceServiceConfig(
            model_name=model_name,
            checkpoint_id=checkpoint_id,
            checkpoint_dir=Path(checkpoint_dir),
        )
        return cls(cfg, forward_fn, tokenizer=tokenizer).start()


__all__ = [
    "InferenceServiceConfig",
    "InferenceService",
]
