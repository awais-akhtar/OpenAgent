"""Phase 1C integration bridge: extended prefetch + model kernels + direct I/O.

This module is the single entry point for Phase 1C.  It wraps a
``Phase1BOptimizedSession`` (or any compatible session) with three Phase 1C
capabilities:

1. **Extended 3+ layer lookahead prefetch with routing affinity clustering**
   (``phase1c_extended_prefetch.ExtendedPrefetchController``).

2. **Model-specific FFN kernels** for GLM, Qwen shared experts, and DeepSeek
   MLA dispatch (``phase1c_model_kernels.Phase1CKernelRegistry``).

3. **Direct / unbuffered I/O** for cold expert weight reads when the underlying
   store exposes a ``set_reader`` hook
   (``phase1c_direct_io.DirectIOReader``).

The same generate/stream/close interface as Phase 1B is preserved.  Metrics
from all three Phase 1C sub-systems are merged into a single ``metrics()`` dict
keyed under ``"phase1c"``.

Lifecycle
---------
``Phase1COptimizedSession`` is opened once per inference session:

1. It delegates to the wrapped Phase 1B session for actual generation.
2. After each committed request it calls
   ``ExtendedPrefetchController.on_layer_complete`` for every layer trace
   reported by the session, submitting 3+ layer look-ahead hints to the
   expert store.
3. On ``close()`` it saves the accumulated affinity cluster map alongside
   the Phase 1B route profile and then closes the Phase 1B session.

Route traces are consumed via an optional ``route_trace_hook`` the caller
may register on the base session.  When no hook is available the Phase 1C
prefetch degrades gracefully: affinity data accumulates but hints cannot
be submitted until the session emits observable route events.

This module starts no threads.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple

import logging

from .failure_reporting import add_failure_note, failure_detail
from .affinity_prefetch import (
    AffinityClusterMap,
    ExtendedPrefetchController,
    affinity_map_path,
    PHASE1C_PREFETCH_DEPTH_DEFAULT,
)
from .model_kernel_registry import Phase1CKernelRegistry
from .direct_io_reader import DirectIOReader

logger = logging.getLogger(__name__)

PHASE1C_STATE_FORMAT = "openagent.phase1c-session-state.v1"


# ---------------------------------------------------------------------------
# Phase1COptimizedSession
# ---------------------------------------------------------------------------

class Phase1COptimizedSession:
    """Wrap a Phase 1B session with Phase 1C capabilities.

    Args:
        phase1b_session: A ``Phase1BOptimizedSession`` (or duck-typed equivalent).
        prefetch_depth: Number of layers ahead to prefetch (>= 3 for Phase 1C).
        enable_direct_io: Use O_DIRECT / Windows unbuffered reads for cold experts.
        model_dir: Root directory of the model checkpoint (needed for DirectIOReader).
    """

    def __init__(
        self,
        phase1b_session: Any,
        *,
        prefetch_depth: int = PHASE1C_PREFETCH_DEPTH_DEFAULT,
        enable_direct_io: bool = True,
        model_dir: Optional[Path] = None,
    ) -> None:
        if phase1b_session is None:
            raise ValueError("phase1b_session is required")
        if prefetch_depth < 1:
            raise ValueError("prefetch_depth must be >= 1")

        self.base_session = phase1b_session
        self._prefetch_depth = prefetch_depth
        self._lock = threading.RLock()
        self._closed = False
        self._phase1c_saved = False

        # --- Phase 1C sub-systems ---

        # 1. Extended prefetch + affinity clustering
        state_dir: Optional[Path] = getattr(phase1b_session, "state_directory", None)
        manifest = getattr(phase1b_session, "manifest", None)
        model_id = (getattr(manifest, "model_identity", None) or
                    getattr(manifest, "model_id", None) or "unknown")

        self._affinity_map = AffinityClusterMap()
        self._affinity_path: Optional[Path] = None
        if state_dir is not None:
            self._affinity_path = affinity_map_path(state_dir, model_id)
            if self._affinity_path.exists():
                self._affinity_map = AffinityClusterMap.load(self._affinity_path)
                logger.info("phase1c: loaded affinity map from %s", self._affinity_path)

        self._prefetch_ctrl = ExtendedPrefetchController(
            affinity_map=self._affinity_map,
            depth=prefetch_depth,
        )

        # 2. Model-specific kernels
        self.kernels = Phase1CKernelRegistry()

        # 3. Direct I/O reader
        self._direct_io: Optional[DirectIOReader] = None
        if enable_direct_io and model_dir is not None:
            self._direct_io = DirectIOReader(Path(model_dir))
            # Wire to store if it exposes set_reader
            stores = getattr(phase1b_session, "_stores", ())
            for store in stores:
                set_reader = getattr(store, "set_direct_io_reader", None)
                if callable(set_reader):
                    set_reader(self._direct_io)
                    logger.info("phase1c: direct I/O reader wired to %s", type(store).__name__)

        # Accumulated route trace for the current forward pass
        self._current_trace: List[Tuple[int, List[str]]] = []

    # -- route trace hook (called by production session if it supports it) ----

    def on_layer_complete(self, layer_id: int, expert_ids: List[str]) -> None:
        """Consume one layer's routing decision.

        Call this hook after each transformer layer's forward pass if the model
        family exposes routing decisions.  The hook is optional: if never called
        the prefetch controller accumulates no data and submits no hints.
        """
        with self._lock:
            self._current_trace.append((layer_id, list(expert_ids)))
            stores = list(getattr(self.base_session, "_stores", ()))
            store = stores[0] if stores else None

        self._prefetch_ctrl.on_layer_complete(
            layer_id=layer_id,
            fired_experts=expert_ids,
            all_layer_traces=list(self._current_trace),
            store=store,
        )

    def _reset_trace(self) -> None:
        with self._lock:
            self._current_trace.clear()

    # -- generation ------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Phase 1C session is closed")

    def generate(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                  **kwargs: Any) -> Any:
        with self._lock:
            self._check_open()
            self._reset_trace()
        result = self.base_session.generate(*args, cancelled=cancelled, **kwargs)
        return result

    def stream(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                **kwargs: Any) -> Iterator[Any]:
        with self._lock:
            self._check_open()
            self._reset_trace()
        yield from self.base_session.stream(*args, cancelled=cancelled, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.base_session, name)

    # -- telemetry -------------------------------------------------------------

    def metrics(self) -> dict:
        """Return merged Phase 1B + Phase 1C metrics."""
        base = {}
        get_metrics = getattr(self.base_session, "metrics", None)
        if callable(get_metrics):
            base = get_metrics()

        phase1c: dict = {
            "prefetch_depth": self._prefetch_depth,
            **self._prefetch_ctrl.metrics(),
            **self.kernels.stats(),
        }
        if self._direct_io is not None:
            phase1c.update(self._direct_io.stats())

        return {**base, "phase1c": phase1c}

    def status(self) -> dict:
        base = {}
        get_status = getattr(self.base_session, "status", None)
        if callable(get_status):
            base = get_status()
        with self._lock:
            return {**base, "phase1c": {
                "closed": self._closed,
                "phase1c_saved": self._phase1c_saved,
                "prefetch_depth": self._prefetch_depth,
                "affinity_path": str(self._affinity_path) if self._affinity_path else None,
                "direct_io_enabled": self._direct_io is not None,
            }}

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Save Phase 1C state, then close the Phase 1B session."""
        with self._lock:
            if self._closed:
                return
            primary: BaseException | None = None

            def record(err: BaseException, label: str) -> None:
                nonlocal primary
                if primary is None:
                    primary = err
                else:
                    add_failure_note(primary, label + ": " + failure_detail(err))

            if not self._phase1c_saved:
                saved = True
                if self._affinity_path is not None:
                    try:
                        self._affinity_map.save(self._affinity_path)
                        logger.info("phase1c: affinity map saved → %s", self._affinity_path)
                    except BaseException as err:
                        saved = False
                        record(err, "phase1c affinity map save failure")
                self._phase1c_saved = saved

            if primary is None:
                try:
                    self.base_session.close()
                except BaseException as err:
                    record(err, "phase1b session close failure")

            if primary is not None:
                raise primary
            self._closed = True

    def __enter__(self) -> "Phase1COptimizedSession":
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def wrap_phase1b_with_phase1c(
    phase1b_session: Any,
    *,
    prefetch_depth: int = PHASE1C_PREFETCH_DEPTH_DEFAULT,
    enable_direct_io: bool = True,
    model_dir: Optional[Path] = None,
) -> Phase1COptimizedSession:
    """Bind Phase 1C capabilities to an existing Phase 1B session.

    Args:
        phase1b_session: An admitted ``Phase1BOptimizedSession``.
        prefetch_depth: Layers ahead to prefetch (Phase 1C default: 3).
        enable_direct_io: Enable O_DIRECT / Windows unbuffered reads.
        model_dir: Checkpoint directory (required for direct I/O).

    Returns:
        A ``Phase1COptimizedSession`` that enhances the Phase 1B session.
    """
    return Phase1COptimizedSession(
        phase1b_session,
        prefetch_depth=prefetch_depth,
        enable_direct_io=enable_direct_io,
        model_dir=model_dir,
    )


__all__ = [
    "PHASE1C_STATE_FORMAT",
    "Phase1COptimizedSession",
    "wrap_phase1b_with_phase1c",
]
