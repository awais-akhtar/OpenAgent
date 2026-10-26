"""Phase 1E integration bridge: AOT kernels + AVX-512 + Intel Arc GPU path.

Wraps a ``Phase1DOptimizedSession`` with Phase 1E capabilities:

1. Replaces ``Phase1DKernelDispatch`` with ``Phase1EKernelDispatch`` which uses
   the AVX shared-library fast path, Numba AOT precompile, and Intel Arc GPU.

2. Exposes ``register_arc_expert`` so the caller can optionally upload expert
   weights to GPU memory at load time.

3. Triggers AOT precompile in the background on session open.

Session stack after Phase 1E::

    base_session
      └── Phase1BOptimizedSession
            └── Phase1COptimizedSession
                  └── Phase1DOptimizedSession
                        └── Phase1EOptimizedSession   ← this module

The same generate / stream / close / metrics interface is preserved.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

import logging
import numpy as np

from .failure_reporting import add_failure_note, failure_detail
from .aot_kernel_cache import (
    Phase1EKernelDispatch,
)

logger = logging.getLogger(__name__)

PHASE1E_STATE_FORMAT = "openagent.phase1e-session-state.v1"


class Phase1EOptimizedSession:
    """Wrap a Phase 1D session with Phase 1E kernel acceleration.

    Args:
        phase1d_session: A ``Phase1DOptimizedSession`` (or duck-typed equiv).
        enable_arc_gpu: Enable Intel Arc GPU expert dispatch.
        aot_cache_dir: Override AOT kernel cache directory.
    """

    def __init__(
        self,
        phase1d_session: Any,
        *,
        enable_arc_gpu: bool = True,
        aot_cache_dir: Optional[Path] = None,
    ) -> None:
        if phase1d_session is None:
            raise ValueError("phase1d_session is required")

        self.base_session = phase1d_session
        self._lock   = threading.RLock()
        self._operation_lock = threading.RLock()
        self._closed = False

        self.kernels = Phase1EKernelDispatch(
            enable_arc_gpu=enable_arc_gpu,
            aot_cache_dir=aot_cache_dir,
        )
        logger.info("phase1e: session opened (arc_gpu=%s)", enable_arc_gpu)

    # -- expert GPU upload (optional) -----------------------------------------

    def register_arc_expert(
        self,
        expert_id: str,
        gate_bytes: bytes,
        up_bytes: bytes,
        down_bytes: bytes,
        d_model: int,
        d_ff: int,
        dtype: np.dtype = np.float16,
    ) -> None:
        """Upload an expert's weights to Arc GPU VRAM for zero-copy dispatch."""
        self.kernels.register_arc_expert(
            expert_id, gate_bytes, up_bytes, down_bytes,
            d_model=d_model, d_ff=d_ff, dtype=dtype,
        )

    # -- routing hook ----------------------------------------------------------

    def on_token_routed(self, layer_id: int, expert_ids: List[str]) -> None:
        """Forward to Phase 1D routing hook."""
        hook = getattr(self.base_session, "on_token_routed", None)
        if callable(hook):
            hook(layer_id, expert_ids)

    # -- generation ------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Phase 1E session is closed")

    def generate(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                  **kwargs: Any) -> Any:
        with self._operation_lock:
            with self._lock:
                self._check_open()
            return self.base_session.generate(*args, cancelled=cancelled, **kwargs)

    def stream(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                **kwargs: Any) -> Iterator[Any]:
        with self._operation_lock:
            with self._lock:
                self._check_open()
            yield from self.base_session.stream(*args, cancelled=cancelled, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.base_session, name)

    # -- telemetry -------------------------------------------------------------

    def metrics(self) -> dict:
        base: dict = {}
        get_m = getattr(self.base_session, "metrics", None)
        if callable(get_m):
            base = get_m()
        return {**base, "phase1e": self.kernels.stats()}

    def status(self) -> dict:
        base: dict = {}
        get_s = getattr(self.base_session, "status", None)
        if callable(get_s):
            base = get_s()
        with self._lock:
            return {**base, "phase1e": {
                "closed": self._closed,
                **self.kernels.stats(),
            }}

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        with self._operation_lock:
            self._close_once()

    def _close_once(self) -> None:
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

            try:
                self.base_session.close()
            except BaseException as err:
                record(err, "phase1d session close failure")

            if primary is not None:
                raise primary
            self._closed = True

    def __enter__(self) -> "Phase1EOptimizedSession":
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def wrap_phase1d_with_phase1e(
    phase1d_session: Any,
    *,
    enable_arc_gpu: bool = True,
    aot_cache_dir: Optional[Path] = None,
) -> Phase1EOptimizedSession:
    """Bind Phase 1E capabilities to an existing Phase 1D session."""
    return Phase1EOptimizedSession(
        phase1d_session,
        enable_arc_gpu=enable_arc_gpu,
        aot_cache_dir=aot_cache_dir,
    )


__all__ = [
    "PHASE1E_STATE_FORMAT",
    "Phase1EOptimizedSession",
    "wrap_phase1d_with_phase1e",
]
