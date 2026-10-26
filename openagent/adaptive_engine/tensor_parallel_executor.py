"""Phase 3 — Tensor Parallelism: split model computation across multiple ranks.

Supports splitting large MoE expert weight matrices across N worker processes
(or threads) so that no single process holds the full weight tensor in memory.

Components
----------
``TensorParallelConfig``      -- rank count, backend, and chunk strategy
``TensorParallelCommunicator``-- allreduce / scatter / gather over shared memory
                                 or multiprocessing queues; falls back to no-op
                                 when world_size == 1
``ColumnParallelLinear``      -- splits output dimension across ranks; each rank
                                 computes a column shard; allreduce not needed
``RowParallelLinear``         -- splits input dimension across ranks; each rank
                                 computes a partial sum; allreduce combines them
``TensorParallelExpertShard`` -- wraps one expert's (gate, up, down) matrices for
                                 one rank; handles local forward + reduction
``Phase3TensorParallelSession``-- wraps a base session (Phase 1 or 2); intercepts
                                 ``on_expert_dispatch`` to route each expert's
                                 computation through the shard assigned to this rank
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TensorParallelConfig
# ---------------------------------------------------------------------------

class TensorParallelBackend(str, Enum):
    SHARED_MEM   = "shared_mem"    # multiprocessing shared memory (same machine)
    QUEUE        = "queue"          # multiprocessing Queue (fallback)
    LOCAL        = "local"          # single process, no IPC (world_size == 1)


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism."""
    world_size:  int   = 1             # total number of parallel ranks
    rank:        int   = 0             # this process's rank (0-indexed)
    backend:     TensorParallelBackend = TensorParallelBackend.LOCAL
    dtype:       str   = "float32"     # numpy dtype for weight shards
    chunk_dim:   str   = "column"      # "column" or "row" split for FFN layers

    def __post_init__(self) -> None:
        if self.rank >= self.world_size:
            raise ValueError(
                f"rank {self.rank} >= world_size {self.world_size}")
        if self.world_size == 1:
            self.backend = TensorParallelBackend.LOCAL


# ---------------------------------------------------------------------------
# TensorParallelCommunicator
# ---------------------------------------------------------------------------

class TensorParallelCommunicator:
    """Handles allreduce and scatter/gather across TP ranks.

    When ``world_size == 1`` every operation is a no-op (identity transform).
    In a real multi-process deployment, replace ``_allreduce_fn`` with an
    NCCL or gloo collective; this class provides the interface and a
    correct single-process fallback.
    """

    def __init__(
        self,
        config: TensorParallelConfig,
        allreduce_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self._config = config
        self._allreduce_fn = allreduce_fn   # injected by multi-process launcher
        self._lock = threading.Lock()
        self._ops_count = 0

    def allreduce(self, tensor: np.ndarray) -> np.ndarray:
        """Sum tensor across all ranks; returns the global sum on every rank."""
        with self._lock:
            self._ops_count += 1
        if self._config.world_size == 1 or self._allreduce_fn is None:
            return tensor
        return self._allreduce_fn(tensor)

    def scatter(
        self, tensor: np.ndarray, dim: int = 0
    ) -> np.ndarray:
        """Return this rank's shard of tensor along dim."""
        if self._config.world_size == 1:
            return tensor
        n = tensor.shape[dim]
        shard_size = (n + self._config.world_size - 1) // self._config.world_size
        start = self._config.rank * shard_size
        end   = min(start + shard_size, n)
        idx = [slice(None)] * tensor.ndim
        idx[dim] = slice(start, end)
        return tensor[tuple(idx)]

    def gather(
        self, shards: List[np.ndarray], dim: int = 0
    ) -> np.ndarray:
        """Concatenate shards from all ranks along dim."""
        return np.concatenate(shards, axis=dim)

    def metrics(self) -> dict:
        with self._lock:
            return {
                "tp_world_size": self._config.world_size,
                "tp_rank":       self._config.rank,
                "tp_backend":    self._config.backend.value,
                "tp_allreduce_ops": self._ops_count,
            }


# ---------------------------------------------------------------------------
# ColumnParallelLinear / RowParallelLinear
# ---------------------------------------------------------------------------

class ColumnParallelLinear:
    """Linear layer split along output dimension (column parallel).

    Each rank owns columns [rank * shard : (rank+1) * shard] of the weight
    matrix.  No allreduce needed; outputs are gathered when required.

    Shape: weight (in_features, full_out_features) → shard (in_features, out_shard)
    """

    def __init__(
        self,
        weight: np.ndarray,         # full weight (in_features, out_features)
        bias: Optional[np.ndarray],
        comm: TensorParallelCommunicator,
    ) -> None:
        self._comm  = comm
        self._bias  = bias
        # Scatter along output (column) dimension
        self._weight_shard = comm.scatter(weight, dim=1)
        self._out_features_shard = self._weight_shard.shape[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (..., in_features) → (..., out_shard)"""
        out = x @ self._weight_shard
        if self._bias is not None:
            bias_shard = self._comm.scatter(self._bias, dim=0)
            out = out + bias_shard
        return out

    @property
    def out_features_shard(self) -> int:
        return self._out_features_shard


class RowParallelLinear:
    """Linear layer split along input dimension (row parallel).

    Each rank owns rows [rank * shard : (rank+1) * shard] of the weight
    matrix.  Partial sums are allreduced across ranks.

    Shape: weight (full_in_features, out_features) → shard (in_shard, out_features)
    """

    def __init__(
        self,
        weight: np.ndarray,         # full weight (in_features, out_features)
        bias: Optional[np.ndarray],
        comm: TensorParallelCommunicator,
    ) -> None:
        self._comm  = comm
        self._bias  = bias
        self._weight_shard = comm.scatter(weight, dim=0)
        self._in_shard = self._weight_shard.shape[0]

    def forward(self, x_shard: np.ndarray) -> np.ndarray:
        """x_shard: (..., in_shard) → (..., out_features); allreduced across ranks."""
        partial = x_shard @ self._weight_shard
        result  = self._comm.allreduce(partial)
        if self._bias is not None and self._comm._config.rank == 0:
            result = result + self._bias
        return result

    @property
    def in_features_shard(self) -> int:
        return self._in_shard


# ---------------------------------------------------------------------------
# TensorParallelExpertShard
# ---------------------------------------------------------------------------

class TensorParallelExpertShard:
    """One expert's weight matrices, split across TP ranks.

    SwiGLU FFN layout: gate (d_model → d_ff), up (d_model → d_ff),
    down (d_ff → d_model).

    Gate and up are column-parallel (each rank holds output columns).
    Down is row-parallel (each rank holds input rows = output cols of gate/up).
    """

    def __init__(
        self,
        expert_id: str,
        gate_w: np.ndarray,    # (d_model, d_ff)
        up_w: np.ndarray,      # (d_model, d_ff)
        down_w: np.ndarray,    # (d_ff, d_model)
        comm: TensorParallelCommunicator,
    ) -> None:
        self.expert_id = expert_id
        self._gate = ColumnParallelLinear(gate_w, None, comm)
        self._up   = ColumnParallelLinear(up_w,   None, comm)
        self._down = RowParallelLinear(down_w,    None, comm)

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """SwiGLU forward with tensor-parallel split.

        hidden: (batch, d_model) → (batch, d_model)
        """
        gate_shard = self._gate.forward(hidden)
        up_shard   = self._up.forward(hidden)
        # SiLU activation on gate shard
        silu = gate_shard / (1.0 + np.exp(-gate_shard))
        activated = silu * up_shard
        # Row-parallel down projection + allreduce
        return self._down.forward(activated)


# ---------------------------------------------------------------------------
# TensorParallelExpertRegistry
# ---------------------------------------------------------------------------

class TensorParallelExpertRegistry:
    """Stores ``TensorParallelExpertShard`` objects keyed by expert_id."""

    def __init__(self, comm: TensorParallelCommunicator) -> None:
        self._comm    = comm
        self._shards: Dict[str, TensorParallelExpertShard] = {}
        self._lock    = threading.Lock()

    def register(
        self,
        expert_id: str,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> TensorParallelExpertShard:
        shard = TensorParallelExpertShard(
            expert_id, gate_w, up_w, down_w, self._comm)
        with self._lock:
            self._shards[expert_id] = shard
        return shard

    def get(self, expert_id: str) -> Optional[TensorParallelExpertShard]:
        with self._lock:
            return self._shards.get(expert_id)

    def forward(
        self, expert_id: str, hidden: np.ndarray
    ) -> Optional[np.ndarray]:
        shard = self.get(expert_id)
        if shard is None:
            return None
        return shard.forward(hidden)

    @property
    def registered_count(self) -> int:
        with self._lock:
            return len(self._shards)


# ---------------------------------------------------------------------------
# Phase3TensorParallelSession
# ---------------------------------------------------------------------------

class Phase3TensorParallelSession:
    """Wraps a base session and intercepts expert dispatch for TP execution.

    When an expert is dispatched (via ``on_expert_dispatch``):
    1. Looks up the expert in the ``TensorParallelExpertRegistry``.
    2. Computes the shard-local output via ``TensorParallelExpertShard.forward``.
    3. Stores the result so Phase 1's kernel dispatch can use it.

    All other session calls are forwarded to the base session.
    """

    def __init__(
        self,
        base_session: Any,
        config: Optional[TensorParallelConfig] = None,
        allreduce_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if base_session is None:
            raise ValueError("base_session required")
        self.base_session = base_session
        self._config = config or TensorParallelConfig()
        self._comm   = TensorParallelCommunicator(self._config, allreduce_fn)
        self._registry = TensorParallelExpertRegistry(self._comm)
        self._lock   = threading.RLock()
        self._closed = False

        # Telemetry
        self._dispatch_hits  = 0
        self._dispatch_miss  = 0

    # ---- expert registration ------------------------------------------------

    def register_expert(
        self,
        expert_id: str,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> None:
        self._registry.register(expert_id, gate_w, up_w, down_w)

    # ---- expert dispatch hook -----------------------------------------------

    def on_expert_dispatch(
        self,
        layer_id: int,
        expert_id: str,
        hidden: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Called by Phase 1 hook wiring for each active expert.

        Returns the TP-computed output, or None to fall back to Phase 1 path.
        """
        result = self._registry.forward(expert_id, hidden)
        with self._lock:
            if result is not None:
                self._dispatch_hits += 1
            else:
                self._dispatch_miss += 1
        return result

    # ---- passthrough --------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Phase3TensorParallelSession is closed")

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._check_open()
        return self.base_session.generate(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._check_open()
        return self.base_session.stream(*args, **kwargs)

    def on_token_routed(self, layer_id: int, expert_ids: List[str]) -> None:
        hook = getattr(self.base_session, "on_token_routed", None)
        if callable(hook):
            hook(layer_id, expert_ids)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.base_session, name)

    # ---- telemetry ----------------------------------------------------------

    def metrics(self) -> dict:
        base: dict = {}
        get_m = getattr(self.base_session, "metrics", None)
        if callable(get_m):
            base = get_m()
        with self._lock:
            return {
                **base,
                **self._comm.metrics(),
                "tp_dispatch_hits":  self._dispatch_hits,
                "tp_dispatch_miss":  self._dispatch_miss,
                "tp_experts_registered": self._registry.registered_count,
            }

    # ---- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        close_fn = getattr(self.base_session, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logger.warning("phase3/tp: base session close error: %s", exc)

    def __enter__(self) -> "Phase3TensorParallelSession":
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def wrap_session_with_tensor_parallel(
    base_session: Any,
    *,
    world_size: int = 1,
    rank: int = 0,
    allreduce_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Phase3TensorParallelSession:
    """Wrap a base session with tensor-parallel expert dispatch."""
    config = TensorParallelConfig(world_size=world_size, rank=rank)
    return Phase3TensorParallelSession(base_session, config, allreduce_fn)


__all__ = [
    "TensorParallelBackend",
    "TensorParallelConfig",
    "TensorParallelCommunicator",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelExpertShard",
    "TensorParallelExpertRegistry",
    "Phase3TensorParallelSession",
    "wrap_session_with_tensor_parallel",
]
