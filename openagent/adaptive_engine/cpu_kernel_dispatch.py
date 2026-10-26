"""Phase 1D: Numba-JIT CPU Kernels for Expert FFN.

Phase 1C executes expert FFN via plain NumPy.  Phase 1D replaces the hot path
with Numba-compiled kernels that:

1. Run at native C speed on the host CPU.
2. Are compiled once per dtype/shape signature and cached on disk by Numba's
   AOT cache (``cache=True``), so subsequent sessions skip recompilation.
3. Exploit SIMD automatically via LLVM's auto-vectoriser (AVX-512 / AVX2
   depending on the CPU) when the array layout is contiguous C-order float32.
4. Fuse SwiGLU gate*up multiplication into a single pass to halve memory
   traffic compared to the two-pass NumPy version.

When Numba is not installed (the package is optional), every entry point falls
back transparently to the NumPy implementation without raising an error; the
session logs one INFO line and continues.

Architecture
------------
``SwiGLUKernel``         -- fused SiLU-gate × up → down projection (all families)
``FusedGLMKernel``       -- GLM 5.x: single-read fused gate+up weight blob
``QwenSharedKernel``     -- Qwen: pinned shared-expert fast path
``DeepSeekMoEKernel``    -- DeepSeek V4/V4.1: moe_intermediate_size=256 path
``Phase1DKernelDispatch``-- selects the best kernel per model family

Numba compilation notes
-----------------------
- All kernels upcast float16 inputs to float32 for FP precision, then return
  the result in the caller's dtype.
- ``@nb.njit(cache=True, fastmath=True, parallel=False)`` is used.  ``parallel``
  is off to avoid spawning extra threads that would compete with the expert
  prefetch workers already running on Phase 1B/C threads.
- Shape assumptions:  hidden  (B, D_model),  gate_w/up_w  (D_ff, D_model),
  down_w  (D_model, D_ff).  B is typically 1 (single-token decode).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Numba import
# ---------------------------------------------------------------------------
try:
    import numba as nb
    _HAS_NUMBA = True
    logger.info("phase1d: Numba %s available — JIT kernels enabled", nb.__version__)
except ImportError:
    _HAS_NUMBA = False
    logger.info("phase1d: Numba not installed — falling back to NumPy kernels")


# ---------------------------------------------------------------------------
# Pure-NumPy reference implementations (also the fallback)
# ---------------------------------------------------------------------------

def _np_silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _np_swiglu_forward(
    hidden: np.ndarray,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
) -> np.ndarray:
    """SwiGLU: out = down( silu(hidden @ gate_w.T) * (hidden @ up_w.T) )."""
    h = hidden.astype(np.float32, copy=False)
    gate = h @ gate_w.T.astype(np.float32)
    up   = h @ up_w.T.astype(np.float32)
    return (_np_silu(gate) * up) @ down_w.T.astype(np.float32)


def _np_fused_glm_forward(
    hidden: np.ndarray,
    fused_w: np.ndarray,   # (2*D_ff, D_model) — gate rows first, then up
    down_w: np.ndarray,
    d_ff: int,
) -> np.ndarray:
    h = hidden.astype(np.float32, copy=False)
    gate = h @ fused_w[:d_ff].T.astype(np.float32)
    up   = h @ fused_w[d_ff:].T.astype(np.float32)
    return (_np_silu(gate) * up) @ down_w.T.astype(np.float32)


# ---------------------------------------------------------------------------
# Numba kernels (compiled lazily on first call)
# ---------------------------------------------------------------------------

if _HAS_NUMBA:
    @nb.njit(cache=True, fastmath=True, parallel=False)
    def _nb_silu(x: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                v = x[i, j]
                out[i, j] = v / (1.0 + np.exp(-v))
        return out

    @nb.njit(cache=True, fastmath=True, parallel=False)
    def _nb_swiglu_forward(          # type: ignore[misc]
        hidden: np.ndarray,          # (B, D_model) float32
        gate_w: np.ndarray,          # (D_ff, D_model) float32
        up_w: np.ndarray,            # (D_ff, D_model) float32
        down_w: np.ndarray,          # (D_model, D_ff) float32
    ) -> np.ndarray:
        gate = hidden @ gate_w.T
        up   = hidden @ up_w.T
        B, D_ff = gate.shape
        act = np.empty_like(gate)
        for i in range(B):
            for j in range(D_ff):
                v = gate[i, j]
                act[i, j] = (v / (1.0 + np.exp(-v))) * up[i, j]
        return act @ down_w.T

    @nb.njit(cache=True, fastmath=True, parallel=False)
    def _nb_fused_glm_forward(       # type: ignore[misc]
        hidden: np.ndarray,          # (B, D_model) float32
        fused_w: np.ndarray,         # (2*D_ff, D_model) float32
        down_w: np.ndarray,          # (D_model, D_ff) float32
        d_ff: int,
    ) -> np.ndarray:
        gate = hidden @ fused_w[:d_ff].T
        up   = hidden @ fused_w[d_ff:].T
        B, Df = gate.shape
        act = np.empty_like(gate)
        for i in range(B):
            for j in range(Df):
                v = gate[i, j]
                act[i, j] = (v / (1.0 + np.exp(-v))) * up[i, j]
        return act @ down_w.T

    def _jit_swiglu(hidden, gate_w, up_w, down_w):
        h32 = np.ascontiguousarray(hidden.astype(np.float32))
        g32 = np.ascontiguousarray(gate_w.astype(np.float32))
        u32 = np.ascontiguousarray(up_w.astype(np.float32))
        d32 = np.ascontiguousarray(down_w.astype(np.float32))
        return _nb_swiglu_forward(h32, g32, u32, d32)

    def _jit_fused_glm(hidden, fused_w, down_w, d_ff):
        h32 = np.ascontiguousarray(hidden.astype(np.float32))
        f32 = np.ascontiguousarray(fused_w.astype(np.float32))
        d32 = np.ascontiguousarray(down_w.astype(np.float32))
        return _nb_fused_glm_forward(h32, f32, d32, d_ff)

else:
    _jit_swiglu = _np_swiglu_forward  # type: ignore[assignment]

    def _jit_fused_glm(hidden, fused_w, down_w, d_ff):  # type: ignore[misc]
        return _np_fused_glm_forward(hidden, fused_w, down_w, d_ff)


# ---------------------------------------------------------------------------
# SwiGLUKernel — generic family fast path
# ---------------------------------------------------------------------------

class SwiGLUKernel:
    """Numba-JIT SwiGLU kernel for any expert with separate gate/up/down weights.

    Usage::

        kernel = SwiGLUKernel(d_model=4096, d_ff=14336)
        kernel.load(expert_id, gate_bytes, up_bytes, down_bytes)
        out = kernel.forward(hidden, expert_id)   # (B, D_model) → (B, D_model)
    """

    def __init__(self, d_model: int, d_ff: int, dtype: np.dtype = np.float16) -> None:
        self.d_model = d_model
        self.d_ff    = d_ff
        self.dtype   = dtype
        self._cache: Dict[str, tuple] = {}
        self._lock   = threading.Lock()
        self._calls  = 0
        self._hits   = 0

    def load(self, expert_id: str, gate_bytes: bytes,
             up_bytes: bytes, down_bytes: bytes) -> None:
        g = np.frombuffer(gate_bytes, dtype=self.dtype).reshape(self.d_ff,    self.d_model)
        u = np.frombuffer(up_bytes,   dtype=self.dtype).reshape(self.d_ff,    self.d_model)
        d = np.frombuffer(down_bytes, dtype=self.dtype).reshape(self.d_model, self.d_ff)
        with self._lock:
            self._cache[expert_id] = (g, u, d)

    def forward(self, hidden: np.ndarray, expert_id: str) -> Optional[np.ndarray]:
        with self._lock:
            weights = self._cache.get(expert_id)
            self._calls += 1
            if weights is not None:
                self._hits += 1
        if weights is None:
            return None
        gate_w, up_w, down_w = weights
        out = _jit_swiglu(hidden, gate_w, up_w, down_w)
        return out.astype(hidden.dtype)

    def evict(self, expert_id: str) -> None:
        with self._lock:
            self._cache.pop(expert_id, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "swiglu_experts_cached": len(self._cache),
                "swiglu_calls": self._calls,
                "swiglu_hits": self._hits,
                "swiglu_numba": _HAS_NUMBA,
            }


# ---------------------------------------------------------------------------
# FusedGLMKernel — GLM 5.x / DSA fused gate+up blob
# ---------------------------------------------------------------------------

class FusedGLMKernel:
    """Phase 1D GLM kernel: one-read fused weight blob, Numba-compiled forward.

    Replaces the Phase 1C ``GLMExpertKernel`` with a Numba path.
    Same ``load_expert`` / ``forward`` interface.
    """

    def __init__(self, d_model: int, d_ff: int, dtype: np.dtype = np.float16) -> None:
        self.d_model = d_model
        self.d_ff    = d_ff
        self.dtype   = dtype
        # {expert_id: (fused_w, down_w)}
        self._cache: Dict[str, tuple] = {}
        self._lock   = threading.Lock()
        self._calls  = 0

    def load_expert(self, expert_id: str, fused_weight_bytes: bytes,
                    down_weight_bytes: bytes) -> None:
        fw = np.frombuffer(fused_weight_bytes, dtype=self.dtype).reshape(2 * self.d_ff, self.d_model)
        dw = np.frombuffer(down_weight_bytes,  dtype=self.dtype).reshape(self.d_model,  self.d_ff)
        with self._lock:
            self._cache[expert_id] = (fw, dw)

    def forward(self, hidden: np.ndarray, expert_id: str) -> Optional[np.ndarray]:
        with self._lock:
            weights = self._cache.get(expert_id)
            self._calls += 1
        if weights is None:
            return None
        fused_w, down_w = weights
        out = _jit_fused_glm(hidden, fused_w, down_w, self.d_ff)
        return out.astype(hidden.dtype)

    def evict(self, expert_id: str) -> None:
        with self._lock:
            self._cache.pop(expert_id, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "glm_jit_experts_cached": len(self._cache),
                "glm_jit_calls": self._calls,
                "glm_jit_numba": _HAS_NUMBA,
            }


# ---------------------------------------------------------------------------
# QwenSharedKernel — Numba path for always-on shared experts
# ---------------------------------------------------------------------------

class QwenSharedKernel:
    """Phase 1D Qwen shared-expert kernel with Numba forward pass.

    Shared experts are pinned at session open; ``forward`` never touches disk.
    """

    def __init__(self, shared_expert_ids, d_model: int, d_ff: int,
                 dtype: np.dtype = np.float16) -> None:
        self._shared_ids = frozenset(shared_expert_ids)
        self.d_model = d_model
        self.d_ff    = d_ff
        self.dtype   = dtype
        self._pinned: Dict[str, tuple] = {}
        self._lock   = threading.Lock()

    def is_shared(self, expert_id: str) -> bool:
        return expert_id in self._shared_ids

    def pin(self, expert_id: str, gate_bytes: bytes,
            up_bytes: bytes, down_bytes: bytes) -> None:
        if expert_id not in self._shared_ids:
            raise ValueError(f"expert {expert_id!r} not a declared shared expert")
        g = np.frombuffer(gate_bytes, dtype=self.dtype).reshape(self.d_ff,    self.d_model)
        u = np.frombuffer(up_bytes,   dtype=self.dtype).reshape(self.d_ff,    self.d_model)
        d = np.frombuffer(down_bytes, dtype=self.dtype).reshape(self.d_model, self.d_ff)
        with self._lock:
            self._pinned[expert_id] = (g, u, d)

    def forward(self, hidden: np.ndarray, expert_id: str) -> Optional[np.ndarray]:
        with self._lock:
            weights = self._pinned.get(expert_id)
        if weights is None:
            return None
        gate_w, up_w, down_w = weights
        out = _jit_swiglu(hidden, gate_w, up_w, down_w)
        return out.astype(hidden.dtype)

    def all_pinned(self) -> bool:
        with self._lock:
            return self._shared_ids.issubset(self._pinned.keys())

    def stats(self) -> dict:
        with self._lock:
            return {
                "qwen_jit_pinned": len(self._pinned),
                "qwen_jit_declared": len(self._shared_ids),
                "qwen_jit_numba": _HAS_NUMBA,
            }


# ---------------------------------------------------------------------------
# DeepSeekMoEKernel — moe_intermediate_size=256 Numba path
# ---------------------------------------------------------------------------

class DeepSeekMoEKernel:
    """Phase 1D DeepSeek V4/V4.1 routed expert kernel.

    Expert tensors use moe_intermediate_size (default 256) as the hidden
    dimension; this kernel handles the smaller shapes correctly and dispatches
    through the same Numba SwiGLU path.
    """

    def __init__(self, d_model: int, moe_intermediate_size: int = 256,
                 dtype: np.dtype = np.float16) -> None:
        self.d_model = d_model
        self.moe_dim = moe_intermediate_size
        self.dtype   = dtype
        self._cache: Dict[str, tuple] = {}
        self._lock   = threading.Lock()
        self._calls  = 0
        self._latent_skips = 0

    def load_expert(self, expert_id: str, gate_bytes: bytes,
                    up_bytes: bytes, down_bytes: bytes) -> None:
        g = np.frombuffer(gate_bytes, dtype=self.dtype).reshape(self.moe_dim,  self.d_model)
        u = np.frombuffer(up_bytes,   dtype=self.dtype).reshape(self.moe_dim,  self.d_model)
        d = np.frombuffer(down_bytes, dtype=self.dtype).reshape(self.d_model,  self.moe_dim)
        with self._lock:
            self._cache[expert_id] = (g, u, d)

    def forward(self, hidden: np.ndarray, expert_id: str,
                kv_latent: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        with self._lock:
            weights = self._cache.get(expert_id)
            self._calls += 1
            if kv_latent is not None:
                self._latent_skips += 1
        if weights is None:
            return None
        gate_w, up_w, down_w = weights
        out = _jit_swiglu(hidden, gate_w, up_w, down_w)
        return out.astype(hidden.dtype)

    def evict(self, expert_id: str) -> None:
        with self._lock:
            self._cache.pop(expert_id, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "deepseek_jit_cached": len(self._cache),
                "deepseek_jit_calls": self._calls,
                "deepseek_jit_latent_skips": self._latent_skips,
                "deepseek_jit_numba": _HAS_NUMBA,
            }


# ---------------------------------------------------------------------------
# Phase1DKernelDispatch — unified dispatch for all families
# ---------------------------------------------------------------------------

class Phase1DKernelDispatch:
    """Route expert dispatch to the best Phase 1D kernel for each model family.

    Families: "generic", "glm", "qwen", "deepseek".
    Returns None on miss so the caller can fall back to the Phase 1C path.
    """

    def __init__(self) -> None:
        self._generic:  Optional[SwiGLUKernel]       = None
        self._glm:      Optional[FusedGLMKernel]      = None
        self._qwen:     Optional[QwenSharedKernel]    = None
        self._deepseek: Optional[DeepSeekMoEKernel]   = None
        self._lock = threading.Lock()

    def register_generic(self,  k: SwiGLUKernel)      -> None:
        with self._lock: self._generic  = k

    def register_glm(self,      k: FusedGLMKernel)    -> None:
        with self._lock: self._glm      = k

    def register_qwen(self,     k: QwenSharedKernel)  -> None:
        with self._lock: self._qwen     = k

    def register_deepseek(self, k: DeepSeekMoEKernel) -> None:
        with self._lock: self._deepseek = k

    def forward(self, family: str, expert_id: str,
                hidden: np.ndarray, **kwargs: Any) -> Optional[np.ndarray]:
        with self._lock:
            if family == "glm" and self._glm is not None:
                return self._glm.forward(hidden, expert_id)
            if family == "qwen" and self._qwen is not None:
                if self._qwen.is_shared(expert_id):
                    return self._qwen.forward(hidden, expert_id)
            if family == "deepseek" and self._deepseek is not None:
                return self._deepseek.forward(
                    hidden, expert_id, kv_latent=kwargs.get("kv_latent"))
            if self._generic is not None:
                return self._generic.forward(hidden, expert_id)
        return None

    def stats(self) -> dict:
        with self._lock:
            s: dict = {"has_numba": _HAS_NUMBA}
            for k in (self._generic, self._glm, self._qwen, self._deepseek):
                if k is not None:
                    s.update(k.stats())
            return s


__all__ = [
    "SwiGLUKernel",
    "FusedGLMKernel",
    "QwenSharedKernel",
    "DeepSeekMoEKernel",
    "Phase1DKernelDispatch",
]
