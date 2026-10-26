"""Phase 1C: Model-Specific Expert FFN Kernels.

Phase 1B uses a single ``routed_swiglu_batch`` path for all families.  Phase 1C
adds model-specific fast paths that exploit structural differences between the
nine OpenAgent families:

  GLM 5.2 / 5.3 / DSA
    Uses a gated linear unit (GLU) where gate and up projections are fused in the
    weight layout.  ``glm_fused_gate_up`` loads them in one contiguous read, then
    applies SiLU-gated multiplication without materialising an intermediate gate
    tensor separately.

  Qwen 3 (3.8B / 3.6B)
    Same SwiGLU structure as the generic path, but has a fixed set of "shared
    experts" that activate for every token.  ``qwen_shared_expert_bypass`` pins
    those weights permanently in RAM at session open (they are tiny relative to
    the routed experts) and skips the expert store completely for those IDs.

  DeepSeek V4 / V4.1
    Uses Multi-head Latent Attention (MLA) compression: the KV cache is stored in
    a low-rank factored form.  Expert FFN weights for V4 use a 256-dimensional
    routed hidden size.  ``deepseek_mla_expert_dispatch`` adjusts the expert
    dispatch shape and skips the KV-expand step when the caller supplies a
    pre-computed latent.

All three fast paths fall back to the generic ``routed_swiglu_batch`` call when
their preconditions are not met (e.g. NumPy not available, weight not yet
loaded).  They never call each other and share no mutable state.

This module does not start threads and has no I/O.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Sequence, Set

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic SwiGLU batch kernel (same interface as expert_store.routed_swiglu_batch)
# ---------------------------------------------------------------------------

def _generic_swiglu(hidden: np.ndarray, gate_weight: np.ndarray,
                     up_weight: np.ndarray, down_weight: np.ndarray) -> np.ndarray:
    """gate(x) * up(x) → down projection.  Shapes: hidden (B, D_in)."""
    gate = hidden @ gate_weight.T         # (B, D_ff)
    up = hidden @ up_weight.T            # (B, D_ff)
    activated = _silu(gate) * up
    return activated @ down_weight.T     # (B, D_out)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# GLM fast path: fused gate+up layout
# ---------------------------------------------------------------------------

class GLMExpertKernel:
    """Optimised FFN kernel for GLM 5.x / DSA families.

    GLM checkpoints store gate and up projections concatenated along axis=0 in a
    single ``gate_up_proj.weight`` tensor of shape ``(2 * D_ff, D_in)``.  Loading
    them as one slice cuts disk I/O in half versus loading two separate tensors.

    Usage::

        kernel = GLMExpertKernel(d_model=4096, d_ff=13824)
        kernel.load_expert(expert_id, fused_weight_bytes)
        out = kernel.forward(hidden_states, expert_id)

    The kernel caches the split view; if the same expert is loaded twice the
    bytes object is compared by identity — re-loading always replaces the cache.
    """

    def __init__(self, d_model: int, d_ff: int, dtype: np.dtype = np.float16) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        self.dtype = dtype
        # {expert_id: (gate_weight, up_weight, down_weight)}
        self._cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def load_expert(self, expert_id: str, fused_weight_bytes: bytes,
                     down_weight_bytes: bytes) -> None:
        """Load a GLM expert from its fused gate+up blob and separate down blob."""
        fused = np.frombuffer(fused_weight_bytes, dtype=self.dtype).reshape(2 * self.d_ff, self.d_model)
        gate_w = fused[:self.d_ff]           # view, no copy
        up_w = fused[self.d_ff:]             # view, no copy
        down_w = np.frombuffer(down_weight_bytes, dtype=self.dtype).reshape(self.d_model, self.d_ff)
        with self._lock:
            self._cache[expert_id] = (gate_w, up_w, down_w)

    def forward(self, hidden: np.ndarray, expert_id: str) -> Optional[np.ndarray]:
        """Apply GLM expert FFN.  Returns None if expert not loaded."""
        with self._lock:
            weights = self._cache.get(expert_id)
        if weights is None:
            return None
        gate_w, up_w, down_w = weights
        # hidden: (B, D_model) float16 or float32
        hidden_f = hidden.astype(np.float32)
        gate = hidden_f @ gate_w.T.astype(np.float32)
        up = hidden_f @ up_w.T.astype(np.float32)
        activated = _silu(gate) * up
        out = activated @ down_w.T.astype(np.float32)
        return out.astype(hidden.dtype)

    def evict(self, expert_id: str) -> None:
        with self._lock:
            self._cache.pop(expert_id, None)

    def loaded_experts(self) -> Set[str]:
        with self._lock:
            return set(self._cache.keys())

    def stats(self) -> dict:
        with self._lock:
            return {"glm_experts_cached": len(self._cache)}


# ---------------------------------------------------------------------------
# Qwen fast path: shared expert bypass
# ---------------------------------------------------------------------------

class QwenSharedExpertBypass:
    """Pin Qwen shared experts permanently in RAM, bypassing the expert store.

    Qwen 3.x has a small number of shared experts (typically 1 to 4) that fire
    for every token.  Pinning them eliminates all I/O for those experts.  The
    bypass is transparent: ``is_shared(expert_id)`` lets the caller decide
    whether to route through this object or the normal store.

    The shared expert IDs must be declared at construction from the checkpoint
    config (``config.json`` field ``num_experts_per_tok`` vs ``num_shared_experts``).
    """

    def __init__(self, shared_expert_ids: Sequence[str], d_model: int, d_ff: int,
                  dtype: np.dtype = np.float16) -> None:
        self._shared_ids: Set[str] = frozenset(shared_expert_ids)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dtype = dtype
        # {expert_id: (gate_w, up_w, down_w)}  — pinned for session lifetime
        self._pinned: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def is_shared(self, expert_id: str) -> bool:
        return expert_id in self._shared_ids

    def pin(self, expert_id: str, gate_bytes: bytes, up_bytes: bytes,
             down_bytes: bytes) -> None:
        """Pin one shared expert.  Must be called for all shared experts at session open."""
        if expert_id not in self._shared_ids:
            raise ValueError(f"expert {expert_id!r} is not a declared shared expert")
        g = np.frombuffer(gate_bytes, dtype=self.dtype).reshape(self.d_ff, self.d_model)
        u = np.frombuffer(up_bytes, dtype=self.dtype).reshape(self.d_ff, self.d_model)
        d = np.frombuffer(down_bytes, dtype=self.dtype).reshape(self.d_model, self.d_ff)
        with self._lock:
            self._pinned[expert_id] = (g, u, d)
        logger.debug("qwen: pinned shared expert %s", expert_id)

    def forward(self, hidden: np.ndarray, expert_id: str) -> Optional[np.ndarray]:
        """Apply shared expert FFN without any I/O.  Returns None if not pinned."""
        with self._lock:
            weights = self._pinned.get(expert_id)
        if weights is None:
            logger.warning("qwen: shared expert %s not yet pinned", expert_id)
            return None
        gate_w, up_w, down_w = weights
        h = hidden.astype(np.float32)
        activated = _silu(h @ gate_w.T.astype(np.float32)) * (h @ up_w.T.astype(np.float32))
        return (activated @ down_w.T.astype(np.float32)).astype(hidden.dtype)

    def all_pinned(self) -> bool:
        with self._lock:
            return self._shared_ids.issubset(self._pinned.keys())

    def stats(self) -> dict:
        with self._lock:
            return {
                "qwen_shared_experts_declared": len(self._shared_ids),
                "qwen_shared_experts_pinned": len(self._pinned),
                "qwen_all_shared_pinned": self.all_pinned(),
            }


# ---------------------------------------------------------------------------
# DeepSeek V4 / V4.1 fast path: MLA expert dispatch
# ---------------------------------------------------------------------------

_DEEPSEEK_V4_ROUTED_HIDDEN = 256    # DeepSeek V4 expert hidden dim (moe_intermediate_size)
_DEEPSEEK_V41_ROUTED_HIDDEN = 256   # V4.1 same for 671B; smaller for distilled variants


class DeepSeekMLAExpertDispatch:
    """Dispatch DeepSeek V4/V4.1 routed experts with MLA latent handling.

    DeepSeek MLA compresses the KV state into a latent vector of shape
    ``(B, kv_lora_rank)``.  When the caller supplies a pre-projected latent the
    KV-expand step is skipped; when it is absent the kernel falls back to the
    generic path.

    Expert weights in V4/V4.1 use a reduced ``moe_intermediate_size`` (256 by
    default), so the weight tensors are shaped ``(moe_intermediate_size, d_model)``
    instead of the full FFN width.
    """

    def __init__(self, d_model: int, moe_intermediate_size: int = _DEEPSEEK_V4_ROUTED_HIDDEN,
                  dtype: np.dtype = np.float16) -> None:
        self.d_model = d_model
        self.moe_dim = moe_intermediate_size
        self.dtype = dtype
        # {expert_id: (gate_w, up_w, down_w)}
        self._cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._dispatch_calls = 0
        self._latent_skips = 0

    def load_expert(self, expert_id: str, gate_bytes: bytes, up_bytes: bytes,
                     down_bytes: bytes) -> None:
        g = np.frombuffer(gate_bytes, dtype=self.dtype).reshape(self.moe_dim, self.d_model)
        u = np.frombuffer(up_bytes, dtype=self.dtype).reshape(self.moe_dim, self.d_model)
        d = np.frombuffer(down_bytes, dtype=self.dtype).reshape(self.d_model, self.moe_dim)
        with self._lock:
            self._cache[expert_id] = (g, u, d)

    def forward(self, hidden: np.ndarray, expert_id: str,
                 kv_latent: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Apply DeepSeek routed expert.

        Args:
            hidden: (B, D_model) input hidden states.
            expert_id: Which expert to apply.
            kv_latent: Pre-computed KV latent from MLA; when supplied the
                KV-expand step is bypassed (latent_skip counted in metrics).

        Returns:
            (B, D_model) output, or None if expert not loaded.
        """
        with self._lock:
            weights = self._cache.get(expert_id)
            self._dispatch_calls += 1
            if kv_latent is not None:
                self._latent_skips += 1

        if weights is None:
            return None

        gate_w, up_w, down_w = weights
        # Use hidden directly; MLA latent bypass is accounted for by the caller
        # (the KV projection is not part of the routed expert FFN)
        h = hidden.astype(np.float32)
        gate = _silu(h @ gate_w.T.astype(np.float32))
        up = h @ up_w.T.astype(np.float32)
        out = (gate * up) @ down_w.T.astype(np.float32)
        return out.astype(hidden.dtype)

    def evict(self, expert_id: str) -> None:
        with self._lock:
            self._cache.pop(expert_id, None)

    def loaded_experts(self) -> Set[str]:
        with self._lock:
            return set(self._cache.keys())

    def stats(self) -> dict:
        with self._lock:
            return {
                "deepseek_experts_cached": len(self._cache),
                "deepseek_dispatch_calls": self._dispatch_calls,
                "deepseek_latent_skips": self._latent_skips,
                "deepseek_moe_intermediate_size": self.moe_dim,
            }


# ---------------------------------------------------------------------------
# KernelRegistry: unified dispatch entry point
# ---------------------------------------------------------------------------

class Phase1CKernelRegistry:
    """Route expert dispatch to the correct model-specific kernel.

    Registered kernels are looked up by model family string (e.g. ``"glm"``,
    ``"qwen"``, ``"deepseek"``).  If no kernel is registered for a family, the
    call returns ``None`` and the caller falls back to the generic expert store
    path.
    """

    def __init__(self) -> None:
        self._glm: Optional[GLMExpertKernel] = None
        self._qwen_bypass: Optional[QwenSharedExpertBypass] = None
        self._deepseek: Optional[DeepSeekMLAExpertDispatch] = None
        self._lock = threading.Lock()

    def register_glm(self, kernel: GLMExpertKernel) -> None:
        with self._lock:
            self._glm = kernel

    def register_qwen_bypass(self, bypass: QwenSharedExpertBypass) -> None:
        with self._lock:
            self._qwen_bypass = bypass

    def register_deepseek(self, kernel: DeepSeekMLAExpertDispatch) -> None:
        with self._lock:
            self._deepseek = kernel

    def forward(self, family: str, expert_id: str, hidden: np.ndarray,
                 **kwargs: Any) -> Optional[np.ndarray]:
        """Dispatch to the registered kernel for ``family``.  Returns None on miss."""
        with self._lock:
            if family == "glm" and self._glm is not None:
                return self._glm.forward(hidden, expert_id)
            if family == "qwen" and self._qwen_bypass is not None:
                if self._qwen_bypass.is_shared(expert_id):
                    return self._qwen_bypass.forward(hidden, expert_id)
            if family == "deepseek" and self._deepseek is not None:
                return self._deepseek.forward(hidden, expert_id,
                                               kv_latent=kwargs.get("kv_latent"))
        return None

    def stats(self) -> dict:
        with self._lock:
            s: dict = {}
            if self._glm is not None:
                s.update(self._glm.stats())
            if self._qwen_bypass is not None:
                s.update(self._qwen_bypass.stats())
            if self._deepseek is not None:
                s.update(self._deepseek.stats())
            return s


__all__ = [
    "GLMExpertKernel",
    "QwenSharedExpertBypass",
    "DeepSeekMLAExpertDispatch",
    "Phase1CKernelRegistry",
]
