"""Phase 1F: Forward-pass hook wiring for all 9 model families.

Phases 1C and 1D built machinery (affinity prefetch, Markov placement, Numba
kernels, grammar speculative decode) that is inert until the actual model
forward pass emits routing decisions.  Phase 1F wires those hooks.

The nine families and their forward-pass patterns:

Family              Hook mechanism
-----------         -------------------------------------------------
OLMoE               ``model.model.layers[i].mlp.experts`` — post_hook
GLM 5.2/5.3/DSA     ``model.transformer.layers[i].mlp`` — post_hook
Inkling             ``model.model.layers[i].block_sparse_moe`` — post_hook
Kimi K3             ``model.model.layers[i].moe_block`` — post_hook
DeepSeek V4/V4.1    ``model.model.layers[i].mlp`` with routing_weights — post_hook
Qwen 3.8B/3.6B      ``model.model.layers[i].mlp.experts`` + shared — post_hook
Mixtral             ``model.model.layers[i].block_sparse_moe`` — post_hook
HF generic          nn.Module register_forward_hook fallback

For non-Torch sessions (native/compiled) we also provide a
``NativeRoutingBridge`` that intercepts the expert-dispatch callback the native
runtime fires (``on_expert_dispatch``).

No imports from Torch are required at module load time; Torch is imported lazily
only if ``wire_torch_model`` is called.

Components
----------
``FamilyHookWirer``     -- detects family and attaches hooks to a Torch model
``NativeRoutingBridge`` -- wires native session callbacks
``HookWiringRegistry``  -- manages active wirings, safe to call close()
``wire_session``        -- high-level entry: wraps any Phase 1D/1E session
"""
from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------

def detect_family(model: Any) -> str:
    """Return a family string for a model object or config dict."""
    cls = type(model).__name__.lower()
    cfg = getattr(model, "config", None) or {}
    arch = (getattr(cfg, "architectures", None) or [cfg] if isinstance(cfg, dict) else [])
    arch_str = " ".join(str(a).lower() for a in arch)

    full = cls + " " + arch_str
    if "olmoe" in full:
        return "olmoe"
    if "deepseek" in full or "deepseekv4" in full or "deepseek_v4" in full:
        return "deepseek"
    if "glm" in full:
        return "glm"
    if "qwen" in full:
        return "qwen"
    if "mixtral" in full:
        return "mixtral"
    if "inkling" in full:
        return "inkling"
    if "kimi" in full or "k3" in full:
        return "kimi"
    return "generic"


# ---------------------------------------------------------------------------
# FamilyHookWirer — attaches PyTorch forward hooks
# ---------------------------------------------------------------------------

class FamilyHookWirer:
    """Attach routing hooks to a PyTorch model's MoE layers.

    The hook fires after each MoE sub-module's forward pass and calls
    ``routing_callback(layer_id, expert_ids)`` where ``expert_ids`` is a
    list of string expert IDs selected in that forward pass.

    Hook handles are stored so they can be removed on ``remove()``.
    """

    def __init__(
        self,
        routing_callback: Callable[[int, List[str]], None],
        family: str = "generic",
    ) -> None:
        self._cb    = routing_callback
        self._family = family
        self._handles: List[Any] = []
        self._lock   = threading.Lock()
        self._wired_layers: Set[int] = set()

    # ------------------------------------------------------------------
    # Internal hook factories
    # ------------------------------------------------------------------

    def _make_hook(self, layer_id: int) -> Callable:
        cb = self._cb
        fam = self._family

        def _hook(module: Any, inputs: Any, output: Any) -> None:
            try:
                expert_ids = _extract_expert_ids(module, inputs, output, fam)
                if expert_ids:
                    cb(layer_id, expert_ids)
            except Exception as exc:
                logger.debug("phase1f: hook error layer %d: %s", layer_id, exc)

        return _hook

    # ------------------------------------------------------------------
    # Public wiring
    # ------------------------------------------------------------------

    def wire_model(self, model: Any) -> int:
        """Attach hooks to all MoE layers in ``model``.

        Returns the number of layers hooked.
        """
        try:
            import torch.nn as nn
        except ImportError:
            logger.info("phase1f: PyTorch not available — hook wiring skipped")
            return 0

        layers = _get_moe_layers(model, self._family)
        if not layers:
            logger.info("phase1f: no MoE layers found for family=%s", self._family)
            return 0

        with self._lock:
            for layer_id, module in layers:
                if layer_id in self._wired_layers:
                    continue
                handle = module.register_forward_hook(self._make_hook(layer_id))
                self._handles.append(handle)
                self._wired_layers.add(layer_id)

        logger.info("phase1f: wired %d MoE layers (family=%s)", len(layers), self._family)
        return len(layers)

    def remove(self) -> None:
        """Remove all registered hooks."""
        with self._lock:
            for h in self._handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._handles.clear()
            self._wired_layers.clear()


# ---------------------------------------------------------------------------
# Helper: layer discovery per family
# ---------------------------------------------------------------------------

def _get_moe_layers(model: Any, family: str) -> List[Tuple[int, Any]]:
    """Return [(layer_id, moe_module), ...] for the given model family."""
    results: List[Tuple[int, Any]] = []

    def _try_layers(path_fragments: List[str], sub_attr: str) -> bool:
        obj = model
        for frag in path_fragments:
            obj = getattr(obj, frag, None)
            if obj is None:
                return False
        layers = obj if hasattr(obj, "__iter__") else None
        if layers is None:
            return False
        for i, layer in enumerate(layers):
            sub = getattr(layer, sub_attr, None)
            if sub is not None:
                results.append((i, sub))
        return bool(results)

    # Family-specific layer paths
    paths: Dict[str, List[Tuple[List[str], str]]] = {
        "olmoe":    [
            (["model", "model", "layers"], "mlp"),
            (["model", "layers"], "mlp"),
        ],
        "deepseek": [
            (["model", "model", "layers"], "mlp"),
            (["model", "layers"], "mlp"),
        ],
        "glm": [
            (["model", "transformer", "layers"], "mlp"),
            (["transformer", "layers"], "mlp"),
        ],
        "qwen": [
            (["model", "model", "layers"], "mlp"),
            (["model", "layers"], "mlp"),
        ],
        "mixtral": [
            (["model", "model", "layers"], "block_sparse_moe"),
            (["model", "layers"], "block_sparse_moe"),
        ],
        "inkling": [
            (["model", "model", "layers"], "block_sparse_moe"),
            (["model", "layers"], "block_sparse_moe"),
        ],
        "kimi": [
            (["model", "model", "layers"], "moe_block"),
            (["model", "layers"], "moe_block"),
        ],
    }

    for path_frags, sub_attr in paths.get(family, []) + [
        (["model", "model", "layers"], "mlp"),
        (["model", "layers"], "mlp"),
    ]:
        if _try_layers(path_frags, sub_attr):
            return results
    return results


# ---------------------------------------------------------------------------
# Expert ID extraction from hook output
# ---------------------------------------------------------------------------

def _extract_expert_ids(module: Any, inputs: Any, output: Any, family: str) -> List[str]:
    """Extract a list of expert IDs from a MoE forward hook.

    Different families store routing indices in different places.
    Falls back to inspecting common attribute names.
    """
    try:
        import torch
    except ImportError:
        return []

    # Many HF MoE layers store routing_weights / selected_experts on the module
    for attr in ("_routing_indices", "_selected_experts", "router_indices"):
        indices = getattr(module, attr, None)
        if indices is not None:
            return _indices_to_ids(indices)

    # Output tuple: some implementations return (hidden_state, router_logits, expert_index)
    if isinstance(output, (tuple, list)):
        for item in output:
            if hasattr(item, "shape") and item.dtype in (torch.int32, torch.int64):
                try:
                    if item.ndim <= 2:
                        return _indices_to_ids(item)
                except Exception:
                    pass

    # Inspect module's last stored routing for Mixtral / OLMoE pattern
    gate = getattr(module, "gate", None)
    if gate is not None:
        last_idx = getattr(gate, "_last_selected", None)
        if last_idx is not None:
            return _indices_to_ids(last_idx)

    return []


def _indices_to_ids(tensor: Any) -> List[str]:
    try:
        flat = tensor.detach().cpu().view(-1).tolist()
        return [f"e{int(i)}" for i in set(flat)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# NativeRoutingBridge — for native (non-Torch) sessions
# ---------------------------------------------------------------------------

class NativeRoutingBridge:
    """Wire the routing callback into a native session's dispatch hook.

    Native sessions produced by OpenAgent's compiled runtime expose an
    ``on_expert_dispatch`` attribute that accepts a callable::

        session.on_expert_dispatch = bridge.on_expert_dispatch

    The callable receives ``(layer_id: int, expert_ids: List[str])``.
    """

    def __init__(
        self,
        session: Any,
        routing_callback: Callable[[int, List[str]], None],
    ) -> None:
        self._session = weakref.ref(session)
        self._cb = routing_callback
        self._wired = False
        self._calls = 0
        self._lock  = threading.Lock()

    def wire(self) -> bool:
        """Attach the callback.  Returns True if successful."""
        session = self._session()
        if session is None:
            return False
        if hasattr(session, "on_expert_dispatch"):
            session.on_expert_dispatch = self.on_expert_dispatch
            with self._lock:
                self._wired = True
            logger.info("phase1f: NativeRoutingBridge wired to %s", type(session).__name__)
            return True
        logger.debug("phase1f: session has no on_expert_dispatch attribute")
        return False

    def on_expert_dispatch(self, layer_id: int, expert_ids: List[str]) -> None:
        with self._lock:
            self._calls += 1
        try:
            self._cb(layer_id, expert_ids)
        except Exception as exc:
            logger.debug("phase1f: routing callback error: %s", exc)

    def remove(self) -> None:
        session = self._session()
        if session is not None:
            try:
                del session.on_expert_dispatch
            except AttributeError:
                pass
        with self._lock:
            self._wired = False

    def stats(self) -> dict:
        with self._lock:
            return {
                "native_bridge_wired": self._wired,
                "native_bridge_calls": self._calls,
            }


# ---------------------------------------------------------------------------
# batch_logits wiring — connect speculative decode to logit source
# ---------------------------------------------------------------------------

def wire_batch_logits(optimized_session: Any, source_session: Any) -> bool:
    """Wire ``batch_logits`` from ``source_session`` into ``optimized_session``.

    This enables Phase 1D's GrammarSpeculativeDecoder to run the verification
    step.  Checks several attribute names the production runtime may use.
    """
    for attr in ("batch_logits", "get_logits", "compute_logits", "forward_logits"):
        fn = getattr(source_session, attr, None)
        if callable(fn):
            optimized_session.batch_logits = fn
            logger.info("phase1f: batch_logits wired from %s.%s", type(source_session).__name__, attr)
            return True
    logger.debug("phase1f: no logit source found on %s", type(source_session).__name__)
    return False


# ---------------------------------------------------------------------------
# HookWiringRegistry — manages all active wirings
# ---------------------------------------------------------------------------

class HookWiringRegistry:
    """Track all hook wirings for a session; remove them on close()."""

    def __init__(self) -> None:
        self._wirers: List[FamilyHookWirer]      = []
        self._bridges: List[NativeRoutingBridge] = []
        self._lock = threading.Lock()
        self._closed = False

    def add_wirer(self, wirer: FamilyHookWirer) -> None:
        with self._lock:
            self._wirers.append(wirer)

    def add_bridge(self, bridge: NativeRoutingBridge) -> None:
        with self._lock:
            self._bridges.append(bridge)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for w in self._wirers:
                try:
                    w.remove()
                except Exception:
                    pass
            for b in self._bridges:
                try:
                    b.remove()
                except Exception:
                    pass
            self._wirers.clear()
            self._bridges.clear()
            self._closed = True

    def stats(self) -> dict:
        with self._lock:
            return {
                "hook_wirers": len(self._wirers),
                "native_bridges": len(self._bridges),
            }


# ---------------------------------------------------------------------------
# wire_session — high-level entry point
# ---------------------------------------------------------------------------

def wire_session(
    optimized_session: Any,
    *,
    torch_model: Optional[Any] = None,
    native_session: Optional[Any] = None,
    family: Optional[str] = None,
) -> HookWiringRegistry:
    """Wire routing hooks from a model or native session into an optimized session.

    The optimized session must expose ``on_token_routed(layer_id, expert_ids)``.

    Args:
        optimized_session: Phase 1D or 1E session with ``on_token_routed``.
        torch_model: A HuggingFace / PyTorch model to wire hooks into.
        native_session: A native OpenAgent session with ``on_expert_dispatch``.
        family: Override model family detection.

    Returns:
        A ``HookWiringRegistry`` whose ``close()`` removes all hooks.
    """
    registry = HookWiringRegistry()
    cb = getattr(optimized_session, "on_token_routed", None)
    if cb is None:
        logger.warning("phase1f: optimized_session has no on_token_routed — hooks not wired")
        return registry

    if torch_model is not None:
        det_family = family or detect_family(torch_model)
        wirer = FamilyHookWirer(cb, family=det_family)
        n = wirer.wire_model(torch_model)
        if n > 0:
            registry.add_wirer(wirer)
        # Try to wire batch_logits for speculative decode
        wire_batch_logits(optimized_session, torch_model)

    if native_session is not None:
        bridge = NativeRoutingBridge(native_session, cb)
        if bridge.wire():
            registry.add_bridge(bridge)
        wire_batch_logits(optimized_session, native_session)

    return registry


__all__ = [
    "detect_family",
    "FamilyHookWirer",
    "NativeRoutingBridge",
    "HookWiringRegistry",
    "wire_session",
    "wire_batch_logits",
]
