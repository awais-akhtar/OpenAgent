"""Phase 1H: Intel NPU offload for the routing MLP.

The routing MLP is the cheapest part of any MoE transformer (a small linear
layer that decides which experts to activate).  Offloading it to the Intel NPU
frees CPU cycles and, crucially, overlaps the routing decision with the CPU
expert load that follows in the next layer, reducing expert prefetch latency.

Design
------
The NPU handles ``routing_mlp(hidden) → router_logits`` for each token.
The CPU handles the expert FFN (via Phases 1C/1D/1E).

Two backends are tried in order:

1. **OpenVINO NPU** (``openvino`` package + Intel NPU driver).
   The routing MLP is exported as ONNX and compiled to an OpenVINO IR blob
   once per model; the blob is cached.  Inference is async: the NPU kernel
   is submitted and the CPU starts loading the next layer's experts
   concurrently; the result is collected just before it is needed.

2. **DirectML NPU** (Windows only, ``onnxruntime-directml``).
   Falls back to the DML NPU/iGPU path when OpenVINO is unavailable.

3. **CPU fallback** — transparent if neither NPU backend is available.

Components
----------
``RoutingMLPSpec``        -- describes one routing MLP (weights + metadata)
``OpenVINONPUBackend``    -- OpenVINO async inference on Intel NPU
``DirectMLNPUBackend``    -- DirectML inference on Intel NPU/iGPU
``NPURoutingDispatch``    -- unified entry point; selects best backend
``Phase1HNPUSession``     -- session wrapper that intercepts routing calls
"""
from __future__ import annotations

import io
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional runtime probes
# ---------------------------------------------------------------------------

try:
    import openvino as ov
    _HAS_OPENVINO = True
except ImportError:
    _HAS_OPENVINO = False

try:
    import onnxruntime as _ort
    _HAS_ORT = True
    _HAS_DIRECTML_NPU = any(
        "DmlExecutionProvider" in p
        for p in _ort.get_available_providers()
    )
except ImportError:
    _HAS_ORT = False
    _HAS_DIRECTML_NPU = False

_CACHE_SUFFIX_OV  = ".phase1h_ov.xml"
_CACHE_SUFFIX_BIN = ".phase1h_ov.bin"
_CACHE_SUFFIX_ORT = ".phase1h_ort.onnx"


# ---------------------------------------------------------------------------
# RoutingMLPSpec — describes one MoE routing MLP
# ---------------------------------------------------------------------------

@dataclass
class RoutingMLPSpec:
    """Weights and metadata for one routing MLP.

    The routing MLP is a single linear projection:
        router_logits = hidden @ weight.T + bias   (bias optional)
    """
    layer_id:     int
    model_id:     str
    hidden_size:  int
    num_experts:  int
    weight:       np.ndarray   # (num_experts, hidden_size) float32
    bias:         Optional[np.ndarray] = None  # (num_experts,) float32

    def apply_numpy(self, hidden: np.ndarray) -> np.ndarray:
        """CPU fallback: (B, hidden_size) → (B, num_experts)."""
        h = hidden.astype(np.float32, copy=False)
        out = h @ self.weight.T.astype(np.float32)
        if self.bias is not None:
            out += self.bias.astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# ONNX export helper (no Torch required — we build ONNX by hand)
# ---------------------------------------------------------------------------

def _build_onnx_routing_mlp(spec: RoutingMLPSpec) -> bytes:
    """Construct a minimal ONNX model for the routing MLP."""
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper

        weight_init = numpy_helper.from_array(
            spec.weight.astype(np.float32), name="weight")
        inputs = [helper.make_tensor_value_info(
            "hidden", TensorProto.FLOAT, [None, spec.hidden_size])]
        outputs = [helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT, [None, spec.num_experts])]

        if spec.bias is not None:
            bias_init = numpy_helper.from_array(
                spec.bias.astype(np.float32), name="bias")
            nodes = [
                helper.make_node("Gemm", ["hidden", "weight", "bias"], ["logits"],
                                 transB=1),
            ]
            graph = helper.make_graph(nodes, "routing", inputs, outputs,
                                      [weight_init, bias_init])
        else:
            nodes = [
                helper.make_node("Gemm", ["hidden", "weight"], ["logits"],
                                 transB=1),
            ]
            graph = helper.make_graph(nodes, "routing", inputs, outputs,
                                      [weight_init])

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model.SerializeToString()
    except ImportError:
        return b""


# ---------------------------------------------------------------------------
# OpenVINO NPU Backend
# ---------------------------------------------------------------------------

class OpenVINONPUBackend:
    """Run the routing MLP on Intel NPU via OpenVINO async inference.

    The compiled model blob is cached to disk so subsequent sessions skip
    model compilation (which can take several seconds on first run).
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._core: Any   = None
        self._compiled: Dict[int, Any] = {}  # layer_id → compiled model
        self._infer_reqs: Dict[int, Any] = {}
        self._cache_dir   = Path(cache_dir) if cache_dir else None
        self._lock        = threading.Lock()
        self._calls       = 0
        self._npu_calls   = 0
        self._async_pending: Dict[int, Tuple[float, Any]] = {}

        if _HAS_OPENVINO:
            try:
                self._core = ov.Core()
                devices = self._core.available_devices
                self._device = "NPU" if "NPU" in devices else (
                    "GPU" if "GPU" in devices else "CPU")
                logger.info("phase1h: OpenVINO backend on device=%s", self._device)
            except Exception as exc:
                logger.warning("phase1h: OpenVINO init failed: %s", exc)
                self._core = None

    def load_spec(self, spec: RoutingMLPSpec, onnx_bytes: bytes) -> bool:
        """Compile and cache the routing MLP for a given layer."""
        if self._core is None or not onnx_bytes:
            return False

        cache_path = None
        if self._cache_dir is not None:
            cache_path = self._cache_dir / f"{spec.model_id}_layer{spec.layer_id}{_CACHE_SUFFIX_OV}"

        try:
            # Try loading from cache first
            if cache_path and (cache_path).exists():
                compiled = self._core.compile_model(str(cache_path), self._device)
            else:
                model = self._core.read_model(bytes=onnx_bytes)
                compiled = self._core.compile_model(model, self._device)
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    compiled.export_model(str(cache_path))

            with self._lock:
                self._compiled[spec.layer_id] = compiled
                self._infer_reqs[spec.layer_id] = compiled.create_infer_request()
            logger.info("phase1h: OpenVINO compiled layer %d on %s",
                        spec.layer_id, self._device)
            return True
        except Exception as exc:
            logger.warning("phase1h: OpenVINO compile failed layer %d: %s",
                           spec.layer_id, exc)
            return False

    def submit_async(self, layer_id: int, hidden: np.ndarray) -> bool:
        """Submit routing MLP inference asynchronously on NPU.

        Returns True if the async request was submitted successfully.
        The CPU can proceed with loading experts; call ``collect(layer_id)``
        just before the router logits are needed.
        """
        with self._lock:
            req = self._infer_reqs.get(layer_id)
        if req is None:
            return False
        try:
            h32 = np.ascontiguousarray(hidden.astype(np.float32))
            req.set_input_tensor(ov.Tensor(h32))
            req.start_async()
            with self._lock:
                self._async_pending[layer_id] = (time.monotonic(), req)
                self._npu_calls += 1
            return True
        except Exception as exc:
            logger.debug("phase1h: OpenVINO async submit failed: %s", exc)
            return False

    def collect(self, layer_id: int) -> Optional[np.ndarray]:
        """Wait for and return async inference result for layer_id."""
        with self._lock:
            pending = self._async_pending.pop(layer_id, None)
        if pending is None:
            return None
        _, req = pending
        try:
            req.wait()
            out = req.get_output_tensor().data.copy()
            with self._lock:
                self._calls += 1
            return out
        except Exception as exc:
            logger.debug("phase1h: OpenVINO collect failed: %s", exc)
            return None

    def forward_sync(self, layer_id: int, hidden: np.ndarray) -> Optional[np.ndarray]:
        """Synchronous inference for layers where async is not needed."""
        if self.submit_async(layer_id, hidden):
            return self.collect(layer_id)
        return None

    def stats(self) -> dict:
        with self._lock:
            return {
                "ov_device": getattr(self, "_device", "none"),
                "ov_compiled_layers": len(self._compiled),
                "ov_calls": self._calls,
                "ov_npu_calls": self._npu_calls,
                "ov_available": self._core is not None,
            }


# ---------------------------------------------------------------------------
# DirectML NPU Backend
# ---------------------------------------------------------------------------

class DirectMLNPUBackend:
    """Run the routing MLP on Intel NPU/iGPU via DirectML (Windows)."""

    def __init__(self) -> None:
        self._sessions: Dict[int, Any] = {}
        self._lock   = threading.Lock()
        self._calls  = 0
        self._available = _HAS_ORT and _HAS_DIRECTML_NPU
        if self._available:
            logger.info("phase1h: DirectML NPU backend available")

    def load_spec(self, spec: RoutingMLPSpec, onnx_bytes: bytes) -> bool:
        if not self._available or not onnx_bytes:
            return False
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            sess = ort.InferenceSession(
                onnx_bytes,
                sess_options=opts,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
            with self._lock:
                self._sessions[spec.layer_id] = sess
            logger.info("phase1h: DirectML compiled layer %d", spec.layer_id)
            return True
        except Exception as exc:
            logger.warning("phase1h: DirectML compile failed layer %d: %s",
                           spec.layer_id, exc)
            return False

    def forward(self, layer_id: int, hidden: np.ndarray) -> Optional[np.ndarray]:
        with self._lock:
            sess = self._sessions.get(layer_id)
        if sess is None:
            return None
        try:
            h32 = np.ascontiguousarray(hidden.astype(np.float32))
            [logits] = sess.run(["logits"], {"hidden": h32})
            with self._lock:
                self._calls += 1
            return logits
        except Exception as exc:
            logger.debug("phase1h: DirectML forward failed: %s", exc)
            return None

    def stats(self) -> dict:
        with self._lock:
            return {
                "dml_sessions": len(self._sessions),
                "dml_calls": self._calls,
                "dml_available": self._available,
            }


# ---------------------------------------------------------------------------
# NPURoutingDispatch — unified entry point
# ---------------------------------------------------------------------------

class NPURoutingDispatch:
    """Route expert routing MLP calls to NPU or CPU.

    Usage::

        dispatch = NPURoutingDispatch(cache_dir=state_dir / "npu_cache")
        dispatch.load_routing_mlp(spec)          # once per layer at session open
        # --- during forward pass ---
        dispatch.submit_async(layer_id, hidden)  # fire-and-forget on NPU
        # ... CPU loads experts for layer_id ...
        logits = dispatch.collect(layer_id)      # NPU result now ready
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._ov  = OpenVINONPUBackend(cache_dir)
        self._dml = DirectMLNPUBackend()
        self._specs: Dict[int, RoutingMLPSpec] = {}
        self._lock   = threading.Lock()
        self._cpu_fallbacks = 0

    def load_routing_mlp(self, spec: RoutingMLPSpec) -> None:
        """Compile the routing MLP for ``spec.layer_id`` onto the NPU."""
        onnx_bytes = _build_onnx_routing_mlp(spec)
        with self._lock:
            self._specs[spec.layer_id] = spec

        # Try OpenVINO first, then DirectML
        if not self._ov.load_spec(spec, onnx_bytes):
            if not self._dml.load_spec(spec, onnx_bytes):
                logger.debug("phase1h: layer %d on CPU (no NPU backend compiled)",
                             spec.layer_id)

    def submit_async(self, layer_id: int, hidden: np.ndarray) -> bool:
        """Fire-and-forget NPU inference for layer_id."""
        return self._ov.submit_async(layer_id, hidden)

    def collect(self, layer_id: int) -> Optional[np.ndarray]:
        """Retrieve NPU result (OpenVINO async path)."""
        result = self._ov.collect(layer_id)
        if result is not None:
            return result
        # Synchronous DML path
        with self._lock:
            spec = self._specs.get(layer_id)
        if spec is not None:
            dml_res = self._dml.forward(layer_id, np.zeros((1, spec.hidden_size), dtype=np.float32))
            if dml_res is not None:
                return dml_res
        return None

    def forward(self, layer_id: int, hidden: np.ndarray) -> np.ndarray:
        """Synchronous routing forward (submit + collect + CPU fallback)."""
        # Try OpenVINO async submit+collect
        if self._ov.submit_async(layer_id, hidden):
            result = self._ov.collect(layer_id)
            if result is not None:
                return result

        # Try DirectML sync
        dml = self._dml.forward(layer_id, hidden)
        if dml is not None:
            return dml

        # CPU fallback
        with self._lock:
            spec = self._specs.get(layer_id)
            self._cpu_fallbacks += 1
        if spec is not None:
            return spec.apply_numpy(hidden)
        # No spec — return zeros (caller should handle gracefully)
        return np.zeros((hidden.shape[0], 0), dtype=np.float32)

    def stats(self) -> dict:
        s: dict = {
            "npu_cpu_fallbacks": self._cpu_fallbacks,
            "npu_layers_loaded": len(self._specs),
        }
        s.update(self._ov.stats())
        s.update(self._dml.stats())
        return s


# ---------------------------------------------------------------------------
# Phase1HNPUSession — session wrapper
# ---------------------------------------------------------------------------

class Phase1HNPUSession:
    """Wrap a Phase 1G session with NPU routing offload.

    This is the final layer of the Phase 1 session stack::

        Phase1GOptimizedSession
          └── Phase1HNPUSession   ← this class

    At session open the caller registers each layer's routing MLP via
    ``register_routing_mlp(layer_id, weight, bias, hidden_size, num_experts)``.

    During the forward pass (wired via Phase 1F hooks) the session intercepts
    ``on_token_routed`` to also submit routing predictions to the NPU for the
    *next* layer ahead of time (overlap NPU routing with CPU expert load).
    """

    def __init__(
        self,
        phase1g_session: Any,
        *,
        cache_dir: Optional[Path] = None,
        model_id: str = "unknown",
    ) -> None:
        if phase1g_session is None:
            raise ValueError("phase1g_session is required")

        self.base_session = phase1g_session
        self._lock   = threading.RLock()
        self._closed = False
        self._model_id = model_id

        # Resolve cache dir from session stack
        if cache_dir is None:
            state_dir = getattr(phase1g_session, "state_directory", None)
            if state_dir:
                cache_dir = Path(state_dir) / "npu_cache"

        self._dispatch = NPURoutingDispatch(cache_dir)
        self._hidden_cache: Dict[int, np.ndarray] = {}  # pending hidden states

    # -- routing MLP registration ---------------------------------------------

    def register_routing_mlp(
        self,
        layer_id: int,
        weight: np.ndarray,
        num_experts: int,
        bias: Optional[np.ndarray] = None,
    ) -> None:
        """Upload routing MLP weights for layer_id to the NPU."""
        spec = RoutingMLPSpec(
            layer_id=layer_id,
            model_id=self._model_id,
            hidden_size=weight.shape[1] if weight.ndim == 2 else weight.shape[0],
            num_experts=num_experts,
            weight=weight.astype(np.float32),
            bias=bias.astype(np.float32) if bias is not None else None,
        )
        self._dispatch.load_routing_mlp(spec)

    # -- overlapping hook: submit NPU ahead of time ---------------------------

    def on_token_routed(self, layer_id: int, expert_ids: List[str]) -> None:
        """Called after CPU routing for layer_id; submit NPU for layer_id+1."""
        hook = getattr(self.base_session, "on_token_routed", None)
        if callable(hook):
            hook(layer_id, expert_ids)

        # Submit routing for next layer if we have a cached hidden state
        next_layer = layer_id + 1
        with self._lock:
            hidden = self._hidden_cache.pop(next_layer, None)
        if hidden is not None:
            self._dispatch.submit_async(next_layer, hidden)

    def prefeed_hidden(self, layer_id: int, hidden: np.ndarray) -> None:
        """Store hidden state for layer_id so it can be routed on the NPU.

        The model forward pass calls this with the hidden state just before
        applying the routing MLP; Phase 1H submits the NPU kernel and later
        collects the result via ``collect_routing(layer_id)``.
        """
        with self._lock:
            self._hidden_cache[layer_id] = hidden
        self._dispatch.submit_async(layer_id, hidden)

    def collect_routing(self, layer_id: int) -> Optional[np.ndarray]:
        """Collect NPU routing logits for layer_id (may block briefly)."""
        return self._dispatch.collect(layer_id)

    def route_sync(self, layer_id: int, hidden: np.ndarray) -> np.ndarray:
        """Synchronous routing (submit + collect + CPU fallback)."""
        return self._dispatch.forward(layer_id, hidden)

    # -- generation ------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Phase 1H session is closed")

    def generate(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                  **kwargs: Any) -> Any:
        with self._lock:
            self._check_open()
        return self.base_session.generate(*args, cancelled=cancelled, **kwargs)

    def stream(self, *args: Any, cancelled: Callable[[], bool] | None = None,
                **kwargs: Any) -> Any:
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
        return {**base, "phase1h": self._dispatch.stats()}

    def status(self) -> dict:
        base: dict = {}
        get_s = getattr(self.base_session, "status", None)
        if callable(get_s):
            base = get_s()
        with self._lock:
            return {**base, "phase1h": {"closed": self._closed, **self._dispatch.stats()}}

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        from .failure_reporting import add_failure_note, failure_detail
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
                record(err, "phase1g session close failure")

            if primary is not None:
                raise primary
            self._closed = True

    def __enter__(self) -> "Phase1HNPUSession":
        self._check_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def wrap_phase1g_with_phase1h(
    phase1g_session: Any,
    *,
    cache_dir: Optional[Path] = None,
    model_id: str = "unknown",
) -> Phase1HNPUSession:
    """Bind Phase 1H NPU offload to an existing Phase 1G session."""
    return Phase1HNPUSession(
        phase1g_session,
        cache_dir=cache_dir,
        model_id=model_id,
    )


__all__ = [
    "RoutingMLPSpec",
    "OpenVINONPUBackend",
    "DirectMLNPUBackend",
    "NPURoutingDispatch",
    "Phase1HNPUSession",
    "wrap_phase1g_with_phase1h",
]
