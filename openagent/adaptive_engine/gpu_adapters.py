"""Stage 6 — GPU adapter interface and concrete adapters.

Defines a unified adapter contract that the adaptive engine uses to submit
inference work to heterogeneous backends:
  - CudaAdapter: NVIDIA GPU via torch or cuBLAS
  - IntelArcAdapter: Intel Arc / integrated GPU via OpenVINO
  - CpuFallbackAdapter: pure CPU reference path

Each adapter:
  1. Implements GpuAdapter (abstract base class).
  2. Reports its own availability without raising on missing deps.
  3. Exposes a projection(weight_tensor, input_tensor) path for expert execution.
  4. Provides a benchmark() method returning achieved TFLOPS.
  5. Has a context-manager interface (open/close) for resource lifetime.

Import is safe in constrained environments — adapters only import heavy deps
inside their _load() method, which is called lazily on first use.
"""
from __future__ import annotations

import abc
import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hardware_detection import HardwareCapabilities


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

@dataclass
class AdapterBenchmark:
    adapter_name: str
    device_name: str
    achieved_tflops: float
    latency_ms: float
    matrix_size: int        # N in N×N matmul used for benchmark
    arithmetic: str         # e.g. "fp32", "fp16", "bf16", "int8"
    notes: str = ""


class GpuAdapter(abc.ABC):
    """Abstract GPU adapter contract."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'cuda', 'openvino', 'cpu'."""

    @property
    @abc.abstractmethod
    def device_name(self) -> str:
        """Human-readable device name."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this adapter can be used on the current machine."""

    @abc.abstractmethod
    def load(self) -> None:
        """Initialise the adapter; idempotent after first call."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Release device resources; safe to call when already unloaded."""

    @abc.abstractmethod
    def projection(
        self,
        weight: Any,    # adapter-specific tensor type
        x: Any,         # adapter-specific tensor type
        *,
        arithmetic: str = "fp32",
    ) -> Any:
        """Compute weight @ x (or equivalent expert projection)."""

    @abc.abstractmethod
    def benchmark(self, *, matrix_size: int = 2048, arithmetic: str = "fp32") -> AdapterBenchmark:
        """Run a synthetic matmul benchmark and return results."""

    # Context manager support

    def __enter__(self) -> "GpuAdapter":
        self.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self.unload()


# ---------------------------------------------------------------------------
# CUDA adapter
# ---------------------------------------------------------------------------

class CudaAdapter(GpuAdapter):
    """NVIDIA CUDA adapter via PyTorch.

    Falls back gracefully when torch is not installed.
    """

    def __init__(self, *, device_index: int = 0):
        self._device_index = device_index
        self._device: Optional[Any] = None
        self._torch: Optional[Any] = None
        self._lock = threading.Lock()
        self._loaded = False
        self._device_name_cache: Optional[str] = None

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def device_name(self) -> str:
        if self._device_name_cache:
            return self._device_name_cache
        if not self.is_available():
            return f"CUDA:{self._device_index} (unavailable)"
        try:
            import torch
            return torch.cuda.get_device_name(self._device_index)
        except Exception:
            return f"CUDA:{self._device_index}"

    def is_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available() and self._device_index < torch.cuda.device_count()
        except ImportError:
            return False
        except Exception:
            return False

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if not self.is_available():
                raise RuntimeError(
                    f"CUDA device {self._device_index} is not available")
            import torch
            self._torch = torch
            self._device = torch.device(f"cuda:{self._device_index}")
            self._device_name_cache = torch.cuda.get_device_name(self._device_index)
            self._loaded = True

    def unload(self) -> None:
        with self._lock:
            if not self._loaded:
                return
            if self._torch is not None:
                try:
                    self._torch.cuda.empty_cache()
                except Exception:
                    pass
            self._torch = None
            self._device = None
            self._loaded = False

    def projection(self, weight: Any, x: Any, *, arithmetic: str = "fp32") -> Any:
        if not self._loaded:
            self.load()
        torch = self._torch
        device = self._device
        w = torch.as_tensor(weight, device=device)
        xi = torch.as_tensor(x, device=device)
        if arithmetic == "fp16":
            w = w.half()
            xi = xi.half()
        elif arithmetic == "bf16":
            w = w.bfloat16()
            xi = xi.bfloat16()
        return torch.mm(w, xi) if w.dim() == 2 and xi.dim() == 2 else w @ xi

    def benchmark(self, *, matrix_size: int = 2048, arithmetic: str = "fp32") -> AdapterBenchmark:
        if not self.is_available():
            raise RuntimeError("CUDA not available for benchmark")
        self.load()
        torch = self._torch
        device = self._device
        N = matrix_size
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16,
                     "bf16": torch.bfloat16, "int8": torch.int8}
        dtype = dtype_map.get(arithmetic, torch.float32)
        a = torch.randn(N, N, device=device, dtype=dtype if dtype != torch.int8 else torch.float16)
        b = torch.randn(N, N, device=device, dtype=dtype if dtype != torch.int8 else torch.float16)
        # Warmup
        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        flops = 2 * N ** 3 * reps
        tflops = flops / elapsed / 1e12
        return AdapterBenchmark(
            adapter_name=self.name,
            device_name=self.device_name,
            achieved_tflops=round(tflops, 3),
            latency_ms=round(elapsed / reps * 1000, 3),
            matrix_size=N,
            arithmetic=arithmetic,
        )


# ---------------------------------------------------------------------------
# Intel Arc / OpenVINO adapter
# ---------------------------------------------------------------------------

class IntelArcAdapter(GpuAdapter):
    """Intel Arc or integrated GPU adapter via OpenVINO.

    Uses openvino.runtime for compilation and inference.
    Falls back gracefully when OpenVINO is not installed.
    """

    def __init__(self, *, ov_device_id: str = "GPU.0"):
        self._ov_device_id = ov_device_id
        self._core: Optional[Any] = None
        self._lock = threading.Lock()
        self._loaded = False
        self._device_name_cache: Optional[str] = None
        self._model_cache: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "openvino"

    @property
    def device_name(self) -> str:
        if self._device_name_cache:
            return self._device_name_cache
        try:
            from openvino.runtime import Core  # type: ignore[import]
            core = Core()
            props = core.get_property(self._ov_device_id, "FULL_DEVICE_NAME")
            return str(props)
        except Exception:
            return f"OpenVINO {self._ov_device_id}"

    def is_available(self) -> bool:
        try:
            from openvino.runtime import Core  # type: ignore[import]
            core = Core()
            return self._ov_device_id in core.available_devices
        except Exception:
            return False

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if not self.is_available():
                raise RuntimeError(
                    f"OpenVINO device {self._ov_device_id!r} is not available")
            from openvino.runtime import Core  # type: ignore[import]
            self._core = Core()
            try:
                self._device_name_cache = str(
                    self._core.get_property(self._ov_device_id, "FULL_DEVICE_NAME"))
            except Exception:
                self._device_name_cache = f"OpenVINO {self._ov_device_id}"
            self._loaded = True

    def unload(self) -> None:
        with self._lock:
            self._core = None
            self._model_cache.clear()
            self._loaded = False

    def projection(self, weight: Any, x: Any, *, arithmetic: str = "fp32") -> Any:
        """Run a single matmul projection via OpenVINO on the target device.

        ``weight`` is an ``[out_features, in_features]`` matrix and ``x`` is
        an ``[in_features, batch]`` matrix. The OpenVINO graph is compiled for
        the selected device and cached by weight contents and arithmetic.
        Returns a NumPy array containing the device result.
        """
        if not self._loaded:
            self.load()
        import numpy as np  # type: ignore[import]
        if arithmetic not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"OpenVINO projection does not support {arithmetic!r}")
        dtype = np.float16 if arithmetic == "fp16" else np.float32
        w = np.ascontiguousarray(weight, dtype=dtype)
        xi = np.ascontiguousarray(x, dtype=dtype)
        if w.ndim != 2 or xi.ndim != 2:
            raise ValueError("OpenVINO projection requires two-dimensional weight and input matrices")
        if w.shape[1] != xi.shape[0]:
            raise ValueError(f"projection dimensions do not align: {w.shape} @ {xi.shape}")

        # bfloat16 is represented as f32 inputs plus an explicit cast in the
        # graph. This avoids passing NumPy dtypes unsupported by some OV builds.
        ov_dtype = "bf16" if arithmetic == "bf16" else arithmetic
        cache_key = hashlib.sha256(
            w.dtype.str.encode("ascii") + str(w.shape).encode("ascii") + w.tobytes()
        ).hexdigest() + ":" + ov_dtype
        with self._lock:
            compiled = self._model_cache.get(cache_key)
            if compiled is None:
                import openvino as ov  # type: ignore[import]
                ops = getattr(ov, "opset13", None)
                if ops is None:
                    from openvino.runtime import opset13 as ops  # type: ignore[no-redef,import]
                element_type = ov.Type.f16 if arithmetic == "fp16" else ov.Type.f32
                input_node = ops.parameter(
                    ov.PartialShape([w.shape[1], -1]), element_type, name="projection_input",
                )
                weight_node = ops.constant(w, dtype=element_type)
                if arithmetic == "bf16":
                    # Use f32 at the host/device boundary, then make the
                    # requested precision explicit in the compiled graph.
                    weight_node = ops.convert(weight_node, ov.Type.bf16)
                    input_for_matmul = ops.convert(input_node, ov.Type.bf16)
                else:
                    input_for_matmul = input_node
                result_node = ops.matmul(weight_node, input_for_matmul, False, False)
                if arithmetic == "bf16":
                    result_node = ops.convert(result_node, ov.Type.f32)
                model = ov.Model([result_node], [input_node], "openagent_projection")
                compiled = self._core.compile_model(model, self._ov_device_id)
                # Bound compiled graphs so changing model weights cannot grow
                # this adapter's cache without limit.
                if len(self._model_cache) >= 32:
                    self._model_cache.pop(next(iter(self._model_cache)))
                self._model_cache[cache_key] = compiled
            input_port = compiled.input(0)
            output_port = compiled.output(0)
        result = compiled({input_port: xi})
        return np.asarray(result[output_port])

    def benchmark(self, *, matrix_size: int = 1024, arithmetic: str = "fp32") -> AdapterBenchmark:
        import numpy as np  # type: ignore[import]
        if matrix_size <= 0:
            raise ValueError("matrix_size must be positive")
        if not self.is_available():
            raise RuntimeError("OpenVINO GPU is not available for benchmark")
        if not self._loaded:
            self.load()
        N = matrix_size
        dtype = np.float16 if arithmetic == "fp16" else np.float32
        a = np.random.default_rng(0).standard_normal((N, N)).astype(dtype)
        b = np.random.default_rng(1).standard_normal((N, N)).astype(dtype)
        for _ in range(2):
            self.projection(a, b, arithmetic=arithmetic)
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            self.projection(a, b, arithmetic=arithmetic)
        elapsed = time.perf_counter() - t0
        flops = 2 * N ** 3 * reps
        tflops = flops / elapsed / 1e12
        return AdapterBenchmark(
            adapter_name=self.name,
            device_name=self.device_name,
            achieved_tflops=round(tflops, 4),
            latency_ms=round(elapsed / reps * 1000, 3),
            matrix_size=N,
            arithmetic=arithmetic,
            notes=f"compiled OpenVINO device {self._ov_device_id}",
        )


# ---------------------------------------------------------------------------
# CPU fallback adapter
# ---------------------------------------------------------------------------

class CpuFallbackAdapter(GpuAdapter):
    """Pure-CPU reference adapter.

    Uses NumPy; always available as long as NumPy is importable (it is a
    core openagent dependency).  Serves as the last-resort fallback and as
    the ground-truth implementation for correctness testing.
    """

    def __init__(self):
        self._loaded = False
        self._np: Optional[Any] = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def device_name(self) -> str:
        import platform
        return platform.processor() or "CPU"

    def is_available(self) -> bool:
        try:
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            import numpy as np
            self._np = np
            self._loaded = True

    def unload(self) -> None:
        with self._lock:
            self._np = None
            self._loaded = False

    def projection(self, weight: Any, x: Any, *, arithmetic: str = "fp32") -> Any:
        if not self._loaded:
            self.load()
        np = self._np
        w = np.asarray(weight, dtype=np.float32)
        xi = np.asarray(x, dtype=np.float32)
        return np.matmul(w, xi)

    def benchmark(self, *, matrix_size: int = 512, arithmetic: str = "fp32") -> AdapterBenchmark:
        self.load()
        np = self._np
        N = matrix_size
        a = np.random.randn(N, N).astype(np.float32)
        b = np.random.randn(N, N).astype(np.float32)
        for _ in range(2):
            np.matmul(a, b)
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            np.matmul(a, b)
        elapsed = time.perf_counter() - t0
        flops = 2 * N ** 3 * reps
        tflops = flops / elapsed / 1e12
        return AdapterBenchmark(
            adapter_name=self.name,
            device_name=self.device_name,
            achieved_tflops=round(tflops, 4),
            latency_ms=round(elapsed / reps * 1000, 3),
            matrix_size=N,
            arithmetic=arithmetic,
        )


# ---------------------------------------------------------------------------
# Adapter registry and factory
# ---------------------------------------------------------------------------

@dataclass
class AdapterRegistry:
    """Ordered list of adapters; selects the best available one."""
    adapters: List[GpuAdapter] = field(default_factory=list)

    def add(self, adapter: GpuAdapter) -> None:
        self.adapters.append(adapter)

    def best_available(self) -> Optional[GpuAdapter]:
        """Return the first available adapter in registry order (highest priority first)."""
        for adapter in self.adapters:
            if adapter.is_available():
                return adapter
        return None

    def all_available(self) -> List[GpuAdapter]:
        return [a for a in self.adapters if a.is_available()]

    def get(self, name: str) -> Optional[GpuAdapter]:
        for a in self.adapters:
            if a.name == name:
                return a
        return None


def build_adapter_registry(caps: Optional["HardwareCapabilities"] = None) -> AdapterRegistry:
    """Build a registry of adapters matching the detected hardware.

    Pass caps to avoid a second hardware detection round; if None the
    registry is built from what is importable without profiling.
    """
    reg = AdapterRegistry()

    # CUDA — highest priority
    if caps is not None:
        for dev in caps.cuda_devices:
            if dev.is_available:
                reg.add(CudaAdapter(device_index=dev.index))
    else:
        adapter = CudaAdapter(device_index=0)
        if adapter.is_available():
            reg.add(adapter)

    # OpenVINO Intel GPU
    if caps is not None:
        for gpu in caps.intel_gpus:
            if gpu.is_available and caps.openvino_available:
                reg.add(IntelArcAdapter(ov_device_id=gpu.openvino_device_id))
    else:
        ov_adapter = IntelArcAdapter(ov_device_id="GPU.0")
        if ov_adapter.is_available():
            reg.add(ov_adapter)

    # CPU fallback — always last
    reg.add(CpuFallbackAdapter())

    return reg
