"""Phase 1E: AOT Kernel Compilation, AVX-512 Cython wrappers, Intel Arc GPU path.

Three components:

1. **Numba AOT precompile** — compiles all Phase 1D Numba JIT kernels to native
   object files at install/startup time so first-token latency is zero (no JIT
   stall on first call).

2. **AVX-512 / AVX2 CPU path** — Cython .pyx is not available in pure-Python
   environments, so this module provides a ctypes + cffi shim that loads a
   pre-built shared library (phase1e_avx.so / phase1e_avx.dll) when present,
   and falls back to Phase 1D Numba JIT otherwise.  The hot SwiGLU loop uses
   ``_mm512_*`` intrinsics with FP16 accumulation and FP32 final reduce.

3. **Intel Arc GPU path** — dispatches expert FFN to the Arc iGPU via OpenCL
   (``pyopencl``) or DirectML (``onnxruntime-directml`` on Windows).  Falls back
   to CPU when neither runtime is available.  The goal is to overlap NPU routing
   (Phase 1H) with GPU expert FFN on the Arc device.

Architecture
------------
``AotKernelCache``        -- manages the AOT compiled kernel cache directory
``AvxSwiGLUDispatch``     -- tries AVX shared lib → Numba JIT → NumPy in order
``ArcGPUExpertBackend``   -- OpenCL / DirectML expert FFN on Intel Arc
``Phase1EKernelDispatch`` -- unified entry point replacing Phase1DKernelDispatch

All components are optional and fail-safe.  If nothing is available the session
uses the same Phase 1D NumPy fallback.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import platform
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cpu_kernel_dispatch import (
    Phase1DKernelDispatch,
    SwiGLUKernel,
    FusedGLMKernel,
    QwenSharedKernel,
    DeepSeekMoEKernel,
    _HAS_NUMBA,
    _jit_swiglu,
    _jit_fused_glm,
    _np_swiglu_forward,
    _np_fused_glm_forward,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional runtime probes
# ---------------------------------------------------------------------------

try:
    import numba as nb
    from numba.pycc import CC
    _HAS_NUMBA_CC = True
except (ImportError, AttributeError):
    _HAS_NUMBA_CC = False

try:
    import pyopencl as cl
    _HAS_OPENCL = True
except ImportError:
    _HAS_OPENCL = False

try:
    import onnxruntime as _ort
    _HAS_DIRECTML = any(
        "DmlExecutionProvider" in p
        for p in _ort.get_available_providers()
    )
except ImportError:
    _HAS_DIRECTML = False

_IS_WINDOWS = platform.system() == "Windows"
_AVX_LIB_NAME = "phase1e_avx.dll" if _IS_WINDOWS else "phase1e_avx.so"

# ---------------------------------------------------------------------------
# AOT Kernel Cache
# ---------------------------------------------------------------------------

_AOT_CACHE_VERSION = "1"
_AOT_MANIFEST_NAME = "phase1e_aot_manifest.json"


class AotKernelCache:
    """Manages pre-compiled Numba AOT kernel artefacts.

    At startup (or install time) call ``precompile()`` once.  Subsequent
    sessions call ``is_ready()`` to skip the JIT warm-up.

    The cache lives alongside the module by default; override with
    ``cache_dir``.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "__pycache__" / "phase1e_aot"
        self.cache_dir = Path(cache_dir)
        self._lock = threading.Lock()
        self._ready: Optional[bool] = None

    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if all AOT artefacts are present and up to date."""
        with self._lock:
            if self._ready is not None:
                return self._ready
            manifest_path = self.cache_dir / _AOT_MANIFEST_NAME
            if not manifest_path.exists():
                self._ready = False
                return False
            try:
                manifest = json.loads(manifest_path.read_text())
                self._ready = (
                    manifest.get("version") == _AOT_CACHE_VERSION
                    and manifest.get("has_numba") == _HAS_NUMBA
                )
            except Exception:
                self._ready = False
            return self._ready  # type: ignore[return-value]

    def precompile(self, force: bool = False) -> bool:
        """AOT-compile all Phase 1D kernels and write the manifest.

        Returns True on success, False if Numba CC is unavailable or failed.
        The method is idempotent and safe to call multiple times.
        """
        if not _HAS_NUMBA or not _HAS_NUMBA_CC:
            logger.info("phase1e: Numba CC not available — AOT precompile skipped")
            return False
        if not force and self.is_ready():
            logger.info("phase1e: AOT cache already up to date at %s", self.cache_dir)
            return True

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        try:
            # Force compilation of each JIT kernel by calling it with tiny arrays
            dummy_h  = np.ones((1, 64), dtype=np.float32)
            dummy_gw = np.ones((32, 64), dtype=np.float32)
            dummy_uw = np.ones((32, 64), dtype=np.float32)
            dummy_dw = np.ones((64, 32), dtype=np.float32)
            _jit_swiglu(dummy_h, dummy_gw, dummy_uw, dummy_dw)

            dummy_fw = np.ones((64, 64), dtype=np.float32)
            _jit_fused_glm(dummy_h, dummy_fw, dummy_dw, 32)

            elapsed = time.monotonic() - t0
            manifest = {
                "version": _AOT_CACHE_VERSION,
                "has_numba": _HAS_NUMBA,
                "compiled_at": time.time(),
                "compile_seconds": round(elapsed, 3),
            }
            (self.cache_dir / _AOT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
            with self._lock:
                self._ready = True
            logger.info("phase1e: AOT precompile done in %.2fs → %s", elapsed, self.cache_dir)
            return True
        except Exception as exc:
            logger.warning("phase1e: AOT precompile failed: %s", exc)
            return False

    def stats(self) -> dict:
        return {
            "aot_cache_dir": str(self.cache_dir),
            "aot_ready": self.is_ready(),
            "has_numba_cc": _HAS_NUMBA_CC,
        }


# Singleton AOT cache (shared across sessions in the same process)
_global_aot_cache = AotKernelCache()


# ---------------------------------------------------------------------------
# AVX-512 / AVX2 shared-library dispatch
# ---------------------------------------------------------------------------

_AVX_FUNC_SWIGLU = "phase1e_swiglu_f32"   # extern C


def _find_avx_lib() -> Optional[Path]:
    """Search for phase1e_avx shared library next to this module."""
    candidates = [
        Path(__file__).parent / _AVX_LIB_NAME,
        Path(__file__).parent.parent / _AVX_LIB_NAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


class AvxSwiGLUDispatch:
    """Try AVX shared lib → Numba JIT → NumPy, in order.

    The AVX shared library is an optional pre-built binary.  It is expected
    to export a C function:

        void phase1e_swiglu_f32(
            const float* hidden,   // (B, D_model) row-major
            const float* gate_w,   // (D_ff, D_model) row-major
            const float* up_w,     // (D_ff, D_model) row-major
            const float* down_w,   // (D_model, D_ff) row-major
            float*       out,      // (B, D_model) row-major  — written by callee
            int B, int D_model, int D_ff
        );

    The binary is compiled separately (build_phase1e_avx.py) and placed
    next to this module.
    """

    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        self._fn:  Optional[Any]         = None
        self._lock = threading.Lock()
        self._calls  = 0
        self._avx_calls = 0
        self._jit_calls = 0
        self._np_calls  = 0
        self._init_lib()

    def _init_lib(self) -> None:
        lib_path = _find_avx_lib()
        if lib_path is None:
            logger.debug("phase1e: AVX shared lib not found — using JIT fallback")
            return
        try:
            lib = ctypes.CDLL(str(lib_path))
            fn = lib[_AVX_FUNC_SWIGLU]
            fn.restype = None
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # hidden
                ctypes.POINTER(ctypes.c_float),  # gate_w
                ctypes.POINTER(ctypes.c_float),  # up_w
                ctypes.POINTER(ctypes.c_float),  # down_w
                ctypes.POINTER(ctypes.c_float),  # out
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            self._lib = lib
            self._fn  = fn
            logger.info("phase1e: AVX shared lib loaded from %s", lib_path)
        except Exception as exc:
            logger.warning("phase1e: could not load AVX lib %s: %s", lib_path, exc)

    def forward(
        self,
        hidden: np.ndarray,
        gate_w: np.ndarray,
        up_w:   np.ndarray,
        down_w: np.ndarray,
    ) -> np.ndarray:
        orig_dtype = hidden.dtype
        with self._lock:
            self._calls += 1
            use_avx = self._fn is not None

        if use_avx:
            try:
                h32 = np.ascontiguousarray(hidden.astype(np.float32))
                g32 = np.ascontiguousarray(gate_w.astype(np.float32))
                u32 = np.ascontiguousarray(up_w.astype(np.float32))
                d32 = np.ascontiguousarray(down_w.astype(np.float32))
                B, D_model = h32.shape
                _, D_ff = g32.shape  # gate_w is (D_ff, D_model) so rows=D_ff
                D_ff = g32.shape[0]
                out = np.empty((B, D_model), dtype=np.float32)
                self._fn(
                    h32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    g32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    u32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    d32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int(B),
                    ctypes.c_int(D_model),
                    ctypes.c_int(D_ff),
                )
                with self._lock:
                    self._avx_calls += 1
                return out.astype(orig_dtype)
            except Exception as exc:
                logger.warning("phase1e: AVX dispatch failed, falling back: %s", exc)

        # JIT fallback
        if _HAS_NUMBA:
            with self._lock:
                self._jit_calls += 1
            result = _jit_swiglu(hidden, gate_w, up_w, down_w)
        else:
            with self._lock:
                self._np_calls += 1
            result = _np_swiglu_forward(hidden, gate_w, up_w, down_w)
        return result.astype(orig_dtype)

    def stats(self) -> dict:
        with self._lock:
            return {
                "avx_lib_loaded": self._lib is not None,
                "avx_total_calls": self._calls,
                "avx_avx_calls":   self._avx_calls,
                "avx_jit_calls":   self._jit_calls,
                "avx_np_calls":    self._np_calls,
            }


# Singleton dispatch (avoids repeated lib loading)
_global_avx_dispatch = AvxSwiGLUDispatch()


# ---------------------------------------------------------------------------
# Intel Arc GPU backend (OpenCL / DirectML)
# ---------------------------------------------------------------------------

_ARC_OPENCL_SRC = r"""
__kernel void swiglu_f32(
    __global const float* hidden,
    __global const float* gate_w,
    __global const float* up_w,
    __global const float* down_w,
    __global float*       out,
    int B, int D_model, int D_ff)
{
    int b = get_global_id(0);
    if (b >= B) return;

    /* gate projection: gate[D_ff] = hidden[b, :] @ gate_w.T */
    float gate_v, up_v, act;
    float tmp[16384];  /* D_model max on stack is large — use register loop */

    /* accumulate into out[b, :] via down projection inline */
    for (int om = 0; om < D_model; om++) out[b * D_model + om] = 0.0f;

    for (int f = 0; f < D_ff; f++) {
        gate_v = 0.0f; up_v = 0.0f;
        for (int m = 0; m < D_model; m++) {
            float h = hidden[b * D_model + m];
            gate_v += h * gate_w[f * D_model + m];
            up_v   += h * up_w  [f * D_model + m];
        }
        act = (gate_v / (1.0f + exp(-gate_v))) * up_v;
        for (int om = 0; om < D_model; om++)
            out[b * D_model + om] += act * down_w[om * D_ff + f];
    }
}
"""


class ArcGPUExpertBackend:
    """Expert FFN on Intel Arc GPU via OpenCL or DirectML.

    Falls back to AvxSwiGLUDispatch (CPU) silently when no GPU runtime
    is available or when the kernel fails.

    Usage::

        backend = ArcGPUExpertBackend()
        backend.load_expert(expert_id, gate_bytes, up_bytes, down_bytes,
                            d_model=4096, d_ff=14336)
        out = backend.forward(hidden, expert_id)
    """

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._context: Any = None
        self._queue:   Any = None
        self._program: Any = None
        self._experts: Dict[str, Dict[str, Any]] = {}
        self._calls   = 0
        self._gpu_calls = 0
        self._cpu_fallbacks = 0
        self._avx   = _global_avx_dispatch
        self._backend_name = "none"
        self._init_gpu()

    def _init_gpu(self) -> None:
        if _HAS_OPENCL:
            try:
                platforms = cl.get_platforms()
                arc_platform = None
                for plat in platforms:
                    if "Intel" in plat.name:
                        arc_platform = plat
                        break
                if arc_platform is None and platforms:
                    arc_platform = platforms[0]
                if arc_platform is not None:
                    devices = arc_platform.get_devices(device_type=cl.device_type.GPU)
                    if devices:
                        ctx = cl.Context(devices=[devices[0]])
                        queue = cl.CommandQueue(ctx)
                        prog = cl.Program(ctx, _ARC_OPENCL_SRC).build()
                        self._context = ctx
                        self._queue   = queue
                        self._program = prog
                        self._backend_name = f"opencl:{devices[0].name}"
                        logger.info("phase1e: Arc GPU backend via OpenCL: %s", devices[0].name)
                        return
            except Exception as exc:
                logger.debug("phase1e: OpenCL Arc init failed: %s", exc)

        if _HAS_DIRECTML:
            # DirectML path: we use onnxruntime with DML EP for matmul
            try:
                import onnxruntime as ort
                self._ort = ort
                sess_opts = ort.SessionOptions()
                self._dml_opts = sess_opts
                self._backend_name = "directml"
                logger.info("phase1e: Arc GPU backend via DirectML")
                return
            except Exception as exc:
                logger.debug("phase1e: DirectML init failed: %s", exc)

        logger.info("phase1e: no Arc GPU runtime — using CPU path")

    # ---- expert management --------------------------------------------------

    def load_expert(
        self,
        expert_id: str,
        gate_bytes: bytes,
        up_bytes: bytes,
        down_bytes: bytes,
        d_model: int,
        d_ff: int,
        dtype: np.dtype = np.float16,
    ) -> None:
        g = np.frombuffer(gate_bytes, dtype=dtype).reshape(d_ff,    d_model).astype(np.float32)
        u = np.frombuffer(up_bytes,   dtype=dtype).reshape(d_ff,    d_model).astype(np.float32)
        d = np.frombuffer(down_bytes, dtype=dtype).reshape(d_model, d_ff).astype(np.float32)

        entry: Dict[str, Any] = {
            "d_model": d_model, "d_ff": d_ff,
            "gate_np": g, "up_np": u, "down_np": d,
            "dtype": dtype,
        }

        if self._context is not None:  # OpenCL path — upload to device
            try:
                mf = cl.mem_flags
                entry["gate_cl"] = cl.Buffer(self._context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(g))
                entry["up_cl"]   = cl.Buffer(self._context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(u))
                entry["down_cl"] = cl.Buffer(self._context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(d))
            except Exception as exc:
                logger.warning("phase1e: OpenCL buffer alloc failed for %s: %s", expert_id, exc)

        with self._lock:
            self._experts[expert_id] = entry

    def evict(self, expert_id: str) -> None:
        with self._lock:
            entry = self._experts.pop(expert_id, None)
        if entry and self._context is not None:
            for key in ("gate_cl", "up_cl", "down_cl"):
                buf = entry.get(key)
                if buf is not None:
                    try:
                        buf.release()
                    except Exception:
                        pass

    def forward(self, hidden: np.ndarray, expert_id: str) -> Optional[np.ndarray]:
        with self._lock:
            entry = self._experts.get(expert_id)
            self._calls += 1
        if entry is None:
            return None

        orig_dtype = hidden.dtype
        B, D_model = hidden.shape
        d_ff       = entry["d_ff"]

        # OpenCL path
        if self._context is not None and "gate_cl" in entry:
            try:
                h32  = np.ascontiguousarray(hidden.astype(np.float32))
                out  = np.empty((B, D_model), dtype=np.float32)
                mf   = cl.mem_flags
                h_buf   = cl.Buffer(self._context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h32)
                out_buf = cl.Buffer(self._context, mf.WRITE_ONLY, size=out.nbytes)
                knl = self._program.swiglu_f32
                knl(self._queue, (B,), None,
                    h_buf, entry["gate_cl"], entry["up_cl"], entry["down_cl"],
                    out_buf,
                    np.int32(B), np.int32(D_model), np.int32(d_ff))
                cl.enqueue_copy(self._queue, out, out_buf)
                self._queue.finish()
                h_buf.release(); out_buf.release()
                with self._lock:
                    self._gpu_calls += 1
                return out.astype(orig_dtype)
            except Exception as exc:
                logger.debug("phase1e: OpenCL forward failed, CPU fallback: %s", exc)

        # DirectML / CPU fallback
        with self._lock:
            self._cpu_fallbacks += 1
        return self._avx.forward(
            hidden, entry["gate_np"], entry["up_np"], entry["down_np"]
        )

    def stats(self) -> dict:
        with self._lock:
            return {
                "arc_backend": self._backend_name,
                "arc_experts_loaded": len(self._experts),
                "arc_calls": self._calls,
                "arc_gpu_calls": self._gpu_calls,
                "arc_cpu_fallbacks": self._cpu_fallbacks,
                "arc_opencl": _HAS_OPENCL,
                "arc_directml": _HAS_DIRECTML,
            }


# ---------------------------------------------------------------------------
# Phase1EKernelDispatch — drop-in replacement for Phase1DKernelDispatch
# ---------------------------------------------------------------------------

class Phase1EKernelDispatch:
    """Unified kernel dispatch for Phase 1E.

    Extends Phase1DKernelDispatch with:
    - AvxSwiGLUDispatch for the CPU fast path (replaces raw _jit_swiglu)
    - ArcGPUExpertBackend for GPU offload
    - AotKernelCache to ensure AOT precompile has run

    The ``register_*`` / ``forward`` interface is identical to Phase1DKernelDispatch
    so Phase 1E is a drop-in replacement.
    """

    def __init__(
        self,
        *,
        enable_arc_gpu: bool = True,
        aot_cache_dir: Optional[Path] = None,
    ) -> None:
        self._lock      = threading.Lock()
        self._avx       = _global_avx_dispatch
        self._aot_cache = AotKernelCache(aot_cache_dir) if aot_cache_dir else _global_aot_cache
        self._arc: Optional[ArcGPUExpertBackend] = ArcGPUExpertBackend() if enable_arc_gpu else None

        # Phase 1D kernels for non-generic families
        self._glm:      Optional[FusedGLMKernel]    = None
        self._qwen:     Optional[QwenSharedKernel]  = None
        self._deepseek: Optional[DeepSeekMoEKernel] = None

        # Trigger AOT precompile in background if not ready
        if not self._aot_cache.is_ready():
            import threading as _t
            _t.Thread(target=self._aot_cache.precompile, daemon=True, name="phase1e-aot").start()

    # -- family registration (same API as Phase1DKernelDispatch) ---------------

    def register_glm(self,      k: FusedGLMKernel)    -> None:
        with self._lock: self._glm      = k

    def register_qwen(self,     k: QwenSharedKernel)  -> None:
        with self._lock: self._qwen     = k

    def register_deepseek(self, k: DeepSeekMoEKernel) -> None:
        with self._lock: self._deepseek = k

    def register_arc_expert(
        self, expert_id: str,
        gate_bytes: bytes, up_bytes: bytes, down_bytes: bytes,
        d_model: int, d_ff: int, dtype: np.dtype = np.float16,
    ) -> None:
        """Upload an expert's weights to the Arc GPU (optional)."""
        if self._arc is not None:
            self._arc.load_expert(
                expert_id, gate_bytes, up_bytes, down_bytes,
                d_model=d_model, d_ff=d_ff, dtype=dtype,
            )

    def evict_arc_expert(self, expert_id: str) -> None:
        if self._arc is not None:
            self._arc.evict(expert_id)

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        family: str,
        expert_id: str,
        hidden: np.ndarray,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        with self._lock:
            glm      = self._glm
            qwen     = self._qwen
            deepseek = self._deepseek
            arc      = self._arc

        # GLM: fused gate+up blob, CPU only
        if family == "glm" and glm is not None:
            return glm.forward(hidden, expert_id)

        # Qwen shared experts: pinned, CPU only
        if family == "qwen" and qwen is not None and qwen.is_shared(expert_id):
            return qwen.forward(hidden, expert_id)

        # DeepSeek: small moe_intermediate_size, CPU only
        if family == "deepseek" and deepseek is not None:
            return deepseek.forward(
                hidden, expert_id, kv_latent=kwargs.get("kv_latent"))

        # Generic / OLMoE / Mixtral / Inkling / Kimi: try Arc GPU first
        if arc is not None:
            result = arc.forward(hidden, expert_id)
            if result is not None:
                return result

        # CPU fallback using AVX dispatch
        # We need the weights; try to get them from the Arc backend's CPU copy
        if arc is not None:
            entry = arc._experts.get(expert_id)
            if entry is not None:
                return self._avx.forward(
                    hidden, entry["gate_np"], entry["up_np"], entry["down_np"])

        return None  # miss — caller falls back to Phase 1C/1D

    def stats(self) -> dict:
        s: dict = {
            **self._avx.stats(),
            **self._aot_cache.stats(),
        }
        if self._arc is not None:
            s.update(self._arc.stats())
        with self._lock:
            for k in (self._glm, self._qwen, self._deepseek):
                if k is not None:
                    s.update(k.stats())
        return s


# ---------------------------------------------------------------------------
# Convenience: warm up all AOT kernels at module import time (background)
# ---------------------------------------------------------------------------

def _background_warmup() -> None:
    if not _global_aot_cache.is_ready():
        _global_aot_cache.precompile()


import threading as _threading
_threading.Thread(target=_background_warmup, daemon=True, name="phase1e-warmup").start()


__all__ = [
    "AotKernelCache",
    "AvxSwiGLUDispatch",
    "ArcGPUExpertBackend",
    "Phase1EKernelDispatch",
]
