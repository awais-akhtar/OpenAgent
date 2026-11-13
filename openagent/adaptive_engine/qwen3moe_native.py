"""ctypes binding for the Q4_K/Q6_K matvec kernels, built with MSVC when needed.

The C kernels release the GIL while they run, so rows are split across a
thread pool and computed in parallel. When the kernel library cannot be built
or loaded, `matvec` falls back to the NumPy dequantizers, which are slower but
compute the same values.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .qwen3moe_gguf import dequantize

logger = logging.getLogger(__name__)

NATIVE_DIR = Path(__file__).resolve().parent / "native"
SOURCE = NATIVE_DIR / "qwen3moe_kernels.c"
LIBRARY = NATIVE_DIR / "qwen3moe_kernels.dll"
VCVARS_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
)
MOE_MAX_TASKS = 8
GGML_Q4_K = 12
GGML_Q6_K = 14
_TYPE_LIBRARY = {GGML_Q4_K: "q4k_matvec_rows", GGML_Q6_K: "q6k_matvec_rows"}
_SPLIT_MIN_ROWS = 128
_lock = threading.Lock()
_library: ctypes.CDLL | None = None
_load_attempted = False
_pool = ThreadPoolExecutor(max_workers=max(1, min(16, os.cpu_count() or 1)), thread_name_prefix="qwen-matvec")


def _find_vcvars() -> str | None:
    for candidate in VCVARS_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def build(force: bool = False) -> Path | None:
    """Compile the kernel library with MSVC. Returns None when no compiler is installed."""
    if not force and LIBRARY.is_file() and LIBRARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return LIBRARY
    vcvars = _find_vcvars()
    if vcvars is None:
        return None
    script = NATIVE_DIR / "build_qwen3moe_kernels.bat"
    lines = [
        "@echo off",
        f'call "{vcvars}" >nul',
        f'cl /nologo /O2 /arch:AVX2 /fp:fast /LD /W3 /Fe:"{LIBRARY.name}" /Fo:qwen3moe_kernels.obj "{SOURCE.name}"',
    ]
    script.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    try:
        result = subprocess.run(
            ["cmd", "/c", str(script)], cwd=str(NATIVE_DIR), capture_output=True, text=True, timeout=300,
        )
    finally:
        script.unlink(missing_ok=True)
    for leftover in ("qwen3moe_kernels.obj", "qwen3moe_kernels.lib", "qwen3moe_kernels.exp"):
        try:
            (NATIVE_DIR / leftover).unlink()
        except FileNotFoundError:
            pass
    if result.returncode != 0 or not LIBRARY.is_file():
        logger.warning("Qwen3-MoE kernel build failed: %s", (result.stdout + result.stderr)[-1500:])
        return None
    return LIBRARY


PF_AVX2_INSTRUCTIONS_AVAILABLE = 40


def cpu_has_avx2() -> bool:
    """Runtime check: the kernels are built with /arch:AVX2, so they must not load without it."""
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.WinDLL("kernel32").IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE))
    except OSError:
        return False


def _load() -> ctypes.CDLL | None:
    global _library, _load_attempted
    with _lock:
        if _load_attempted:
            return _library
        _load_attempted = True
        if not cpu_has_avx2():
            logger.warning("Qwen3-MoE native kernels need AVX2; using the NumPy fallback")
            return None
        path = build()
        if path is None:
            logger.warning("Qwen3-MoE kernels unavailable: no MSVC build and no prebuilt library")
            return None
        library = ctypes.CDLL(str(path))
        for name in _TYPE_LIBRARY.values():
            function = getattr(library, name)
            function.restype = None
            function.argtypes = [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_void_p, ctypes.c_void_p,
            ]
        library.moe_pool_init.restype = ctypes.c_int
        library.moe_pool_init.argtypes = [ctypes.c_int]
        library.moe_pool_shutdown.restype = None
        library.moe_pool_shutdown.argtypes = []
        library.moe_layer.restype = ctypes.c_int
        library.moe_layer.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        _library = library
        return _library


def native_available() -> bool:
    return _load() is not None


def q8_matvec(ggml_type: int, raw, rows: int, cols: int, x: np.ndarray) -> np.ndarray:
    """y = W x with int8-quantized activations (Q4_K or Q6_K weights), rows split across the pool."""
    library = _load()
    if library is None or ggml_type not in _TYPE_LIBRARY or cols > 8192:
        return matvec(ggml_type, raw, rows, cols, x)
    x = np.ascontiguousarray(x, dtype=np.float32)
    buffer = np.frombuffer(raw, dtype=np.uint8)
    y = np.empty(rows, dtype=np.float32)
    function = library.q8_matvec_rows
    function.restype = None
    function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                         ctypes.c_void_p, ctypes.c_void_p]
    base, x_ptr, y_ptr = buffer.ctypes.data, x.ctypes.data, y.ctypes.data
    if rows < _SPLIT_MIN_ROWS:
        function(ggml_type, base, cols, 0, rows, x_ptr, y_ptr)
        return y
    workers = max(1, min(_pool._max_workers, rows // _SPLIT_MIN_ROWS))
    step = -(-rows // workers)
    futures = [_pool.submit(function, ggml_type, base, cols, start, min(rows, start + step), x_ptr, y_ptr)
               for start in range(0, rows, step)]
    for future in futures:
        future.result()
    return y


def matvec(ggml_type: int, raw, rows: int, cols: int, x: np.ndarray, *, split: bool = True) -> np.ndarray:
    """y = W x, where W is `rows` x `cols` quantized row-major and x has `cols` values.

    Large matrices are split across the thread pool; pass split=False when the
    caller already runs several matvecs in parallel.
    """
    if cols % 256:
        raise ValueError("quantized rows must be a whole number of 256-value blocks")
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape != (cols,):
        raise ValueError("input vector length does not match the matrix")
    library = _load() if ggml_type in _TYPE_LIBRARY else None
    if library is None:
        weights = dequantize(raw, ggml_type, rows * cols).reshape(rows, cols)
        return weights @ x
    buffer = np.frombuffer(raw, dtype=np.uint8)
    if buffer.size != rows * (cols // 256) * (144 if ggml_type == GGML_Q4_K else 210):
        raise ValueError("quantized matrix has the wrong byte length")
    y = np.empty(rows, dtype=np.float32)
    function = getattr(library, _TYPE_LIBRARY[ggml_type])
    base = buffer.ctypes.data
    x_ptr = x.ctypes.data
    y_ptr = y.ctypes.data
    if not split or rows < _SPLIT_MIN_ROWS:
        function(base, cols, 0, rows, x_ptr, y_ptr)
        return y
    workers = max(1, min(_pool._max_workers, rows // _SPLIT_MIN_ROWS))
    step = -(-rows // workers)
    futures = []
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        futures.append(_pool.submit(function, base, cols, start, stop, x_ptr, y_ptr))
    for future in futures:
        future.result()
    return y


def moe_pool_init(threads: int) -> bool:
    """Create the persistent expert worker pool once. Returns False without the native library."""
    library = _load()
    if library is None or not hasattr(library, "moe_pool_init"):
        return False
    return library.moe_pool_init(int(threads)) > 0


def moe_pool_shutdown() -> None:
    library = _load()
    if library is not None and hasattr(library, "moe_pool_shutdown"):
        library.moe_pool_shutdown()


def moe_layer_available() -> bool:
    library = _load()
    return library is not None and hasattr(library, "moe_layer")


def moe_layer(gate_ptrs, up_ptrs, down_ptrs, gate_type: int, up_type: int, down_type: int, n_embd: int, n_ff: int,
              weights: np.ndarray, h: np.ndarray, scratch: np.ndarray, ffn: np.ndarray, out: np.ndarray) -> None:
    """Fused routed-expert pass for one layer. Pointers are integer addresses of leased expert bytes."""
    library = _load()
    if library is None:
        raise RuntimeError("native MoE pass is unavailable")
    tasks = len(gate_ptrs)
    gate_array = (ctypes.c_void_p * tasks)(*gate_ptrs)
    up_array = (ctypes.c_void_p * tasks)(*up_ptrs)
    down_array = (ctypes.c_void_p * tasks)(*down_ptrs)
    status = library.moe_layer(
        ctypes.cast(gate_array, ctypes.c_void_p), ctypes.cast(up_array, ctypes.c_void_p),
        ctypes.cast(down_array, ctypes.c_void_p), gate_type, up_type, down_type, n_embd, n_ff, tasks,
        weights.ctypes.data, h.ctypes.data, scratch.ctypes.data, ffn.ctypes.data, out.ctypes.data,
    )
    if status != 0:
        raise RuntimeError("native MoE pass rejected its inputs")
