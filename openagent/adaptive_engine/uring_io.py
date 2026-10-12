"""Linux liburing/O_DIRECT reads of immutable files; no fallback or installs.

The packaged C source is lazily compiled against system liburing into a cache.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading


class UringIOUnsupported(OSError):
    """Platform, dependencies, kernel or target rejects this backend."""


_lock = threading.Lock()
_library = None
_REJECTED = {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL, errno.EPERM}


def _load_library():
    global _library
    if not sys.platform.startswith("linux"):
        raise UringIOUnsupported("io_uring requires Linux; unavailable on Windows")
    with _lock:
        if _library is not None:
            return _library
        compiler = next((shutil.which(c) for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
        if compiler is None:
            raise UringIOUnsupported("system C compiler and liburing development files required")
        source = Path(__file__).with_name("native") / "uring_reader.c"
        try:
            digest = hashlib.sha256(source.read_bytes() + os.fsencode(compiler) + os.uname().machine.encode()).hexdigest()
            root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "openagent" / "uring"
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            target = root / (digest + ".so")
            if not target.exists():
                with tempfile.TemporaryDirectory(prefix="build-", dir=root) as directory:
                    output = Path(directory) / "reader.so"
                    result = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", str(source), "-o", str(output), "-luring"], capture_output=True, text=True, timeout=60)
                    if result.returncode:
                        raise UringIOUnsupported("system liburing compilation failed: " + result.stderr[-4000:])
                    os.replace(output, target)
            library = ctypes.CDLL(str(target))
            library.oa_uring_probe.argtypes = []
            library.oa_uring_probe.restype = ctypes.c_int
            library.oa_uring_read.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
            library.oa_uring_read.restype = ctypes.c_int
        except (OSError, subprocess.TimeoutExpired, AttributeError) as exc:
            if isinstance(exc, UringIOUnsupported):
                raise
            raise UringIOUnsupported(f"liburing bridge unavailable: {exc}") from exc
        _library = library
        return library


def capability_report() -> dict:
    """Build/load and probe ring creation; target support remains independent."""
    try:
        rc = _load_library().oa_uring_probe()
        if rc != 0:
            raise UringIOUnsupported(-rc, f"io_uring kernel probe failed: {os.strerror(-rc)}")
    except UringIOUnsupported as exc:
        return {"available": False, "reason": str(exc), "backend": "liburing", "direct_io": True}
    return {"available": True, "reason": "", "backend": "liburing", "direct_io": True}


def read_region(path: str | os.PathLike[str], offset: int, size: int) -> bytes:
    """Return up to size bytes at offset, stopping at the initial EOF."""
    if not isinstance(offset, int) or not isinstance(size, int):
        raise TypeError("offset and size must be integers")
    if offset < 0 or size < 0:
        raise ValueError("offset and size must be non-negative")
    if offset + size > (1 << 63) - 1 or size > sys.maxsize:
        raise OverflowError("range exceeds native signed offset/address limits")
    encoded = os.fsencode(path)
    if b"\x00" in encoded:
        raise ValueError("embedded null byte in path")
    library = _load_library()
    buffer = ctypes.create_string_buffer(max(1, size))
    received = ctypes.c_uint64()
    rc = library.oa_uring_read(encoded, offset, size, buffer, ctypes.byref(received))
    if rc < 0:
        code = -rc
        exception = UringIOUnsupported if code in _REJECTED else OSError
        raise exception(code, os.strerror(code), os.fspath(path))
    if rc != 0 or received.value > size:
        raise OSError(errno.EIO, "invalid liburing bridge result")
    return buffer.raw[:received.value]
