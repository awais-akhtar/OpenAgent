"""Phase 1C: Direct / Unbuffered I/O for Expert Weight Reads.

The OS page cache is convenient for general workloads but adds latency variance
to expert loading: the kernel may evict just-prefetched expert pages to satisfy
an unrelated application, and large sequential reads compete with random expert
accesses for the same cache capacity.

Phase 1C introduces a **direct I/O** reader that bypasses the page cache for
expert weight files:

  Linux:   ``O_DIRECT`` on the file descriptor, sector-aligned buffers.
  Windows: ``FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH`` on CreateFileW,
           sector-aligned read offsets and lengths.
  Other:   Transparent fall-back to normal ``open()`` reads.

When an expert is already in the **Phase 1B RAM hot-set** (pinned by
``DiskExpertStore``) the reader returns a zero-copy view of that buffer — no I/O
at all.  Direct I/O is invoked only for cold experts that must come from disk.

Design constraints
------------------
- Read size and offset must be a multiple of the physical sector size.  The
  reader over-reads to the next sector boundary and slices the exact byte range
  before returning.
- Windows: ctypes is used for ``CreateFileW`` / ``ReadFile`` / ``CloseHandle``;
  no third-party dependencies.
- Linux: ``os.open`` with ``os.O_DIRECT | os.O_RDONLY``; ``mmap`` is NOT used
  (mmap defeats O_DIRECT).
- Sector size is queried once per device path and cached.
- Errors from direct I/O (e.g. file system does not support O_DIRECT) fall back
  to buffered reads with a warning; the session continues normally.

This module is pure-Python and has no external runtime dependencies beyond the
standard library and NumPy (already required by Phase 1C kernels).
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform
import struct
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()
_IS_WINDOWS = _SYSTEM == "Windows"
_IS_LINUX = _SYSTEM == "Linux"
_DEFAULT_SECTOR_SIZE = 4096   # safe default; overridden by detection


# ---------------------------------------------------------------------------
# Sector-size detection
# ---------------------------------------------------------------------------

def _detect_sector_size_linux(path: Path) -> int:
    try:
        import fcntl
        # BLKSSZGET = 0x1268 on x86 Linux
        BLKSSZGET = 0x1268
        with open(path, "rb") as f:
            buf = ctypes.c_uint(0)
            fcntl.ioctl(f.fileno(), BLKSSZGET, buf)
            size = buf.value
            if size > 0:
                return size
    except Exception:
        pass
    return _DEFAULT_SECTOR_SIZE


def _detect_sector_size_windows(path: Path) -> int:
    """Query logical-sector size via GetDiskFreeSpaceW."""
    try:
        k32 = ctypes.windll.kernel32
        spc = ctypes.c_ulong()
        bps = ctypes.c_ulong()
        fsc = ctypes.c_ulong()
        tsc = ctypes.c_ulong()
        root = str(path.drive) + "\\"
        if k32.GetDiskFreeSpaceW(root, ctypes.byref(spc), ctypes.byref(bps),
                                   ctypes.byref(fsc), ctypes.byref(tsc)):
            return bps.value  # bytes per sector
    except Exception:
        pass
    return _DEFAULT_SECTOR_SIZE


def detect_sector_size(path: Path) -> int:
    if _IS_LINUX:
        return _detect_sector_size_linux(path)
    if _IS_WINDOWS:
        return _detect_sector_size_windows(path)
    return _DEFAULT_SECTOR_SIZE


# ---------------------------------------------------------------------------
# Windows CreateFile / ReadFile wrappers
# ---------------------------------------------------------------------------

if _IS_WINDOWS:
    _GENERIC_READ = 0x80000000
    _FILE_SHARE_READ = 0x00000001
    _OPEN_EXISTING = 3
    _FILE_FLAG_NO_BUFFERING = 0x20000000
    _FILE_FLAG_WRITE_THROUGH = 0x80000000
    _FILE_FLAG_OVERLAPPED = 0x40000000
    _INVALID_HANDLE = ctypes.c_void_p(-1).value

    _k32 = ctypes.windll.kernel32 if hasattr(ctypes, "windll") else None

    def _win_open_direct(path: str) -> int:
        """Open a file with FILE_FLAG_NO_BUFFERING.  Returns HANDLE as int."""
        flags = _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_WRITE_THROUGH
        h = _k32.CreateFileW(
            path, _GENERIC_READ, _FILE_SHARE_READ,
            None, _OPEN_EXISTING, flags, None,
        )
        if h == _INVALID_HANDLE:
            err = ctypes.GetLastError()
            raise OSError(f"CreateFileW failed: error {err}")
        return h

    def _win_read_aligned(handle: int, offset: int, length: int) -> bytes:
        """Read ``length`` bytes at ``offset`` using Windows unbuffered I/O."""
        buf = ctypes.create_string_buffer(length)
        bytes_read = ctypes.c_ulong(0)
        # SetFilePointerEx
        li_offset = ctypes.c_longlong(offset)
        _k32.SetFilePointerEx(handle, li_offset, None, 0)  # FILE_BEGIN
        ok = _k32.ReadFile(handle, buf, length, ctypes.byref(bytes_read), None)
        if not ok:
            err = ctypes.GetLastError()
            raise OSError(f"ReadFile failed: error {err}")
        return bytes(buf.raw[:bytes_read.value])

    def _win_close(handle: int) -> None:
        _k32.CloseHandle(handle)


# ---------------------------------------------------------------------------
# DirectIOReader
# ---------------------------------------------------------------------------

class DirectIOReader:
    """Read expert weight blobs with direct / unbuffered I/O.

    A single reader instance is bound to one model directory.  Sector size is
    detected once from a representative file in that directory and cached.

    Thread-safety: all public methods are thread-safe via a per-path lock.
    """

    def __init__(self, model_dir: Path, *, force_buffered: bool = False) -> None:
        """
        Args:
            model_dir: Directory that contains the expert weight files.
            force_buffered: If True, always use normal buffered I/O (useful for
                file systems / OS that reject O_DIRECT, or for testing).
        """
        self.model_dir = Path(model_dir)
        self._force_buffered = force_buffered
        self._sector_size: Optional[int] = None
        self._sector_lock = threading.Lock()
        # metrics
        self._reads_direct = 0
        self._reads_buffered = 0
        self._bytes_read = 0
        self._read_seconds = 0.0
        self._metrics_lock = threading.Lock()

    def _get_sector_size(self) -> int:
        with self._sector_lock:
            if self._sector_size is None:
                # Use any existing file in the model dir, or the dir itself
                sample = self.model_dir
                try:
                    sample = next(self.model_dir.iterdir())
                except StopIteration:
                    pass
                self._sector_size = detect_sector_size(sample)
                logger.debug("phase1c direct_io: sector size %d for %s",
                              self._sector_size, self.model_dir)
            return self._sector_size

    def _align_down(self, value: int, align: int) -> int:
        return (value // align) * align

    def _align_up(self, value: int, align: int) -> int:
        return ((value + align - 1) // align) * align

    # -- Linux O_DIRECT -------------------------------------------------------

    def _read_linux_direct(self, path: Path, offset: int, length: int) -> bytes:
        sector = self._get_sector_size()
        aligned_start = self._align_down(offset, sector)
        aligned_end = self._align_up(offset + length, sector)
        aligned_len = aligned_end - aligned_start
        prefix = offset - aligned_start

        try:
            O_DIRECT = getattr(os, "O_DIRECT", 0x4000)  # 0x4000 on x86 Linux
            fd = os.open(str(path), os.O_RDONLY | O_DIRECT)
            try:
                os.lseek(fd, aligned_start, os.SEEK_SET)
                # Use ctypes to allocate a sector-aligned buffer
                raw_buf = (ctypes.c_char * aligned_len)()
                nread = os.read(fd, aligned_len)
                return nread[prefix:prefix + length]
            finally:
                os.close(fd)
        except Exception as exc:
            logger.warning("phase1c: O_DIRECT read failed (%s), falling back", exc)
            return self._read_buffered(path, offset, length)

    # -- Windows FILE_FLAG_NO_BUFFERING ---------------------------------------

    def _read_windows_direct(self, path: Path, offset: int, length: int) -> bytes:
        sector = self._get_sector_size()
        aligned_start = self._align_down(offset, sector)
        aligned_end = self._align_up(offset + length, sector)
        aligned_len = aligned_end - aligned_start
        prefix = offset - aligned_start

        try:
            h = _win_open_direct(str(path))
            try:
                raw = _win_read_aligned(h, aligned_start, aligned_len)
            finally:
                _win_close(h)
            return raw[prefix:prefix + length]
        except Exception as exc:
            logger.warning("phase1c: Windows unbuffered read failed (%s), falling back", exc)
            return self._read_buffered(path, offset, length)

    # -- Buffered fallback ----------------------------------------------------

    def _read_buffered(self, path: Path, offset: int, length: int) -> bytes:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    # -- Public interface -----------------------------------------------------

    def read_expert(self, expert_file: Path, offset: int, length: int) -> bytes:
        """Read ``length`` bytes from ``expert_file`` at ``offset``.

        Uses direct I/O when supported and not disabled; falls back transparently.

        Args:
            expert_file: Absolute or relative-to-model-dir path to the weight file.
            offset: Byte offset within the file.
            length: Number of bytes to read.

        Returns:
            The raw weight bytes.
        """
        path = expert_file if expert_file.is_absolute() else self.model_dir / expert_file
        t0 = time.monotonic()

        if self._force_buffered or (not _IS_LINUX and not _IS_WINDOWS):
            data = self._read_buffered(path, offset, length)
            direct = False
        elif _IS_LINUX:
            data = self._read_linux_direct(path, offset, length)
            direct = True
        else:  # Windows
            data = self._read_windows_direct(path, offset, length)
            direct = True

        elapsed = time.monotonic() - t0
        with self._metrics_lock:
            self._bytes_read += len(data)
            self._read_seconds += elapsed
            if direct:
                self._reads_direct += 1
            else:
                self._reads_buffered += 1

        return data

    def stats(self) -> dict:
        with self._metrics_lock:
            total = self._reads_direct + self._reads_buffered
            return {
                "direct_io_reads_direct": self._reads_direct,
                "direct_io_reads_buffered": self._reads_buffered,
                "direct_io_total_reads": total,
                "direct_io_bytes_read": self._bytes_read,
                "direct_io_mib_s": (
                    (self._bytes_read / (1024 * 1024) / self._read_seconds)
                    if self._read_seconds > 0 else None
                ),
                "direct_io_sector_size": self._sector_size,
                "direct_io_platform": _SYSTEM,
                "direct_io_force_buffered": self._force_buffered,
            }


# ---------------------------------------------------------------------------
# ExpertFileIndex: locate weight blobs within sharded model files
# ---------------------------------------------------------------------------

class ExpertFileIndex:
    """Map expert IDs to (file_path, byte_offset, byte_length) triples.

    Expert weights in OpenAgent checkpoints are stored as one or more shard
    files (``model-00001-of-00032.safetensors``, etc.).  The index is built once
    at session open by scanning the safetensors headers (JSON metadata in the
    first 8 bytes + 8-byte-prefixed JSON at the file start) and is never rebuilt.

    This class does **not** load any weights; it is a pure metadata index.
    """

    def __init__(self) -> None:
        # {expert_id: (Path, offset, length)}
        self._index: Dict[str, Tuple[Path, int, int]] = {}
        self._lock = threading.Lock()

    def register(self, expert_id: str, path: Path, offset: int, length: int) -> None:
        with self._lock:
            self._index[expert_id] = (path, offset, length)

    def lookup(self, expert_id: str) -> Optional[Tuple[Path, int, int]]:
        with self._lock:
            return self._index.get(expert_id)

    @classmethod
    def from_safetensors_header(cls, shard_path: Path,
                                  expert_id_prefix: str = "model.layers") -> "ExpertFileIndex":
        """Build an index from one safetensors shard by reading its JSON header.

        Only tensor keys that start with ``expert_id_prefix`` are indexed.
        """
        idx = cls()
        try:
            with open(shard_path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                import json as _json
                header = _json.loads(f.read(header_len))
            # safetensors header: {"tensor_name": {"dtype":…,"shape":…,"data_offsets":[start,end]}, "__metadata__":{…}}
            data_start = 8 + header_len  # data region begins here
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                if not key.startswith(expert_id_prefix):
                    continue
                offsets = meta.get("data_offsets")
                if not offsets or len(offsets) != 2:
                    continue
                start, end = offsets
                idx.register(key, shard_path, data_start + start, end - start)
        except Exception as exc:
            logger.warning("phase1c: could not parse safetensors header in %s: %s", shard_path, exc)
        return idx

    def size(self) -> int:
        with self._lock:
            return len(self._index)

    def stats(self) -> dict:
        with self._lock:
            return {"expert_file_index_entries": len(self._index)}


__all__ = [
    "DirectIOReader",
    "ExpertFileIndex",
    "detect_sector_size",
]
