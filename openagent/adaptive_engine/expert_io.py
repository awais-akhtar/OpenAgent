"""Synchronous aligned direct reads; checksum validation belongs to the store.

Files must remain immutable during a read. No buffered fallback or io_uring is
used. Unsupported filesystems/devices explicitly raise DirectIOUnsupported.
"""

from __future__ import annotations

import ctypes
import errno
import mmap
import os
import sys


class DirectIOUnsupported(OSError):
    """The platform or target rejected the direct-I/O contract."""


def _expanded(offset: int, size: int, alignment: int) -> tuple[int, int]:
    start = offset // alignment * alignment
    end = (offset + size + alignment - 1) // alignment * alignment
    return start, end - start


def read_region(path: str | os.PathLike[str], offset: int, size: int) -> bytes:
    """Return up to size bytes at offset, stopping at EOF, without file caching.

    Negative ranges are invalid. Missing files and permissions retain normal OS
    errors; unsupported direct I/O is distinguishable via DirectIOUnsupported.
    """
    if not isinstance(offset, int) or not isinstance(size, int):
        raise TypeError("offset and size must be integers")
    if offset < 0 or size < 0:
        raise ValueError("offset and size must be non-negative")
    if sys.platform == "win32":
        return _read_windows(os.fsdecode(path), offset, size)
    if sys.platform.startswith("linux"):
        return _read_linux(path, offset, size)
    raise DirectIOUnsupported("direct I/O is supported only on Windows and Linux")


def _read_linux(path: str | os.PathLike[str], offset: int, size: int) -> bytes:
    if not hasattr(os, "O_DIRECT") or not hasattr(os, "preadv"):
        raise DirectIOUnsupported("Linux O_DIRECT and preadv are required")
    rejected = {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        info = os.fstat(fd)
        wanted = min(size, max(0, info.st_size - offset))
        if not wanted:
            return b""
        alignment = max(4096, mmap.PAGESIZE, info.st_blksize)
        start, length = _expanded(offset, wanted, alignment)
        capacity = max(alignment, (1024 * 1024 // alignment) * alignment)
        result = bytearray()
        # Overallocate and align the view even when filesystem alignment exceeds
        # the page alignment guaranteed by anonymous mmap.
        with mmap.mmap(-1, capacity + alignment) as buffer:
            address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
            padding = (-address) % alignment
            with memoryview(buffer)[padding:padding + capacity] as view:
                while length:
                    count = min(length, capacity)
                    with view[:count] as target:
                        received = os.preadv(fd, [target], start)
                    lower = max(0, offset - start)
                    upper = min(received, offset + wanted - start)
                    if upper > lower:
                        result.extend(view[lower:upper])
                    if received < count:
                        break
                    start += count
                    length -= count
        return bytes(result)
    except OSError as exc:
        if exc.errno in rejected:
            raise DirectIOUnsupported(exc.errno, "Linux target rejected aligned O_DIRECT", os.fspath(path)) from exc
        raise
    finally:
        if fd is not None:
            os.close(fd)


def _read_windows(path: str, offset: int, size: int) -> bytes:
    from ctypes import wintypes as w

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    def bind(name, args, result):
        function = getattr(kernel, name)
        function.argtypes = args
        function.restype = result
        return function

    create = bind("CreateFileW", [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE], w.HANDLE)
    close = bind("CloseHandle", [w.HANDLE], w.BOOL)
    alloc = bind("VirtualAlloc", [w.LPVOID, ctypes.c_size_t, w.DWORD, w.DWORD], w.LPVOID)
    free = bind("VirtualFree", [w.LPVOID, ctypes.c_size_t, w.DWORD], w.BOOL)
    length_fn = bind("GetFileSizeEx", [w.HANDLE, ctypes.POINTER(ctypes.c_longlong)], w.BOOL)
    seek = bind("SetFilePointerEx", [w.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), w.DWORD], w.BOOL)
    read = bind("ReadFile", [w.HANDLE, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD), w.LPVOID], w.BOOL)
    storage = bind("GetFileInformationByHandleEx", [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD], w.BOOL)

    def fail():
        code = ctypes.get_last_error()
        if code in {1, 50, 87}:
            raise DirectIOUnsupported(code, "Windows target rejected aligned unbuffered I/O", path)
        raise ctypes.WinError(code)

    handle = create(path, 0x80000000, 7, None, 3, 0x20000000, None)
    if handle == ctypes.c_void_p(-1).value:
        fail()
    allocation = None
    try:
        file_size = ctypes.c_longlong()
        if not length_fn(handle, ctypes.byref(file_size)):
            fail()
        wanted = min(size, max(0, file_size.value - offset))
        if not wanted:
            return b""
        # FILE_STORAGE_INFO: logical sector, three physical sector sizes,
        # flags and two sector-alignment offsets (seven DWORDs).
        info = (w.DWORD * 7)()
        if not storage(handle, 16, ctypes.byref(info), ctypes.sizeof(info)):
            fail()
        alignment = max(4096, *info[:4])
        start, remaining = _expanded(offset, wanted, alignment)
        capacity = max(alignment, (1024 * 1024 // alignment) * alignment)
        if capacity > 0xFFFFFFFF:
            raise DirectIOUnsupported("Windows sector alignment exceeds ReadFile limits")
        allocation = alloc(None, capacity + alignment, 0x3000, 4)
        if not allocation:
            fail()
        address = allocation + (-allocation) % alignment
        if not seek(handle, start, None, 0):
            fail()
        result = bytearray()
        while remaining:
            count = min(remaining, capacity)
            received = w.DWORD()
            if not read(handle, address, count, ctypes.byref(received), None):
                if ctypes.get_last_error() == 38:  # ERROR_HANDLE_EOF
                    break
                fail()
            lower = max(0, offset - start)
            upper = min(received.value, offset + wanted - start)
            if upper > lower:
                result.extend(ctypes.string_at(address + lower, upper - lower))
            if received.value < count:
                break
            start += count
            remaining -= count
        return bytes(result)
    finally:
        if allocation:
            free(allocation, 0, 0x8000)
        close(handle)
