import ctypes
import errno
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from openagent.adaptive_engine import uring_io


class UringIOTests(unittest.TestCase):
    def test_bounds(self):
        with patch.object(uring_io, "_load_library") as load:
            for offset, size, error in [(-1, 1, ValueError), (0, -1, ValueError),
                                        (0.5, 1, TypeError), (2**63, 0, OverflowError),
                                        (2**63 - 1, 1, OverflowError)]:
                with self.assertRaises(error):
                    uring_io.read_region("unused", offset, size)
            with self.assertRaises(ValueError):
                uring_io.read_region("bad\x00path", 0, 1)
            load.assert_not_called()

    def test_abi_and_eof(self):
        def read(path, offset, size, buffer, received):
            self.assertEqual((path, offset, size), (b"weights", 7, 5))
            ctypes.memmove(buffer, b"a\x00b", 3)
            ctypes.cast(received, ctypes.POINTER(ctypes.c_uint64))[0] = 3
            return 0
        with patch.object(uring_io, "_load_library", return_value=Mock(oa_uring_read=read)):
            self.assertEqual(uring_io.read_region("weights", 7, 5), b"a\x00b")

    def test_negative_errors(self):
        for code, exception in [(errno.EIO, OSError), (errno.ENOENT, FileNotFoundError),
                                 (errno.EACCES, PermissionError), (errno.EOPNOTSUPP, uring_io.UringIOUnsupported)]:
            with self.subTest(code=code), patch.object(uring_io, "_load_library", return_value=Mock(oa_uring_read=Mock(return_value=-code))):
                with self.assertRaises(exception) as caught:
                    uring_io.read_region("weights", 0, 4)
                self.assertEqual(caught.exception.errno, code)

    def test_invalid_bridge_length(self):
        def read(path, offset, size, buffer, received):
            ctypes.cast(received, ctypes.POINTER(ctypes.c_uint64))[0] = size + 1
            return 0
        with patch.object(uring_io, "_load_library", return_value=Mock(oa_uring_read=read)):
            with self.assertRaises(OSError):
                uring_io.read_region("weights", 0, 1)

    def test_missing_compiler(self):
        with patch.object(uring_io.sys, "platform", "linux"), patch.object(uring_io, "_library", None), patch.object(uring_io.shutil, "which", return_value=None):
            self.assertFalse(uring_io.capability_report()["available"])
            with self.assertRaises(uring_io.UringIOUnsupported):
                uring_io.read_region("unused", 0, 0)

    def test_probe_rejected(self):
        with patch.object(uring_io, "_load_library", return_value=Mock(oa_uring_probe=Mock(return_value=-errno.EPERM))):
            self.assertFalse(uring_io.capability_report()["available"])

    def test_missing_liburing_build_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(uring_io.sys, "platform", "linux"), \
             patch.object(uring_io, "_library", None), \
             patch.object(uring_io.os, "uname", return_value=Mock(machine="x86_64"), create=True), \
             patch.dict(os.environ, {"XDG_CACHE_HOME": directory}), \
             patch.object(uring_io.shutil, "which", return_value="/usr/bin/cc"), \
             patch.object(uring_io.subprocess, "run", return_value=Mock(returncode=1, stderr="liburing.h missing")) as compile_call, \
             patch.object(uring_io.ctypes, "CDLL") as load:
            report = uring_io.capability_report()
            self.assertFalse(report["available"])
            self.assertIn("liburing.h missing", report["reason"])
            self.assertIn("-luring", compile_call.call_args.args[0])
            load.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "real unsupported Windows check")
    def test_real_windows_unsupported(self):
        self.assertFalse(uring_io.capability_report()["available"])
        with self.assertRaises(uring_io.UringIOUnsupported):
            uring_io.read_region("unused", 0, 1)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux native unverified on this host")
    def test_real_linux(self):
        report = uring_io.capability_report()
        if not report["available"]:
            if os.environ.get("OPENAGENT_REQUIRE_URING") == "1":
                self.fail("Required io_uring kernel unavailable: " + report["reason"])
            self.skipTest("Linux native unverified: " + report["reason"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights"
            data = bytes(range(256)) * 9000 + b"tail"
            path.write_bytes(data)
            for offset, size in [(0, 1), (7, 31), (4095, 3), (17, 1100000),
                                  (len(data) - 3, 99), (len(data) + 1, 10), (0, 0)]:
                try:
                    actual = uring_io.read_region(path, offset, size)
                except uring_io.UringIOUnsupported as exc:
                    if os.environ.get("OPENAGENT_REQUIRE_URING") == "1":
                        self.fail("Required io_uring target unavailable: " + str(exc))
                    self.skipTest("Linux native target unverified: " + str(exc))
                self.assertEqual(actual, data[offset:offset + size])
            path.unlink()
            path.touch()
            self.assertEqual(uring_io.read_region(path, 0, 10), b"")
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                uring_io.read_region(path, 0, 1)
