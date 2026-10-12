from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine import expert_io


class ExpertIOTests(unittest.TestCase):
    def test_alignment_expansion(self):
        self.assertEqual(expert_io._expanded(4095, 2, 4096), (0, 8192))
        self.assertEqual(expert_io._expanded(4096, 4096, 4096), (4096, 4096))

    def test_invalid_ranges(self):
        for offset, size in [(-1, 1), (0, -1)]:
            with self.assertRaises(ValueError):
                expert_io.read_region("unused", offset, size)
        with self.assertRaises(TypeError):
            expert_io.read_region("unused", 0.5, 1)

    def test_unsupported_platform(self):
        with patch.object(expert_io.sys, "platform", "darwin"):
            with self.assertRaises(expert_io.DirectIOUnsupported):
                expert_io.read_region("unused", 0, 1)

    @unittest.skipUnless(sys.platform == "win32" or sys.platform.startswith("linux"), "requires native direct I/O")
    def test_native_ranges_match_buffered_reference_and_release_handles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            data = bytes(range(256)) * 9000 + b"unaligned-tail"
            path.write_bytes(data)
            ranges = [(0, 1), (1, 17), (4095, 3), (4096, 4096),
                      (17, 1100000), (len(data) - 7, 99),
                      (len(data), 20), (len(data) + 99, 20), (3, 0)]
            for offset, size in ranges:
                with self.subTest(offset=offset, size=size):
                    try:
                        actual = expert_io.read_region(path, offset, size)
                    except expert_io.DirectIOUnsupported as exc:
                        self.skipTest(f"explicit native direct-I/O rejection: {exc}")
                    with path.open("rb") as reference:
                        reference.seek(offset)
                        self.assertEqual(actual, reference.read(size))
            path.unlink()
            empty = Path(directory) / "empty.bin"
            empty.touch()
            self.assertEqual(expert_io.read_region(empty, 0, 10), b"")

    def test_linux_rejection_is_explicit_and_closes_descriptor(self):
        with patch.object(expert_io.os, "O_DIRECT", 16384, create=True), \
             patch.object(expert_io.os, "preadv", create=True), \
             patch.object(expert_io.os, "open", return_value=42), \
             patch.object(expert_io.os, "fstat", return_value=os.stat_result((0,) * 10)) as stat, \
             patch.object(expert_io.os, "close") as close:
            stat.side_effect = OSError(errno.EOPNOTSUPP, "unsupported")
            with self.assertRaises(expert_io.DirectIOUnsupported):
                expert_io._read_linux("weights", 0, 10)
            close.assert_called_once_with(42)

    @unittest.skipUnless(sys.platform == "win32", "requires real Windows unbuffered I/O")
    def test_windows_store_integration(self):
        from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            data = bytes(range(256)) * 40 + b"tail"
            path.write_bytes(data)
            expected = data[7:5010]
            region = ExpertRegion(path.name, 7, len(expected), hashlib.sha256(expected).hexdigest())
            try:
                with DiskExpertStore(directory, {"expert": region}, capacity_bytes=8192,
                                     io_mode="direct") as store:
                    with store.acquire("expert") as payload:
                        self.assertEqual(bytes(payload), expected)
            except expert_io.DirectIOUnsupported as exc:
                self.skipTest(f"explicit Windows direct-I/O rejection: {exc}")
            bad = ExpertRegion(path.name, 7, len(expected), "0" * 64)
            with DiskExpertStore(directory, {"expert": bad}, capacity_bytes=8192,
                                 io_mode="direct") as store:
                with self.assertRaises(ValueError):
                    with store.acquire("expert"):
                        self.fail("store accepted an invalid checksum")
            path.unlink()

    @unittest.skipUnless(sys.platform == "win32" or sys.platform.startswith("linux"), "requires native direct I/O")
    def test_missing_file_is_not_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                expert_io.read_region(Path(directory) / "missing", 0, 1)


if __name__ == "__main__":
    unittest.main()
