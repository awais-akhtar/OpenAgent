"""Real Linux certification. Run as a module; skips are not certification."""
import hashlib
import io
import os
from pathlib import Path
import platform
import sys
import tempfile
import unittest

from openagent.adaptive_engine import expert_io, uring_io


class LinuxEngineCertification(unittest.TestCase):
    def setUp(self):
        if os.environ.get("OPENAGENT_CERTIFY_LINUX") != "1":
            self.skipTest("Linux certification not requested; not hardware verification")
        # Deliberately fail, rather than skip, on unsupported hosts or kernels.
        self.assertTrue(sys.platform.startswith("linux"), "certification requires real Linux")
        self.assertTrue(hasattr(os, "O_DIRECT") and hasattr(os, "preadv"))
        report = uring_io.capability_report()
        self.assertTrue(report["available"], report["reason"])

    def test_real_reads_and_store_integrity(self):
        from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion

        with tempfile.TemporaryDirectory(prefix="openagent-cert-") as directory:
            path = Path(directory) / "weights.bin"
            data = bytes(range(256)) * 9000 + b"unaligned-tail"
            path.write_bytes(data)
            for mode, reader in (("direct", expert_io.read_region), ("uring", uring_io.read_region)):
                with self.subTest(mode=mode):
                    for offset, size in ((0, 1), (1, 17), (4095, 3), (4096, 4096),
                                         (17, 1100000), (len(data) - 7, 99),
                                         (len(data), 20), (len(data) + 99, 20), (3, 0)):
                        with self.subTest(offset=offset, size=size):
                            self.assertEqual(reader(path, offset, size), data[offset:offset + size])
                    expected = data[7:5010]
                    region = ExpertRegion(path.name, 7, len(expected), hashlib.sha256(expected).hexdigest())
                    with DiskExpertStore(directory, {"expert": region}, capacity_bytes=8192,
                                         io_mode=mode) as store:
                        with store.acquire("expert") as payload:
                            self.assertEqual(bytes(payload), expected)
                    bad = ExpertRegion(path.name, 7, len(expected), "0" * 64)
                    with DiskExpertStore(directory, {"expert": bad}, capacity_bytes=8192,
                                         io_mode=mode) as store:
                        with self.assertRaisesRegex(ValueError, "checksum"):
                            with store.acquire("expert"):
                                self.fail("invalid checksum accepted")
                    empty = Path(directory) / (mode + "-empty")
                    empty.touch()
                    self.assertEqual(reader(empty, 0, 10), b"")
                    empty.unlink()
                    with self.assertRaises(FileNotFoundError):
                        reader(empty, 0, 1)
            path.unlink()


class StrictCertificationResult(unittest.TextTestResult):
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        windows_only = {
            "tests.test_uring_io.UringIOTests.test_real_windows_unsupported",
            "tests.test_expert_io.ExpertIOTests.test_windows_store_integration",
        }
        if test.id() not in windows_only or not sys.platform.startswith("linux"):
            self.failures.append((test, "Certification rejects skipped test: " + reason))


class CertificationRunnerTests(unittest.TestCase):
    def test_skip_is_failure(self):
        class Skipped(unittest.TestCase):
            def runTest(self):
                self.skipTest("native backend unavailable")

        result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=StrictCertificationResult).run(Skipped())
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(len(result.failures), 1)

    def test_success_is_success(self):
        result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=StrictCertificationResult).run(
            unittest.FunctionTestCase(lambda: None))
        self.assertTrue(result.wasSuccessful())


def main():
    print("Linux certification:", platform.platform(), "temp root:", tempfile.gettempdir(), flush=True)
    os.environ["OPENAGENT_REQUIRE_URING"] = "1"
    os.environ["OPENAGENT_CERTIFY_LINUX"] = "1"
    names = [__name__ + ".LinuxEngineCertification", *sys.argv[1:]]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2, resultclass=StrictCertificationResult).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
