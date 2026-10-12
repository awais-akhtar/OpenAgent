"""Fail-closed runner for real accelerator certification, not device evidence."""

import io
import unittest


class NoSkipCertificationResult(unittest.TextTestResult):
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.failures.append((test, "Hardware certification rejects skipped test: " + reason))


def run_certification(suite, *, stream=None):
    result = unittest.TextTestRunner(
        stream=stream, verbosity=2, resultclass=NoSkipCertificationResult,
    ).run(suite)
    return 0 if result.wasSuccessful() and result.testsRun > 0 else 1


class CertificationRunnerContractTests(unittest.TestCase):
    def run_suite(self, *tests):
        return run_certification(unittest.TestSuite(tests), stream=io.StringIO())

    def test_success(self):
        self.assertEqual(self.run_suite(unittest.FunctionTestCase(lambda: None)), 0)

    def test_empty_suite_rejected(self):
        self.assertEqual(self.run_suite(), 1)

    def test_skip_rejected(self):
        class Skipped(unittest.TestCase):
            def runTest(self):
                self.skipTest("no hardware")

        self.assertEqual(self.run_suite(Skipped()), 1)

    def test_setup_skip_rejected(self):
        class Skipped(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise unittest.SkipTest("unavailable runtime")

            def test_execution(self):
                pass

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Skipped)
        self.assertEqual(run_certification(suite, stream=io.StringIO()), 1)

    def test_error_and_failure_rejected(self):
        def error():
            raise RuntimeError("device failed")

        def failure():
            raise AssertionError("numerical mismatch")

        for function in (error, failure):
            with self.subTest(function=function.__name__):
                self.assertEqual(self.run_suite(unittest.FunctionTestCase(function)), 1)
