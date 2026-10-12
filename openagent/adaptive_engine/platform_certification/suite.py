"""Package-owned fail-closed suite runner."""
import sys
import unittest
import json


class CertificationResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successful_tests = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successful_tests.append(test.id())

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.failures.append((test, "Certification rejects skipped checks: " + reason))

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.failures.append((test, "Certification rejects expected failures"))


def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(sys.argv[1:])
    result = unittest.TextTestRunner(verbosity=2, resultclass=CertificationResult).run(suite)
    passed = bool(sys.argv[1:] and result.testsRun and not loader.errors
                  and result.wasSuccessful() and not result.skipped
                  and not result.expectedFailures and not result.unexpectedSuccesses)
    print("OPENAGENT_SUITE_EVIDENCE " + json.dumps({
        "successful_tests": result.successful_tests, "passed": passed,
        "tests_run": result.testsRun, "skipped": len(result.skipped),
        "failures": len(result.failures), "errors": len(result.errors),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "load_errors": loader.errors,
    }), flush=True)
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
