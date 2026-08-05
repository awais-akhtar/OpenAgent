from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.system_metrics import system_metrics


class SystemMetricsTests(unittest.TestCase):
    def test_system_metrics_are_local_and_structured(self) -> None:
        with TemporaryDirectory() as tmp:
            metrics = system_metrics(tmp)

        self.assertEqual(metrics["status"], "ok")
        self.assertGreaterEqual(metrics["cpu"]["count"], 1)
        self.assertIn("usage_percent", metrics["cpu"])
        self.assertIn("total_gb", metrics["memory"])
        self.assertIn("free_gb", metrics["disk"])
        self.assertIn("local OS", metrics["privacy_boundary"])


if __name__ == "__main__":
    unittest.main()
