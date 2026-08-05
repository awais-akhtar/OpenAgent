from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.hard_check import run_project_hard_check


class HardCheckTests(unittest.TestCase):
    def test_project_hard_check_reports_local_and_windows_readiness(self) -> None:
        with TemporaryDirectory() as tmp:
            result = run_project_hard_check(tmp, project="core")

            self.assertTrue(result["ok"], result)
            self.assertTrue(result["local_runtime_ok"])
            self.assertTrue(result["windows_native_ok"])
            self.assertTrue(result["installer_assets_ok"])
            self.assertEqual(result["runtime_audit"]["failed"], 0)
            self.assertEqual(result["desktop_status"]["mode"], "native-desktop")
            self.assertFalse(result["desktop_status"]["browser_required"])
            self.assertEqual(result["project"], "core")
            self.assertGreaterEqual(len(result["deployment_level_remaining_work"]), 1)


if __name__ == "__main__":
    unittest.main()
