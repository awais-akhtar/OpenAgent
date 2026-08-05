from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.mission_control import MissionControl
from openagent.storage import OpenAgentStore


class MissionControlTests(unittest.TestCase):
    def test_feature_matrix_and_full_local_check(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            mission = MissionControl(store)

            matrix = mission.feature_matrix(project="mission-test")
            self.assertEqual(matrix["status"], "ready")
            self.assertGreaterEqual(matrix["feature_count"], 50)
            self.assertTrue(matrix["features"])
            self.assertIn("privacy_boundary", matrix["features"][0])

            result = mission.run_full_check(project="mission-test")
            self.assertEqual(result["status"], "passed", result)
            self.assertEqual(result["failed"], 0, result)
            self.assertEqual(result["passed"], result["total"])
            self.assertGreaterEqual(result["total"], 18)
            self.assertTrue(all(check["evidence"] for check in result["checks"]))


if __name__ == "__main__":
    unittest.main()
