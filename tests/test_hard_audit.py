from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.hard_audit import run_hard_audit
from openagent.storage import OpenAgentStore


class HardAuditTests(unittest.TestCase):
    def test_hard_audit_exercises_all_local_runtime_areas(self) -> None:
        with TemporaryDirectory() as tmp:
            result = run_hard_audit(OpenAgentStore(tmp), project="core")

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["passed"], result["total"])
            areas = {check["area"] for check in result["checks"]}
            self.assertSetEqual(
                areas,
                {
                    "coverage",
                    "ai",
                    "workspace",
                    "knowledge",
                    "memory",
                    "agents",
                    "automation",
                    "assistant",
                    "voice",
                    "local-ai",
                    "platform",
                    "production",
                    "deployment",
                },
            )


if __name__ == "__main__":
    unittest.main()
