from __future__ import annotations

import unittest

from openagent.capabilities import vision_coverage


class CapabilitiesTests(unittest.TestCase):
    def test_vision_coverage_reports_honest_statuses(self) -> None:
        coverage = vision_coverage()
        self.assertEqual(coverage["summary"]["verdict"], "implemented_local_ai_os")
        self.assertEqual(coverage["summary"]["current_release"], "0.13.0")
        self.assertGreaterEqual(coverage["summary"]["counts"]["implemented"], 15)
        self.assertEqual(coverage["summary"]["counts"].get("not_implemented", 0), 0)
        self.assertEqual(coverage["summary"]["counts"].get("scaffolded", 0), 0)
        self.assertEqual(coverage["summary"]["counts"].get("foundation", 0), 0)
        self.assertEqual(coverage["summary"]["completed_stages"], list(range(16)))

        items = {
            item["name"]: item
            for section in coverage["sections"]
            for item in section["items"]
        }
        self.assertEqual(items["Self-hosted local core"]["status"], "implemented")
        voice_item = items[
            "Voice conversations, wake word, STT, TTS, cloning, transcription, translation, and personas"
        ]
        enterprise_item = items[
            "RBAC, multi-user workspaces, org memory, policies, SSO validation, backups, and disaster recovery"
        ]
        self.assertEqual(voice_item["status"], "implemented")
        self.assertEqual(items["Desktop automation, desktop app operation, and form filling"]["status"], "implemented")
        self.assertIn("openagent/enterprise.py", enterprise_item["evidence"])


if __name__ == "__main__":
    unittest.main()
