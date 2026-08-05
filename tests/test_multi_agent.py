from __future__ import annotations

from tempfile import TemporaryDirectory
import json
import unittest

from openagent.multi_agent import ApprovalRequired, MultiAgentRuntime
from openagent.storage import OpenAgentStore


class MultiAgentRuntimeTests(unittest.TestCase):
    def test_permissions_delegation_parallel_verification_and_marketplace(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = MultiAgentRuntime(OpenAgentStore(tmp))

            self.assertTrue(runtime.has_permission("executive", "agent.run", project="core"))
            marketplace = runtime.marketplace()
            self.assertIn("executive", {agent["id"] for agent in marketplace})

            checked = runtime.run_agent("planning", "Plan Stage 3", project="core")
            self.assertEqual(checked["run"]["agent"], "planning")
            self.assertIn(checked["verification"]["status"], {"passed", "warning"})

            delegated = runtime.delegate(
                "executive",
                "research",
                "Find local knowledge gaps",
                project="core",
            )
            self.assertEqual(delegated["from_agent"], "executive")
            self.assertEqual(delegated["to_agent"], "research")

            parallel = runtime.run_parallel(
                [
                    {"agent": "planning", "goal": "Plan memory work"},
                    {"agent": "security", "goal": "Review permissions"},
                ],
                project="core",
            )
            self.assertEqual(parallel["status"], "completed")
            self.assertEqual(len(parallel["results"]), 2)

    def test_human_approval_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = MultiAgentRuntime(OpenAgentStore(tmp))

            with self.assertRaises(ApprovalRequired) as raised:
                runtime.run_agent(
                    "automation",
                    "Run a sensitive automation",
                    project="core",
                    require_approval=True,
                )
            approval = json.loads(str(raised.exception))
            resolved = runtime.resolve_approval(approval["id"], approved=True)
            self.assertEqual(resolved["status"], "approved")

            result = runtime.run_agent(
                "automation",
                "Run a sensitive automation",
                project="core",
                require_approval=True,
                approval_id=approval["id"],
            )
            self.assertEqual(result["run"]["agent"], "automation")


if __name__ == "__main__":
    unittest.main()

