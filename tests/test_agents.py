from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.context import ContextEngine
from openagent.agents import AgentRegistry
from openagent.storage import OpenAgentStore


class AgentTests(unittest.TestCase):
    def test_executive_agent_uses_local_context(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            context = ContextEngine(store)
            context.ingest(
                "Roadmap",
                "The next milestone is a local model gateway.",
                project="core",
            )
            registry = AgentRegistry(store)

            run = registry.run("executive", "Plan local model gateway", project="core")

            self.assertEqual(run["agent"], "executive")
            self.assertIn("local model gateway", run["goal"])
            self.assertGreaterEqual(run["confidence"], 0.7)
            self.assertEqual(store.health()["agent_run_count"], 1)

    def test_unknown_agent_raises_error(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = AgentRegistry(OpenAgentStore(tmp))
            with self.assertRaises(ValueError):
                registry.run("missing", "Do work")

    def test_registry_exposes_all_builtin_agent_roles(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = AgentRegistry(OpenAgentStore(tmp))
            ids = {agent["id"] for agent in registry.specs()}
            self.assertSetEqual(
                ids,
                {
                    "executive",
                    "research",
                    "coding",
                    "writing",
                    "planning",
                    "automation",
                    "knowledge",
                    "voice",
                    "communication",
                    "security",
                    "operations",
                },
            )


if __name__ == "__main__":
    unittest.main()
