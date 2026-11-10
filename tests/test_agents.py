from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.context import ContextEngine
from openagent.agents import AgentRegistry
from openagent.model_gateway import ProviderConfigurationError
from openagent.storage import OpenAgentStore


class AgentTests(unittest.TestCase):
    def test_cancelled_agent_run_does_not_persist(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            registry = AgentRegistry(store)
            with self.assertRaises(InterruptedError):
                registry.run("executive", "Plan a task", cancelled=lambda: True)
            self.assertEqual(store.health()["agent_run_count"], 0)

    def test_invalid_cpu_provider_does_not_become_successful_fallback_run(self) -> None:
        class RejectedGateway:
            def complete(self, *_args, **_kwargs):
                raise ProviderConfigurationError("CPU model exceeds storage budget")

        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            registry = AgentRegistry(store, RejectedGateway())
            with self.assertRaisesRegex(ProviderConfigurationError, "storage budget"):
                registry.run("executive", "Plan a task")
            self.assertEqual(store.health()["agent_run_count"], 0)

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
                    "scientific",
                },
            )


if __name__ == "__main__":
    unittest.main()
