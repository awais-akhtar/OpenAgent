from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.agents import AgentRegistry
from openagent.local_ai import LocalAIInfrastructure
from openagent.model_gateway import ModelResult, StreamChunk
from openagent.multi_agent import MultiAgentRuntime
from openagent.research import ResearchEngine
from openagent.storage import OpenAgentStore
from openagent.worker import WorkerRuntime
from openagent.workflows import WorkflowEngine


class RecordingGateway:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.providers = {"local": object(), "auto": object()}

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return ModelResult(
            provider="engine-provider",
            model="engine-model",
            content="ENGINE_RESULT: completed by the shared adaptive model runtime.",
            metadata={
                "cache_hit": False,
                "latency_ms": 12.5,
                "inference": {"engine": "openagent-adaptive-cpu-v1", "task": "agent"},
            },
        )

    def stream_complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "stream": True, **kwargs})
        yield StreamChunk(
            provider="engine-provider",
            model="engine-model",
            content="ENGINE_RESEARCH_RESULT",
            done=True,
            metadata={"inference": {"engine": "openagent-adaptive-cpu-v1", "task": "research"}},
        )

    def specs(self):
        return [{"id": "local", "available": True}]


class SharedModelEngineTests(unittest.TestCase):
    def test_agent_run_is_model_backed_and_auditable(self) -> None:
        with TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            gateway = RecordingGateway()
            run = AgentRegistry(store, gateway).run("coding", "Implement a parser", project="core")

            self.assertIn("ENGINE_RESULT", run["response"])
            self.assertEqual(run["metadata"]["provider"], "engine-provider")
            self.assertEqual(run["metadata"]["engine"], "openagent-adaptive-cpu-v1")
            self.assertFalse(bool(gateway.calls[0]["allow_external"]))
            prompt = gateway.calls[0]["messages"][-1]["content"]
            self.assertTrue(prompt.endswith("CURRENT TASK (ANSWER THIS NOW):\nImplement a parser"))
            audit = store.list_audit_events(project="core", limit=1)[0]
            self.assertEqual(audit["metadata"]["execution"]["model"], "engine-model")

    def test_runtime_services_share_one_gateway_instance(self) -> None:
        with TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            gateway = RecordingGateway()
            workflow = WorkflowEngine(store, gateway)
            multi_agent = MultiAgentRuntime(store, gateway)
            research = ResearchEngine(store, gateway)
            worker = WorkerRuntime(store, worker_id="shared-engine-test", gateway=gateway)
            local_ai = LocalAIInfrastructure(store, gateway)

            self.assertIs(workflow.gateway, gateway)
            self.assertIs(workflow.chat.gateway, gateway)
            self.assertIs(workflow.agents.gateway, gateway)
            self.assertIs(multi_agent.registry.gateway, gateway)
            self.assertIs(research.gateway, gateway)
            self.assertIs(worker.gateway, gateway)
            self.assertIs(local_ai.gateway, gateway)

    def test_research_returns_actual_engine_provider_evidence(self) -> None:
        with TemporaryDirectory() as temp:
            gateway = RecordingGateway()
            result = ResearchEngine(OpenAgentStore(temp), gateway).run(
                "How is model execution shared?",
                project="core",
                max_iterations=1,
            )

            self.assertEqual(result["report"], "ENGINE_RESEARCH_RESULT")
            self.assertEqual(result["provider"], "engine-provider")
            self.assertEqual(result["engine"]["engine"], "openagent-adaptive-cpu-v1")


if __name__ == "__main__":
    unittest.main()
