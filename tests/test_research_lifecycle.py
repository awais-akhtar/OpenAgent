from __future__ import annotations

from tempfile import TemporaryDirectory
import threading
import unittest
import json
from urllib.request import urlopen

from openagent.model_gateway import StreamChunk
from openagent.research import ResearchEngine
from openagent.research_lifecycle import ResearchLifecycle
from openagent.routes.research_routes import ROUTES
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class ResearchLifecycleTests(unittest.TestCase):
    def test_research_synthesis_uses_streaming_long_form_budget(self) -> None:
        class Gateway:
            def __init__(self) -> None:
                self.max_output_tokens = 0

            def stream_complete(self, messages, **kwargs):
                self.max_output_tokens = int(kwargs.get("max_output_tokens", 0))
                yield StreamChunk("local", "test", "Complete source-backed report.", True, {})

        with TemporaryDirectory() as temp:
            engine = ResearchEngine(OpenAgentStore(temp))
            gateway = Gateway()
            engine.gateway = gateway
            report = engine._synthesize(
                "local privacy",
                "inspect evidence",
                [],
                ["No matching sources; state uncertainty."],
                provider="local",
                model=None,
                allow_external=False,
            )

        self.assertEqual(report, "Complete source-backed report.")
        self.assertEqual(gateway.max_output_tokens, 768)

    def test_durable_completion_sources_graph_branch_and_retention(self) -> None:
        with TemporaryDirectory() as temp:
            lifecycle = ResearchLifecycle(OpenAgentStore(temp))
            session = lifecycle.create("Local AI acceleration", project="core", max_attempts=2)
            lifecycle.start(session["id"])
            completed = lifecycle.complete(
                session["id"],
                {
                    "plan": "Inspect hardware, then verify execution.",
                    "report": "The runtime must return execution evidence.",
                    "provider": "openvino-direct",
                    "model": "local-research-model",
                    "sources": [{"id": "s1", "title": "Local benchmark", "type": "local", "snippet": "NPU verified"}],
                },
            )
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["provider"], "openvino-direct")
            self.assertEqual(completed["model"], "local-research-model")
            self.assertEqual(len(completed["sources"]), 1)
            self.assertEqual(len(lifecycle.source_graph(session["id"])["edges"]), 1)
            branch = lifecycle.branch(session["id"], label="Follow-up")
            self.assertEqual(branch["parent_id"], session["id"])
            self.assertTrue(lifecycle.archive(branch["id"])["archived"])
            self.assertTrue(lifecycle.delete(branch["id"])["deleted"])

    def test_cancel_and_retry_are_server_side_states(self) -> None:
        with TemporaryDirectory() as temp:
            lifecycle = ResearchLifecycle(OpenAgentStore(temp))
            session = lifecycle.create("Cancellation", max_attempts=2)
            cancelled = lifecycle.cancel(session["id"])
            self.assertEqual(cancelled["status"], "cancelled")
            retried = lifecycle.retry(session["id"])
            self.assertEqual(retried["status"], "queued")
            self.assertFalse(retried["cancel_requested"])

    def test_route_contract_contains_full_research_lifecycle(self) -> None:
        paths = {path for _, path in ROUTES}
        self.assertIn("/api/research/sessions/{id}/cancel", paths)
        self.assertIn("/api/research/sessions/{id}/source-graph", paths)
        self.assertIn("/api/research/sessions/{id}/report", paths)
        self.assertIn("/api/research/sessions/{id}/delete", paths)
        self.assertIn("/api/research/sessions/{id}/stream", paths)

    def test_progress_stream_replays_checkpoints_and_completes(self) -> None:
        with TemporaryDirectory() as temp:
            lifecycle = ResearchLifecycle(OpenAgentStore(temp))
            session = lifecycle.create("Stream durable progress")
            lifecycle.start(session["id"])
            lifecycle.complete(session["id"], {"report": "done", "sources": []})
            events = list(lifecycle.stream_events(session["id"], timeout_seconds=1))
            self.assertEqual(events[0]["type"], "session")
            self.assertTrue(any(event["type"] == "checkpoint" for event in events))
            self.assertEqual(events[-1]["type"], "complete")
            self.assertEqual(events[-1]["session"]["status"], "completed")

    def test_http_progress_stream_uses_event_stream_contract(self) -> None:
        with TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            lifecycle = ResearchLifecycle(store)
            session = lifecycle.create("HTTP stream")
            lifecycle.complete(session["id"], {"report": "done", "sources": []})
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                response = urlopen(
                    f"http://127.0.0.1:{server.server_address[1]}/api/research/sessions/{session['id']}/stream?timeout=1",
                    timeout=5,
                )
                self.assertEqual(response.headers.get_content_type(), "text/event-stream")
                body = response.read().decode("utf-8")
                self.assertIn('"type": "checkpoint"', body)
                self.assertIn('"type": "complete"', body)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_http_report_returns_persisted_report_and_sources(self) -> None:
        with TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            lifecycle = ResearchLifecycle(store)
            session = lifecycle.create("Persisted report")
            lifecycle.start(session["id"])
            lifecycle.complete(session["id"], {"report": "A durable local report.", "sources": [{"id": "s1", "title": "Local source"}]})
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                payload = json.loads(urlopen(
                    f"http://127.0.0.1:{server.server_address[1]}/api/research/sessions/{session['id']}/report",
                    timeout=5,
                ).read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(payload["report"], "A durable local report.")
        self.assertEqual(payload["sources"][0]["title"], "Local source")


if __name__ == "__main__":
    unittest.main()
