from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from openagent.inference_accelerator import AdaptiveCPUInferenceEngine, compact_messages, estimate_messages_tokens
from openagent.model_gateway import ModelGateway, ModelResult
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class InferenceAcceleratorTests(unittest.TestCase):
    def test_long_conversation_is_compacted_while_latest_turn_is_preserved(self) -> None:
        messages = [{"role": "system", "content": "Keep answers grounded."}]
        messages.extend(
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index} " + ("detail " * 300)}
            for index in range(12)
        )
        compacted, omitted = compact_messages(messages, token_budget=600)

        self.assertGreater(omitted, 0)
        self.assertEqual(compacted[-1]["content"], messages[-1]["content"])
        self.assertTrue(any("Earlier conversation digest" in item["content"] for item in compacted))

    def test_exact_completion_cache_avoids_second_model_call(self) -> None:
        calls = {"count": 0}

        def complete(provider, messages, *, model=None):
            calls["count"] += 1
            return ModelResult("ollama", model or "tiny", "real local answer", {"local_network": True})

        with patch("openagent.model_gateway.OllamaProvider.available", return_value=True), patch(
            "openagent.model_gateway.OllamaProvider.complete", complete
        ):
            gateway = ModelGateway()
            first = gateway.complete([{"role": "user", "content": "hello cache"}], provider_id="auto")
            second = gateway.complete([{"role": "user", "content": "hello cache"}], provider_id="auto")

        self.assertEqual(calls["count"], 1)
        self.assertFalse(first.metadata["cache_hit"])
        self.assertTrue(second.metadata["cache_hit"])
        self.assertEqual(second.content, "real local answer")

    def test_large_attachment_turn_stays_inside_context_budget(self) -> None:
        messages = [
            {"role": "system", "content": "Keep answers grounded. " * 1200},
            {"role": "user", "content": "Read this CV.\n<attached_evidence>\n" + ("work experience and education " * 900)},
        ]
        compacted, _ = compact_messages(messages, token_budget=1000)

        self.assertLessEqual(estimate_messages_tokens(compacted), 1010)
        self.assertIn("Read this CV", compacted[-1]["content"])
        self.assertIn("attached_evidence", compacted[-1]["content"])

    def test_engine_overhead_benchmark_is_measured(self) -> None:
        result = AdaptiveCPUInferenceEngine().benchmark(iterations=10)

        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["planning_ms"]["mean"], 0.0)
        self.assertGreaterEqual(result["cache_lookup_mean_ms"], 0.0)

    def test_activity_load_tracks_live_requests_and_returns_to_idle(self) -> None:
        engine = AdaptiveCPUInferenceEngine()
        plan = engine.plan(
            [{"role": "user", "content": "Measure this request"}],
            provider_id="local",
            model="test",
            allow_external=False,
        )
        self.assertEqual(0, engine.status()["activity"]["load_percent"])
        with engine.activate(plan):
            activity = engine.status()["activity"]
            self.assertEqual(1, activity["active_requests"])
            self.assertGreater(activity["load_percent"], 0)
        self.assertEqual(0, engine.status()["activity"]["active_requests"])
        self.assertEqual(0, engine.status()["activity"]["load_percent"])

    def test_embedding_requests_use_value_preserving_engine_cache(self) -> None:
        gateway = ModelGateway()
        first = gateway.embed(["private retrieval text"], provider_id="local")
        second = gateway.embed(["private retrieval text"], provider_id="local")

        self.assertFalse(first.metadata["cache_hit"])
        self.assertTrue(second.metadata["cache_hit"])
        self.assertEqual(first.embeddings, second.embeddings)
        self.assertEqual(second.metadata["inference"]["engine"], "openagent-adaptive-cpu-v1")
        self.assertGreaterEqual(gateway.inference_status()["embedding_cache"]["hits"], 1)

    def test_long_form_plan_uses_explicit_output_budget(self) -> None:
        plan = AdaptiveCPUInferenceEngine().plan(
            [{"role": "user", "content": "Write an evidence-grounded academic document."}],
            provider_id="ollama",
            model="local-writer",
            allow_external=False,
            max_output_tokens=900,
        )

        self.assertEqual(plan.options["num_predict"], 900)
        self.assertGreaterEqual(plan.options["num_ctx"], 1024)

    def test_inference_status_and_benchmark_http_contracts(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                status = json.loads(urlopen(f"{base}/api/inference/status", timeout=10).read())
                request = Request(
                    f"{base}/api/inference/benchmark",
                    data=json.dumps({"project": "core", "iterations": 10}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                benchmark = json.loads(urlopen(request, timeout=10).read())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(status["engine"], "openagent-adaptive-cpu-v1")
        self.assertEqual(benchmark["status"], "ok")
        self.assertEqual(benchmark["iterations"], 10)


if __name__ == "__main__":
    unittest.main()
