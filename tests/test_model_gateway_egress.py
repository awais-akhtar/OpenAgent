from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import unittest
from unittest.mock import patch

from openagent.egress import EgressResponse
from openagent.model_gateway import (
    ModelGateway,
    get_json,
    local_http_available,
    post_json,
    stream_json_lines,
    stream_sse_json,
)


class ModelHandler(BaseHTTPRequestHandler):
    first_line_sent = threading.Event()
    release_stream = threading.Event()

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            body = {"models": [{"name": "local-model"}]}
        else:
            body = {"data": [{"id": "local-model"}]}
        self._json(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if self.path.startswith("/v1/") else "application/x-ndjson")
            self.end_headers()
            if self.path.startswith("/v1/"):
                lines = [b'data: {"choices":[{"delta":{"content":"first"}}]}\n', b'data: [DONE]\n']
            else:
                lines = [b'{"message":{"content":"first"},"done":false}\n', b'{"done":true}\n']
            self.wfile.write(lines[0])
            self.wfile.flush()
            self.first_line_sent.set()
            self.release_stream.wait(5)
            self.wfile.write(lines[1])
            self.wfile.flush()
            return
        self._json({"message": {"content": "local answer"}})

    def _json(self, value):
        body = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ModelGatewayEgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(5)

    def setUp(self):
        ModelHandler.first_line_sent.clear()
        ModelHandler.release_stream.clear()

    def test_loopback_discovery_and_post_work_without_external_approval(self):
        self.assertTrue(local_http_available(f"{self.base_url}/api/tags"))
        self.assertEqual(get_json(f"{self.base_url}/api/tags", timeout_seconds=1)["models"][0]["name"], "local-model")
        result = post_json(f"{self.base_url}/api/chat", {"stream": False}, headers={})
        self.assertEqual(result["message"]["content"], "local answer")

    def test_loopback_streams_yield_before_response_ends(self):
        for endpoint, stream in (
            ("/api/chat", stream_json_lines),
            ("/v1/chat/completions", stream_sse_json),
        ):
            with self.subTest(endpoint=endpoint):
                ModelHandler.first_line_sent.clear()
                ModelHandler.release_stream.clear()
                events = stream(f"{self.base_url}{endpoint}", {"stream": True}, headers={})
                try:
                    first = next(events)
                    self.assertTrue(ModelHandler.first_line_sent.wait(1))
                    self.assertFalse(ModelHandler.release_stream.is_set())
                    self.assertEqual(
                        first.get("message", {}).get("content") or first.get("choices", [{}])[0].get("delta", {}).get("content"),
                        "first",
                    )
                    ModelHandler.release_stream.set()
                    self.assertEqual(list(events), [{"done": True}])
                finally:
                    ModelHandler.release_stream.set()
                    events.close()

    def test_stopping_a_stream_ends_its_producer(self):
        release = threading.Event()
        stopped = threading.Event()

        def fake_fetch(self, url, **kwargs):
            try:
                kwargs["on_line"]('{"message":{"content":"first"}}')
                release.wait(2)
                while True:
                    kwargs["on_line"]('{"message":{"content":"later"}}')
            finally:
                stopped.set()

        with patch("openagent.model_gateway.EgressClient.fetch", fake_fetch):
            events = stream_json_lines(f"{self.base_url}/api/chat", {"stream": True}, headers={})
            self.assertEqual(next(events)["message"]["content"], "first")
            events.close()
            release.set()
            self.assertTrue(stopped.wait(2), "Closing the stream left its producer running.")

    def test_public_destinations_are_denied_before_connection(self):
        with patch("openagent.egress.EgressClient._open_response") as connect:
            self.assertFalse(local_http_available("http://8.8.8.8/models"))
            with self.assertRaises(PermissionError):
                get_json("http://8.8.8.8/models", timeout_seconds=1)
            with self.assertRaises(PermissionError):
                post_json("http://8.8.8.8/chat/completions", {}, headers={})
            with self.assertRaises(PermissionError):
                list(stream_sse_json("http://8.8.8.8/chat/completions", {}, headers={}))
            connect.assert_not_called()

    def test_remote_ollama_setting_requires_approval_even_when_provider_is_local(self):
        with patch.dict(os.environ, {"OPENAGENT_OLLAMA_URL": "http://8.8.8.8:11434"}), patch(
            "openagent.egress.EgressClient._open_response"
        ) as connect:
            gateway = ModelGateway()
            self.assertEqual(
                gateway.complete([{"role": "user", "content": "unapproved"}], provider_id="ollama").provider,
                "local",
            )
            self.assertEqual([item["provider"] for item in gateway.discover_models()], ["local"])
            self.assertTrue(
                all(call.args[0].local for call in connect.call_args_list),
                "Unapproved model discovery must never connect to a public address.",
            )

            calls = []

            def fake_fetch(self, url, **kwargs):
                calls.append(kwargs)
                body = b'{"models":[{"name":"remote-model"}]}' if url.endswith("/api/tags") else b'{"message":{"content":"approved answer"}}'
                return EgressResponse(url, 200, {"content-type": "application/json"}, body)

            with patch("openagent.model_gateway.EgressClient.fetch", fake_fetch):
                result = gateway.complete(
                    [{"role": "user", "content": "approved"}], provider_id="ollama", allow_external=True
                )
            self.assertEqual(result.content, "approved answer")
            self.assertEqual(result.model, "remote-model")
            self.assertTrue(all(call["approved"] for call in calls))

    def test_approved_external_provider_paths_use_egress(self):
        calls = []

        def fake_fetch(self, url, **kwargs):
            calls.append((url, kwargs))
            if kwargs.get("on_line"):
                kwargs["on_line"]('data: {"choices":[{"delta":{"content":"token"}}]}')
                kwargs["on_line"]("data: [DONE]")
                body = b""
            elif url.endswith("/embeddings"):
                body = b'{"data":[{"embedding":[0.5,0.25]}]}'
            else:
                body = b'{"choices":[{"message":{"content":"answer"}}]}'
            return EgressResponse(url, 200, {"content-type": "application/json"}, body)

        environment = {"OPENAGENT_OPENAI_BASE_URL": "https://models.example/v1", "OPENAGENT_OPENAI_API_KEY": "secret"}
        with patch.dict(os.environ, environment), patch("openagent.model_gateway.EgressClient.fetch", fake_fetch):
            gateway = ModelGateway()
            self.assertEqual(gateway.complete([{"role": "user", "content": "one"}], provider_id="openai-compatible", allow_external=True).content, "answer")
            self.assertEqual("".join(item.content for item in gateway.stream_complete([{"role": "user", "content": "two"}], provider_id="openai-compatible", allow_external=True)), "token")
            self.assertEqual(gateway.embed(["three"], provider_id="openai-compatible", allow_external=True).embeddings, [[0.5, 0.25]])
            self.assertEqual(gateway.complete_vision([{"role": "user", "content": "four"}], b"image", provider_id="openai-compatible", allow_external=True).content, "answer")
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(options["approved"] and options["allow_loopback"] for _, options in calls))
        self.assertTrue(all(options["max_redirects"] == 0 for _, options in calls))


if __name__ == "__main__":
    unittest.main()
