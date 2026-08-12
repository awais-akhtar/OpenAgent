from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.model_cookbook import ModelCookbook
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class FakeModelEndpoint(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({"data": [{"id": "tiny-local"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "OK"}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class ModelCookbookTests(unittest.TestCase):
    def test_hosts_serve_logs_cached_and_hardware_fit(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            cookbook = ModelCookbook(store)

            host = cookbook.add_host(label="CPU host", project="core")
            self.assertEqual(host["label"], "CPU host")

            session = cookbook.serve(project="core", host_id=host["id"], model_ref="tiny.gguf", runner="python", args=["--version"])
            self.assertEqual(session["status"], "planned")
            self.assertEqual(cookbook.logs(session["id"])[0]["level"], "info")

            adopted = cookbook.adopt(project="core", model_ref="external", runner="ollama", pid=123)
            self.assertEqual(adopted["status"], "adopted")
            self.assertEqual(cookbook.stop(adopted["id"])["status"], "stopped")

            fit = cookbook.hardware_fit(project="core", model_ref="qwen-coder", parameter_count_b=3.0)
            self.assertIn("recommended_bits", fit)
            repeated_fit = cookbook.hardware_fit(project="core", model_ref="qwen-coder", parameter_count_b=3.0)
            self.assertEqual(repeated_fit["recommended_bits"], fit["recommended_bits"])
            self.assertIsInstance(cookbook.cached_files(project="core"), list)

            with self.assertRaises(ValueError):
                cookbook.serve(project="core", model_ref="bad", runner="cmd", args=["/c", "echo"])

    def test_remote_serving_certification_is_probed_signed_and_revocable(self) -> None:
        endpoint = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelEndpoint)
        thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as tmp:
                cookbook = ModelCookbook(OpenAgentStore(tmp))
                certification = cookbook.certify_endpoint(
                    endpoint=f"http://127.0.0.1:{endpoint.server_address[1]}", project="core"
                )
                self.assertEqual(certification["status"], "certified")
                self.assertTrue(certification["report"]["openai_compatible"])
                self.assertEqual(len(certification["report_signature"]), 64)
                self.assertEqual(cookbook.revoke_certification(certification["id"])["status"], "revoked")
        finally:
            endpoint.shutdown()
            endpoint.server_close()
            thread.join(timeout=2)


class ModelCookbookRouteTests(unittest.TestCase):
    def test_model_cookbook_routes_are_callable(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                host_req = Request(
                    f"{base_url}/api/model-cookbook/hosts",
                    data=json.dumps({"project": "core", "label": "CPU host"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                host = json.loads(urlopen(host_req, timeout=5).read().decode("utf-8"))["host"]
                self.assertEqual(host["label"], "CPU host")

                serve_req = Request(
                    f"{base_url}/api/model-cookbook/serve",
                    data=json.dumps(
                        {
                            "project": "core",
                            "host_id": host["id"],
                            "model_ref": "tiny.gguf",
                            "runner": "python",
                            "args": ["--version"],
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                session = json.loads(urlopen(serve_req, timeout=5).read().decode("utf-8"))["session"]
                self.assertEqual(session["status"], "planned")

                logs = json.loads(urlopen(f"{base_url}/api/model-cookbook/logs/{session['id']}", timeout=5).read().decode("utf-8"))
                self.assertEqual(len(logs["logs"]), 1)

                fit = json.loads(urlopen(f"{base_url}/api/model-cookbook/hardware-fit?project=core&model_ref=tiny", timeout=5).read().decode("utf-8"))
                self.assertIn("expected_memory_gb", fit)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
