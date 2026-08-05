from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class ResponseEnvelopeTests(unittest.TestCase):
    def _start_server(self, store: OpenAgentStore):
        server = build_http_server("127.0.0.1", 0, store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_default_response_has_v2_envelope_and_legacy_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server, thread = self._start_server(store)
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                body = json.loads(urlopen(f"{base_url}/api/health", timeout=15).read().decode("utf-8"))
                self.assertTrue(body["success"])
                self.assertIsNone(body["error"])
                self.assertEqual(body["data"]["status"], "ok")
                self.assertIn("status", body)
            finally:
                server.shutdown()
                thread.join(timeout=2)

    def test_legacy_response_via_header_returns_raw_shape(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server, thread = self._start_server(store)
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                success_request = Request(f"{base_url}/api/health", headers={"X-Response-Envelope": "legacy"})
                body = json.loads(urlopen(success_request, timeout=15).read().decode("utf-8"))
                self.assertNotIn("success", body)
                self.assertEqual(body["status"], "ok")
            finally:
                server.shutdown()
                thread.join(timeout=2)

    def test_envelope_wraps_success_and_error(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server, thread = self._start_server(store)
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                success_body = json.loads(urlopen(f"{base_url}/api/health", timeout=15).read().decode("utf-8"))
                self.assertTrue(success_body["success"])
                self.assertIsNone(success_body["error"])
                self.assertEqual(success_body["data"]["status"], "ok")
                self.assertIn("request_id", success_body["meta"])
                self.assertIn("ts", success_body["meta"])

                error_request = Request(
                    f"{base_url}/api/document/does-not-exist", headers={"X-Response-Envelope": "v2"}
                )
                from urllib.error import HTTPError

                try:
                    urlopen(error_request, timeout=15)
                    self.fail("expected HTTPError")
                except HTTPError as exc:
                    error_body = json.loads(exc.read().decode("utf-8"))
                self.assertFalse(error_body["success"])
                self.assertIsNone(error_body["data"])
                self.assertEqual(error_body["error"]["code"], "not_found")
                self.assertIn("document not found", error_body["error"]["message"])
            finally:
                server.shutdown()
                thread.join(timeout=2)

    def test_legacy_via_query_param(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server, thread = self._start_server(store)
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                body = json.loads(urlopen(f"{base_url}/api/health?envelope=legacy", timeout=15).read().decode("utf-8"))
                self.assertNotIn("success", body)
                self.assertEqual(body["status"], "ok")
            finally:
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
