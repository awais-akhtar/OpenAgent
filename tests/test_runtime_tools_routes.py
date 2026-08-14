from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class RuntimeToolsRouteTests(unittest.TestCase):
    def test_status_is_available_without_optional_browser_or_office_packages(self) -> None:
        with TemporaryDirectory() as temp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(temp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                payload = json.loads(urlopen(f"{base_url}/api/runtime-tools/status", timeout=10).read().decode("utf-8"))
                self.assertIn("browser", payload)
                self.assertIn("office_exports", payload)
                self.assertIn("colab", payload)
                self.assertIn("openai-compatible", payload["providers"]["registered"])
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

    def test_external_browser_navigation_requires_explicit_approval(self) -> None:
        with TemporaryDirectory() as temp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(temp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            request = Request(
                f"{base_url}/api/runtime-tools/browser-test",
                data=json.dumps({"url": "https://example.com", "approved_external": False}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with self.assertRaises(HTTPError) as captured:
                    urlopen(request, timeout=10)
                self.assertEqual(captured.exception.code, 400)
                try:
                    payload = json.loads(captured.exception.read().decode("utf-8"))
                    self.assertIn("explicit approval", payload["error"]["message"])
                finally:
                    captured.exception.close()
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()


if __name__ == "__main__":
    unittest.main()