from __future__ import annotations

import base64
import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.communication import CommunicationHub
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class ProviderCertificationTests(unittest.TestCase):
    def test_provider_matrix_callback_and_attachment_privacy(self) -> None:
        with TemporaryDirectory() as tmp:
            hub = CommunicationHub(OpenAgentStore(tmp))

            matrix = hub.provider_certification_matrix()
            provider_ids = {provider["id"] for provider in matrix["providers"]}
            self.assertIn("google", provider_ids)
            self.assertIn("microsoft", provider_ids)
            self.assertIn("generic", provider_ids)

            callback = hub.record_oauth_callback(state="state-1", code="secret-code", project="core", owner="u1")
            self.assertTrue(callback["code_received"])
            self.assertNotEqual(callback["code_fingerprint"], "secret-code")
            self.assertEqual(hub.oauth_callbacks(project="core,missing"), [])
            self.assertEqual(hub.oauth_callbacks(project="core", owner="u1")[0]["state"], "state-1")

            attachment = hub.ingest_attachment(
                filename="message.txt",
                content=b"attachment text for local ingestion",
                project="core",
                owner="u1",
                message_id="msg-1",
            )
            self.assertEqual(attachment["filename"], "message.txt")
            self.assertTrue(attachment["local_path"])
            self.assertEqual(hub.attachments(project="core", owner="u1")[0]["message_id"], "msg-1")


class ProviderCertificationRouteTests(unittest.TestCase):
    def test_provider_certification_routes_are_callable(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                matrix = json.loads(urlopen(f"{base_url}/api/communication/provider-certification", timeout=5).read().decode("utf-8"))
                self.assertIn("providers", matrix)

                callback = json.loads(
                    urlopen(
                        f"{base_url}/api/communication/oauth/callback?project=core&owner=u1&state=abc&code=secret",
                        timeout=5,
                    )
                    .read()
                    .decode("utf-8")
                )["callback"]
                self.assertTrue(callback["code_received"])
                self.assertNotEqual(callback["code_fingerprint"], "secret")

                attachment_req = Request(
                    f"{base_url}/api/communication/attachments",
                    data=json.dumps(
                        {
                            "project": "core",
                            "filename": "attachment.txt",
                            "content_base64": base64.b64encode(b"hello attachment").decode("ascii"),
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                attachment = json.loads(urlopen(attachment_req, timeout=5).read().decode("utf-8"))["attachment"]
                self.assertEqual(attachment["filename"], "attachment.txt")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
