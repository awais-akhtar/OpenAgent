from __future__ import annotations

import base64
import hashlib
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from openagent.chat_artifacts import ChatArtifactService
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class ChatArtifactTests(unittest.TestCase):
    def test_renders_pdf_and_docx_into_local_storage(self) -> None:
        with TemporaryDirectory() as tmp:
            service = ChatArtifactService(OpenAgentStore(tmp))

            pdf = service.render(
                title="Local test document",
                content="# Summary\n\nA short locally rendered document.",
                file_format="pdf",
                project="artifact-test",
            )
            docx = service.render(
                title="Editable test document",
                content="# Summary\n\nAn editable locally rendered document.",
                file_format="docx",
                project="artifact-test",
            )

            pdf_bytes = base64.b64decode(pdf["base64"])
            docx_bytes = base64.b64decode(docx["base64"])
            self.assertTrue(pdf_bytes.startswith(b"%PDF"))
            self.assertTrue(docx_bytes.startswith(b"PK"))
            self.assertEqual(pdf["sha256"], hashlib.sha256(pdf_bytes).hexdigest())
            self.assertEqual(docx["sha256"], hashlib.sha256(docx_bytes).hexdigest())
            self.assertEqual(Path(pdf["local_path"]).read_bytes(), pdf_bytes)
            self.assertEqual(Path(docx["local_path"]).read_bytes(), docx_bytes)

    def test_chat_artifact_http_route_returns_downloadable_file(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_address[1]}/api/chat/artifact",
                    data=json.dumps(
                        {
                            "title": "Chat note",
                            "content": "A file produced from chat.",
                            "format": "txt",
                            "project": "route-test",
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                response = urlopen(request, timeout=5)
                payload = json.loads(response.read().decode("utf-8"))

                self.assertEqual(response.status, 201)
                artifact = payload["artifact"]
                self.assertEqual(artifact["format"], "txt")
                self.assertEqual(base64.b64decode(artifact["base64"]), b"A file produced from chat.")
                self.assertTrue(Path(artifact["local_path"]).is_file())
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
