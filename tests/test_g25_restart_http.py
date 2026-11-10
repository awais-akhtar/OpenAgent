"""G25 uploaded documents and indexed retrieval survive a real HTTP restart."""

from __future__ import annotations

import base64
import json
import threading
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class G25RestartHttpTests(unittest.TestCase):
    def test_owner_bound_document_and_index_survive_server_restart(self) -> None:
        with TemporaryDirectory() as directory, patch.dict("os.environ", {"OPENAGENT_AUTH_ENFORCED": "1"}):
            store = OpenAgentStore(directory)
            auth = AuthManager(store)
            alice = auth.register("g25-restart-a", "restart-a@example.test", "strong-password-1")
            bob = auth.register("g25-restart-b", "restart-b@example.test", "strong-password-2")
            scopes = ["g25:read", "g25:write"]
            alice_token = auth.create_api_token(alice["user_id"], "restart-a", scopes)["token"]
            bob_token = auth.create_api_token(bob["user_id"], "restart-b", scopes)["token"]

            def start():
                server = build_http_server("127.0.0.1", 0, OpenAgentStore(directory))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                return server, thread

            def stop(server, thread):
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            def call(server, token, method, path, payload=None):
                headers = {"Authorization": f"Bearer {token}"}
                data = None
                if payload is not None:
                    headers["Content-Type"] = "application/json"
                    data = json.dumps(payload).encode()
                request = Request(f"http://127.0.0.1:{server.server_address[1]}{path}", data=data, headers=headers, method=method)
                try:
                    with urlopen(request, timeout=10) as response:
                        return response.status, json.loads(response.read())
                except HTTPError as error:
                    with error:
                        return error.code, json.loads(error.read())

            server, thread = start()
            try:
                status, uploaded = call(server, alice_token, "POST", "/api/g25/documents/upload", {
                    "project": "alpha", "filename": "notes.txt",
                    "content": base64.b64encode(b"Owner alpha renewable energy research notes.").decode(),
                })
                self.assertEqual(status, 201, uploaded)
                document = uploaded.get("data", uploaded)
                doc_id = document["document_id"]
                status, indexed = call(server, alice_token, "POST", "/api/g25/rag/index", {"project": "alpha", "doc_id": doc_id})
                self.assertEqual(status, 201, indexed)
                status, source_node = call(server, alice_token, "POST", "/api/g25/graph/nodes", {"project": "alpha", "label": "Source", "properties": {"kind": "document"}})
                self.assertEqual(status, 201, source_node)
                status, target_node = call(server, alice_token, "POST", "/api/g25/graph/nodes", {"project": "alpha", "label": "Finding", "properties": {"kind": "result"}})
                self.assertEqual(status, 201, target_node)
                status, linked = call(server, alice_token, "POST", "/api/g25/graph/edges", {
                    "project": "alpha", "source_id": source_node.get("data", source_node)["id"],
                    "target_id": target_node.get("data", target_node)["id"], "relationship": "SUPPORTS",
                })
                self.assertEqual(status, 201, linked)
                status, chain = call(server, alice_token, "POST", "/api/g25/reasoning/chains", {
                    "project": "alpha", "topic": "Renewable energy findings",
                })
                self.assertEqual(status, 201, chain)
                chain_id = chain.get("data", chain)["chain_id"]
            finally:
                stop(server, thread)

            restarted, restarted_thread = start()
            try:
                status, saved = call(restarted, alice_token, "GET", f"/api/g25/documents/{doc_id}?project=alpha")
                self.assertEqual(status, 200, saved)
                self.assertIn("renewable energy", saved.get("data", saved)["text"])
                status, context = call(restarted, alice_token, "GET", "/api/g25/rag/context?project=alpha&q=renewable")
                self.assertEqual(status, 200, context)
                self.assertTrue(context.get("data", context)["document_context"], context)
                status, paths = call(restarted, alice_token, "GET", "/api/g25/graph/paths?project=alpha&start=Source&end=Finding")
                self.assertEqual(status, 200, paths)
                self.assertEqual(len(paths.get("data", paths)["paths"]), 1)
                status, saved_chain = call(restarted, alice_token, "GET", f"/api/g25/reasoning/chains/{chain_id}?project=alpha")
                self.assertEqual(status, 200, saved_chain)
                self.assertEqual(saved_chain.get("data", saved_chain)["topic"], "Renewable energy findings")

                status, _ = call(restarted, bob_token, "GET", f"/api/g25/documents/{doc_id}?project=alpha")
                self.assertEqual(status, 404)
                status, foreign = call(restarted, bob_token, "GET", "/api/g25/rag/context?project=alpha&q=renewable")
                self.assertEqual(status, 200, foreign)
                self.assertEqual(foreign.get("data", foreign)["document_context"], [])
                status, hidden_paths = call(restarted, bob_token, "GET", "/api/g25/graph/paths?project=alpha&start=Source&end=Finding")
                self.assertEqual(status, 200, hidden_paths)
                self.assertEqual(hidden_paths.get("data", hidden_paths)["paths"], [])
                status, _ = call(restarted, bob_token, "GET", f"/api/g25/reasoning/chains/{chain_id}?project=alpha")
                self.assertEqual(status, 404)
            finally:
                stop(restarted, restarted_thread)


if __name__ == "__main__":
    unittest.main()
