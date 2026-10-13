from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager, required_scope_for_path
from openagent.egress import EgressError, EgressPolicy
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class ProtectedApiAcceptanceTests(unittest.TestCase):
    def test_protected_api_rejects_missing_invalid_and_under_scoped_tokens(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENAGENT_AUTH_ENFORCED": "1"}):
            store = OpenAgentStore(tmp)
            manager = AuthManager(store)
            user = manager.register("reader", "reader@example.test", "a sufficiently long password")
            read_token = manager.create_api_token(user["user_id"], "read-only", scopes=["docs:read"])["token"]
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                cases = (
                    ({}, 401),
                    ({"Authorization": "Bearer invalid-token"}, 401),
                    ({"Authorization": f"Bearer {read_token}"}, 403),
                )
                for headers, expected_status in cases:
                    request = Request(
                        f"{base}/api/document",
                        data=json.dumps({"title": "must not be created", "content": "isolated"}).encode(),
                        headers={"Content-Type": "application/json", **headers},
                        method="POST",
                    )
                    with self.subTest(status=expected_status):
                        with self.assertRaises(HTTPError) as ctx:
                            urlopen(request, timeout=10)
                        self.assertEqual(ctx.exception.code, expected_status)
                        ctx.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_authenticated_document_owner_cannot_be_spoofed_or_read_cross_user(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENAGENT_AUTH_ENFORCED": "1"}):
            store = OpenAgentStore(tmp)
            manager = AuthManager(store)
            alice = manager.register("alice", "alice@example.test", "a sufficiently long password")
            bob = manager.register("bob", "bob@example.test", "another sufficiently long password")
            alice_token = manager.create_api_token(
                alice["user_id"], "alice-docs", scopes=["docs:read", "docs:write"]
            )["token"]
            bob_token = manager.create_api_token(
                bob["user_id"], "bob-docs", scopes=["docs:read", "docs:write"]
            )["token"]
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def request(path: str, token: str, *, method: str = "GET", payload: dict | None = None):
                body = json.dumps(payload).encode() if payload is not None else None
                headers = {"Authorization": f"Bearer {token}"}
                if body is not None:
                    headers["Content-Type"] = "application/json"
                req = Request(f"{base}{path}", data=body, headers=headers, method=method)
                try:
                    with urlopen(req, timeout=10) as response:
                        return response.status, json.loads(response.read().decode())
                except HTTPError as error:
                    try:
                        return error.code, json.loads(error.read().decode())
                    finally:
                        error.close()

            try:
                status, created = request(
                    "/api/document", alice_token, method="POST",
                    payload={"title": "Alice private document", "content": "private", "project": "shared", "owner": bob["user_id"]},
                )
                self.assertEqual(status, 201)
                document_id = created["document"]["doc_id"]
                self.assertEqual(request(f"/api/document/{document_id}", alice_token)[0], 200)
                self.assertEqual(request(f"/api/document/{document_id}", bob_token)[0], 404)
                # Supplying Alice's owner id in the URL query cannot widen Bob's scope.
                self.assertEqual(request(f"/api/document/{document_id}?owner={alice['user_id']}", bob_token)[0], 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


class EgressNegativeAcceptanceTests(unittest.TestCase):
    def test_local_and_mixed_dns_destinations_are_denied_even_when_approved(self) -> None:
        def resolver_for(*addresses: str):
            return lambda host, port, **kwargs: [(2, 1, 6, "", (address, port)) for address in addresses]

        candidates = (
            ("https://localhost/resource", resolver_for("127.0.0.1")),
            ("https://service.example.test/resource", resolver_for("10.0.0.8")),
            ("https://rebound.example.test/resource", resolver_for("93.184.216.34", "192.168.1.4")),
            ("http://169.254.169.254/latest/meta-data/", resolver_for("169.254.169.254")),
        )
        for url, resolver in candidates:
            with self.subTest(url=url), self.assertRaises(EgressError):
                EgressPolicy(resolver=resolver).authorize(url, approved=True)

    def test_sensitive_route_scope_contracts_are_enforced_by_method(self) -> None:
        self.assertEqual(required_scope_for_path("/api/documents/library", "GET"), "docs:read")
        self.assertEqual(required_scope_for_path("/api/document", "POST"), "docs:write")
        self.assertEqual(required_scope_for_path("/api/security/egress/requests", "GET"), "security:read")
        self.assertEqual(required_scope_for_path("/api/security/egress/resolve", "POST"), "security:admin")
        self.assertIsNone(required_scope_for_path("/health", "GET"))


if __name__ == "__main__":
    unittest.main()
