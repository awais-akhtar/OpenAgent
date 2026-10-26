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
from openagent.models import AuditEvent
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class ProtectedApiAcceptanceTests(unittest.TestCase):
    def _assert_rejected(self, request, expected_status: int) -> None:
        try:
            with urlopen(request, timeout=10) as response:
                self.fail(f"expected HTTP {expected_status}, received {response.status}")
        except HTTPError as error:
            try:
                self.assertEqual(error.code, expected_status)
            finally:
                error.close()
        except ConnectionAbortedError as error:
            # A Windows loopback abort can happen before this pre-dispatch
            # authorization rejection reaches the client under suite socket
            # churn. Retry only that specific transient; all three requests
            # are rejected before a protected handler can mutate state.
            if getattr(error, "winerror", None) != 10053:
                raise
            with self.assertRaises(HTTPError) as retry:
                urlopen(request, timeout=10)
            try:
                self.assertEqual(retry.exception.code, expected_status)
            finally:
                retry.exception.close()

    def test_cross_project_audit_feed_requires_security_admin_scope(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENAGENT_AUTH_ENFORCED": "1"}):
            store = OpenAgentStore(tmp)
            auth = AuthManager(store)
            user = auth.register("auditor", "auditor@example.test", "a sufficiently long password")
            read_token = auth.create_api_token(user["user_id"], "security-reader", scopes=["security:read"])["token"]
            admin_token = auth.create_api_token(user["user_id"], "security-admin", scopes=["security:admin"])["token"]
            store.add_audit_event(AuditEvent(actor="alpha-user", action="read", resource="alpha-secret", project="alpha"))
            store.add_audit_event(AuditEvent(actor="beta-user", action="read", resource="beta-secret", project="beta"))
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def get(path: str, token: str | None = None):
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                try:
                    with urlopen(Request(f"{base}{path}", headers=headers), timeout=10) as response:
                        return response.status, json.loads(response.read().decode())
                except HTTPError as error:
                    try:
                        return error.code, json.loads(error.read().decode())
                    finally:
                        error.close()

            try:
                self.assertEqual(get("/api/audit")[0], 401)
                self.assertEqual(get("/api/audit", read_token)[0], 403)
                status, payload = get("/api/audit", admin_token)
                self.assertEqual(status, 200)
                self.assertTrue({"alpha", "beta"}.issubset(
                    {event["project"] for event in payload["events"]}
                ))
                scoped_status, scoped_payload = get("/api/audit?project=alpha&limit=10", admin_token)
                self.assertEqual(scoped_status, 200)
                self.assertEqual({event["project"] for event in scoped_payload["events"]}, {"alpha"})
                self.assertEqual({event["resource"] for event in scoped_payload["events"]}, {"alpha-secret"})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

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
                        self._assert_rejected(request, expected_status)
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
        self.assertEqual(required_scope_for_path("/health", "GET"), "production:read")


if __name__ == "__main__":
    unittest.main()
