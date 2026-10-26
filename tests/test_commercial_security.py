from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openagent.auth import AuthError, AuthManager, LOGIN_FAILURE_LIMIT, PBKDF2_ITERATIONS
from openagent.config import StartupConfigurationError, startup_settings
from openagent.crypto import CryptoError, decrypt_text, encrypt_text
from openagent.egress import EgressClient, EgressDecision, EgressError, EgressPolicy
from openagent.http_security import RateLimit, SlidingWindowRateLimiter, request_rate_class
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore
from openagent.vault import EncryptedVault


def legacy_password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return "pbkdf2$sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode(),
        base64.b64encode(derived).decode(),
    )


def legacy_vault_ciphertext(value: str, passphrase: str) -> str:
    salt = secrets.token_bytes(32)
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 200_000, dklen=32)
    stream = bytearray()
    block = key
    raw = value.encode()
    while len(stream) < len(raw):
        block = hashlib.sha256(block).digest()
        stream.extend(block)
    ciphertext = bytes(left ^ right for left, right in zip(raw, stream))
    mac = hmac.new(key, ciphertext, hashlib.sha256).digest()
    return base64.b64encode(salt + mac + ciphertext).decode()


class StartupPolicyTests(unittest.TestCase):
    def test_non_loopback_requires_team_auth(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(StartupConfigurationError):
                startup_settings("0.0.0.0", 8765)

    def test_team_auth_and_explicit_unsafe_override_are_distinct(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAGENT_DEPLOYMENT_MODE": "team", "OPENAGENT_AUTH_ENFORCED": "1"},
            clear=True,
        ):
            self.assertFalse(startup_settings("0.0.0.0", 8765).loopback_only)
        with patch.dict(
            os.environ,
            {"OPENAGENT_UNSAFE_ALLOW_UNAUTHENTICATED_NETWORK": "1"},
            clear=True,
        ):
            self.assertTrue(startup_settings("0.0.0.0", 8765).unsafe_network_override)


class CryptoMigrationTests(unittest.TestCase):
    def test_aead_detects_tampering_and_binds_purpose(self) -> None:
        payload = encrypt_text("private", "correct horse battery staple", purpose="test.record")
        self.assertEqual(
            decrypt_text(payload, "correct horse battery staple", purpose="test.record"),
            "private",
        )
        with self.assertRaises(CryptoError):
            decrypt_text(payload, "correct horse battery staple", purpose="different.record")
        damaged = dict(payload)
        raw = bytearray(base64.b64decode(damaged["ciphertext"]))
        raw[-1] ^= 1
        damaged["ciphertext"] = base64.b64encode(raw).decode()
        with self.assertRaises(CryptoError):
            decrypt_text(damaged, "correct horse battery staple", purpose="test.record")

    def test_pbkdf2_password_is_rehashed_after_valid_login(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            auth = AuthManager(store)
            user = auth.register("owner", "owner@example.com", "initial-password")
            with store.session() as connection:
                connection.execute(
                    "UPDATE auth_users SET password_hash = ? WHERE id = ?",
                    (legacy_password_hash("initial-password"), user["user_id"]),
                )
            auth.login("owner", "initial-password")
            with store.session() as connection:
                encoded = connection.execute(
                    "SELECT password_hash FROM auth_users WHERE id = ?", (user["user_id"],)
                ).fetchone()[0]
            self.assertTrue(encoded.startswith("$argon2id$"))

    def test_legacy_vault_record_migrates_only_after_successful_decrypt(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            vault = EncryptedVault(store)
            entry_id = "legacy-id-kept-for-compatibility"
            encrypted = legacy_vault_ciphertext("retained value", "old-passphrase")
            with store.session() as connection:
                connection.execute(
                    "INSERT INTO vault_entries VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        entry_id,
                        "default",
                        "secret",
                        "Legacy",
                        "[]",
                        encrypted,
                        hashlib.sha256(b"fp:old-passphrase").hexdigest()[:16],
                        "",
                        0,
                        "",
                        "2026-01-01T00:00:00+00:00",
                        "2026-01-01T00:00:00+00:00",
                    ),
                )
            with self.assertRaises(ValueError):
                vault.retrieve_secret(entry_id, "wrong-passphrase")
            with store.session() as connection:
                self.assertEqual(connection.execute("SELECT ciphertext FROM vault_entries").fetchone()[0], encrypted)
            result = vault.retrieve_secret(entry_id, "old-passphrase")
            self.assertEqual(result["secret"], "retained value")
            with store.session() as connection:
                migrated = connection.execute("SELECT ciphertext FROM vault_entries").fetchone()[0]
            self.assertEqual(json.loads(migrated)["algorithm"], "AES-256-GCM")


class AuthenticationBoundaryTests(unittest.TestCase):
    def test_team_bootstrap_requires_exact_token_and_later_owner_authorization(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"OPENAGENT_DEPLOYMENT_MODE": "team", "OPENAGENT_BOOTSTRAP_TOKEN": "one-time-bootstrap"},
            clear=True,
        ):
            auth = AuthManager(OpenAgentStore(tmp))
            with self.assertRaises(AuthError):
                auth.register("owner", "owner@example.com", "long-password")
            owner = auth.register(
                "owner",
                "owner@example.com",
                "long-password",
                bootstrap_token="one-time-bootstrap",
            )
            self.assertEqual(owner["role"], "owner")
            with self.assertRaises(AuthError):
                auth.register("member", "member@example.com", "long-password")
            member = auth.register(
                "member",
                "member@example.com",
                "long-password",
                authorized=True,
            )
            self.assertEqual(member["role"], "member")

    def test_login_is_throttled_after_repeated_failures(self) -> None:
        with TemporaryDirectory() as tmp:
            auth = AuthManager(OpenAgentStore(tmp))
            auth.register("owner", "owner@example.com", "long-password")
            for _ in range(LOGIN_FAILURE_LIMIT):
                with self.assertRaisesRegex(AuthError, "invalid"):
                    auth.login("owner", "wrong-password")
            with self.assertRaisesRegex(AuthError, "too many"):
                auth.login("owner", "long-password")


class EgressBoundaryTests(unittest.TestCase):
    @staticmethod
    def _resolver(*addresses: str):
        def resolve(host: str, port: int, **_: object):
            return [(2, 1, 6, "", (address, port)) for address in addresses]
        return resolve

    def test_external_access_requires_approval(self) -> None:
        policy = EgressPolicy(resolver=self._resolver("93.184.216.34"))
        with self.assertRaises(PermissionError):
            policy.authorize("https://example.com/data", approved=False)
        decision = policy.authorize("https://example.com/data", approved=True)
        self.assertEqual(decision.addresses, ("93.184.216.34",))

    def test_private_and_mixed_dns_answers_are_rejected(self) -> None:
        private = EgressPolicy(resolver=self._resolver("10.0.0.7"))
        with self.assertRaises(EgressError):
            private.authorize("https://internal.example/data", approved=True)
        mixed = EgressPolicy(resolver=self._resolver("93.184.216.34", "127.0.0.1"))
        with self.assertRaises(EgressError):
            mixed.authorize("https://rebind.example/data", approved=True)

    def test_loopback_transport_is_available_only_when_feature_allows_it(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                payload = b"local evidence"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}/health"
        try:
            client = EgressClient()
            with self.assertRaises(EgressError):
                client.fetch(url, approved=True, purpose="blocked local request")
            response = client.fetch(
                url,
                approved=False,
                allow_loopback=True,
                purpose="local browser check",
                allowed_content_types={"text/plain"},
            )
            self.assertEqual(response.body, b"local evidence")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_large_download_resumes_without_buffering_the_existing_prefix(self) -> None:
        payload = b"abcdefghij"

        class Handler(BaseHTTPRequestHandler):
            invalid_range = False
            short_response = False

            def do_GET(self) -> None:
                start = int(self.headers.get("Range", "bytes=0-").removeprefix("bytes=").split("-", 1)[0])
                body = payload[start:]
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(body)))
                if start:
                    range_start = 0 if self.invalid_range else start
                    self.send_header("Content-Range", f"bytes {range_start}-{len(payload) - 1}/{len(payload)}")
                self.end_headers()
                self.wfile.write(body[:2] if self.short_response else body)

            def log_message(self, *_: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        class TestPolicy:
            def authorize(self, url: str, **_: object) -> EgressDecision:
                return EgressDecision(
                    url=url,
                    hostname="example.test",
                    port=server.server_address[1],
                    addresses=("127.0.0.1",),
                    local=False,
                )

        try:
            with TemporaryDirectory() as tmp:
                target = Path(tmp) / "model.part"
                target.write_bytes(payload[:4])
                result = EgressClient(policy=TestPolicy()).download(  # type: ignore[arg-type]
                    f"http://example.test:{server.server_address[1]}/model",
                    target,
                    approved=True,
                    purpose="test model download",
                    max_total_bytes=len(payload),
                )
                self.assertTrue(result["resumed"])
                self.assertEqual(target.read_bytes(), payload)
                target.write_bytes(payload[:4])
                Handler.invalid_range = True
                with self.assertRaisesRegex(EgressError, "unexpected partial range"):
                    EgressClient(policy=TestPolicy()).download(  # type: ignore[arg-type]
                        f"http://example.test:{server.server_address[1]}/model",
                        target,
                        approved=True,
                        purpose="invalid range check",
                        max_total_bytes=len(payload),
                    )
                self.assertEqual(target.read_bytes(), payload[:4])
                Handler.invalid_range = False
                Handler.short_response = True
                with self.assertRaisesRegex(EgressError, "declared byte count"):
                    EgressClient(policy=TestPolicy()).download(  # type: ignore[arg-type]
                        f"http://example.test:{server.server_address[1]}/model",
                        target,
                        approved=True,
                        purpose="short response check",
                        max_total_bytes=len(payload),
                    )
                self.assertEqual(target.read_bytes(), payload[:6])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_bounded_line_callback_runs_before_stream_finishes(self) -> None:
        continue_server = threading.Event()
        callback_was_live: list[bool] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                self.wfile.write(b'{"status":"downloading"}\n')
                self.wfile.flush()
                callback_was_live.append(continue_server.wait(timeout=2))
                self.wfile.write(b'{"status":"success"}\n')

            def log_message(self, *_: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        lines: list[str] = []
        try:
            response = EgressClient().fetch(
                f"http://127.0.0.1:{server.server_address[1]}/pull",
                approved=False,
                allow_loopback=True,
                purpose="stream progress test",
                max_response_bytes=1024,
                on_line=lambda line: (lines.append(line), continue_server.set()),
            )
            self.assertEqual(callback_was_live, [True])
            self.assertEqual(len(lines), 2)
            self.assertIn(b'"success"', response.body)
        finally:
            continue_server.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class HttpBoundaryTests(unittest.TestCase):
    def test_trusted_origin_preflight_allows_document_updates(self) -> None:
        origin = "https://workbench.example"
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAGENT_TRUSTED_ORIGINS": origin}):
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(Path(tmp)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                cases = (
                    ("/api/document/example", origin, 204),
                    ("/v1/messages", origin, 204),
                    ("/api/document/example", "https://untrusted.example", 403),
                )
                for path, request_origin, expected in cases:
                    with self.subTest(path=path, origin=request_origin):
                        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
                        try:
                            connection.request("OPTIONS", path, headers={
                                "Origin": request_origin, "Access-Control-Request-Method": "PUT",
                            })
                            response = connection.getresponse()
                            response.read()
                            self.assertEqual(response.status, expected)
                            if expected == 204:
                                self.assertIn("PUT", response.getheader("Access-Control-Allow-Methods").split(", "))
                                self.assertEqual(response.getheader("Access-Control-Allow-Origin"), origin)
                                allowed_headers = response.getheader("Access-Control-Allow-Headers").lower()
                                if path == "/v1/messages":
                                    self.assertIn("x-api-key", allowed_headers)
                                    self.assertIn("anthropic-version", allowed_headers)
                                else:
                                    self.assertNotIn("x-api-key", allowed_headers)
                            else:
                                self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
                        finally:
                            connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_anthropic_version_header_reaches_compatibility_route(self) -> None:
        payload = json.dumps({"model": "local", "messages": [
            {"role": "user", "content": "hello"}], "max_tokens": 2})
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "0"}):
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(Path(tmp)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for headers, expected, error_type in (
                    ({"Content-Type": "application/json"}, 400, "invalid_request_error"),
                    ({"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
                     503, "api_error"),
                ):
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", server.server_address[1], timeout=10)
                    try:
                        connection.request("POST", "/v1/messages", body=payload, headers=headers)
                        response = connection.getresponse()
                        body = json.loads(response.read().decode("utf-8"))
                        self.assertEqual(response.status, expected)
                        self.assertEqual(body["type"], "error")
                        self.assertEqual(body["error"]["type"], error_type)
                    finally:
                        connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_anthropic_auth_gate_uses_anthropic_error_envelope(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(Path(tmp)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for credential_headers in ({}, {"x-api-key": "invalid-local-token"}):
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", server.server_address[1], timeout=10)
                    try:
                        connection.request("POST", "/v1/messages", body="{}",
                            headers={"Content-Type": "application/json", **credential_headers})
                        response = connection.getresponse()
                        body = json.loads(response.read().decode("utf-8"))
                        self.assertEqual(response.status, 401)
                        self.assertEqual(body["type"], "error")
                        self.assertEqual(body["error"]["type"], "authentication_error")
                    finally:
                        connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_all_mutation_verbs_are_limited_before_dispatch(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(Path(tmp)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                # Do not race the server thread's first select/accept cycle on
                # Windows. The socket is bound before serve_forever begins,
                # so an immediate POST can otherwise be aborted by the host
                # while the thread is still entering its request loop.
                ready = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
                try:
                    ready.request("GET", "/api/health")
                    response = ready.getresponse()
                    response.read()
                    self.assertEqual(response.status, 200)
                finally:
                    ready.close()
                with patch.object(SlidingWindowRateLimiter, "consume", return_value=(False, 9)):
                    for method in ("POST", "PUT", "PATCH", "DELETE"):
                        with self.subTest(method=method):
                            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
                            try:
                                # Rate limiting precedes JSON parsing, so this
                                # probe needs no body. Sending an unread body
                                # while the handler closes a rejected request
                                # can reset the response on Windows TCP.
                                connection.request(method, "/api/document/missing")
                                response = connection.getresponse()
                                response.read()
                                self.assertEqual(response.status, 429)
                                self.assertEqual(response.getheader("Retry-After"), "9")
                            finally:
                                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_rate_limit_classes_and_retry_window(self) -> None:
        limiter = SlidingWindowRateLimiter({"general": RateLimit(2, 10.0), "auth": RateLimit(2, 10.0)})
        self.assertEqual(request_rate_class("/api/auth/login", "POST"), "auth")
        self.assertEqual(request_rate_class("/api/auth/me"), "general")
        self.assertEqual(request_rate_class("/api/local-ai/models"), "general")
        self.assertEqual(request_rate_class("/api/local-ai/models/download", "POST"), "model_job")
        self.assertTrue(limiter.consume("client", "auth", now=1.0)[0])
        self.assertTrue(limiter.consume("client", "auth", now=2.0)[0])
        allowed, retry_after = limiter.consume("client", "auth", now=3.0)
        self.assertFalse(allowed)
        self.assertEqual(retry_after, 8)

    def test_malformed_json_is_400_and_security_headers_are_present(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(Path(tmp)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
                connection.request(
                    "POST",
                    "/api/workflows",
                    body=b"{not-json",
                    headers={"Content-Type": "application/json", "Content-Length": "9"},
                )
                response = connection.getresponse()
                body = json.loads(response.read().decode())
                self.assertEqual(response.status, 400)
                self.assertFalse(body["success"])
                self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
                self.assertEqual(response.getheader("X-Frame-Options"), "DENY")
                self.assertTrue(response.getheader("X-Request-ID"))
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
