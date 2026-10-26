from __future__ import annotations

import json
import http.client
import os
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from openagent.auth import AuthError, AuthManager, personal_remote_public_route, required_scope_for_path
from openagent.config import StartupConfigurationError, startup_settings
from openagent.routes import route_contracts
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


BOOTSTRAP = "batch2b-test-bootstrap-secret-32-chars-minimum"


def remote_environment(**extra: str) -> dict[str, str]:
    return {
        "OPENAGENT_DEPLOYMENT_MODE": "personal_remote",
        "OPENAGENT_AUTH_ENFORCED": "1",
        "OPENAGENT_PERSONAL_REMOTE_HTTPS_TERMINATED": "1",
        "OPENAGENT_PERSONAL_REMOTE_HOSTS": "remote.test",
        "OPENAGENT_BOOTSTRAP_TOKEN": BOOTSTRAP,
        **extra,
    }


class PersonalRemoteConfigTests(unittest.TestCase):
    def test_local_default_and_personal_remote_configuration_are_fail_closed(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(startup_settings("127.0.0.1", 8765).deployment_mode, "single_user")
            with self.assertRaises(StartupConfigurationError):
                startup_settings("0.0.0.0", 8765)
        with patch.dict(os.environ, remote_environment(OPENAGENT_AUTH_ENFORCED="0"), clear=True):
            with self.assertRaisesRegex(StartupConfigurationError, "AUTH_ENFORCED=0"):
                startup_settings("127.0.0.1", 8765)
        with patch.dict(os.environ, remote_environment(OPENAGENT_PERSONAL_REMOTE_HOSTS=""), clear=True):
            with self.assertRaisesRegex(StartupConfigurationError, "HOSTS allowlist"):
                startup_settings("127.0.0.1", 8765)
        with patch.dict(os.environ, remote_environment(OPENAGENT_UNSAFE_ALLOW_UNAUTHENTICATED_NETWORK="1"), clear=True):
            with self.assertRaisesRegex(StartupConfigurationError, "personal_remote"):
                startup_settings("0.0.0.0", 8765)
        with patch.dict(os.environ, {"OPENAGENT_DEPLOYMENT_MODE": "typo"}, clear=True):
            with self.assertRaisesRegex(StartupConfigurationError, "must be single_user"):
                startup_settings("127.0.0.1", 8765)

    def test_every_declared_private_route_has_an_explicit_scope(self) -> None:
        public = {
            ("GET", "/api/health"),
            ("GET", "/health"),
            ("POST", "/api/auth/login"),
            ("POST", "/api/auth/refresh"),
            ("POST", "/api/auth/register"),
        }
        missing = []
        for group in route_contracts():
            for route in group["routes"]:
                method = str(route["method"])
                path = str(route["path"])
                if (method, path) in public:
                    self.assertTrue(personal_remote_public_route(path, method))
                    continue
                sample_path = path
                for param in ("id", "token_id", "task_id", "project_id", "run_id", "session_id"):
                    sample_path = sample_path.replace("{" + param + "}", "sample")
                if required_scope_for_path(sample_path, method) is None:
                    missing.append((group["group"], method, path))
        self.assertEqual(missing, [])
        self.assertIsNone(required_scope_for_path("/api/unregistered/private", "GET"))


class PersonalRemoteAccountTests(unittest.TestCase):
    def test_one_owner_bootstrap_and_multisession_tokens(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, remote_environment(), clear=True):
            manager = AuthManager(OpenAgentStore(tmp))
            self.assertIsNone(manager.validate_personal_remote_accounts())
            owner = manager.register(
                "solo-owner", "solo@example.test", "temporary-owner-password",
                bootstrap_token=BOOTSTRAP,
            )
            self.assertEqual(owner["role"], "owner")
            self.assertEqual(manager.validate_personal_remote_accounts(), owner["user_id"])
            with self.assertRaisesRegex(AuthError, "additional account registration is disabled"):
                manager.register("second", "second@example.test", "temporary-second-password", bootstrap_token=BOOTSTRAP, authorized=True)
            with self.assertRaisesRegex(ValueError, "explicit least-privilege"):
                manager.create_api_token(owner["user_id"], "implicit-broad-token")
            token = manager.create_api_token(owner["user_id"], "read-token", scopes=["knowledge:read"])
            self.assertEqual(token["scopes"], ["knowledge:read"])

    def test_ambiguous_legacy_store_is_refused_without_migration(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = AuthManager(store)
            owner = manager.register("owner", "owner@example.test", "temporary-owner-password")
            manager.register("legacy-member", "member@example.test", "temporary-member-password", authorized=True)
            with patch.dict(os.environ, remote_environment(), clear=True):
                with self.assertRaisesRegex(AuthError, "exactly one active owner"):
                    manager.validate_personal_remote_accounts()
            with store.session() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM auth_users").fetchone()[0], 2)

    def test_personal_remote_server_bootstraps_owner_and_gates_private_routes(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, remote_environment(), clear=True):
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                def request(method: str, path: str, *, token: str = "", body: dict[str, object] | None = None,
                            host: str = "remote.test", proto: str = "https", origin: str = "",
                            bootstrap: str = "") -> tuple[int, dict[str, object], dict[str, str]]:
                    payload = json.dumps(body).encode("utf-8") if body is not None else None
                    headers = {"Host": host, "X-Forwarded-Proto": proto}
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                    if body is not None:
                        headers["Content-Type"] = "application/json"
                    if origin:
                        headers["Origin"] = origin
                    if bootstrap:
                        headers["X-OpenAgent-Bootstrap-Token"] = bootstrap
                    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
                    connection.request(method, path, body=payload, headers=headers)
                    response = connection.getresponse()
                    raw = response.read()
                    status = response.status
                    response_headers = dict(response.getheaders())
                    connection.close()
                    try:
                        parsed = json.loads(raw.decode("utf-8")) if raw else {}
                    except json.JSONDecodeError:
                        parsed = {}
                    return status, parsed, response_headers

                status, payload, _ = request("POST", "/api/auth/register", body={
                    "username": "remote-owner", "email": "remote@example.test", "password": "temporary-remote-password",
                })
                self.assertEqual(status, 403, payload)  # No bootstrap proof.
                status, payload, _ = request("POST", "/api/auth/register", bootstrap=BOOTSTRAP, body={
                    "username": "remote-owner", "email": "remote@example.test", "password": "temporary-remote-password",
                    "bootstrap_token": "must-be-ignored",
                })
                self.assertEqual(status, 201)
                created = payload["data"]["user"]
                manager = AuthManager(store)
                status, _, _ = request("POST", "/api/auth/register", token=manager.login("remote-owner", "temporary-remote-password")["access_token"], bootstrap=BOOTSTRAP, body={
                    "username": "second", "email": "second@example.test", "password": "temporary-second-password",
                })
                self.assertEqual(status, 403)
                login_status, login_payload, _ = request("POST", "/api/auth/login", body={
                    "username_or_email": "remote-owner", "password": "temporary-remote-password",
                })
                self.assertEqual(login_status, 200)
                session = login_payload
                token = manager.create_api_token(created["user_id"], "smoke-owner", scopes=[
                    "knowledge:read", "knowledge:write", "media:read", "media:write", "settings:read", "settings:write",
                ])["token"]

                self.assertEqual(request("GET", "/api/health")[0], 200)
                self.assertEqual(request("GET", "/api/knowledge/documents?project=default")[0], 401)
                self.assertEqual(request("GET", "/api/research/sessions?project=default")[0], 401)
                self.assertEqual(request("GET", "/api/media/assets?project=default")[0], 401)
                foreign_store = OpenAgentStore(os.path.join(tmp, "foreign-installation"))
                foreign_auth = AuthManager(foreign_store)
                foreign_owner = foreign_auth.register("foreign-owner", "foreign@example.test", "temporary-foreign-password", bootstrap_token=BOOTSTRAP)
                foreign_token = foreign_auth.create_api_token(foreign_owner["user_id"], "foreign-token", scopes=["knowledge:read"])["token"]
                wrong_status, _, _ = request("GET", "/api/knowledge/documents?project=default", token=foreign_token)
                self.assertEqual(wrong_status, 401)
                self.assertEqual(request("GET", "/api/knowledge/documents?project=default", token=token)[0], 200)
                self.assertEqual(request("GET", "/api/research/sessions?project=default", token=token)[0], 200)
                self.assertEqual(request("GET", "/api/media/assets?project=default", token=token)[0], 200)
                self.assertEqual(request("GET", "/api/private-unknown", token=token)[0], 403)
                self.assertEqual(request("GET", "/api/routes/handlers")[0], 401)
                self.assertEqual(request("GET", "/api/routes/handlers", token=token)[0], 403)
                path_status, path_payload, _ = request("POST", "/api/media/ingest", token=token,
                    body={"project": "default", "path": str(store.data_dir / "openagent.sqlite3")})
                self.assertEqual(path_status, 403)
                self.assertIn("server-path", str(path_payload.get("error", "")))
                self.assertEqual(request("GET", "/api/knowledge/documents", token=token, host="attacker.test")[0], 421)
                self.assertEqual(request("GET", "/api/knowledge/documents", token=token, proto="http")[0], 400)
                self.assertEqual(request("GET", "/api/knowledge/documents", token=token, origin="https://attacker.test")[0], 403)
                _, _, headers = request("GET", "/api/knowledge/documents", token=token)
                self.assertIn("no-store", headers.get("Cache-Control", ""))
                self.assertTrue(session.get("access_token"))  # Browser/session login remains supported.
            finally:
                server.shutdown()
                thread.join(timeout=3)
                server.server_close()

    def test_existing_ambiguous_database_refuses_personal_remote_server_startup(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = AuthManager(store)
            manager.register("owner", "owner@example.test", "temporary-owner-password")
            manager.register("member", "member@example.test", "temporary-member-password", authorized=True)
            with patch.dict(os.environ, remote_environment(), clear=True):
                with self.assertRaisesRegex(StartupConfigurationError, "exactly one active owner"):
                    build_http_server("127.0.0.1", 0, store)


if __name__ == "__main__":
    unittest.main()
