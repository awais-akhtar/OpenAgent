from __future__ import annotations

import os
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager, required_scope_for_path
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class AuthEnforcementTests(unittest.TestCase):
    def _start_server(self, store: OpenAgentStore):
        server = build_http_server("127.0.0.1", 0, store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _assert_http_error(self, request, status: int) -> None:
        with self.assertRaises(HTTPError) as ctx:
            try:
                urlopen(request, timeout=15)
            except ConnectionError:
                # Windows may abort a loopback connection under suite-wide
                # socket churn before the server returns its auth rejection.
                # These requests are rejected before route dispatch, so one
                # retry cannot repeat a protected operation.
                urlopen(request, timeout=15)
        try:
            self.assertEqual(ctx.exception.code, status)
        finally:
            ctx.exception.close()

    def test_protected_routes_are_open_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server, thread = self._start_server(store)
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                for path in ("/api/developer/status", "/api/local-ai/status", "/api/security/status"):
                    with urlopen(f"{base_url}{path}", timeout=15) as response:
                        self.assertEqual(response.status, 200)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

    def test_main_chat_and_backup_routes_require_their_declared_scopes(self) -> None:
        self.assertEqual(required_scope_for_path("/api/chat", "POST"), "chat:write")
        self.assertEqual(required_scope_for_path("/api/chat/stream", "POST"), "chat:write")
        self.assertEqual(required_scope_for_path("/api/backups", "GET"), "backup:read")
        self.assertEqual(required_scope_for_path("/api/backups/restore", "POST"), "backup:admin")

        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server, thread = self._start_server(store)
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
                    for path, method in (
                        ("/api/chat", "POST"),
                        ("/api/backups", "GET"),
                        ("/api/backups/restore", "POST"),
                    ):
                        request = Request(f"{base_url}{path}", data=b"{}" if method == "POST" else None, method=method)
                        with self.subTest(path=path, method=method):
                            self._assert_http_error(request, 401)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

    def test_enforcement_blocks_and_allows_correctly_when_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            auth = AuthManager(store)
            user = auth.register("guard", "guard@example.com", "a-strong-password-1")
            session = auth.login("guard", "a-strong-password-1")
            full_token = session["access_token"]
            restricted = auth.create_api_token(user["user_id"], "restricted", scopes=["chat:read"])

            server, thread = self._start_server(store)
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
                    self._assert_http_error(f"{base_url}/api/developer/status", 401)
                    self._assert_http_error(Request(f"{base_url}/api/chat", data=b"{}", method="POST"), 401)
                    self._assert_http_error(f"{base_url}/api/backups", 401)
                    self._assert_http_error(Request(
                        f"{base_url}/api/developer/status",
                        headers={"Authorization": f"Bearer {restricted['token']}"},
                    ), 403)

                    with urlopen(Request(
                        f"{base_url}/api/developer/status",
                        headers={"Authorization": f"Bearer {full_token}"},
                    ), timeout=15) as ok_response:
                        self.assertEqual(ok_response.status, 200)

                    with urlopen(Request(
                        f"{base_url}/api/backups", headers={"Authorization": f"Bearer {full_token}"}),
                        timeout=15) as backup_list:
                        self.assertEqual(backup_list.status, 200)

                    self._assert_http_error(Request(
                        f"{base_url}/api/chat", data=b"{}", method="POST",
                        headers={"Authorization": f"Bearer {restricted['token']}", "Content-Type": "application/json"},
                    ), 403)

                    # Unprotected routes remain open even while enforcement is on.
                    with urlopen(f"{base_url}/api/health", timeout=15) as health:
                        self.assertEqual(health.status, 200)

                    # The expanded scope map now also gates previously-open route groups.
                    self._assert_http_error(f"{base_url}/api/personal-os/status", 401)
                    with urlopen(
                        Request(
                            f"{base_url}/api/personal-os/status",
                            headers={"Authorization": f"Bearer {full_token}"},
                        ),
                        timeout=15,
                    ) as personal_os_ok:
                        self.assertEqual(personal_os_ok.status, 200)

                    self._assert_http_error(f"{base_url}/api/terminal/runs", 401)
                    with urlopen(
                        Request(
                            f"{base_url}/api/terminal/runs?project=default",
                            headers={"Authorization": f"Bearer {full_token}"},
                        ),
                        timeout=15,
                    ) as terminal_ok:
                        self.assertEqual(terminal_ok.status, 200)
                    self._assert_http_error(Request(
                        f"{base_url}/api/terminal/runs",
                        headers={"Authorization": f"Bearer {restricted['token']}"},
                    ), 403)

                    # POST/PUT/PATCH/DELETE require the write/admin scope, not the read scope.
                    write_only = auth.create_api_token(user["user_id"], "write-only", scopes=["personal_os:write"])
                    self._assert_http_error(Request(
                        f"{base_url}/api/personal-os/status",
                        headers={"Authorization": f"Bearer {write_only['token']}"},
                    ), 403)
                with urlopen(f"{base_url}/api/developer/status", timeout=15) as unrestricted_again:
                    self.assertEqual(unrestricted_again.status, 200)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
