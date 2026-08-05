from __future__ import annotations

import os
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class AuthEnforcementTests(unittest.TestCase):
    def _start_server(self, store: OpenAgentStore):
        server = build_http_server("127.0.0.1", 0, store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_protected_routes_are_open_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server, thread = self._start_server(store)
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                for path in ("/api/developer/status", "/api/local-ai/status", "/api/security/status"):
                    response = urlopen(f"{base_url}{path}", timeout=15)
                    self.assertEqual(response.status, 200)
            finally:
                server.shutdown()
                thread.join(timeout=2)

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
                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(f"{base_url}/api/developer/status", timeout=15)
                    self.assertEqual(ctx.exception.code, 401)

                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(
                            Request(
                                f"{base_url}/api/developer/status",
                                headers={"Authorization": f"Bearer {restricted['token']}"},
                            ),
                            timeout=15,
                        )
                    self.assertEqual(ctx.exception.code, 403)

                    ok_response = urlopen(
                        Request(
                            f"{base_url}/api/developer/status",
                            headers={"Authorization": f"Bearer {full_token}"},
                        ),
                        timeout=15,
                    )
                    self.assertEqual(ok_response.status, 200)

                    # Unprotected routes remain open even while enforcement is on.
                    health = urlopen(f"{base_url}/api/health", timeout=15)
                    self.assertEqual(health.status, 200)

                    # The expanded scope map now also gates previously-open route groups.
                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(f"{base_url}/api/personal-os/status", timeout=15)
                    self.assertEqual(ctx.exception.code, 401)
                    personal_os_ok = urlopen(
                        Request(
                            f"{base_url}/api/personal-os/status",
                            headers={"Authorization": f"Bearer {full_token}"},
                        ),
                        timeout=15,
                    )
                    self.assertEqual(personal_os_ok.status, 200)

                    # POST/PUT/PATCH/DELETE require the write/admin scope, not the read scope.
                    write_only = auth.create_api_token(user["user_id"], "write-only", scopes=["personal_os:write"])
                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(
                            Request(
                                f"{base_url}/api/personal-os/status",
                                headers={"Authorization": f"Bearer {write_only['token']}"},
                            ),
                            timeout=15,
                        )
                    self.assertEqual(ctx.exception.code, 403)
                unrestricted_again = urlopen(f"{base_url}/api/developer/status", timeout=15)
                self.assertEqual(unrestricted_again.status, 200)
            finally:
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
