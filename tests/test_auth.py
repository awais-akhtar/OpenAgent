from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthError, AuthManager
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class AuthManagerTests(unittest.TestCase):
    def test_register_login_and_session_lifecycle(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = AuthManager(store)

            user = manager.register("alice", "alice@example.com", "correct horse battery")
            self.assertEqual(user["role"], "owner")

            second = manager.register("bob", "bob@example.com", "another-strong-pass")
            self.assertEqual(second["role"], "member")

            with self.assertRaises(AuthError):
                manager.register("alice", "someone-else@example.com", "whatever-password")

            with self.assertRaises(AuthError):
                manager.login("alice", "wrong-password")

            session = manager.login("alice", "correct horse battery", ip="127.0.0.1", user_agent="unittest")
            self.assertIn("access_token", session)
            self.assertIn("chat:read", session["scopes"])

            current = manager.current_user(session["access_token"])
            self.assertIsNotNone(current)
            self.assertEqual(current["username"], "alice")

            sessions = manager.list_sessions(user["user_id"])
            self.assertEqual(len(sessions), 1)
            self.assertTrue(sessions[0]["active"])

            revoked = manager.revoke_session(user["user_id"], sessions[0]["session_id"])
            self.assertTrue(revoked["revoked"])
            self.assertIsNone(manager.current_user(session["access_token"]))

            with self.assertRaises(KeyError):
                manager.revoke_session(user["user_id"], "does-not-exist")

    def test_two_factor_setup_and_enforced_login(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = AuthManager(store)
            user = manager.register("carol", "carol@example.com", "a-very-strong-password")

            setup = manager.setup_two_factor(user["user_id"])
            self.assertIn("secret", setup)
            self.assertEqual(len(setup["backup_codes"]), 8)

            with self.assertRaises(AuthError):
                manager.verify_two_factor(user["user_id"], "000000")

            from openagent.auth import _totp_code  # internal helper, same-package test

            valid_code = _totp_code(setup["secret"])
            verified = manager.verify_two_factor(user["user_id"], valid_code)
            self.assertTrue(verified["enabled"])

            with self.assertRaises(AuthError):
                manager.login("carol", "a-very-strong-password")

            valid_code_2 = _totp_code(setup["secret"])
            session = manager.login("carol", "a-very-strong-password", otp_code=valid_code_2)
            self.assertIn("access_token", session)

            disabled = manager.disable_two_factor(user["user_id"], _totp_code(setup["secret"]))
            self.assertFalse(disabled["enabled"])

    def test_api_tokens_can_be_scoped_and_revoked(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = AuthManager(store)
            user = manager.register("dave", "dave@example.com", "another-strong-password")

            issued = manager.create_api_token(user["user_id"], "ci-bot", scopes=["chat:read", "docs:write"])
            self.assertEqual(set(issued["scopes"]), {"chat:read", "docs:write"})

            resolved = manager.resolve_bearer(issued["token"])
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved["user_id"], user["user_id"])
            self.assertEqual(set(resolved["scopes"]), {"chat:read", "docs:write"})

            tokens = manager.list_api_tokens(user["user_id"])
            self.assertEqual(len(tokens), 1)
            self.assertNotIn("token", tokens[0])

            manager.revoke_api_token(user["user_id"], issued["token_id"])
            self.assertIsNone(manager.resolve_bearer(issued["token"]))

            with self.assertRaises(KeyError):
                manager.revoke_api_token(user["user_id"], "missing-token-id")

    def test_scope_catalog_is_stable(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = AuthManager(store)
            scopes = manager.list_scopes()
            self.assertGreaterEqual(len(scopes), 20)
            self.assertTrue(all("scope" in entry and "description" in entry for entry in scopes))


class AuthHttpRouteTests(unittest.TestCase):
    def test_register_login_and_me_over_http(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                register_request = Request(
                    f"{base_url}/api/auth/register",
                    data=json.dumps({
                        "username": "erin",
                        "email": "erin@example.com",
                        "password": "yet-another-strong-pass",
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                registered = json.loads(urlopen(register_request, timeout=15).read().decode("utf-8"))
                self.assertEqual(registered["user"]["username"], "erin")

                login_request = Request(
                    f"{base_url}/api/auth/login",
                    data=json.dumps({
                        "username_or_email": "erin",
                        "password": "yet-another-strong-pass",
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                session = json.loads(urlopen(login_request, timeout=15).read().decode("utf-8"))
                token = session["access_token"]
                refresh_token = session["refresh_token"]
                self.assertTrue(token)
                self.assertTrue(refresh_token)

                me_request = Request(f"{base_url}/api/auth/me", headers={"Authorization": f"Bearer {token}"})
                me = json.loads(urlopen(me_request, timeout=15).read().decode("utf-8"))
                self.assertEqual(me["user"]["username"], "erin")

                refresh_request = Request(
                    f"{base_url}/api/auth/refresh",
                    data=json.dumps({"refresh_token": refresh_token}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                refreshed = json.loads(urlopen(refresh_request, timeout=15).read().decode("utf-8"))
                self.assertNotEqual(refreshed["access_token"], token)
                self.assertEqual(refreshed["session_id"], session["session_id"])

                old_me_request = Request(f"{base_url}/api/auth/me", headers={"Authorization": f"Bearer {token}"})
                with self.assertRaises(HTTPError) as old_ctx:
                    urlopen(old_me_request, timeout=15)
                self.assertEqual(old_ctx.exception.code, 401)
                old_ctx.exception.close()

                unauthorized = Request(f"{base_url}/api/auth/me")
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(unauthorized, timeout=15)
                self.assertEqual(ctx.exception.code, 401)
                ctx.exception.close()

                scopes = json.loads(urlopen(f"{base_url}/api/auth/scopes", timeout=15).read().decode("utf-8"))
                self.assertGreaterEqual(len(scopes["scopes"]), 20)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
