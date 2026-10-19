from __future__ import annotations

from email.message import EmailMessage
import json
import os
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager
from openagent.communication import CommunicationHub
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class ResilientSyncTests(unittest.TestCase):
    def test_duplicate_messages_are_skipped_on_second_sync(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            hub = CommunicationHub(store)
            hub.configure_connector("slack", enabled=True)

            payload = {
                "messages": [
                    {"sender": "alice", "recipients": ["bob"], "subject": "Hi", "body": "hello there", "thread_id": "t1"},
                ]
            }
            first = hub.sync_connector_resilient("slack", payload=payload)
            self.assertEqual(first["duplicates_skipped"], 0)
            self.assertEqual(first["attempts"], 1)
            self.assertEqual(first["imported_messages"], 1)

            second = hub.sync_connector_resilient("slack", payload=payload)
            self.assertEqual(second["duplicates_skipped"], 1)
            self.assertEqual(second["imported_messages"], 0)

    def test_unknown_connector_retries_then_dead_letters(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            hub = CommunicationHub(store)

            result = hub.sync_connector_resilient("does-not-exist", payload={}, max_attempts=2)
            self.assertEqual(result["status"], "dead_lettered")
            self.assertEqual(result["attempts"], 2)
            self.assertTrue(result["dead_letter_id"])

            dead_letters = hub.list_dead_letters()
            self.assertEqual(len(dead_letters), 1)
            self.assertEqual(dead_letters[0]["platform"], "does-not-exist")
            self.assertEqual(dead_letters[0]["attempts"], 2)

    def test_original_sync_connector_has_no_cross_call_dedupe(self) -> None:
        """sync_connector itself keeps its original (non-deduping) behavior."""
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            hub = CommunicationHub(store)
            hub.configure_connector("slack", enabled=True)
            first = hub.sync_connector("slack", payload={
                "messages": [
                    {"sender": "alice", "recipients": ["bob"], "subject": "Hi", "body": "hello there", "thread_id": "t1"},
                ]
            })
            second = hub.sync_connector("slack", payload={
                "messages": [
                    {"sender": "alice", "recipients": ["bob"], "subject": "Hi again", "body": "hello again", "thread_id": "t2"},
                ]
            })
            self.assertEqual(first["imported_messages"], 1)
            self.assertEqual(second["imported_messages"], 1)  # unchanged: no dedupe in the original method

    def test_email_imap_provider_sync_imports_messages(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            hub = CommunicationHub(store)
            hub.configure_connector(
                "email",
                enabled=True,
                config={
                    "imap_host": "mail.local",
                    "imap_username": "user",
                    "imap_password": "pass",
                    "imap_ssl": True,
                },
            )

            message = EmailMessage()
            message["From"] = "alice@example.com"
            message["To"] = "bob@example.com"
            message["Subject"] = "Provider message"
            message["Message-ID"] = "<provider-message@example.com>"
            message.set_content("Hello from IMAP.")

            class FakeImap:
                def __init__(self, host, port, timeout=20):
                    self.host = host
                    self.port = port
                    self.timeout = timeout

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def login(self, username, password):
                    self.username = username
                    self.password = password
                    return "OK", []

                def select(self, mailbox):
                    return "OK", []

                def search(self, charset, query):
                    return "OK", [b"1"]

                def fetch(self, message_id, spec):
                    return "OK", [(b"1 (RFC822)", message.as_bytes())]

            with patch("openagent.communication.imaplib.IMAP4_SSL", FakeImap):
                run = hub.sync_connector("email", project="core", allow_external=True)

            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["imported_messages"], 1)
            inbox = hub.inbox(project="core")
            self.assertEqual(inbox[0]["subject"], "Provider message")

    def test_email_smtp_send_uses_configured_provider_after_approval(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            hub = CommunicationHub(store)
            hub.configure_connector(
                "email",
                enabled=True,
                config={
                    "smtp_host": "smtp.local",
                    "smtp_username": "user",
                    "smtp_password": "pass",
                    "smtp_from": "me@example.com",
                    "smtp_ssl": True,
                },
            )
            draft = hub.create_draft(
                platform="email",
                recipients=["you@example.com"],
                subject="Hello",
                body="External SMTP body",
                project="core",
            )
            approval = hub.request_send_approval(draft["id"], project="core")
            hub.resolve_send_approval(approval["id"], approved=True)
            sent_messages = []

            class FakeSmtp:
                def __init__(self, host, port, timeout=20):
                    self.host = host
                    self.port = port
                    self.timeout = timeout

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def login(self, username, password):
                    self.username = username
                    self.password = password

                def send_message(self, message):
                    sent_messages.append(message)

            with patch("openagent.communication.smtplib.SMTP_SSL", FakeSmtp):
                sent = hub.send_draft(draft["id"], approval_id=approval["id"], allow_external=True)

            self.assertEqual(sent["status"], "sent_external")
            self.assertEqual(len(sent_messages), 1)
            self.assertEqual(sent_messages[0]["To"], "you@example.com")

    def test_oauth_onboarding_refresh_and_xoauth2_imap_sync(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            hub = CommunicationHub(store)
            account = hub.configure_oauth_account(
                provider="google",
                client_id="client-id",
                client_secret="client-secret",
                redirect_uri="http://127.0.0.1/callback",
                owner="alice",
            )
            auth_url = hub.oauth_authorization_url(account["id"], owner="alice", state="state-1", code_challenge="challenge")
            self.assertIn("code_challenge=challenge", auth_url["authorization_url"])
            self.assertIn("access_type=offline", auth_url["authorization_url"])

            class FakeTokenResponse:
                status = 200

                def __init__(self) -> None:
                    self.body = json.dumps({
                        "access_token": "access-token",
                        "refresh_token": "refresh-token",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    }).encode("utf-8")

            captured_token_requests: list[bytes] = []

            def fake_fetch(url, **kwargs):
                captured_token_requests.append(kwargs["body"])
                return FakeTokenResponse()

            with patch.object(hub.egress, "fetch", fake_fetch):
                refreshed = hub.refresh_oauth_token(
                    account["id"],
                    owner="alice",
                    allow_external=True,
                    authorization_code="oauth-code",
                    code_verifier="verifier",
                )
            self.assertEqual(refreshed["status"], "token_ready")
            self.assertTrue(refreshed["account"]["token_present"])
            self.assertIn(b"authorization_code", captured_token_requests[0])

            hub.configure_connector(
                "email",
                enabled=True,
                config={
                    "provider": "google",
                    "imap_host": "mail.local",
                    "imap_username": "alice@example.com",
                    "imap_mailbox": "archive",
                    "oauth_account_id": account["id"],
                    "imap_ssl": True,
                },
            )
            message = EmailMessage()
            message["From"] = "sender@example.com"
            message["To"] = "alice@example.com"
            message["Subject"] = "OAuth inbox"
            message["Message-ID"] = "<oauth-message@example.com>"
            message.set_content("Hello through XOAUTH2.")

            class FakeImapOauth:
                last_mailbox = ""
                authenticated = False

                def __init__(self, host, port, timeout=20):
                    self.host = host
                    self.port = port
                    self.timeout = timeout

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def authenticate(self, mechanism, callback):
                    self.__class__.authenticated = mechanism == "XOAUTH2" and b"Bearer access-token" in callback(b"")
                    return "OK", []

                def select(self, mailbox):
                    self.__class__.last_mailbox = mailbox
                    return "OK", []

                def search(self, charset, query):
                    return "OK", [b"1"]

                def fetch(self, message_id, spec):
                    return "OK", [(b"1 (RFC822)", message.as_bytes())]

            with patch("openagent.communication.imaplib.IMAP4_SSL", FakeImapOauth):
                run = hub.sync_connector("email", project="core", owner="alice", allow_external=True)

            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["imported_messages"], 1)
            self.assertTrue(FakeImapOauth.authenticated)
            self.assertEqual(FakeImapOauth.last_mailbox, "[Gmail]/All Mail")


class ResilientSyncHttpRouteTests(unittest.TestCase):
    def test_sync_resilient_route_and_dead_letters_route(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = Request(
                    f"{base_url}/api/communication/sync-resilient",
                    data=json.dumps({
                        "platform": "unknown-platform",
                        "max_attempts": 1,
                        "payload": {},
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                result = json.loads(urlopen(request, timeout=15).read().decode("utf-8"))
                self.assertEqual(result["status"], "dead_lettered")

                dead_letters = json.loads(urlopen(f"{base_url}/api/communication/dead-letters", timeout=15).read().decode("utf-8"))
                self.assertEqual(len(dead_letters["dead_letters"]), 1)
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()

    def test_communication_routes_hide_cross_user_messages_when_auth_is_enforced(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            auth = AuthManager(store)
            alice = auth.register("alice", "alice-comm@example.com", "strong-password-1")
            auth.register("bob", "bob-comm@example.com", "strong-password-2")
            alice_token = auth.login("alice", "strong-password-1")["access_token"]
            bob_token = auth.login("bob", "strong-password-2")["access_token"]

            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
                    import_request = Request(
                        f"{base_url}/api/communication/messages/import",
                        data=json.dumps(
                            {
                                "platform": "email",
                                "project": "core",
                                "owner": "spoofed-owner-is-ignored",
                                "sender": "alice@example.com",
                                "subject": "Private",
                                "body": "Only Alice should see this.",
                            }
                        ).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {alice_token}",
                        },
                        method="POST",
                    )
                    message = json.loads(urlopen(import_request, timeout=15).read().decode("utf-8"))
                    self.assertEqual(message["message"]["owner"], alice["user_id"])

                    bob_inbox = Request(
                        f"{base_url}/api/communication/inbox?project=core",
                        headers={"Authorization": f"Bearer {bob_token}"},
                    )
                    inbox = json.loads(urlopen(bob_inbox, timeout=15).read().decode("utf-8"))
                    self.assertEqual(inbox["messages"], [])

                    bob_reply = Request(
                        f"{base_url}/api/communication/drafts/reply",
                        data=json.dumps({"project": "core", "message_id": message["message"]["id"]}).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {bob_token}",
                        },
                        method="POST",
                    )
                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(bob_reply, timeout=15)
                    self.assertEqual(ctx.exception.code, 400)
                    ctx.exception.close()
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
