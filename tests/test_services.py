from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.auth import AuthManager
from openagent.calendar_sync import CalendarSyncManager
from openagent.calendar_system import CalendarSystem
from openagent.communication import CommunicationHub
from openagent.documents import DocumentManager
from openagent.local_ai import LocalAIInfrastructure
from openagent.services.auth import (
    AuthIdentityService,
    AuthScopeService,
    AuthSessionService,
    AuthTokenService,
)
from openagent.services.calendar_sync import CalendarSyncService
from openagent.services.communication import CommunicationProviderService
from openagent.services.documents import DocumentLifecycleService
from openagent.services.model_ops import ModelOpsLifecycleService
from openagent.storage import OpenAgentStore


class ServiceLayerTests(unittest.TestCase):
    def test_auth_services_cover_identity_sessions_scopes_and_tokens(self) -> None:
        with TemporaryDirectory() as tmp:
            auth = AuthManager(OpenAgentStore(tmp))
            identity = AuthIdentityService(auth)
            sessions = AuthSessionService(auth)
            scopes = AuthScopeService(auth)
            tokens = AuthTokenService(auth)

            user = identity.register(
                username="alice",
                email="alice@example.test",
                password="strong-password-1",
                display_name="Alice",
            )
            login = identity.login(identifier="alice", password="strong-password-1")
            self.assertEqual(identity.me(login["access_token"])["user_id"], user["user_id"])
            self.assertTrue(scopes.is_allowed(login["scopes"], "/api/calendar/events", "GET"))

            api_token = tokens.create(
                user_id=user["user_id"],
                name="calendar-reader",
                scopes=["calendar:read"],
            )
            self.assertEqual(tokens.resolve(api_token["token"])["user_id"], user["user_id"])
            self.assertEqual(tokens.list(user["user_id"])[0]["token_id"], api_token["token_id"])
            self.assertEqual(tokens.revoke(user["user_id"], api_token["token_id"])["revoked"], True)

            session_id = sessions.list_sessions(user["user_id"])[0]["session_id"]
            self.assertEqual(sessions.revoke(user["user_id"], session_id)["revoked"], True)

    def test_document_service_covers_versions_restore_and_drafts(self) -> None:
        with TemporaryDirectory() as tmp:
            documents = DocumentLifecycleService(DocumentManager(OpenAgentStore(tmp)))
            doc = documents.create_document(title="Plan", content="v1")
            updated = documents.update_document(doc["doc_id"], content="v2", if_version=1)
            self.assertEqual(updated["version"], 2)
            self.assertEqual([v["version"] for v in documents.versions(doc["doc_id"])], [2, 1])
            self.assertEqual(documents.restore_version(doc["doc_id"], 1)["new_version"], 3)

            patched = documents.patch_document(doc["doc_id"], tags=["ops"], folder="team")
            self.assertEqual(patched["folder"], "team")
            self.assertEqual(documents.library(tag="ops")["count"], 1)

            draft = documents.create_draft(doc_id=doc["doc_id"], content="typing", base_version=3)
            fetched = documents.get_draft(draft["draft_id"])
            self.assertEqual(fetched["content"], "typing")
            self.assertEqual(documents.update_draft(draft["draft_id"], content="done")["content"], "done")

    def test_communication_calendar_and_model_services_are_operational(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)

            communication = CommunicationProviderService(CommunicationHub(store))
            providers = communication.oauth_providers()
            self.assertTrue(any(provider["id"] == "generic" for provider in providers))
            account = communication.configure_oauth_account(
                provider="google",
                label="Local Google",
                client_id="client-id",
                redirect_uri="http://127.0.0.1/callback",
            )
            self.assertFalse(account["token_present"])
            auth_url = communication.authorization_url(account["account_id"], state="state-1")
            self.assertIn("state-1", auth_url["authorization_url"])
            self.assertIn("inbox", communication.mailbox_policy(account_id=account["account_id"])["mailboxes"])
            sync = communication.sync_resilient(platform="email", payload={"messages": []})
            self.assertIn(sync["status"], {"ok", "completed", "failed", "dead_lettered"})

            calendar = CalendarSystem(store)
            calendar_sync = CalendarSyncService(CalendarSyncManager(store, calendar))
            provider = calendar_sync.register_provider(provider="manual", config={"ics_seed": ""})
            self.assertEqual(calendar_sync.providers()[0]["provider_id"], provider["provider_id"])
            job = calendar_sync.run_sync(provider["provider_id"])
            self.assertEqual(calendar_sync.sync_job(job["sync_job_id"])["state"], "completed")
            self.assertEqual(calendar_sync.conflicts(), [])

            local_ai = LocalAIInfrastructure(store)
            model = local_ai.register_model(name="Service Model", provider="local")
            lifecycle = ModelOpsLifecycleService(local_ai)
            self.assertEqual(lifecycle.transition(model["id"], "discovered")["state"], "discovered")
            self.assertEqual(lifecycle.diagnostics(model["id"])[0]["state"], "discovered")


if __name__ == "__main__":
    unittest.main()
