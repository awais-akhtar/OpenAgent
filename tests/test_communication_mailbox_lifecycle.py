from __future__ import annotations

from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory
import unittest

from openagent.communication import CommunicationHub
from openagent.communication import CommunicationMessageNotFoundError
from openagent.routes.communication_routes import ROUTES
from openagent.routes.communication_routes import CommunicationRouteDependencies, handle_get, handle_post
from openagent.storage import OpenAgentStore


class CommunicationMailboxLifecycleTests(unittest.TestCase):
    def test_draft_and_sent_lists_are_project_and_owner_scoped(self) -> None:
        with TemporaryDirectory() as temp:
            hub = CommunicationHub(OpenAgentStore(temp))
            own = hub.create_draft(platform="email", recipients=["a@example.test"], subject="Own", body="Private", project="core", owner="alice")
            hub.create_draft(platform="email", recipients=["b@example.test"], subject="Other", body="Other user's content", project="core", owner="bob")
            approval = hub.request_send_approval(own["id"], project="core", owner="alice")
            hub.resolve_send_approval(approval["id"], approved=True, owner="alice")
            hub.send_draft(own["id"], approval_id=approval["id"], owner="alice")

            self.assertEqual([draft["id"] for draft in hub.list_drafts(project="core", owner="alice")], [own["id"]])
            sent = hub.list_sent_messages(project="core", owner="alice")
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0]["draft_id"], own["id"])
            self.assertEqual(sent[0]["status"], "simulated_local")
            self.assertEqual(hub.list_sent_messages(project="other", owner="alice"), [])

    def test_routes_cover_sent_and_draft_reads(self) -> None:
        paths = {path for _, path in ROUTES}
        self.assertIn("/api/communication/sent", paths)
        self.assertIn("/api/communication/drafts", paths)

    def test_sent_and_draft_read_routes_forward_project_owner_and_limit(self) -> None:
        class FakeCommunication:
            def list_drafts(self, *, project, owner, limit):
                return [{"project": project, "owner": owner, "limit": limit}]

            def list_sent_messages(self, *, project, owner, limit):
                return [{"project": project, "owner": owner, "limit": limit}]

        deps = CommunicationRouteDependencies(
            communication=FakeCommunication(), first=lambda params, key: (params.get(key) or [None])[0],
            parse_bool=lambda value, default=False: default, parse_int=lambda value, default=0: int(value) if value else default,
            query_owner=lambda params: (params.get("owner") or [""])[0], payload_owner=lambda payload, default="local": default,
            read_json=lambda: {},
        )
        drafts = handle_get("/api/communication/drafts", "project=core&owner=alice&limit=7", deps)
        sent = handle_get("/api/communication/sent", "project=core&owner=alice&limit=9", deps)
        self.assertEqual(drafts.payload["drafts"], [{"project": "core", "owner": "alice", "limit": 7}])
        self.assertEqual(sent.payload["sent"], [{"project": "core", "owner": "alice", "limit": 9}])

    def test_mailbox_mutations_approvals_and_scheduled_send(self) -> None:
        with TemporaryDirectory() as temp:
            hub = CommunicationHub(OpenAgentStore(temp))
            message = hub.import_message(platform="email", sender="a@example.test", body="Please reply", project="core")
            archived = hub.mutate_message(message["id"], read=True, archived=True)
            self.assertTrue(archived["read"])
            self.assertEqual(archived["folder"], "archive")
            draft = hub.create_draft(platform="email", recipients=["a@example.test"], subject="Re", body="Done", project="core")
            approval = hub.request_send_approval(draft["id"], project="core")
            self.assertEqual(len(hub.pending_approvals(project="core")), 1)
            hub.resolve_send_approval(approval["id"], approved=True)
            scheduled = hub.schedule_draft(
                draft["id"], approval_id=approval["id"], project="core",
                send_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
            )
            result = hub.process_scheduled_sends(project="core")
            self.assertEqual(result["count"], 1)
            self.assertEqual(hub.get_scheduled_send(scheduled["id"])["status"], "sent")

    def test_live_inbox_message_can_be_moved_to_trash_without_deleting_record(self) -> None:
        with TemporaryDirectory() as temp:
            hub = CommunicationHub(OpenAgentStore(temp))
            message = hub.import_message(
                platform="email", sender="owner@example.test", subject="Retention notice",
                body="Synthetic local message", project="default", owner="local-owner",
            )
            trashed = hub.mutate_message(message["id"], owner="local-owner", folder="trash")
            self.assertEqual(trashed["folder"], "trash")
            self.assertEqual(hub.get_message(message["id"], owner="local-owner")["folder"], "trash")
            self.assertIsNone(hub.get_message(message["id"], owner="another-owner"))
            other_project_message = hub.import_message(
                platform="email", sender="owner@example.test", subject="Other project",
                body="Must remain in inbox", project="other", owner="local-owner",
            )
            with self.assertRaises(CommunicationMessageNotFoundError):
                hub.mutate_message(other_project_message["id"], project="default", owner="local-owner", folder="trash")
            self.assertEqual(hub.get_message(other_project_message["id"], owner="local-owner")["folder"], "inbox")

    def test_live_message_edit_is_project_and_owner_scoped_and_refreshes_context(self) -> None:
        with TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            hub = CommunicationHub(store)
            message = hub.import_message(
                platform="email", sender="old@example.test", subject="Obsolete subject", body="Comet violet signal",
                project="core", owner="alice",
            )
            hub.import_message(
                platform="email", sender="old@example.test", subject="Obsolete subject", body="Different owner record",
                project="core", owner="bob",
            )
            updated = hub.update_message(
                message["id"], project="core", owner="alice", subject="Fresh subject",
                sender="new@example.test", body="Amber copper orbit",
            )
            self.assertEqual((updated["subject"], updated["sender"], updated["body"]),
                             ("Fresh subject", "new@example.test", "Amber copper orbit"))
            self.assertEqual(store.search_events("Comet violet signal", project="core"), [])
            self.assertTrue(store.search_events("Amber copper orbit", project="core"))
            with self.assertRaises(CommunicationMessageNotFoundError):
                hub.update_message(
                    message["id"], project="core", owner="bob", subject="Forged",
                    sender="bob@example.test", body="Must not change",
                )
            with self.assertRaises(CommunicationMessageNotFoundError):
                hub.update_message(
                    message["id"], project="other", owner="alice", subject="Forged",
                    sender="alice@example.test", body="Must not change",
                )
            self.assertEqual(hub.get_message(message["id"], owner="alice")["subject"], "Fresh subject")

    def test_routes_cover_mailbox_lifecycle(self) -> None:
        paths = {path for _, path in ROUTES}
        self.assertIn("/api/communication/messages/state", paths)
        self.assertIn("/api/communication/scheduled/process", paths)

    def test_route_adapter_passes_project_and_owner_to_scheduled_mail(self) -> None:
        payload = {
            "project": "core",
            "draft_id": "draft-1",
            "approval_id": "approval-1",
            "send_at": "2030-01-01T00:00:00+00:00",
        }

        class FakeCommunication:
            def schedule_draft(self, draft_id, *, send_at, approval_id, project, owner):
                return {"draft_id": draft_id, "send_at": send_at, "approval_id": approval_id, "project": project, "owner": owner}

            def process_scheduled_sends(self, *, project, owner, allow_external):
                return {"project": project, "owner": owner, "allow_external": allow_external}

        deps = CommunicationRouteDependencies(
            communication=FakeCommunication(),
            first=lambda params, key: (params.get(key) or [None])[0],
            parse_bool=lambda value, default=False: bool(value) if value is not None else default,
            parse_int=lambda value, default=0: default,
            query_owner=lambda params: "",
            payload_owner=lambda body, default="local": "local-user" if default else "local-user",
            read_json=lambda: payload,
        )

        scheduled = handle_post("/api/communication/scheduled", deps)
        processed = handle_post("/api/communication/scheduled/process", deps)

        self.assertEqual(scheduled.payload["scheduled"]["project"], "core")
        self.assertEqual(scheduled.payload["scheduled"]["owner"], "local-user")
        self.assertEqual(processed.payload["project"], "core")

    def test_route_adapter_passes_project_and_owner_to_mailbox_mutation(self) -> None:
        payload = {"project": "core", "message_id": "message-1", "folder": "trash"}

        class FakeCommunication:
            def mutate_message(self, message_id, *, project, owner, folder, read, answered, spam, archived):
                return {"id": message_id, "project": project, "owner": owner, "folder": folder}

        deps = CommunicationRouteDependencies(
            communication=FakeCommunication(),
            first=lambda params, key: (params.get(key) or [None])[0],
            parse_bool=lambda value, default=False: default,
            parse_int=lambda value, default=0: default,
            query_owner=lambda params: "",
            payload_owner=lambda body, default="local": "owner-1",
            read_json=lambda: payload,
        )
        result = handle_post("/api/communication/messages/state", deps)
        self.assertEqual(result.payload["message"], {"id": "message-1", "project": "core", "owner": "owner-1", "folder": "trash"})


if __name__ == "__main__":
    unittest.main()
