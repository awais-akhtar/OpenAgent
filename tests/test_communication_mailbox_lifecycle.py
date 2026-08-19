from __future__ import annotations

from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory
import unittest

from openagent.communication import CommunicationHub
from openagent.routes.communication_routes import ROUTES
from openagent.routes.communication_routes import CommunicationRouteDependencies, handle_post
from openagent.storage import OpenAgentStore


class CommunicationMailboxLifecycleTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
