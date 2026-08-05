from __future__ import annotations

from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory
import unittest

from openagent.communication import CommunicationHub
from openagent.routes.communication_routes import ROUTES
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


if __name__ == "__main__":
    unittest.main()
