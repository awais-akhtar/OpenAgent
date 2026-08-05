from __future__ import annotations

from tempfile import TemporaryDirectory
import json
import unittest

from openagent.communication import CommunicationApprovalRequired, CommunicationHub
from openagent.storage import OpenAgentStore


class CommunicationHubTests(unittest.TestCase):
    def test_initialize_upgrades_legacy_ownerless_tables_before_indexes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            with store.session() as connection:
                connection.executescript(
                    """
                    CREATE TABLE contacts (
                        id TEXT PRIMARY KEY,
                        project TEXT NOT NULL,
                        name TEXT NOT NULL
                    );
                    CREATE TABLE inbox_messages (
                        id TEXT PRIMARY KEY,
                        project TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )

            CommunicationHub(store)

            with store.session() as connection:
                contact_columns = {row["name"] for row in connection.execute("PRAGMA table_info(contacts)")}
                inbox_columns = {row["name"] for row in connection.execute("PRAGMA table_info(inbox_messages)")}
                indexes = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%owner%'"
                    )
                }
            self.assertIn("owner", contact_columns)
            self.assertIn("owner", inbox_columns)
            self.assertIn("idx_contacts_owner_project_name", indexes)
            self.assertIn("idx_inbox_owner_project_created", indexes)

    def test_inbox_contacts_calendar_draft_approval_and_analytics(self) -> None:
        with TemporaryDirectory() as tmp:
            hub = CommunicationHub(OpenAgentStore(tmp))

            connectors = hub.connectors()
            self.assertIn("email", {connector["id"] for connector in connectors})
            self.assertIn("caldav", {connector["id"] for connector in connectors})
            self.assertIn("slack", {connector["id"] for connector in connectors})

            message = hub.import_message(
                platform="email",
                sender="awais@example.com",
                recipients=["me@example.com"],
                subject="OpenAgent follow up",
                body="Can you please follow up by Friday? This is urgent.",
                project="core",
            )

            self.assertTrue(message["follow_up"])
            self.assertEqual(message["priority"], "high")
            self.assertEqual(len(hub.contacts(project="core")), 1)
            self.assertEqual(len(hub.inbox(project="core", follow_up=True)), 1)

            draft = hub.draft_reply(
                message["id"],
                project="core",
                instructions="I will send the update today.",
            )
            with self.assertRaises(CommunicationApprovalRequired) as raised:
                hub.send_draft(draft["id"])
            approval = json.loads(str(raised.exception))
            resolved = hub.resolve_send_approval(approval["id"], approved=True)
            self.assertEqual(resolved["status"], "approved")

            sent = hub.send_draft(draft["id"], approval_id=approval["id"])
            self.assertEqual(sent["status"], "simulated_local")

            event = hub.add_calendar_event(
                title="Planning",
                starts_at="2026-06-25T10:00:00+00:00",
                ends_at="2026-06-25T10:30:00+00:00",
                project="core",
                attendees=["awais@example.com"],
                source="caldav",
            )
            self.assertEqual(event["source"], "caldav")
            self.assertEqual(len(hub.calendar(project="core")), 1)

            analytics = hub.analytics(project="core")
            self.assertEqual(analytics["messages"], 1)
            self.assertEqual(analytics["follow_ups"], 1)

    def test_connector_sync_worker_imports_provider_payload(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            hub = CommunicationHub(store)

            run = hub.sync_connector(
                "email",
                project="core",
                payload={
                    "contacts": [
                        {
                            "name": "OpenAgent Lead",
                            "handles": {"email": "lead@example.com"},
                        }
                    ],
                    "messages": [
                        {
                            "sender": "lead@example.com",
                            "recipients": ["me@example.com"],
                            "subject": "Provider sync",
                            "body": "Please follow up after the roadmap review.",
                        }
                    ],
                    "calendar_events": [
                        {
                            "title": "Roadmap review",
                            "starts_at": "2026-06-25T10:00:00+00:00",
                            "ends_at": "2026-06-25T10:30:00+00:00",
                            "attendees": ["lead@example.com"],
                        }
                    ],
                },
            )

            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["imported_contacts"], 1)
            self.assertEqual(run["imported_messages"], 1)
            self.assertEqual(run["imported_calendar_events"], 1)
            self.assertEqual(len(hub.inbox(project="core", follow_up=True)), 1)
            self.assertEqual(len(hub.calendar(project="core")), 1)
            self.assertEqual(hub.sync_runs(project="core")[0]["id"], run["id"])

            audit_actions = {event["action"] for event in store.list_audit_events(project="core", limit=20)}
            self.assertIn("connector.sync", audit_actions)


if __name__ == "__main__":
    unittest.main()
