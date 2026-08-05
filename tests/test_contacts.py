from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.communication import CommunicationHub
from openagent.contacts import ContactsManager
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class ContactsManagerTests(unittest.TestCase):
    def test_contacts_crm_import_export_update_delete(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            communication = CommunicationHub(store)
            contacts = ContactsManager(store)

            created = contacts.create_contact(
                name="Awais",
                project="core",
                owner="local",
                handles={"email": "awais@example.test"},
                notes="Founder",
            )
            communication.import_message(
                platform="email",
                sender="awais@example.test",
                body="Follow up on the local AI roadmap.",
                project="core",
                owner="local",
            )

            found = contacts.search("awais", project="core", owner="local")
            self.assertEqual(found[0]["id"], created["id"])
            self.assertEqual(found[0]["message_count"], 1)

            updated = contacts.update_contact(created["id"], owner="local", notes="Founder and operator")
            self.assertEqual(updated["notes"], "Founder and operator")

            imported = contacts.import_contacts(
                [{"name": "OpenAgent Team", "handles": {"slack": "team-openagent"}}],
                project="core",
                owner="local",
            )
            self.assertEqual(imported["count"], 1)
            exported = contacts.export_contacts(project="core", owner="local")
            self.assertEqual(exported["count"], 2)

            deleted = contacts.delete_contact(created["id"], owner="local")
            self.assertTrue(deleted["deleted"])
            self.assertEqual(len(contacts.search("awais", project="core", owner="local")), 0)


class ContactsRouteTests(unittest.TestCase):
    def test_contacts_routes_cover_crud_and_export(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                create = Request(
                    f"{base_url}/api/contacts",
                    data=json.dumps(
                        {
                            "project": "core",
                            "name": "Local User",
                            "handles": {"email": "local@example.test"},
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                created = json.loads(urlopen(create, timeout=5).read().decode("utf-8"))
                contact_id = created["contact"]["id"]

                patch = Request(
                    f"{base_url}/api/contacts/{contact_id}",
                    data=json.dumps({"notes": "Updated"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="PATCH",
                )
                updated = json.loads(urlopen(patch, timeout=5).read().decode("utf-8"))
                self.assertEqual(updated["contact"]["notes"], "Updated")

                listed = json.loads(urlopen(f"{base_url}/api/contacts?project=core&q=local", timeout=5).read().decode("utf-8"))
                self.assertEqual(len(listed["contacts"]), 1)

                exported = json.loads(urlopen(f"{base_url}/api/contacts/export?project=core", timeout=5).read().decode("utf-8"))
                self.assertEqual(exported["count"], 1)

                delete = Request(f"{base_url}/api/contacts/{contact_id}", data=b"", method="DELETE")
                deleted = json.loads(urlopen(delete, timeout=5).read().decode("utf-8"))
                self.assertTrue(deleted["deleted"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
