from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.backup_restore import BackupRestoreManager
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class BackupRestoreTests(unittest.TestCase):
    def test_create_backup_and_restore_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            store.create_conversation("Backup me", project="core")
            manager = BackupRestoreManager(store)

            backup = manager.create_backup(project="core")
            self.assertTrue(backup["path"].endswith(".zip"))
            self.assertGreater(backup["size_bytes"], 0)

            dry_run = manager.restore_dry_run(backup["path"])
            self.assertEqual(dry_run["status"], "ready")
            self.assertFalse(dry_run["will_write"])
            self.assertIn("current_database_would_be_replaced", dry_run["conflicts"])
            self.assertEqual(len(manager.backups()), 1)


class BackupRouteTests(unittest.TestCase):
    def test_backup_routes_create_list_and_dry_run(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                create = Request(
                    f"{base_url}/api/backups/create",
                    data=json.dumps({"project": "core"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                created = json.loads(urlopen(create, timeout=5).read().decode("utf-8"))
                path = created["backup"]["path"]

                listed = json.loads(urlopen(f"{base_url}/api/backups", timeout=5).read().decode("utf-8"))
                self.assertEqual(len(listed["backups"]), 1)

                dry = Request(
                    f"{base_url}/api/backups/restore-dry-run",
                    data=json.dumps({"path": path}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                dry_run = json.loads(urlopen(dry, timeout=5).read().decode("utf-8"))
                self.assertEqual(dry_run["status"], "ready")
                self.assertFalse(dry_run["will_write"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
