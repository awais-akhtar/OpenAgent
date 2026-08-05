from __future__ import annotations

import json
import os
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager
from openagent.personal_os import PersonalOS, PersonalOSError
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class PersonalOSTests(unittest.TestCase):
    def test_personal_os_manager_persists_local_product_features(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            personal = PersonalOS(store)

            preset = personal.upsert_preset(
                name="Research local",
                project="core",
                kind="research",
                instructions="Use local knowledge first.",
                model="phi4-mini",
                parameters={"temperature": 0.2},
            )
            self.assertEqual(preset["name"], "Research local")
            self.assertEqual(preset["parameters"]["temperature"], 0.2)

            signature = personal.upsert_signature(
                label="Default",
                content="Regards,\nOpenAgent",
                project="core",
                is_default=True,
            )
            self.assertTrue(signature["is_default"])

            draft = personal.save_draft(title="Launch note", content="v1", project="core")
            draft = personal.save_draft(draft_id=draft["id"], title="Launch note", content="v2", project="core")
            self.assertEqual(draft["version"], 2)
            self.assertEqual([item["version"] for item in personal.draft_versions(draft["id"])], [2, 1])

            note = personal.save_note(
                title="Follow up",
                body="Prepare release checklist",
                tags=["Release", "today"],
                due_at="2026-07-09",
                project="core",
            )
            self.assertEqual(note["tags"], ["release", "today"])
            self.assertEqual(personal.notes(project="core", query="checklist")[0]["id"], note["id"])

            asset = personal.add_gallery_asset(
                title="Architecture diagram",
                content="diagram bytes",
                tags=["design"],
                metadata={"kind": "diagram"},
                project="core",
            )
            asset = personal.set_gallery_favorite(asset_id=asset["id"], favorite=True, project="core")
            self.assertTrue(asset["favorite"])
            self.assertEqual(personal.gallery_stats(project="core")["favorites"], 1)

            compare = personal.start_compare(
                prompt="Pick best answer",
                left_output="Short",
                right_output="Detailed",
                project="core",
            )
            compare = personal.vote_compare(compare_id=compare["id"], vote="B", project="core")
            self.assertEqual(compare["winner"], "B")

            bundle = personal.export_bundle(project="core")
            self.assertEqual(len(bundle["drafts"][0]["versions"]), 2)
            imported = personal.import_bundle(bundle, project="imported")
            self.assertGreaterEqual(imported["imported"]["presets"], 1)
            self.assertGreaterEqual(imported["imported"]["compares"], 1)
            imported_draft = personal.drafts(project="imported")[0]
            self.assertEqual(len(personal.draft_versions(imported_draft["id"])), 2)
            self.assertEqual(personal.compare_history(project="imported")[0]["winner"], "B")
            self.assertEqual(personal.status(project="core")["counts"]["compares"], 1)
            self.assertGreaterEqual(len(store.list_audit_events(project="core")), 8)

            with self.assertRaises(PersonalOSError):
                personal.save_note(title="", project="core")

    def test_personal_os_routes_are_runnable(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                status = read_json(f"{base_url}/api/personal-os/status?project=core")
                self.assertEqual(status["status"], "ready")

                preset = post_json(
                    f"{base_url}/api/personal-os/presets",
                    {"project": "core", "name": "Chat preset", "instructions": "Stay local."},
                )
                self.assertEqual(preset["preset"]["name"], "Chat preset")

                signature = post_json(
                    f"{base_url}/api/personal-os/signatures",
                    {"project": "core", "label": "Default", "content": "Regards", "is_default": True},
                )
                self.assertTrue(signature["signature"]["is_default"])

                draft = post_json(
                    f"{base_url}/api/personal-os/drafts",
                    {"project": "core", "title": "Draft", "content": "First"},
                )
                draft = post_json(
                    f"{base_url}/api/personal-os/drafts",
                    {"project": "core", "draft_id": draft["draft"]["id"], "title": "Draft", "content": "Second"},
                )
                self.assertEqual(draft["draft"]["version"], 2)
                versions = read_json(f"{base_url}/api/personal-os/drafts/{draft['draft']['id']}/versions")
                self.assertEqual(len(versions["versions"]), 2)

                note = post_json(
                    f"{base_url}/api/personal-os/notes",
                    {"project": "core", "title": "Note", "body": "Find this", "tags": ["local"]},
                )
                self.assertEqual(note["note"]["tags"], ["local"])

                asset = post_json(
                    f"{base_url}/api/personal-os/gallery",
                    {"project": "core", "title": "Asset", "content": "local media"},
                )
                favorite = post_json(
                    f"{base_url}/api/personal-os/gallery/favorite",
                    {"project": "core", "asset_id": asset["asset"]["id"], "favorite": True},
                )
                self.assertTrue(favorite["asset"]["favorite"])

                compare = post_json(
                    f"{base_url}/api/personal-os/compare/start",
                    {"project": "core", "prompt": "Choose", "left_output": "A", "right_output": "B"},
                )
                vote = post_json(
                    f"{base_url}/api/personal-os/compare/vote",
                    {"project": "core", "compare_id": compare["compare"]["id"], "vote": "A"},
                )
                self.assertEqual(vote["compare"]["winner"], "A")

                bundle = read_json(f"{base_url}/api/personal-os/export?project=core")
                self.assertEqual(bundle["schema"], "openagent-personal-os-v1")
                self.assertEqual(len(bundle["drafts"][0]["versions"]), 2)
                imported = post_json(
                    f"{base_url}/api/personal-os/import",
                    {"project": "copy", "bundle": bundle},
                )
                self.assertGreaterEqual(imported["imported"]["drafts"], 1)
                self.assertGreaterEqual(imported["imported"]["compares"], 1)

                notes = read_json(f"{base_url}/api/personal-os/notes?project=core&q=find")
                self.assertEqual(len(notes["notes"]), 1)

                with self.assertRaises(HTTPError) as ctx:
                    post_json(f"{base_url}/api/personal-os/notes", {"project": "core", "title": ""})
                self.assertEqual(ctx.exception.code, 400)
                ctx.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_personal_os_routes_hide_cross_user_records_when_auth_is_enforced(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            auth = AuthManager(store)
            alice = auth.register("alice", "alice-personal@example.com", "strong-password-1")
            auth.register("bob", "bob-personal@example.com", "strong-password-2")
            alice_token = auth.login("alice", "strong-password-1")["access_token"]
            bob_token = auth.login("bob", "strong-password-2")["access_token"]

            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
                    note_request = Request(
                        f"{base_url}/api/personal-os/notes",
                        data=json.dumps(
                            {
                                "project": "core",
                                "owner": "spoofed-owner-is-ignored",
                                "title": "Private note",
                                "body": "Only Alice can list this.",
                            }
                        ).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {alice_token}",
                        },
                        method="POST",
                    )
                    note = json.loads(urlopen(note_request, timeout=15).read().decode("utf-8"))
                    self.assertEqual(note["note"]["owner"], alice["user_id"])

                    bob_notes = Request(
                        f"{base_url}/api/personal-os/notes?project=core",
                        headers={"Authorization": f"Bearer {bob_token}"},
                    )
                    notes = json.loads(urlopen(bob_notes, timeout=15).read().decode("utf-8"))
                    self.assertEqual(notes["notes"], [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


def read_json(url: str) -> dict[str, object]:
    response = urlopen(url, timeout=10)
    try:
        return json.loads(response.read().decode("utf-8"))
    finally:
        response.close()


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = urlopen(request, timeout=20)
    try:
        return json.loads(response.read().decode("utf-8"))
    finally:
        response.close()


if __name__ == "__main__":
    unittest.main()
