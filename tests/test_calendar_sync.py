from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager
from openagent.calendar_sync import CalendarSyncManager
from openagent.calendar_system import CalendarSystem
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


def _ics(uid: str, title: str, starts: str, ends: str, description: str = "", location: str = "") -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"SUMMARY:{title}",
        f"DTSTART:{starts}",
        f"DTEND:{ends}",
    ]
    if description:
        lines.append(f"DESCRIPTION:{description}")
    if location:
        lines.append(f"LOCATION:{location}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\n".join(lines)


class CalendarSyncManagerTests(unittest.TestCase):
    def test_pull_is_idempotent_and_updates_when_remote_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            calendar_system = CalendarSystem(store)
            sync = CalendarSyncManager(store, calendar_system)

            ics_v1 = _ics("evt-1@example.com", "Team Sync", "20260710T090000Z", "20260710T093000Z", "Weekly sync")
            provider = sync.register_provider("manual", config={"ics_seed": ics_v1})
            self.assertEqual(provider["provider"], "manual")

            first_run = sync.run_sync(provider["provider_id"])
            self.assertEqual(first_run["state"], "completed")
            self.assertEqual(first_run["pulled"], 1)
            self.assertEqual(first_run["conflicts"], 0)

            events = calendar_system.list_events(project="default")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["title"], "Team Sync")

            # Re-running with the same remote content is a no-op (idempotent).
            second_run = sync.run_sync(provider["provider_id"])
            self.assertEqual(second_run["pulled"], 0)
            self.assertEqual(second_run["conflicts"], 0)

            # Remote changes, local is untouched -> safe auto-apply.
            ics_v2 = _ics("evt-1@example.com", "Team Sync (moved)", "20260710T100000Z", "20260710T103000Z", "Weekly sync")
            with store.session() as connection:
                connection.execute(
                    "UPDATE calendar_sync_providers SET config_json = ? WHERE id = ?",
                    (json.dumps({"ics_seed": ics_v2}), provider["provider_id"]),
                )
            third_run = sync.run_sync(provider["provider_id"])
            self.assertEqual(third_run["pulled"], 1)
            self.assertEqual(third_run["conflicts"], 0)
            updated_events = calendar_system.list_events(project="default")
            self.assertEqual(updated_events[0]["title"], "Team Sync (moved)")

    def test_conflict_detected_when_both_sides_change_then_resolved(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            calendar_system = CalendarSystem(store)
            sync = CalendarSyncManager(store, calendar_system)

            ics_v1 = _ics("evt-2@example.com", "Planning", "20260711T090000Z", "20260711T100000Z")
            provider = sync.register_provider("manual", config={"ics_seed": ics_v1})
            sync.run_sync(provider["provider_id"])

            events = calendar_system.list_events(project="default")
            local_event_id = events[0]["id"]
            calendar_system.update_event(local_event_id, updates={"title": "Planning (local edit)"})

            ics_v2 = _ics("evt-2@example.com", "Planning (remote edit)", "20260711T090000Z", "20260711T100000Z")
            with store.session() as connection:
                connection.execute(
                    "UPDATE calendar_sync_providers SET config_json = ? WHERE id = ?",
                    (json.dumps({"ics_seed": ics_v2}), provider["provider_id"]),
                )

            run = sync.run_sync(provider["provider_id"], conflict_policy="manual")
            self.assertEqual(run["conflicts"], 1)
            self.assertEqual(run["pulled"], 0)

            conflicts = sync.list_conflicts(project="default")
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["status"], "open")

            resolved = sync.resolve_conflict(conflicts[0]["conflict_id"], "server_wins")
            self.assertEqual(resolved["status"], "resolved")
            final_event = calendar_system.get_event(local_event_id)
            self.assertEqual(final_event["title"], "Planning (remote edit)")

            # Re-resolving an already-resolved conflict is a safe no-op.
            again = sync.resolve_conflict(conflicts[0]["conflict_id"], "client_wins")
            self.assertEqual(again["status"], "resolved")

    def test_register_provider_validates_kind(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            calendar_system = CalendarSystem(store)
            sync = CalendarSyncManager(store, calendar_system)
            with self.assertRaises(ValueError):
                sync.register_provider("not-a-real-provider")

    def test_get_sync_job_missing_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            calendar_system = CalendarSystem(store)
            sync = CalendarSyncManager(store, calendar_system)
            with self.assertRaises(KeyError):
                sync.get_sync_job("does-not-exist")

    def test_caldav_push_uses_http_put_and_fails_on_remote_error(self) -> None:
        captured: list[dict[str, object]] = []

        class PutHandler(BaseHTTPRequestHandler):
            def do_PUT(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                captured.append({"path": self.path, "body": body.decode("utf-8"), "auth": self.headers.get("Authorization", "")})
                self.send_response(201)
                self.end_headers()

            def log_message(self, format, *args):  # noqa: A002
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), PutHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as tmp:
                store = OpenAgentStore(tmp)
                calendar_system = CalendarSystem(store)
                sync = CalendarSyncManager(store, calendar_system)
                calendar_system.create_event("Local Push", "20260712T100000Z", "20260712T103000Z")
                provider = sync.register_provider(
                    "caldav_http",
                    base_url=f"http://127.0.0.1:{server.server_address[1]}/calendar",
                    config={"username": "alice", "password": "secret"},
                )

                run = sync.run_sync(
                    provider["provider_id"],
                    direction="push",
                    allow_external=True,
                    allow_local_transport=True,
                )
                self.assertEqual(run["state"], "completed")
                self.assertEqual(run["pushed"], 1)
                self.assertEqual(len(captured), 1)
                self.assertTrue(captured[0]["path"].endswith(".ics"))
                self.assertIn("BEGIN:VCALENDAR", captured[0]["body"])
                self.assertTrue(str(captured[0]["auth"]).startswith("Basic "))
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

        class FailingPutHandler(BaseHTTPRequestHandler):
            def do_PUT(self):  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(500)
                self.end_headers()

            def log_message(self, format, *args):  # noqa: A002
                return

        failing = ThreadingHTTPServer(("127.0.0.1", 0), FailingPutHandler)
        failing_thread = threading.Thread(target=failing.serve_forever, daemon=True)
        failing_thread.start()
        try:
            with TemporaryDirectory() as tmp:
                store = OpenAgentStore(tmp)
                calendar_system = CalendarSystem(store)
                sync = CalendarSyncManager(store, calendar_system)
                calendar_system.create_event("Failed Push", "20260712T110000Z", "20260712T113000Z")
                provider = sync.register_provider(
                    "caldav_http",
                    base_url=f"http://127.0.0.1:{failing.server_address[1]}/calendar",
                )

                run = sync.run_sync(
                    provider["provider_id"],
                    direction="push",
                    allow_external=True,
                    allow_local_transport=True,
                )
                self.assertEqual(run["state"], "failed")
                self.assertEqual(run["pushed"], 0)
                self.assertIn("CalDAV PUT failed", run["error"])
        finally:
            failing.shutdown()
            failing_thread.join(timeout=2)
            failing.server_close()

    def test_caldav_report_sync_token_etag_put_and_remote_delete(self) -> None:
        state = {
            "mode": "event",
            "put_if_match": "",
            "put_body": "",
        }

        class ReportHandler(BaseHTTPRequestHandler):
            def do_REPORT(self):  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(207)
                self.send_header("Content-Type", "application/xml")
                self.end_headers()
                if state["mode"] == "deleted":
                    body = """<?xml version="1.0" encoding="utf-8"?>
<D:multistatus xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:sync-token>token-2</D:sync-token>
  <D:response>
    <D:href>/calendar/report-event.ics</D:href>
    <D:status>HTTP/1.1 404 Not Found</D:status>
  </D:response>
</D:multistatus>"""
                else:
                    ics = "\n".join([
                        "BEGIN:VCALENDAR",
                        "VERSION:2.0",
                        "BEGIN:VEVENT",
                        "UID:report-event@example.com",
                        "RECURRENCE-ID:20260714T100000Z",
                        "SUMMARY:Report Event",
                        "DTSTART:20260714T100000Z",
                        "DTEND:20260714T103000Z",
                        "END:VEVENT",
                        "END:VCALENDAR",
                    ])
                    body = f"""<?xml version="1.0" encoding="utf-8"?>
<D:multistatus xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:sync-token>token-1</D:sync-token>
  <D:response>
    <D:href>/calendar/report-event.ics</D:href>
    <D:propstat>
      <D:prop>
        <D:getetag>"etag-1"</D:getetag>
        <C:calendar-data><![CDATA[{ics}]]></C:calendar-data>
      </D:prop>
      <D:status>HTTP/1.1 200 OK</D:status>
    </D:propstat>
  </D:response>
</D:multistatus>"""
                self.wfile.write(body.encode("utf-8"))

            def do_PUT(self):  # noqa: N802
                state["put_if_match"] = self.headers.get("If-Match", "")
                state["put_body"] = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8")
                self.send_response(204)
                self.send_header("ETag", '"etag-2"')
                self.end_headers()

            def log_message(self, format, *args):  # noqa: A002
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), ReportHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as tmp:
                store = OpenAgentStore(tmp)
                calendar_system = CalendarSystem(store)
                sync = CalendarSyncManager(store, calendar_system)
                provider = sync.register_provider(
                    "caldav_http",
                    base_url=f"http://127.0.0.1:{server.server_address[1]}/calendar",
                    config={"use_report": True},
                )

                pull = sync.run_sync(
                    provider["provider_id"],
                    direction="pull",
                    allow_external=True,
                    allow_local_transport=True,
                )
                self.assertEqual(pull["state"], "completed")
                self.assertEqual(pull["pulled"], 1)
                event = calendar_system.list_events(project="default")[0]
                self.assertEqual(event["metadata"]["recurrence_id"], "2026-07-14T10:00:00+00:00")
                provider_after_pull = sync.get_provider(provider["provider_id"])
                self.assertEqual(provider_after_pull["sync_token"], "token-1")

                calendar_system.update_event(event["id"], updates={"title": "Report Event Local Edit"})
                push = sync.run_sync(
                    provider["provider_id"],
                    direction="push",
                    allow_external=True,
                    allow_local_transport=True,
                )
                self.assertEqual(push["state"], "completed")
                self.assertEqual(push["pushed"], 1)
                self.assertEqual(state["put_if_match"], '"etag-1"')
                self.assertIn("Report Event Local Edit", state["put_body"])

                state["mode"] = "deleted"
                delete_pull = sync.run_sync(
                    provider["provider_id"],
                    direction="pull",
                    allow_external=True,
                    allow_local_transport=True,
                )
                self.assertEqual(delete_pull["state"], "completed")
                self.assertEqual(delete_pull["pulled"], 1)
                self.assertEqual(calendar_system.list_events(project="default"), [])
                provider_after_delete = sync.get_provider(provider["provider_id"])
                self.assertEqual(provider_after_delete["sync_token"], "token-2")
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


class CalendarSyncHttpRouteTests(unittest.TestCase):
    def test_register_sync_and_fetch_job_over_http(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                ics = _ics("evt-http@example.com", "HTTP Synced Event", "20260712T090000Z", "20260712T093000Z")
                register_request = Request(
                    f"{base_url}/api/calendar/providers/register",
                    data=json.dumps({"provider": "manual", "config": {"ics_seed": ics}}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                provider = json.loads(urlopen(register_request, timeout=15).read().decode("utf-8"))
                provider_id = provider["provider_id"]

                sync_request = Request(
                    f"{base_url}/api/calendar/sync",
                    data=json.dumps({"provider_id": provider_id}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                sync_result = json.loads(urlopen(sync_request, timeout=15).read().decode("utf-8"))
                self.assertEqual(sync_result["state"], "completed")
                self.assertEqual(sync_result["pulled"], 1)

                job_url = f"{base_url}/api/calendar/sync/{sync_result['sync_job_id']}"
                job = json.loads(urlopen(job_url, timeout=15).read().decode("utf-8"))
                self.assertEqual(job["state"], "completed")

                providers = json.loads(urlopen(f"{base_url}/api/calendar/providers", timeout=15).read().decode("utf-8"))
                self.assertEqual(len(providers["providers"]), 1)
            finally:
                server.shutdown()
                thread.join(timeout=2)

    def test_calendar_provider_routes_hide_cross_user_records_when_auth_is_enforced(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            auth = AuthManager(store)
            alice = auth.register("alice", "alice-cal@example.com", "strong-password-1")
            auth.register("bob", "bob-cal@example.com", "strong-password-2")
            alice_token = auth.login("alice", "strong-password-1")["access_token"]
            bob_token = auth.login("bob", "strong-password-2")["access_token"]

            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with patch.dict(os.environ, {"OPENAGENT_AUTH_ENFORCED": "1"}):
                    register_request = Request(
                        f"{base_url}/api/calendar/providers/register",
                        data=json.dumps(
                            {
                                "provider": "manual",
                                "owner": "spoofed-owner-is-ignored",
                                "config": {"ics_seed": _ics("owned@example.com", "Owned", "20260712T090000Z", "20260712T093000Z")},
                            }
                        ).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {alice_token}",
                        },
                        method="POST",
                    )
                    provider = json.loads(urlopen(register_request, timeout=15).read().decode("utf-8"))
                    provider_id = provider["provider_id"]
                    self.assertEqual(provider["owner"], alice["user_id"])

                    bob_list = Request(
                        f"{base_url}/api/calendar/providers",
                        headers={"Authorization": f"Bearer {bob_token}"},
                    )
                    providers = json.loads(urlopen(bob_list, timeout=15).read().decode("utf-8"))
                    self.assertEqual(providers["providers"], [])

                    bob_sync = Request(
                        f"{base_url}/api/calendar/sync",
                        data=json.dumps({"provider_id": provider_id}).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {bob_token}",
                        },
                        method="POST",
                    )
                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(bob_sync, timeout=15)
                    self.assertEqual(ctx.exception.code, 404)
            finally:
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
