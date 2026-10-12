from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.calendar_sync import CalendarSyncManager
from openagent.calendar_system import CalendarSystem
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class _CalendarHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = (
            "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\n"
            "UID:approved@example.com\nSUMMARY:Approved sync\n"
            "DTSTART:20260915T090000Z\nDTEND:20260915T093000Z\n"
            "END:VEVENT\nEND:VCALENDAR"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/calendar")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class CalendarSyncEgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _CalendarHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _manager_and_provider(self, tmp: str) -> tuple[CalendarSyncManager, dict[str, object]]:
        store = OpenAgentStore(tmp)
        manager = CalendarSyncManager(store, CalendarSystem(store))
        provider = manager.register_provider(
            "caldav_http",
            base_url=f"http://127.0.0.1:{self.server.server_address[1]}/calendar.ics",
        )
        return manager, provider

    def test_caldav_requires_run_approval(self) -> None:
        with TemporaryDirectory() as tmp:
            manager, provider = self._manager_and_provider(tmp)
            run = manager.run_sync(str(provider["provider_id"]))
            self.assertEqual(run["state"], "failed")
            self.assertIn("allow_external=true", run["error"])

    def test_loopback_requires_explicit_local_transport(self) -> None:
        with TemporaryDirectory() as tmp:
            manager, provider = self._manager_and_provider(tmp)
            run = manager.run_sync(str(provider["provider_id"]), allow_external=True)
            self.assertEqual(run["state"], "failed")
            self.assertIn("Loopback", run["error"])

    def test_approved_loopback_transport_can_pull(self) -> None:
        with TemporaryDirectory() as tmp:
            manager, provider = self._manager_and_provider(tmp)
            run = manager.run_sync(
                str(provider["provider_id"]),
                allow_external=True,
                allow_local_transport=True,
            )
            self.assertEqual(run["state"], "completed")
            self.assertEqual(run["pulled"], 1)

    def test_http_route_forwards_run_approval_and_local_transport(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            manager = CalendarSyncManager(store, CalendarSystem(store))
            provider = manager.register_provider(
                "caldav_http",
                base_url=f"http://127.0.0.1:{self.server.server_address[1]}/calendar.ics",
            )
            app = build_http_server("127.0.0.1", 0, store)
            app_thread = threading.Thread(target=app.serve_forever, daemon=True)
            app_thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{app.server_address[1]}/api/calendar/sync",
                    data=json.dumps(
                        {
                            "provider_id": provider["provider_id"],
                            "allow_external": True,
                            "allow_local_transport": True,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                result = json.loads(urlopen(request, timeout=15).read().decode("utf-8"))
                self.assertEqual(result["state"], "completed")
                self.assertEqual(result["pulled"], 1)
            finally:
                app.shutdown()
                app.server_close()
                app_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
