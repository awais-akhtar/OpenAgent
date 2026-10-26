from __future__ import annotations

import unittest

from openagent.routes import calendar_routes
from openagent.auth import required_scope_for_path


class CalendarSyncRouteContractTests(unittest.TestCase):
    def test_calendar_sync_routes_are_declared(self) -> None:
        contract_routes = {(route["method"], route["path"]) for route in calendar_routes.contract()["routes"]}

        self.assertIn(("POST", "/api/calendar/providers/register"), contract_routes)
        self.assertIn(("POST", "/api/calendar/sync"), contract_routes)
        self.assertIn(("GET", "/api/calendar/sync/{sync_job_id}"), contract_routes)

    def test_event_update_and_delete_are_declared_as_calendar_write_operations(self) -> None:
        contract_routes = {(route["method"], route["path"]) for route in calendar_routes.contract()["routes"]}
        self.assertIn(("PATCH", "/api/calendar/events/{event_id}"), contract_routes)
        self.assertIn(("DELETE", "/api/calendar/events/{event_id}"), contract_routes)
        self.assertEqual(required_scope_for_path("/api/calendar/events/event-1", "PATCH"), "calendar:write")
        self.assertEqual(required_scope_for_path("/api/calendar/events/event-1", "DELETE"), "calendar:write")

    def test_reminder_acknowledgement_uses_calendar_write_scope(self) -> None:
        self.assertEqual(required_scope_for_path("/api/calendar/reminders/rem-1/acknowledge", "POST"), "calendar:write")


if __name__ == "__main__":
    unittest.main()
