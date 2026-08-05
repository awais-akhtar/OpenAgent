from __future__ import annotations

import unittest

from openagent.routes import calendar_routes


class CalendarSyncRouteContractTests(unittest.TestCase):
    def test_calendar_sync_routes_are_declared(self) -> None:
        contract_routes = {(route["method"], route["path"]) for route in calendar_routes.contract()["routes"]}

        self.assertIn(("POST", "/api/calendar/providers/register"), contract_routes)
        self.assertIn(("POST", "/api/calendar/sync"), contract_routes)
        self.assertIn(("GET", "/api/calendar/sync/{sync_job_id}"), contract_routes)


if __name__ == "__main__":
    unittest.main()
