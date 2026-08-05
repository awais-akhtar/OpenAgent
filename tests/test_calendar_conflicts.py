from __future__ import annotations

import unittest

from openagent.routes import calendar_routes


class CalendarConflictRouteContractTests(unittest.TestCase):
    def test_calendar_conflict_routes_are_declared(self) -> None:
        contract_routes = {(route["method"], route["path"]) for route in calendar_routes.contract()["routes"]}

        self.assertIn(("GET", "/api/calendar/conflicts"), contract_routes)
        self.assertIn(("POST", "/api/calendar/conflicts/{conflict_id}/resolve"), contract_routes)


if __name__ == "__main__":
    unittest.main()
