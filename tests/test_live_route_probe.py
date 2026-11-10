"""Every registered read route must answer on a clean install.

The route inventory is generated from declarations: a path is listed because a
module declares it, not because anything behind it works. The React-to-Python
smoke proves 51 workflows end to end, which is the strongest evidence here, but
it covers a small share of the registry. This closes the gap between the two for
the read surface by calling every registered GET against a real server over a
temporary store, with one credential holding every scope.

A clean install holds no records, so an empty collection is the correct answer
for most routes and counts as a pass. What fails this test is a 500, a route
that answers 404 for its own bare path because nothing dispatches it, a handler
that raises, or a success carrying an error in its body.

The probe itself is scripts/probe_live_routes.py, which prints a per-route
report and can write the full record set as JSON.
"""

from __future__ import annotations

import unittest

from scripts.probe_live_routes import main


class LiveRouteProbeTests(unittest.TestCase):
    def test_no_registered_read_route_fails_on_a_clean_install(self) -> None:
        # main() returns 1 when any route failed outright or answered 404 on its
        # own path, and prints which ones. Run it with no --json so the report
        # goes to the captured output of this test.
        self.assertEqual(main([]), 0, "scripts/probe_live_routes.py reported failing routes; see captured output")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
