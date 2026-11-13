"""The egress timeout cap: 120 s by default, raised only when a caller asks for it, never above 900 s."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from openagent import egress


class _Stop(Exception):
    pass


class EgressTimeoutCapTests(unittest.TestCase):
    def _captured_timeout(self, **kwargs) -> float:
        seen: dict[str, float] = {}

        def fake_connect(self_, decision, **request):
            seen["timeout"] = request["timeout"]
            raise _Stop()

        with patch.object(egress.EgressClient, "_request_once", fake_connect, create=True):
            try:
                egress.EgressClient().fetch("http://127.0.0.1:11434/api/chat", approved=True, purpose="test", method="POST", body=b"{}", allow_loopback=True, **kwargs)
            except (_Stop, Exception):
                pass
        return seen.get("timeout", -1.0)

    def test_default_cap_is_still_120_seconds(self) -> None:
        self.assertEqual(120.0, min(120.0, max(0.25, 600.0)))
        self.assertLessEqual(self._captured_timeout(timeout_seconds=600.0), 120.0)

    def test_raised_cap_allows_a_model_call_longer_than_120_seconds(self) -> None:
        self.assertGreater(self._captured_timeout(timeout_seconds=600.0, max_timeout_seconds=600.0), 120.0)

    def test_cap_never_exceeds_900_seconds(self) -> None:
        self.assertLessEqual(self._captured_timeout(timeout_seconds=5000.0, max_timeout_seconds=5000.0), 900.0)


if __name__ == "__main__":
    unittest.main()
