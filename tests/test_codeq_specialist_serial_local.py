"""Specialist consults run one at a time for local providers and in parallel otherwise."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from openagent.codebase.service import CodebaseService


class SpecialistConcurrencyTests(unittest.TestCase):
    def _max_in_flight(self, provider: str) -> int:
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()

        def fake_model_json(run, messages, role=None):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            threading.Event().wait(0.05)
            with lock:
                state["active"] -= 1
            return {"summary": role or "", "acceptance_criteria": []}

        service = CodebaseService.__new__(CodebaseService)
        service._event = lambda *args, **kwargs: None
        service._model_json = fake_model_json
        service._compact_map = CodebaseService._compact_map
        run = {"id": "r1", "user_prompt": "Fix the bug", "provider": provider}
        project = {"memory": {}}
        project_map = {"database": "postgres", "framework": "fastapi", "file_count": 3}
        with patch("openagent.codebase.service.classify_request", return_value={"requires_ui": True, "architecture": "full-stack", "kind": "service"}):
            service._consult_specialists(run, project, project_map, {})
        return state["peak"]

    def test_local_provider_consults_one_specialist_at_a_time(self) -> None:
        self.assertEqual(1, self._max_in_flight("ollama"))

    def test_remote_provider_keeps_parallel_consults(self) -> None:
        self.assertGreater(self._max_in_flight("openai"), 1)


if __name__ == "__main__":
    unittest.main()
