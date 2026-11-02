"""An unreachable model must stop a run at once, not after the full budget.

When the gateway cannot reach the requested provider it answers with its
offline stub, which returns prose rather than the JSON actions a run needs.
Every later iteration then fails for the same reason, so a run used to spend
24 iterations -- about 25 minutes -- before reporting `iteration_limit` and
blaming the model for output it never produced. Planning is the first point a
model has to answer, so that is where the run now stops.
"""

from __future__ import annotations

import json
import unittest
from tempfile import TemporaryDirectory
from uuid import uuid4

from openagent.codebase.service import CodebaseService
from openagent.models import utc_now
from openagent.storage import OpenAgentStore


class FallbackProviderTrackingTests(unittest.TestCase):
    """_fallback_providers is the evidence the gate acts on."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.store = OpenAgentStore(self._tmp.name)
        # A real instance, so the schema this reads from is the one the service
        # creates rather than one the test invented.
        self.service = CodebaseService(self.store, None)  # type: ignore[arg-type]

    def _run(self) -> str:
        """A run row for events to reference.

        codebase_runs points at a project and a workspace, which a test of
        event reading has no reason to build, so referential checks are off for
        this one insert. What is under test is which providers
        _fallback_providers reports for a run id, not the run's own integrity.
        """
        run_id = uuid4().hex
        with self.store.session() as connection:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute(
                "INSERT INTO codebase_runs (id, codebase_project_id, workspace_id, project,"
                " owner, user_prompt, status, started_at, updated_at)"
                " VALUES (?, ?, ?, 'default', 'local', 'gate test', 'running', ?, ?)",
                (run_id, uuid4().hex, uuid4().hex, utc_now(), utc_now()),
            )
        return run_id

    def _event(self, run_id: str, event_type: str, payload: dict) -> None:
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO codebase_events (id, run_id, type, payload_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (uuid4().hex, run_id, event_type, json.dumps(payload), utc_now()),
            )

    def test_no_fallback_means_nothing_to_report(self) -> None:
        run_id = self._run()
        self._event(run_id, "model.completed", {"provider": "ollama"})
        self.assertEqual(self.service._fallback_providers(run_id), set())

    def test_a_substituted_provider_is_reported(self) -> None:
        run_id = self._run()
        self._event(run_id, "model.fallback", {"requested_provider": "ollama", "error": "timed out"})
        self.assertEqual(self.service._fallback_providers(run_id), {"ollama"})

    def test_several_substitutions_are_collected_once_each(self) -> None:
        run_id = self._run()
        for provider in ("ollama", "llamacpp", "ollama"):
            self._event(run_id, "model.fallback", {"requested_provider": provider})
        self.assertEqual(self.service._fallback_providers(run_id), {"llamacpp", "ollama"})

    def test_other_runs_are_not_mixed_in(self) -> None:
        mine, theirs = self._run(), self._run()
        self._event(mine, "model.fallback", {"requested_provider": "ollama"})
        self._event(theirs, "model.fallback", {"requested_provider": "deepseek"})
        self.assertEqual(self.service._fallback_providers(mine), {"ollama"})

    def test_unusable_payloads_are_skipped_rather_than_raising(self) -> None:
        run_id = self._run()
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO codebase_events (id, run_id, type, payload_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (uuid4().hex, run_id, "model.fallback", "{not json", utc_now()),
            )
        self._event(run_id, "model.fallback", {"requested_provider": ""})
        self._event(run_id, "model.fallback", {"requested_provider": "ollama"})
        # A corrupt row and a blank name must not hide the usable one.
        self.assertEqual(self.service._fallback_providers(run_id), {"ollama"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
