"""Semantic memory must still serve its sessions after a restart.

The service used to write only per-session counts to its state file and to
discard them again on load, so a restart logged "Loaded memory state: 3
sessions" while every one of those session ids answered "Session not found".
It also wrote the file into the process working directory, which put it in the
repository whenever the server was started from a checkout.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openagent.semantic_memory_service import SemanticMemoryService


class SemanticMemoryDurabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.state_dir = Path(self._tmp.name)

    def _service(self) -> SemanticMemoryService:
        return SemanticMemoryService(state_dir=self.state_dir)

    def test_the_state_file_stays_in_the_given_directory(self) -> None:
        service = self._service()
        service.create_session()
        self.assertEqual(service.state_file.parent, self.state_dir)
        self.assertTrue((self.state_dir / "semantic_memory_state.json").exists())

    def test_messages_survive_a_restart(self) -> None:
        first = self._service()
        session_id = first.create_session(project="p", owner="o")["session_id"]
        first.store_message(session_id, "user", "remember this", project="p", owner="o")

        restarted = self._service()
        history = restarted.get_conversation_history(session_id, project="p", owner="o")
        self.assertNotIn("error", history)
        self.assertEqual([m["content"] for m in history["messages"]], ["remember this"])

    def test_the_restored_session_keeps_its_project_and_owner(self) -> None:
        first = self._service()
        session_id = first.create_session(project="p", owner="o")["session_id"]
        first.store_message(session_id, "user", "scoped", project="p", owner="o")

        restarted = self._service()
        # Another owner must not reach a session restored from disk.
        denied = restarted.get_conversation_history(session_id, project="p", owner="someone-else")
        self.assertIn("error", denied)
        self.assertEqual([s["id"] for s in restarted.list_sessions(project="p", owner="o")], [session_id])

    def test_context_memories_survive_a_restart(self) -> None:
        first = self._service()
        session_id = first.create_session()["session_id"]
        first.store_context(session_id, "topic", "durability")

        restarted = self._service()
        self.assertEqual(restarted.retrieve_context(session_id, "topic"), "durability")
        self.assertEqual(restarted.get_session_context(session_id)["context_size"], 1)

    def test_a_recovered_session_reports_what_it_holds(self) -> None:
        first = self._service()
        session_id = first.create_session()["session_id"]
        first.store_message(session_id, "user", "one")
        first.store_context(session_id, "k", "v")

        recovered = self._service().recover_session(session_id)
        self.assertTrue(recovered["recovered"])
        self.assertEqual(recovered["message_count"], 1)
        self.assertEqual(recovered["context_memory_size"], 1)

    def test_a_counts_only_state_file_is_ignored_rather_than_half_restored(self) -> None:
        legacy = {"sessions": {"old-id": {"created_at": 1.0, "message_count": 4}}, "context_memories": {}}
        (self.state_dir / "semantic_memory_state.json").write_text(json.dumps(legacy), encoding="utf-8")
        service = self._service()
        # The old file carries no messages, so claiming the session exists would
        # only produce a session that answers nothing.
        self.assertEqual(service.sessions, {})
        self.assertIn("error", service.recover_session("old-id"))

    def test_an_unreadable_state_file_does_not_stop_a_start(self) -> None:
        (self.state_dir / "semantic_memory_state.json").write_text("{not json", encoding="utf-8")
        service = self._service()
        self.assertEqual(service.sessions, {})
        session_id = service.create_session()["session_id"]
        self.assertEqual(self._service().recover_session(session_id)["recovered"], True)

    def test_a_save_leaves_no_temporary_file_behind(self) -> None:
        service = self._service()
        service.create_session()
        self.assertEqual(
            sorted(path.name for path in self.state_dir.iterdir()),
            ["semantic_memory_state.json"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
