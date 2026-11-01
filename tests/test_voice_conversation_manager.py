"""Persistence and selection behaviour for the voice conversation manager.

The fuller implementation of this manager sat at the repository root, outside
the package, where nothing could import it. The wired copy declared a state
file it never read or wrote, so conversations were lost on restart, and the
transcription-model and synthesis-voice selection its services supported was
not reachable. These tests cover what was brought across.
"""

from __future__ import annotations

from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

from openagent.voice_conversation_manager import VoiceConversationManager


class VoiceConversationPersistenceTests(unittest.TestCase):
    def test_conversations_survive_a_restart_with_their_project_and_owner(self) -> None:
        with TemporaryDirectory() as state_dir:
            manager = VoiceConversationManager(state_dir=state_dir)
            alice = manager.create_conversation(project="pA", owner="alice")
            manager.create_conversation(project="pB", owner="bob")
            conversation_id = alice["conversation_id"]

            restarted = VoiceConversationManager(state_dir=state_dir)

            self.assertEqual(len(restarted.list_conversations(project="pA", owner="alice")), 1)
            self.assertEqual(len(restarted.list_conversations(project="pB", owner="bob")), 1)
            # Scoping has to survive the reload, not just the first run.
            self.assertIsNone(restarted.get_conversation(conversation_id, project="pB", owner="bob"))
            self.assertIsNotNone(restarted.get_conversation(conversation_id, project="pA", owner="alice"))

    def test_closing_a_conversation_persists(self) -> None:
        with TemporaryDirectory() as state_dir:
            manager = VoiceConversationManager(state_dir=state_dir)
            conversation_id = manager.create_conversation(project="pA", owner="alice")["conversation_id"]
            self.assertTrue(manager.close_conversation(conversation_id, project="pA", owner="alice"))

            reloaded = VoiceConversationManager(state_dir=state_dir).get_conversation(
                conversation_id, project="pA", owner="alice",
            )
            self.assertEqual(reloaded["status"], "closed")

    def test_another_owner_cannot_close_a_conversation(self) -> None:
        with TemporaryDirectory() as state_dir:
            manager = VoiceConversationManager(state_dir=state_dir)
            conversation_id = manager.create_conversation(project="pA", owner="alice")["conversation_id"]
            self.assertFalse(manager.close_conversation(conversation_id, project="pB", owner="bob"))
            still_open = manager.get_conversation(conversation_id, project="pA", owner="alice")
            self.assertEqual(still_open["status"], "active")

    def test_state_file_is_written_beside_the_supplied_directory(self) -> None:
        with TemporaryDirectory() as state_dir:
            manager = VoiceConversationManager(state_dir=state_dir)
            manager.create_conversation(project="pA", owner="alice")
            # A bare relative name would land in the process working directory.
            self.assertTrue(manager.state_file.is_file())
            self.assertEqual(manager.state_file.parent.resolve(), Path(state_dir).resolve())


class VoiceSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.manager = VoiceConversationManager(state_dir=self._tmp.name)

    def test_transcription_models_are_listed_with_the_current_choice(self) -> None:
        reported = self.manager.get_transcription_models()
        self.assertTrue(reported["models"], "no transcription models were offered")
        self.assertIn(reported["current"], reported["models"])

    def test_synthesis_voices_are_listed_with_the_current_choice(self) -> None:
        reported = self.manager.get_synthesis_voices()
        self.assertTrue(reported["voices"], "no synthesis voices were offered")
        self.assertIn(reported["current"], reported["voices"])

    def test_selecting_an_offered_model_and_voice_sticks(self) -> None:
        model = [name for name in self.manager.get_transcription_models()["models"]
                 if name != self.manager.get_transcription_models()["current"]][0]
        self.assertTrue(self.manager.set_transcription_model(model))
        self.assertEqual(self.manager.get_transcription_models()["current"], model)

        voice = [name for name in self.manager.get_synthesis_voices()["voices"]
                 if name != self.manager.get_synthesis_voices()["current"]][0]
        self.assertTrue(self.manager.set_synthesis_voice(voice))
        self.assertEqual(self.manager.get_synthesis_voices()["current"], voice)

    def test_an_unoffered_model_or_voice_is_refused_and_changes_nothing(self) -> None:
        before_model = self.manager.get_transcription_models()["current"]
        before_voice = self.manager.get_synthesis_voices()["current"]

        self.assertFalse(self.manager.set_transcription_model("not-a-model"))
        # set_voice previously accepted any string and returned True.
        self.assertFalse(self.manager.set_synthesis_voice("not-a-voice"))

        self.assertEqual(self.manager.get_transcription_models()["current"], before_model)
        self.assertEqual(self.manager.get_synthesis_voices()["current"], before_voice)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
