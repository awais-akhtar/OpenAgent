"""A refusal must carry the status it deserves, not a blanket 400 or 200.

Four services report a record belonging to another project or owner as an
``error`` key rather than by raising. The route layer answered those with a
blanket ``400 Bad Request`` -- and for semantic memory with ``200`` and
``success: true`` -- so a caller, and the screen, could not tell "your request
was malformed" from "this is not yours" or "there is no such record".
"""

from __future__ import annotations

import unittest
from http import HTTPStatus

from openagent.routes.g25_advanced_routes import _memory_result
from openagent.routes.model_serve_routes import _serve_status
from openagent.routes.voice_conversation_routes import _conversation_status


class MemoryResultStatusTests(unittest.TestCase):
    def test_a_usable_result_is_ok(self) -> None:
        self.assertEqual(_memory_result({"messages": []}).status, HTTPStatus.OK)

    def test_a_foreign_session_is_forbidden(self) -> None:
        result = _memory_result({"error": "Session does not belong to this project/owner"})
        self.assertEqual(result.status, HTTPStatus.FORBIDDEN)

    def test_an_unknown_session_is_not_found(self) -> None:
        self.assertEqual(_memory_result({"error": "Session not found"}).status, HTTPStatus.NOT_FOUND)


class ConversationStatusTests(unittest.TestCase):
    def test_a_foreign_conversation_is_forbidden(self) -> None:
        self.assertEqual(
            _conversation_status("Conversation does not belong to this project/owner"),
            HTTPStatus.FORBIDDEN,
        )

    def test_an_unknown_conversation_is_not_found(self) -> None:
        self.assertEqual(_conversation_status("Conversation abc not found"), HTTPStatus.NOT_FOUND)

    def test_a_provider_failure_stays_a_bad_request(self) -> None:
        # Not an ownership question, so it must not be reported as one.
        self.assertEqual(_conversation_status("Transcription failed: no engine"), HTTPStatus.BAD_REQUEST)


class ServeStatusTests(unittest.TestCase):
    def test_a_foreign_session_is_forbidden(self) -> None:
        self.assertEqual(
            _serve_status("Session does not belong to this project/owner"),
            HTTPStatus.FORBIDDEN,
        )

    def test_an_unknown_session_is_not_found(self) -> None:
        self.assertEqual(_serve_status("Session abc not found"), HTTPStatus.NOT_FOUND)

    def test_a_missing_model_file_stays_a_bad_request(self) -> None:
        # The session exists and is the caller's; the host is not ready for it,
        # so this is not a "no such session" answer.
        self.assertEqual(
            _serve_status("Model file not found: /models/absent.gguf"),
            HTTPStatus.BAD_REQUEST,
        )

    def test_an_undownloaded_model_stays_a_bad_request(self) -> None:
        # The caller may make this request once the model is downloaded, which
        # is what requires_download in the body tells them.
        self.assertEqual(
            _serve_status("Model big-model is not downloaded. Download it first."),
            HTTPStatus.BAD_REQUEST,
        )


class EveryOwnershipRefusalIsMappedTests(unittest.TestCase):
    """The services' own wording is what these mappers key on."""

    SERVICES = (
        "openagent.model_serve_manager",
        "openagent.semantic_memory_service",
        "openagent.ultra_model_service",
        "openagent.voice_conversation_manager",
    )

    def test_each_service_still_uses_the_wording_the_mappers_match(self) -> None:
        import importlib
        import inspect

        for name in self.SERVICES:
            with self.subTest(service=name):
                source = inspect.getsource(importlib.import_module(name))
                self.assertIn("does not belong", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
