"""The Ollama `think` switch is sent only when OPENAGENT_OLLAMA_THINK turns thinking off."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from openagent import model_gateway


class OllamaThinkSwitchTests(unittest.TestCase):
    def test_unset_keeps_request_unchanged(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAGENT_OLLAMA_THINK", None)
            self.assertEqual({}, model_gateway._ollama_think_payload())

    def test_false_turns_thinking_off(self) -> None:
        for value in ("false", "FALSE", "0", "off", "no"):
            with patch.dict(os.environ, {"OPENAGENT_OLLAMA_THINK": value}):
                self.assertEqual({"think": False}, model_gateway._ollama_think_payload(), value)

    def test_true_or_other_values_leave_thinking_to_the_model(self) -> None:
        for value in ("true", "1", "on", "yes"):
            with patch.dict(os.environ, {"OPENAGENT_OLLAMA_THINK": value}):
                self.assertEqual({}, model_gateway._ollama_think_payload(), value)


if __name__ == "__main__":
    unittest.main()
