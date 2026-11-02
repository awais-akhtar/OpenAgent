"""The per-response token cap has to be survivable on a local model.

A cap is an upper bound rather than a target, but the previous 16,384 for
implementation roles let a single call generate for about an hour against a 30B
model measured at roughly 4 tokens a second, so runs stalled before touching a
file. The default is now modest and the ceiling is configurable.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from openagent.codebase.service import (
    DEFAULT_IMPLEMENTATION_OUTPUT_TOKENS,
    IMPLEMENTATION_ROLES,
    codeq_output_token_budget,
)


class OutputBudgetTests(unittest.TestCase):
    def test_implementation_roles_get_the_default_budget(self) -> None:
        for role in sorted(IMPLEMENTATION_ROLES):
            with self.subTest(role=role):
                self.assertEqual(codeq_output_token_budget(role), DEFAULT_IMPLEMENTATION_OUTPUT_TOKENS)

    def test_other_roles_get_half_because_they_return_short_reports(self) -> None:
        half = DEFAULT_IMPLEMENTATION_OUTPUT_TOKENS // 2
        self.assertEqual(codeq_output_token_budget("tester"), half)
        self.assertEqual(codeq_output_token_budget("reviewer"), half)
        self.assertEqual(codeq_output_token_budget(""), half)

    def test_the_default_is_affordable_at_a_local_generation_rate(self) -> None:
        # The rate measured on the host that exposed this: ~4.3 tokens/second.
        # A single call must stay in minutes, not an hour.
        minutes = DEFAULT_IMPLEMENTATION_OUTPUT_TOKENS / 4.3 / 60
        self.assertLess(minutes, 20, f"a worst-case call would take {minutes:.0f} minutes")

    def test_an_operator_can_raise_it_for_a_faster_model(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAGENT_CODEQ_OUTPUT_TOKENS": "12000"}):
            self.assertEqual(codeq_output_token_budget("programmer"), 12000)
            self.assertEqual(codeq_output_token_budget("tester"), 6000)

    def test_the_value_is_clamped_and_bad_input_falls_back(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAGENT_CODEQ_OUTPUT_TOKENS": "50"}):
            self.assertEqual(codeq_output_token_budget("programmer"), 512)
        with mock.patch.dict(os.environ, {"OPENAGENT_CODEQ_OUTPUT_TOKENS": "999999"}):
            self.assertEqual(codeq_output_token_budget("programmer"), 32768)
        for bad in ("nonsense", "", "  ", "-1"):
            with self.subTest(value=bad), mock.patch.dict(os.environ, {"OPENAGENT_CODEQ_OUTPUT_TOKENS": bad}):
                self.assertEqual(
                    codeq_output_token_budget("programmer"), DEFAULT_IMPLEMENTATION_OUTPUT_TOKENS
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
