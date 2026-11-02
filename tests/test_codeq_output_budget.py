"""The per-response token cap has to be survivable on a local model.

A cap is an upper bound rather than a target, but 16,384 for implementation
roles let a single call generate for about an hour against a 30B model measured
at roughly 4 tokens a second, so runs stalled before touching a file.

An earlier attempt gave non-implementation roles half the budget. That starved
``project_manager``, whose reply is the whole task plan: the JSON was truncated,
stopped parsing, and the run fell back to a generic plan. Every role now shares
one budget, set to the value those roles already worked at.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from openagent.codebase.service import DEFAULT_OUTPUT_TOKENS, codeq_output_token_budget


class OutputBudgetTests(unittest.TestCase):
    def test_every_role_shares_one_budget(self) -> None:
        # project_manager returns the plan and must not be starved relative to
        # the roles that return actions.
        for role in ("programmer", "debugger", "architect", "project_manager", "tester", ""):
            with self.subTest(role=role):
                self.assertEqual(codeq_output_token_budget(role), DEFAULT_OUTPUT_TOKENS)

    def test_the_default_does_not_reduce_what_already_worked(self) -> None:
        # The planning and specialist roles were passing at 8,192; dropping
        # below that is what broke plan parsing.
        self.assertGreaterEqual(DEFAULT_OUTPUT_TOKENS, 8192)

    def test_the_default_is_well_under_the_previous_implementation_cap(self) -> None:
        # The point of the change: halve the worst-case implementation call.
        self.assertLessEqual(DEFAULT_OUTPUT_TOKENS, 16384 // 2)

    def test_an_operator_can_raise_it_for_a_faster_model(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAGENT_CODEQ_OUTPUT_TOKENS": "20000"}):
            self.assertEqual(codeq_output_token_budget("programmer"), 20000)
            self.assertEqual(codeq_output_token_budget("project_manager"), 20000)

    def test_the_value_is_clamped_at_both_ends(self) -> None:
        with mock.patch.dict(os.environ, {"OPENAGENT_CODEQ_OUTPUT_TOKENS": "50"}):
            self.assertEqual(codeq_output_token_budget("programmer"), 512)
        with mock.patch.dict(os.environ, {"OPENAGENT_CODEQ_OUTPUT_TOKENS": "999999"}):
            self.assertEqual(codeq_output_token_budget("programmer"), 32768)

    def test_unusable_input_falls_back_to_the_default(self) -> None:
        for bad in ("nonsense", "", "  ", "-1", "0"):
            with self.subTest(value=bad), mock.patch.dict(
                os.environ, {"OPENAGENT_CODEQ_OUTPUT_TOKENS": bad}
            ):
                self.assertEqual(codeq_output_token_budget("programmer"), DEFAULT_OUTPUT_TOKENS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
