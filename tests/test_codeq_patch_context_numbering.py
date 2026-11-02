"""The patch tool must accept text quoted from a numbered context excerpt.

`RepositoryIntelligence._relevant_excerpt` prefixes every line it shows with
``N: ``, so a caller that quotes its own context verbatim produced a search
string that could never match the file it came from. Every patch_file attempt
in a real run failed with "expected 1 occurrence(s), found 0", which burned the
run's iteration budget.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openagent.codebase.repository import RepositoryIntelligence
from openagent.codebase.tools import CodebaseTools


class ContextLineNumberStrippingTests(unittest.TestCase):
    strip = staticmethod(CodebaseTools._strip_context_line_numbers)

    def test_numbered_excerpt_is_reduced_to_the_original_text(self) -> None:
        self.assertEqual(
            self.strip("1: def add(a, b):\n2:     return a + b"),
            "def add(a, b):\n    return a + b",
        )

    def test_plain_text_is_untouched(self) -> None:
        self.assertEqual(self.strip("def add(a, b):"), "def add(a, b):")

    def test_content_that_merely_starts_with_digits_is_untouched(self) -> None:
        # Not our format, so it must be treated as real file content.
        self.assertEqual(self.strip("42 is the answer"), "42 is the answer")
        self.assertEqual(self.strip("2024-01-01: released"), "2024-01-01: released")

    def test_partially_numbered_text_is_untouched(self) -> None:
        # Ambiguous, so the conservative choice is to change nothing.
        self.assertEqual(self.strip("1: def add():\nreturn 1"), "1: def add():\nreturn 1")

    def test_blank_lines_are_preserved(self) -> None:
        self.assertEqual(self.strip("1: a\n\n3: b"), "a\n\nb")


class ContextExcerptMatchesPatchContractTests(unittest.TestCase):
    def test_excerpt_text_can_be_used_as_a_patch_search(self) -> None:
        """The two halves of the contract have to line up.

        What select_context shows is numbered; what patch_file matches is the
        raw file. Stripping has to bridge exactly that, for the real excerpt
        rather than a hand-written approximation of it.
        """
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            context = RepositoryIntelligence(root).select_context("add calc", limit=5)
            excerpt = next(item["content"] for item in context if item["path"] == "calc.py")

            # The excerpt as shown cannot match the file; that is the bug.
            self.assertNotIn(excerpt, (root / "calc.py").read_text(encoding="utf-8"))
            # Stripped, it must match exactly once.
            stripped = CodebaseTools._strip_context_line_numbers(excerpt)
            self.assertEqual((root / "calc.py").read_text(encoding="utf-8").count(stripped), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
