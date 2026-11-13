"""A replacement that drops the indentation of the block it replaces keeps that indentation."""

from __future__ import annotations

import unittest

from openagent.codebase.tools import CodebaseTools


class BlockIndentationTests(unittest.TestCase):
    def test_unindented_replacement_of_an_indented_block_is_indented(self) -> None:
        out = CodebaseTools._keep_block_indentation("    return x", "if x:\n    return 1\nreturn 2")
        self.assertEqual("    if x:\n        return 1\n    return 2", out)

    def test_replacement_with_its_own_indentation_is_unchanged(self) -> None:
        replace = "    if x:\n        return 1"
        self.assertEqual(replace, CodebaseTools._keep_block_indentation("    return x", replace))

    def test_unindented_block_is_unchanged(self) -> None:
        self.assertEqual("import os", CodebaseTools._keep_block_indentation("import sys", "import os"))


if __name__ == "__main__":
    unittest.main()
