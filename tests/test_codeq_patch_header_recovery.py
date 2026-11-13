"""Unified-diff recovery: a miscounted header is repaired only when the hunk describes the file."""

from __future__ import annotations

import unittest

from openagent.codebase.tools import CodebaseTools

ORIGINAL = "def f(v):\n    return v\n"


class PatchHeaderRecoveryTests(unittest.TestCase):
    def test_correct_header_applies(self) -> None:
        patch = "@@ -1,2 +1,2 @@\n def f(v):\n-    return v\n+    return v.strip()\n"
        self.assertEqual("def f(v):\n    return v.strip()\n", CodebaseTools._apply_unified_patch(ORIGINAL, patch))

    def test_miscounted_header_recovers_when_the_hunk_matches_the_file(self) -> None:
        patch = "@@ -1,2 +1,2 @@\n-    return v\n+    return v.strip()\n"
        self.assertEqual(
            "def f(v):\n    return v.strip()\n",
            CodebaseTools._apply_unified_patch(ORIGINAL, patch, trust_hunk_body=True),
        )

    def test_hunk_claiming_more_lines_than_the_file_is_refused_instead_of_duplicating_code(self) -> None:
        # The file has two lines; the hunk claims six. Applying only its matching body would
        # leave the unmatched original lines in place, so the recovery must refuse.
        patch = "@@ -1,6 +1,6 @@\n def f(v):\n+    w = v\n+    return w\n"
        with self.assertRaisesRegex(ValueError, "claims more original lines"):
            CodebaseTools._apply_unified_patch(ORIGINAL, patch, trust_hunk_body=True)

    def test_strict_mode_still_rejects_any_miscount(self) -> None:
        patch = "@@ -1,6 +1,6 @@\n def f(v):\n+    w = v\n"
        with self.assertRaisesRegex(ValueError, "line count does not match"):
            CodebaseTools._apply_unified_patch(ORIGINAL, patch)


if __name__ == "__main__":
    unittest.main()
