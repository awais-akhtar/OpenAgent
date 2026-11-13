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
from openagent.codebase.sandbox import SandboxManager
from openagent.codebase.service import CodebaseService


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
    def test_identical_write_returns_explicit_no_op_feedback(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "example.js").write_text("export const value = 1;\n", encoding="utf-8")
            sandbox = SandboxManager(root / "sandbox")
            sandbox.create("workspace", source=source)
            tools = CodebaseTools(sandbox, "workspace")

            result = tools.execute(
                "write_file", {"path": "example.js", "content": "export const value = 2;\n"},
            )
            self.assertTrue(result["changed"])
            self.assertNotIn("no_op", result)

            no_op = tools.execute(
                "write_file", {"path": "example.js", "content": "export const value = 2;\n"},
            )
            self.assertFalse(no_op["changed"])
            self.assertTrue(no_op["no_op"])
            self.assertIn("Do not repeat the same write", no_op["message"])
            self.assertEqual("", no_op["patch"])

    def test_model_context_omits_display_only_line_numbers(self) -> None:
        cleaned = CodebaseService._model_context([
            {"path": "calc.py", "score": 3, "content": "1: def add(a, b):\n2:     return a + b"},
        ])
        self.assertEqual(cleaned[0]["content"], "def add(a, b):\n    return a + b")
        self.assertEqual(cleaned[0]["path"], "calc.py")

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

    def test_patch_search_strips_repeated_context_number_prefixes(self) -> None:
        self.assertEqual(
            "import unittest\n\nclass Example:\n    pass",
            CodebaseTools._strip_context_line_numbers(
                "1: 1: import unittest\n\n3: 3: class Example:\n4: 4:     pass"
            ),
        )

    def test_write_file_strips_numbered_context_and_allows_explicit_preservation(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            target = source / "example.py"
            target.write_text("original = True\n", encoding="utf-8")
            sandbox = SandboxManager(root / "sandbox")
            sandbox.create("workspace", source=source)
            tools = CodebaseTools(sandbox, "workspace")
            result = tools.execute(
                "write_file",
                {"path": "example.py", "content": "1: import unittest\n2: class Example: pass"},
            )
            self.assertTrue(result["context_line_numbers_stripped"])
            self.assertEqual(
                "import unittest\nclass Example: pass",
                (sandbox.path("workspace") / "example.py").read_text(encoding="utf-8"),
            )
            tools.execute(
                "write_file",
                {
                    "path": "example.py",
                    "content": "1: import unittest\n2: class Example: pass",
                    "preserve_line_numbers": True,
                },
            )
            self.assertEqual(
                "1: import unittest\n2: class Example: pass",
                (sandbox.path("workspace") / "example.py").read_text(encoding="utf-8"),
            )

    def test_patch_file_applies_exact_unified_diff(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            sandbox = SandboxManager(root / "sandbox")
            sandbox.create("workspace", source=source)
            tools = CodebaseTools(sandbox, "workspace")
            unified_diff = (
                "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n-    return a + b\n+    return int(a) + int(b)\n"
            )
            result = tools.execute("patch_file", {"file_path": "calc.py", "patch": unified_diff})
            self.assertIn("int(a) + int(b)", result["patch"])
            self.assertEqual(
                "def add(a, b):\n    return int(a) + int(b)\n",
                (sandbox.path("workspace") / "calc.py").read_text(encoding="utf-8"),
            )
            self.assertEqual("", CodebaseService._action_argument_error(
                "patch_file", {"path": "calc.py", "patch": unified_diff},
            ))

    def test_patch_file_recovers_miscalculated_hunk_counts_when_context_matches(self) -> None:
        """A bad hunk count should not burn retries when its exact body is sound."""
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            original = "def add(a, b):\n    return a + b\n"
            (source / "calc.py").write_text(original, encoding="utf-8")
            sandbox = SandboxManager(root / "sandbox")
            sandbox.create("workspace", source=source)
            tools = CodebaseTools(sandbox, "workspace")
            malformed_counts = (
                "--- a/calc.py\n+++ b/calc.py\n@@ -1,4 +1,4 @@\n"
                " def add(a, b):\n-    return a + b\n+    return int(a) + int(b)\n"
            )

            tools.execute("patch_file", {"path": "calc.py", "patch": malformed_counts})

            self.assertEqual(
                "def add(a, b):\n    return int(a) + int(b)\n",
                (sandbox.path("workspace") / "calc.py").read_text(encoding="utf-8"),
            )

    def test_hunk_count_recovery_rejects_contextless_insertions(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            original = "safe = True\n"
            (source / "calc.py").write_text(original, encoding="utf-8")
            sandbox = SandboxManager(root / "sandbox")
            sandbox.create("workspace", source=source)
            tools = CodebaseTools(sandbox, "workspace")
            malformed_insert = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,3 @@\n+unsafe = True\n"

            with self.assertRaisesRegex(ValueError, "no exact context to verify"):
                tools.execute("patch_file", {"path": "calc.py", "patch": malformed_insert})

            self.assertEqual(original, (sandbox.path("workspace") / "calc.py").read_text(encoding="utf-8"))

    def test_patch_file_recovers_unprefixed_blank_context_only_on_exact_match(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            original = "def add(a, b):\n    return a + b\n\ndef next_value():\n    return 1\n"
            (source / "calc.py").write_text(original, encoding="utf-8")
            sandbox = SandboxManager(root / "sandbox")
            sandbox.create("workspace", source=source)
            tools = CodebaseTools(sandbox, "workspace")
            diff_with_blank_context = (
                "--- a/calc.py\n+++ b/calc.py\n@@ -1,5 +1,5 @@\n"
                " def add(a, b):\n-    return a + b\n+    return int(a) + int(b)\n\n def next_value():\n"
                "     return 1\n"
            )

            tools.execute("patch_file", {"path": "calc.py", "patch": diff_with_blank_context})

            self.assertEqual(
                "def add(a, b):\n    return int(a) + int(b)\n\ndef next_value():\n    return 1\n",
                (sandbox.path("workspace") / "calc.py").read_text(encoding="utf-8"),
            )

    def test_unified_diff_with_stale_context_does_not_modify_file(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            sandbox = SandboxManager(root / "sandbox")
            sandbox.create("workspace", source=source)
            tools = CodebaseTools(sandbox, "workspace")
            with self.assertRaisesRegex(ValueError, "unchanged"):
                tools.execute("patch_file", {"path": "calc.py", "search": "return", "replace": "return"})
            with self.assertRaisesRegex(ValueError, "found 2"):
                tools.execute("patch_file", {"path": "calc.py", "search": "b", "replace": "x"})
            with self.assertRaisesRegex(ValueError, "refused"):
                tools.execute("patch_file", {"path": "calc.py", "search": "return", "replace": "raise"})
            tools.execute("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a - b\n"})
            stale_diff = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a + b\n+    return int(a) + int(b)\n"
            with self.assertRaisesRegex(ValueError, "stale or ambiguous"):
                tools.execute("patch_file", {"path": "calc.py", "patch": stale_diff})
            self.assertEqual(
                "def add(a, b):\n    return a - b\n",
                (sandbox.path("workspace") / "calc.py").read_text(encoding="utf-8"),
            )

    def test_patch_file_accepts_structured_search_replace_payload(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "calc.py").write_text("value = 1\n", encoding="utf-8")
            sandbox = SandboxManager(root / "sandbox")
            sandbox.create("workspace", source=source)
            tools = CodebaseTools(sandbox, "workspace")
            tools.execute(
                "patch_file",
                {"path": "calc.py", "patch": {"search": "value = 1", "replace": "value = 2"}},
            )
            self.assertEqual("value = 2\n", (sandbox.path("workspace") / "calc.py").read_text(encoding="utf-8"))

    def test_numbered_context_passed_as_patch_returns_actionable_error_without_write(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            original = "def add(a, b):\n    return a + b\n"
            (source / "calc.py").write_text(original, encoding="utf-8")
            sandbox = SandboxManager(root / "sandbox")
            sandbox.create("workspace", source=source)
            tools = CodebaseTools(sandbox, "workspace")
            with self.assertRaisesRegex(ValueError, "numbered source context.*write_file"):
                tools.execute("patch_file", {
                    "path": "calc.py",
                    "patch": "1: def add(a, b):\n2:     return a + b",
                })
            self.assertEqual(original, (sandbox.path("workspace") / "calc.py").read_text(encoding="utf-8"))

    def test_grep_accepts_search_as_a_query_alias(self) -> None:
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            sandbox = SandboxManager(root / "sandbox")
            sandbox.create("workspace", source=source)
            tools = CodebaseTools(sandbox, "workspace")
            result = tools.execute("grep", {"path": "calc.py", "search": "add"})
            self.assertTrue(any("add" in item.get("text", "") for item in result["matches"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
