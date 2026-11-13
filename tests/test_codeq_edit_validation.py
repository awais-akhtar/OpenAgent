"""Autonomous CodeQ edits are checked by file type before they are saved; interactive edits are not."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from openagent.codebase.sandbox import SandboxManager
from openagent.codebase.tools import CodebaseTools


def _workspace(root: Path, files: dict[str, str], validate: bool) -> tuple[CodebaseTools, Path]:
    source = root / "source"
    source.mkdir()
    for name, text in files.items():
        (source / name).write_text(text, encoding="utf-8")
    sandbox = SandboxManager(root / "sandbox")
    sandbox.create("workspace", source=source)
    return CodebaseTools(sandbox, "workspace", validate_edits=validate), sandbox.path("workspace")


class EditValidationTests(unittest.TestCase):
    def test_python_edit_that_breaks_the_file_is_refused_and_not_saved(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tools, ws = _workspace(Path(tmp), {"app.py": "def f(v):\n    return v\n"}, validate=True)
            with self.assertRaisesRegex(ValueError, "syntax error"):
                tools.execute("write_file", {"path": "app.py", "content": "def f(v):\nreturn v\n"})
            self.assertEqual("def f(v):\n    return v\n", (ws / "app.py").read_text(encoding="utf-8"))

    def test_json_edit_that_breaks_the_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tools, ws = _workspace(Path(tmp), {"data.json": "{\"a\": 1}\n"}, validate=True)
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                tools.execute("write_file", {"path": "data.json", "content": "{\"a\": }"})
            self.assertEqual("{\"a\": 1}\n", (ws / "data.json").read_text(encoding="utf-8"))

    def test_javascript_edit_is_checked_when_node_is_installed(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node is not installed")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tools, ws = _workspace(Path(tmp), {"app.js": "const x = 1;\n"}, validate=True)
            with self.assertRaisesRegex(ValueError, "JavaScript syntax error"):
                tools.execute("write_file", {"path": "app.js", "content": "const x = ;\n"})
            self.assertEqual("const x = 1;\n", (ws / "app.js").read_text(encoding="utf-8"))

    def test_unknown_file_types_are_not_checked(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tools, ws = _workspace(Path(tmp), {"notes.md": "a\n"}, validate=True)
            tools.execute("write_file", {"path": "notes.md", "content": "anything {{ not code\n"})
            self.assertEqual("anything {{ not code\n", (ws / "notes.md").read_text(encoding="utf-8"))

    def test_interactive_tools_keep_the_previous_behaviour(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tools, ws = _workspace(Path(tmp), {"app.py": "x = 1\n"}, validate=False)
            tools.execute("write_file", {"path": "app.py", "content": "def broken(:\n"})
            self.assertEqual("def broken(:\n", (ws / "app.py").read_text(encoding="utf-8"))


class StaleSearchTests(unittest.TestCase):
    def test_patch_is_refused_after_a_failed_search_until_write_file_replaces_the_file(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            original = "def f(v):\n    return v\n"
            tools, ws = _workspace(Path(tmp), {"util.py": original}, validate=True)
            with self.assertRaisesRegex(ValueError, "search text was not present"):
                tools.execute("patch_file", {"path": "util.py", "search": "return missing", "replace": "return 1"})
            with self.assertRaisesRegex(ValueError, "refused"):
                tools.execute("patch_file", {"path": "util.py", "search": "return v", "replace": "return v.strip()"})
            tools.execute("write_file", {"path": "util.py", "content": "def f(v):\n    return v.strip()\n"})
            tools.execute("patch_file", {"path": "util.py", "search": "return v.strip()", "replace": "return v"})
            self.assertEqual(original, (ws / "util.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
