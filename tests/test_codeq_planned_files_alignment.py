"""A planned task's write set follows the file that defines the symbol its title names."""

from __future__ import annotations

import unittest

from openagent.codebase.service import CodebaseService

PROJECT_MAP = {
    "symbols": [
        {"name": "clean_local_part", "path": "util.py"},
        {"name": "split_address", "path": "domain.py"},
    ],
}


class PlannedFilesAlignmentTests(unittest.TestCase):
    def test_wrong_declared_file_is_replaced_by_the_defining_file(self) -> None:
        tasks = [{"title": "Fix clean_local_part to trim input", "files": ["domain.py"]}]
        aligned = CodebaseService._align_planned_files_with_titles(tasks, PROJECT_MAP)
        self.assertEqual(["util.py"], aligned[0]["files"])

    def test_correct_declaration_is_left_alone(self) -> None:
        tasks = [{"title": "Fix clean_local_part to trim input", "files": ["util.py"]}]
        self.assertEqual(["util.py"], CodebaseService._align_planned_files_with_titles(tasks, PROJECT_MAP)[0]["files"])

    def test_test_file_declarations_are_kept(self) -> None:
        tasks = [{"title": "Fix clean_local_part to trim input", "files": ["domain.py", "tests/test_util.py"]}]
        aligned = CodebaseService._align_planned_files_with_titles(tasks, PROJECT_MAP)
        self.assertEqual(["tests/test_util.py", "util.py"], aligned[0]["files"])

    def test_title_without_a_known_symbol_is_unchanged(self) -> None:
        tasks = [{"title": "Update the README", "files": ["README.md"]}]
        self.assertEqual(["README.md"], CodebaseService._align_planned_files_with_titles(tasks, PROJECT_MAP)[0]["files"])

    def test_first_named_symbol_is_the_subject_when_a_second_is_mentioned(self) -> None:
        tasks = [{"title": "Fix is_valid_email to use split_address", "files": ["domain.py"]}]
        map_ = {"symbols": [{"name": "is_valid_email", "path": "validators.py"}, {"name": "split_address", "path": "domain.py"}]}
        self.assertEqual(["validators.py"], CodebaseService._align_planned_files_with_titles(tasks, map_)[0]["files"])

    def test_ambiguous_symbol_is_not_guessed(self) -> None:
        tasks = [{"title": "Fix clean_local_part", "files": ["domain.py"]}]
        ambiguous = {"symbols": [{"name": "clean_local_part", "path": "a.py"}, {"name": "clean_local_part", "path": "b.py"}]}
        self.assertEqual(["domain.py"], CodebaseService._align_planned_files_with_titles(tasks, ambiguous)[0]["files"])


if __name__ == "__main__":
    unittest.main()
