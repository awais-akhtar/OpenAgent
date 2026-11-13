"""Phase 4: a CodeQ run completes only with current, executed, passing evidence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openagent.codebase.evidence import completion_gate, executed_passing_checks, source_fingerprint

FP = "a" * 64


def _check(kind: str, status: str = "passed", exit_code: int = 0, fingerprint: str = FP) -> dict:
    return {"evidence_id": f"evidence-{kind}", "kind": kind, "status": status, "exit_code": exit_code, "source_fingerprint": fingerprint}


def _validation(checks: list[dict], passed: bool | None = None, fingerprint: str = FP) -> dict:
    if passed is None:
        passed = bool(checks) and all(item["status"] == "passed" for item in checks)
    return {"passed": passed, "checks": checks, "source_fingerprint": fingerprint}


class SourceFingerprintTests(unittest.TestCase):
    def test_stable_until_a_source_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / "app.py").write_text("print('a')\n", encoding="utf-8")
            first = source_fingerprint(root)
            self.assertEqual(first, source_fingerprint(root))
            (base / "app.py").write_text("print('b')\n", encoding="utf-8")
            self.assertNotEqual(first, source_fingerprint(root))

    def test_dependency_and_vcs_directories_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / "app.py").write_text("x = 1\n", encoding="utf-8")
            before = source_fingerprint(root)
            (base / "node_modules").mkdir()
            (base / "node_modules" / "dep.js").write_text("// noise", encoding="utf-8")
            (base / ".git").mkdir()
            (base / ".git" / "HEAD").write_text("ref", encoding="utf-8")
            self.assertEqual(before, source_fingerprint(root))

    def test_added_or_removed_file_changes_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / "a.py").write_text("a", encoding="utf-8")
            first = source_fingerprint(root)
            (base / "b.py").write_text("b", encoding="utf-8")
            second = source_fingerprint(root)
            (base / "b.py").unlink()
            self.assertNotEqual(first, second)
            self.assertEqual(first, source_fingerprint(root))

    def test_unreadable_workspace_file_fails_fingerprint_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "secret.py").write_text("x = 1\n", encoding="utf-8")
            with patch.object(Path, "open", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    source_fingerprint(root)


class CompletionGateTests(unittest.TestCase):
    def test_forged_done_without_any_check_is_blocked(self) -> None:
        # A model's done claim is not an input: no executed evidence, so no completion.
        gate = completion_gate(_validation([], passed=True), FP)
        self.assertEqual(gate["status"], "blocked")
        self.assertIn("no executed verification evidence", gate["reason"])
        self.assertTrue(gate["next_action"])

    def test_artifact_scan_alone_does_not_complete(self) -> None:
        validation = _validation([_check("artifacts")], passed=True)
        self.assertEqual(completion_gate(validation, FP)["status"], "blocked")

    def test_failed_command_blocks_with_the_failing_kind(self) -> None:
        validation = _validation([_check("test", "failed", 1)], passed=False)
        gate = completion_gate(validation, FP)
        self.assertEqual(gate["status"], "blocked")
        self.assertIn("test", gate["reason"])
        self.assertIn("repair", gate["next_action"])

    def test_stale_test_result_after_source_change_is_blocked(self) -> None:
        # The recorded results were produced from an older fingerprint.
        validation = _validation([_check("test", fingerprint="old")], passed=True, fingerprint="old")
        gate = completion_gate(validation, FP)
        self.assertEqual(gate["status"], "blocked")
        self.assertTrue(gate["stale"])
        self.assertIn("stale", gate["reason"])

    def test_checks_from_older_sources_do_not_count_as_evidence(self) -> None:
        validation = _validation([_check("test", fingerprint="old")], passed=True, fingerprint=FP)
        self.assertEqual(executed_passing_checks(validation, FP), [])
        self.assertEqual(completion_gate(validation, FP)["status"], "blocked")

    def test_status_passed_with_nonzero_exit_is_not_evidence(self) -> None:
        validation = _validation([_check("build", "passed", 2)], passed=True)
        self.assertEqual(executed_passing_checks(validation), [])
        self.assertEqual(completion_gate(validation, FP)["status"], "blocked")

    def test_unidentified_check_cannot_complete(self) -> None:
        validation = _validation([_check("test")], passed=True)
        validation["checks"][0]["evidence_id"] = ""
        self.assertEqual(executed_passing_checks(validation, FP), [])
        self.assertEqual(completion_gate(validation, FP)["status"], "blocked")

    def test_blocking_review_finding_blocks_completion(self) -> None:
        validation = _validation([_check("test")], passed=True)
        gate = completion_gate(validation, FP, review_blocking=2)
        self.assertEqual(gate["status"], "blocked")
        self.assertIn("review finding", gate["reason"])

    def test_unresolved_acceptance_item_blocks_completion(self) -> None:
        # Example: a UI task finished but the backend or persistence layer is still missing.
        validation = _validation([_check("build")], passed=True)
        gate = completion_gate(validation, FP, unresolved=[{"layer": "backend"}])
        self.assertEqual(gate["status"], "blocked")
        self.assertIn("acceptance item", gate["reason"])

    def test_current_passing_evidence_completes(self) -> None:
        validation = _validation([_check("test"), _check("artifacts")], passed=True)
        gate = completion_gate(validation, FP)
        self.assertEqual(gate["status"], "completed")
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["evidence"], ["test"])
        self.assertEqual(gate["next_action"], "")

    def test_empty_or_missing_validation_is_not_complete(self) -> None:
        self.assertEqual(completion_gate({}, FP)["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
