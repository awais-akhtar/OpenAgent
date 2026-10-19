"""Tests for packaging checks that must never imply full release certification."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.product_release_gate import check_product

ROOT = Path(__file__).resolve().parents[1]


class ProductReleaseGateTests(unittest.TestCase):
    def test_source_snapshot_without_compiled_react_is_blocked(self) -> None:
        with TemporaryDirectory() as temp:
            report = check_product(ROOT, Path(temp))
        self.assertEqual(report["overallStatus"], "BLOCKED")
        self.assertIn("production-react-build", report["blockedChecks"])

    def test_explicit_good_build_passes_bundle_checks_not_full_release(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "index.html").write_text(
                '<script type="module" src="/app.js"></script>'
                '<link rel="stylesheet" href="/styles.css">', encoding="utf-8")
            (root / "app.js").write_text("// bundled", encoding="utf-8")
            (root / "styles.css").write_text("body{}", encoding="utf-8")
            report = check_product(ROOT, root)
        self.assertNotIn("production-react-build", report["blockedChecks"])
        self.assertNotIn("compiled-entrypoint", report["blockedChecks"])
        self.assertNotIn("no-demo-fixture-artifacts", report["blockedChecks"])
        self.assertIn("separate mandatory release gates", report["scope"])

    def test_demo_fixture_in_release_bundle_is_blocked(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "index.html").write_text('<script src="/app.js"></script>', encoding="utf-8")
            (root / "app.js").write_text("// bundled", encoding="utf-8")
            (root / "reference-fixtures.json").write_text("{}", encoding="utf-8")
            report = check_product(ROOT, root)
        self.assertIn("no-demo-fixture-artifacts", report["blockedChecks"])


if __name__ == '__main__':
    unittest.main()
