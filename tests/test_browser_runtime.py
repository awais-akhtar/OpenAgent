from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openagent.tools.browser import BrowserPolicy, BrowserRuntime, BrowserRuntimeError


class BrowserRuntimeTests(unittest.TestCase):
    def test_local_navigation_is_allowed_without_approval(self) -> None:
        self.assertEqual(BrowserPolicy().authorize("http://127.0.0.1:8765/"), "127.0.0.1")

    def test_external_navigation_requires_approval_or_allowlist(self) -> None:
        policy = BrowserPolicy({"example.com"})
        self.assertEqual(policy.authorize("https://example.com/path"), "example.com")
        with self.assertRaises(PermissionError):
            policy.authorize("https://unapproved.example/path")
        self.assertEqual(policy.authorize("https://unapproved.example/path", approved=True), "unapproved.example")

    def test_invalid_urls_are_rejected_before_browser_startup(self) -> None:
        with self.assertRaises(BrowserRuntimeError):
            BrowserPolicy().authorize("file:///tmp/private.txt")

    def test_missing_playwright_reports_optional_dependency_guidance(self) -> None:
        with TemporaryDirectory() as temp, patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            with self.assertRaisesRegex(BrowserRuntimeError, r"pip install openagent\[browser\]"):
                BrowserRuntime(temp).run("http://localhost:8765/")


if __name__ == "__main__":
    unittest.main()