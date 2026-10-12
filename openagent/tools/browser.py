"""Local browser automation with explicit network boundaries.

The runtime imports Python Playwright only when browser automation is used.
This keeps the base local-first install dependency-free while making browser
automation an opt-in capability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from ..egress import EgressError, EgressPolicy


class BrowserRuntimeError(RuntimeError):
    """Raised when browser automation cannot run safely or locally."""


class BrowserPolicy:
    """Validate browser destinations before a browser process is started."""

    def __init__(self, allowed_domains: set[str] | None = None) -> None:
        self.allowed_domains = {domain.lower().strip() for domain in (allowed_domains or set()) if domain.strip()}
        self.egress = EgressPolicy()

    def authorize(self, url: str, *, approved: bool = False) -> str:
        try:
            host = urlparse(url).hostname or ""
            domain_approved = host.casefold() in self.allowed_domains
            decision = self.egress.authorize(
                url, approved=approved or domain_approved, allow_loopback=True
            )
        except EgressError as exc:
            raise BrowserRuntimeError(str(exc)) from exc
        if self.allowed_domains and not decision.local and decision.hostname not in self.allowed_domains and not approved:
            raise PermissionError(f"Browser navigation to '{decision.hostname}' is outside the configured allowlist.")
        return decision.hostname


class BrowserRuntime:
    """Execute a short browser action sequence and retain its local evidence."""

    def __init__(self, data_dir: Path | str, *, policy: BrowserPolicy | None = None) -> None:
        self.artifact_dir = Path(data_dir).expanduser().resolve() / "browser-artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy or BrowserPolicy()

    def run(
        self,
        url: str,
        *,
        actions: list[dict[str, Any]] | None = None,
        approved: bool = False,
        timeout_seconds: int = 120,
    ) -> dict[str, Any]:
        hostname = self.policy.authorize(url, approved=approved)
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserRuntimeError(
                "Browser automation is optional. Install it with `pip install openagent[browser]` and `playwright install chromium`."
            ) from exc

        run_id = uuid4().hex
        screenshot_path = self.artifact_dir / f"{run_id}.png"
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    console_errors: list[str] = []
                    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
                    page.on("pageerror", lambda error: console_errors.append(str(error)))
                    def authorize_request(route: Any) -> None:
                        request_url = str(route.request.url)
                        scheme = urlparse(request_url).scheme.casefold()
                        if scheme in {"data", "blob", "about"}:
                            route.continue_()
                            return
                        try:
                            self.policy.authorize(request_url, approved=approved)
                        except (BrowserRuntimeError, PermissionError):
                            route.abort("blockedbyclient")
                            return
                        route.continue_()

                    page.route("**/*", authorize_request)
                    page.goto(url, wait_until="domcontentloaded", timeout=max(1, int(timeout_seconds)) * 1000)
                    evidence = [{"action": "navigate", "url": page.url, "title": page.title()}]
                    for action in list(actions or [])[:30]:
                        self._apply_action(page, action)
                        evidence.append({"action": str(action.get("action", "")), "url": page.url})
                    page.screenshot(path=str(screenshot_path), full_page=True)
                    browser_evidence = {
                        "url": page.url,
                        "title": page.title(),
                        "console_errors": console_errors,
                        "evidence": evidence,
                    }
                finally:
                    browser.close()
        except BrowserRuntimeError:
            raise
        except Exception as exc:
            raise BrowserRuntimeError(str(exc) or "Browser automation failed.") from exc
        return {
            "run_id": run_id,
            "hostname": hostname,
            "approved": bool(approved),
            "browser": browser_evidence,
            "artifact": str(screenshot_path) if screenshot_path.is_file() else "",
        }

    @staticmethod
    def _apply_action(page: Any, action: dict[str, Any]) -> None:
        kind = str(action.get("action", "")).strip()
        if kind == "wait":
            page.wait_for_timeout(max(0, int(action.get("ms", 250))))
            return
        if action.get("test_id"):
            locator = page.get_by_test_id(str(action["test_id"]))
        elif action.get("role"):
            locator = page.get_by_role(str(action["role"]), name=action.get("name"))
        elif action.get("selector"):
            locator = page.locator(str(action["selector"]))
        else:
            raise BrowserRuntimeError("Browser actions need selector, test_id, or role.")
        if kind == "click":
            locator.click()
        elif kind == "fill":
            locator.fill(str(action.get("value", "")))
        elif kind == "press":
            locator.press(str(action.get("key", "Enter")))
        else:
            raise BrowserRuntimeError(f"Unsupported browser action '{kind}'.")
