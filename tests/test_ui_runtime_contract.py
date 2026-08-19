from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

from openagent.routes import route_contracts


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
APP = (ROOT / "web" / "app.js").read_text(encoding="utf-8")


class _ButtonParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.buttons: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "button":
            self.buttons.append({key: value or "" for key, value in attrs})


def _button_is_wired(attrs: dict[str, str]) -> bool:
    element_id = attrs.get("id", "")
    if element_id:
        selector = re.escape(f"#{element_id}")
        if re.search(rf"qs\([\"']{selector}[\"']\)\?\.addEventListener", APP):
            return True
        if re.search(rf"qsa\([\"'][^\"']*{selector}[^\"']*[\"']\)", APP):
            return True
        variable = re.search(
            rf"(?:const|let)\s+(\w+)\s*=\s*qs\([\"']{selector}[\"']\)[\s\S]{{0,600}}?\1\.addEventListener",
            APP,
        )
        if variable:
            return True
    inline = attrs.get("onclick", "")
    if inline:
        function_name = inline.split("(", 1)[0].strip()
        return bool(
            re.search(
                rf"(?:(?:async\s+)?function\s+{re.escape(function_name)}\b|const\s+{re.escape(function_name)}\s*=|\b{re.escape(function_name)},)",
                APP,
            )
        )
    return any(
        key in attrs
        for key in (
            "data-view-link", "data-view-jump", "data-command", "data-theme-select", "data-nav-group",
            "data-dashboard-window", "data-provider-filter", "data-incident-filter",
            "data-codebase-file", "data-codebase-approval", "data-codebase-detail", "data-codebase-output",
            "data-developer-section",
        )
    )


def _frontend_api_paths() -> set[str]:
    paths: set[str] = set()
    for match in re.finditer(r"\bapi\(\s*([\"'`])(/api/.*?)(?<!\\)\1", APP, re.DOTALL):
        raw = match.group(2).split("?", 1)[0]
        normalized = re.sub(r"\$\{[^}]+\}", "{value}", raw)
        paths.add(normalized.rstrip("/") or "/")
    return paths


def _route_matches(frontend: str, backend: str) -> bool:
    frontend_parts = frontend.strip("/").split("/")
    backend_parts = backend.strip("/").split("/")
    if len(frontend_parts) != len(backend_parts):
        return False
    return all(
        left == right or left.startswith("{") or right.startswith("{")
        for left, right in zip(frontend_parts, backend_parts, strict=True)
    )


class UIRuntimeContractTests(unittest.TestCase):
    def test_every_static_button_has_a_handler(self) -> None:
        parser = _ButtonParser()
        parser.feed(INDEX)
        missing = [attrs for attrs in parser.buttons if not _button_is_wired(attrs)]
        self.assertEqual([], missing, f"Visible buttons without UI wiring: {missing}")

    def test_every_literal_frontend_api_path_has_a_backend_contract(self) -> None:
        backend_paths = {
            str(route["path"]).rstrip("/") or "/"
            for contract in route_contracts()
            for route in contract["routes"]
        }
        missing = sorted(
            path for path in _frontend_api_paths()
            if not any(_route_matches(path, backend) for backend in backend_paths)
        )
        self.assertEqual([], missing, f"Frontend API paths without backend route contracts: {missing}")


if __name__ == "__main__":
    unittest.main()
