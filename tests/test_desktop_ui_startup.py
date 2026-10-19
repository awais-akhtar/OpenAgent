"""Regressions for local desktop UI asset validation and cold startup."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from openagent.desktop import build_parser as build_desktop_parser
from openagent.workbench import (
    DesktopAssetError,
    _wait_for_ui_server,
    build_parser as build_workbench_parser,
    launch_workbench,
    validate_desktop_ui,
)
from openagent.config import web_root


class DesktopUIStartupTests(unittest.TestCase):
    def test_default_source_ui_is_frontend_second_production_build(self) -> None:
        selected = web_root().resolve()
        self.assertEqual(selected, (Path(__file__).resolve().parents[1] / "frontend-second" / "dist").resolve())
        self.assertTrue((selected / "index.html").is_file())

    def test_missing_ui_fails_with_build_instructions(self) -> None:
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(DesktopAssetError, "missing index.html"):
                validate_desktop_ui(Path(folder))

    def test_rejects_missing_frontend_chunks(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "index.html").write_text(
                '<script type="module" src="/app.js"></script><link rel="stylesheet" href="/styles.css">',
                encoding="utf-8",
            )
            (root / "app.js").write_text("// app", encoding="utf-8")
            with self.assertRaisesRegex(DesktopAssetError, "styles.css"):
                validate_desktop_ui(root)

    def test_accepts_complete_hashed_ui_and_ignores_remote_resources(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            assets = root / "assets"
            assets.mkdir()
            (root / "index.html").write_text(
                '<script type="module" src="/assets/app-abc.js"></script>'
                '<link rel="modulepreload" href="/assets/react.js">'
                '<link rel="stylesheet" href="/assets/style-abc.css">'
                '<script src="https://example.com/nope.js"></script>',
                encoding="utf-8",
            )
            for filename in ("app-abc.js", "react.js", "style-abc.css"):
                (assets / filename).write_text("// asset", encoding="utf-8")
            self.assertEqual(len(validate_desktop_ui(root)), 3)

    def test_rejects_uncompiled_react_source(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "src"
            source.mkdir()
            (root / "index.html").write_text(
                '<script type="module" src="/src/main.tsx"></script>', encoding="utf-8"
            )
            (source / "main.tsx").write_text("// source", encoding="utf-8")
            with self.assertRaisesRegex(DesktopAssetError, "uncompiled React source"):
                validate_desktop_ui(root)

    def test_rejects_resources_that_escape_the_selected_ui_root(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "index.html").write_text('<script src="/../not-in-ui.js"></script>', encoding="utf-8")
            with self.assertRaises(DesktopAssetError):
                validate_desktop_ui(root)

    def test_server_readiness_checks_html_health_and_each_asset(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in {"/", "/api/health", "/app.js", "/styles.css"}:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")
                else:
                    self.send_error(404)

            def log_message(self, *_args: object) -> None:
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                _wait_for_ui_server(
                    f"http://127.0.0.1:{server.server_port}",
                    ["app.js", "styles.css"],
                    timeout_seconds=2,
                )
            finally:
                server.shutdown()
                worker.join(timeout=2)

    def test_isolated_ui_root_flag_is_available_on_both_launchers(self) -> None:
        self.assertEqual(build_workbench_parser().parse_args(["--ui-root", "frontend-second/dist"]).ui_root, "frontend-second/dist")
        self.assertEqual(build_desktop_parser().parse_args(["--ui-root", "frontend-second/dist"]).ui_root, "frontend-second/dist")

    def test_explicit_ui_root_never_reuses_a_running_default_instance(self) -> None:
        with TemporaryDirectory() as folder, \
             patch("openagent.workbench.WorkbenchInstanceLock.acquire", return_value=False), \
             patch("openagent.workbench.open_app_window") as open_window:
            with self.assertRaisesRegex(DesktopAssetError, "explicit --ui-root was not applied"):
                launch_workbench(folder, ui_root=Path(folder) / "dist")
            open_window.assert_not_called()

    def test_explicit_ui_root_is_loopback_only(self) -> None:
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(DesktopAssetError, "loopback"):
                launch_workbench(folder, host="0.0.0.0", ui_root=Path(folder) / "dist")


if __name__ == "__main__":
    unittest.main()
