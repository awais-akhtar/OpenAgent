"""Safety regression for the installed desktop launcher startup assets."""
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from openagent.workbench import DesktopAssetError, validate_desktop_ui, _wait_for_ui_server

class DesktopUIStartupTests(unittest.TestCase):
    def test_missing_ui_is_a_clear_error(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(DesktopAssetError, 'missing index.html'):
                validate_desktop_ui(Path(folder))

    def test_rejects_missing_frontend_chunks(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'index.html').write_text('<script type="module" src="/app.js"></script><link rel="stylesheet" href="/styles.css">')
            (root / 'app.js').write_text('// app')
            with self.assertRaisesRegex(DesktopAssetError, 'styles.css'):
                validate_desktop_ui(root)

    def test_accepts_complete_hashed_ui_and_ignores_remote_resources(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); (root / 'assets').mkdir()
            (root / 'index.html').write_text('<script type="module" src="/assets/app-abc.js"></script><link rel="modulepreload" href="/assets/react.js"><link rel="stylesheet" href="/assets/style-abc.css"><script src="https://example.com/nope.js"></script>')
            for file in ['app-abc.js', 'react.js', 'style-abc.css']:
                (root / 'assets' / file).write_text('// asset')
            self.assertEqual(len(validate_desktop_ui(root)), 3)

    def test_rejects_uncompiled_react_source(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); (root / 'src').mkdir()
            (root / 'index.html').write_text('<script type="module" src="/src/main.tsx"></script>')
            (root / 'src' / 'main.tsx').write_text('// source')
            with self.assertRaisesRegex(DesktopAssetError, 'uncompiled React source'):
                validate_desktop_ui(root)

    def test_rejects_root_escape(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'index.html').write_text('<script src="/../not-in-ui.js"></script>')
            with self.assertRaises(DesktopAssetError):
                validate_desktop_ui(root)

    def test_server_readiness_checks_real_http_routes(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in {'/', '/api/health', '/app.js', '/styles.css'}:
                    self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
                else:
                    self.send_error(404)
            def log_message(self, *_args): pass
        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
            try:
                _wait_for_ui_server(f'http://127.0.0.1:{server.server_port}', ['app.js', 'styles.css'], timeout_seconds=2)
            finally:
                server.shutdown(); worker.join(timeout=2)

if __name__ == '__main__': unittest.main()
