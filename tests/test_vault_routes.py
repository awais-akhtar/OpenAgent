from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.routes.vault_routes import VaultRouteDependencies, contract, handle_get, handle_post
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore
from openagent.vault import EncryptedVault


class VaultRouteTests(unittest.TestCase):
    def test_declared_routes_are_handled_and_preserve_metadata_only_listing(self) -> None:
        with TemporaryDirectory() as tmp:
            payload: dict[str, object] = {
                "project": "core", "label": "Local key", "secret": "value",
                "passphrase": "test-passphrase", "old_passphrase": "test-passphrase",
                "new_passphrase": "new-test-passphrase",
            }
            deps = VaultRouteDependencies(
                vault=EncryptedVault(OpenAgentStore(tmp)),
                first=lambda params, key: params.get(key, [None])[0],
                parse_int=lambda value, default=0: default if value is None else int(value),
                read_json=lambda: payload,
            )
            created = handle_post("/api/vault/entries", deps)
            self.assertEqual(created.status, HTTPStatus.CREATED)
            payload["entry_id"] = created.payload["entry"]["id"]
            listing = handle_get("/api/vault/entries", "project=core", deps).payload["entries"]
            self.assertEqual(listing[0]["id"], payload["entry_id"])
            self.assertNotIn("secret", listing[0])
            self.assertNotIn("ciphertext", listing[0])
            access = handle_get("/api/vault/access-log", "project=core&limit=10", deps).payload["access_log"]
            self.assertEqual(access[0]["entry_id"], payload["entry_id"])
            self.assertEqual(access[0]["action"], "store")
            self.assertNotIn("secret", access[0])
            self.assertNotIn("ciphertext", access[0])
            for route in contract()["routes"]:
                with self.subTest(route=route):
                    result = (
                        handle_get(route["path"], "project=core", deps)
                        if route["method"] == "GET"
                        else handle_post(route["path"], deps)
                    )
                    self.assertIsNotNone(result)
            self.assertIsNone(handle_get("/api/chat", "", deps))
            self.assertIsNone(handle_post("/api/chat", deps))

    def test_http_entries_alias_preserves_store_contract(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = Request(
                    f"{base}/api/vault/entries",
                    data=json.dumps({"project": "core", "label": "Key", "secret": "private", "passphrase": "test-passphrase"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlopen(request, timeout=20) as response:
                    self.assertEqual(response.status, HTTPStatus.CREATED)
                    created = json.load(response)
                with urlopen(f"{base}/api/vault/entries?project=core", timeout=10) as response:
                    listed = json.load(response)["entries"]
                self.assertEqual(listed[0]["id"], created["entry"]["id"])
                self.assertNotIn("secret", listed[0])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_runtime_has_no_inline_vault_dispatch(self) -> None:
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/vault/', runtime)


if __name__ == "__main__":
    unittest.main()
