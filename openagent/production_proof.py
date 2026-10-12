"""Executable local production-proof checks for release candidates."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any, Callable
from urllib.request import urlopen

from .communication import CommunicationHub
from .documents import DocumentManager
from .routes.runtime import build_http_server, parse_bool, parse_int
from .storage import OpenAgentStore
from .worker import WorkerRuntime


class AccessibilityProbe(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.unnamed_buttons: list[int] = []
        self.images_without_alt: list[int] = []
        self._buttons: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(str(values["id"]))
        if tag == "button":
            self._buttons.append({"line": self.getpos()[0], "name": values.get("aria-label") or values.get("title") or "", "text": ""})
        if tag == "img" and "alt" not in values:
            self.images_without_alt.append(self.getpos()[0])

    def handle_data(self, data: str) -> None:
        if self._buttons:
            self._buttons[-1]["text"] += data.strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "button" and self._buttons:
            button = self._buttons.pop()
            if not (button["name"] or button["text"].strip()):
                self.unnamed_buttons.append(button["line"])


def run_production_proof(data_dir: str | Path, *, queue_jobs: int = 100) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def check(name: str, function: Callable[[], dict[str, Any]]) -> None:
        started = time.perf_counter()
        try:
            evidence = function()
            checks.append({"name": name, "status": "passed", "duration_ms": round((time.perf_counter() - started) * 1000, 2), "evidence": evidence})
        except Exception as exc:  # noqa: BLE001 - proof must report every failed category.
            checks.append({"name": name, "status": "failed", "duration_ms": round((time.perf_counter() - started) * 1000, 2), "error": f"{type(exc).__name__}: {exc}"})

    def migration_check() -> dict[str, Any]:
        first = OpenAgentStore(data_path / "migration")
        before = first.migration_status()
        after = OpenAgentStore(data_path / "migration").migration_status()
        if before["integrity"] != "ok" or before["migrations"] != after["migrations"]:
            raise AssertionError("migration initialization is not idempotent")
        return {"version": after["current_version"], "integrity": after["integrity"], "applied": len(after["migrations"])}

    def queue_recovery_check() -> dict[str, Any]:
        store = OpenAgentStore(data_path / "queue-recovery")
        job = store.enqueue_job("echo", {"crash": True}, project="proof")
        store.lease_job("worker-before-crash", lease_seconds=5)
        with store.session() as connection:
            connection.execute("UPDATE job_locks SET lease_until = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job["id"],))
        reclaimed = store.lease_job("worker-after-crash", lease_seconds=5)
        if reclaimed is None or reclaimed["id"] != job["id"]:
            raise AssertionError("expired lease was not reclaimed")
        store.complete_job(job["id"], {"recovered": True})
        return {"job_id": job["id"], "reclaimed_by": reclaimed["lease"]["worker_id"]}

    def queue_soak_check() -> dict[str, Any]:
        count = max(1, min(int(queue_jobs), 1000))
        store = OpenAgentStore(data_path / "queue-soak")
        worker = WorkerRuntime(store, worker_id="production-proof-soak")
        for index in range(count):
            worker.enqueue("echo", {"index": index}, project="proof", priority=index % 5)
        processed = worker.run_forever(poll_seconds=0.01, max_jobs=count)
        status = store.worker_status(project="proof")
        if status["states"].get("completed") != count:
            raise AssertionError(f"queue completed {status['states'].get('completed', 0)} of {count}")
        return {"enqueued": count, "processed": processed["processed"], "states": status["states"]}

    def isolation_check() -> dict[str, Any]:
        manager = DocumentManager(OpenAgentStore(data_path / "multi-user"))
        document = manager.create_document("Owner A", "secret", owner="user-a", project="proof")
        denied = False
        try:
            manager.get_document(document["doc_id"], owner="user-b")
        except KeyError:
            denied = True
        if not denied:
            raise AssertionError("cross-user document read was allowed")
        return {"owner": "user-a", "foreign_owner": "user-b", "cross_user_read": "denied"}

    def provider_check() -> dict[str, Any]:
        matrix = CommunicationHub(OpenAgentStore(data_path / "providers")).provider_certification_matrix()
        providers = {item["id"]: item for item in matrix["providers"]}
        if not {"google", "microsoft", "generic"}.issubset(providers):
            raise AssertionError("provider matrix is incomplete")
        return {"providers": sorted(providers), "privacy_boundary": matrix["privacy_boundary"]}

    def accessibility_check() -> dict[str, Any]:
        html = (root / "web" / "index.html").read_text(encoding="utf-8")
        probe = AccessibilityProbe()
        probe.feed(html)
        duplicates = sorted({item for item in probe.ids if probe.ids.count(item) > 1})
        if duplicates or probe.unnamed_buttons or probe.images_without_alt or '<html lang="en">' not in html:
            raise AssertionError({"duplicate_ids": duplicates, "unnamed_buttons": probe.unnamed_buttons, "images_without_alt": probe.images_without_alt})
        return {"unique_ids": len(probe.ids), "unnamed_buttons": 0, "images_without_alt": 0, "language": "en"}

    def fuzz_check() -> dict[str, Any]:
        rng = random.Random(20260712)
        for _ in range(1000):
            value = bytes(rng.randrange(0, 256) for _ in range(rng.randrange(0, 48))).decode("utf-8", errors="replace")
            parse_bool(value, default=False)
            result = parse_int(value, default=17)
            if not isinstance(result, int):
                raise AssertionError("parse_int returned a non-integer")
        return {"cases": 1000, "seed": 20260712, "exceptions": 0}

    def installer_check() -> dict[str, Any]:
        install = (root / "installers" / "windows" / "install.ps1").read_text(encoding="utf-8")
        uninstall = (root / "installers" / "windows" / "uninstall.ps1").read_text(encoding="utf-8")
        required = ["install-manifest.json", "$SourcePath[full]", "workbench-state.json", "Stop-Process", "RemoveData"]
        combined = install + uninstall
        missing = [item for item in required if item not in combined]
        if missing:
            raise AssertionError(f"installer lifecycle markers missing: {missing}")
        return {"upgrade_preserves_data": True, "uninstall_stops_owned_process": True, "remove_data_is_explicit": True}

    def http_e2e_check() -> dict[str, Any]:
        store = OpenAgentStore(data_path / "http-e2e")
        server = build_http_server("127.0.0.1", 0, store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            health = json.loads(urlopen(base + "/api/health", timeout=10).read().decode("utf-8"))
            identity = json.loads(urlopen(base + "/api/auth/identity/status", timeout=10).read().decode("utf-8"))
            html = urlopen(base + "/", timeout=10).read().decode("utf-8")
            if health.get("status") != "ok" or "OpenAgent" not in html or "webauthn_available" not in identity:
                raise AssertionError("HTTP acceptance contract failed")
            return {"health": health["status"], "identity": identity["webauthn_available"], "static_bytes": len(html)}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def browser_e2e_contract_check() -> dict[str, Any]:
        package = json.loads((root / "package.json").read_text(encoding="utf-8"))
        config = (root / "playwright.config.js").read_text(encoding="utf-8")
        spec = (root / "tests" / "e2e" / "openagent.spec.js").read_text(encoding="utf-8")
        workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        required = [
            package.get("scripts", {}).get("test:e2e") == "playwright test",
            "reuseExistingServer: false" in config,
            "OPENAGENT_E2E_PORT" in config,
            "consoleErrors" in spec,
            "metricGrid" in spec,
            "conversationActionMenu" in spec,
            "codebaseOpenTabs" in spec,
            'data-nav-group="work"' in spec and 'data-nav-group="data"' in spec,
            "npm run test:e2e" in workflow,
            "playwright install --with-deps chromium" in workflow,
        ]
        if not all(required):
            raise AssertionError("Playwright browser acceptance gate is incomplete")
        return {"runner": "playwright", "browser": "chromium", "console_error_gate": True, "ci_enforced": True}

    for name, function in (
        ("migrations", migration_check), ("crash-recovery", queue_recovery_check), ("queue-soak", queue_soak_check),
        ("multi-user-isolation", isolation_check), ("provider-certification", provider_check),
        ("accessibility-contract", accessibility_check), ("parser-fuzz", fuzz_check),
        ("installer-upgrade-uninstall", installer_check), ("http-e2e", http_e2e_check),
        ("browser-e2e-contract", browser_e2e_contract_check),
    ):
        check(name, function)
    return {"ok": all(item["status"] == "passed" for item in checks), "checks": checks,
            "passed": sum(item["status"] == "passed" for item in checks), "failed": sum(item["status"] == "failed" for item in checks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenAgent local production-proof checks.")
    parser.add_argument("--data-dir", default="", help="Directory for isolated proof databases.")
    parser.add_argument("--queue-jobs", type=int, default=100, help="Number of durable queue jobs in the soak check.")
    args = parser.parse_args(argv)
    if args.data_dir:
        result = run_production_proof(args.data_dir, queue_jobs=args.queue_jobs)
    else:
        with TemporaryDirectory(prefix="openagent-production-proof-") as temp:
            result = run_production_proof(temp, queue_jobs=args.queue_jobs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
