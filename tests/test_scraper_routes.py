from __future__ import annotations

import json
import threading
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.routes.runtime import parse_bool
from openagent.routes.runtime import build_http_server
from openagent.routes.scraper_routes import ScraperRouteDependencies, contract, handle_get, handle_post
from openagent.storage import OpenAgentStore


class ScraperRouteTests(TestCase):
    def test_http_schedule_and_list_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = Request(
                    f"{base}/api/scraper/schedule",
                    data=json.dumps({"project": "core", "url": "https://example.org"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, HTTPStatus.CREATED)
                    created = json.load(response)["job"]
                with urlopen(f"{base}/api/scraper/jobs?project=core", timeout=10) as response:
                    jobs = json.load(response)["jobs"]
                self.assertIn(created["job_id"], {item["id"] for item in jobs})
                invalid = Request(
                    f"{base}/api/scraper/schedule", data=b"{}",
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with self.assertRaises(HTTPError) as error:
                    urlopen(invalid, timeout=10)
                error.exception.close()
                self.assertEqual(error.exception.code, HTTPStatus.BAD_REQUEST)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_runtime_has_no_inline_scraper_dispatch(self) -> None:
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/scraper/', runtime)

    def setUp(self) -> None:
        self.payload: dict[str, object] = {}
        self.scraper = Mock()
        self.scraper.status.return_value = {"project": "core", "scheduled_jobs": 1}
        self.scraper.list_scheduled.return_value = [{"id": "job-1"}]
        self.scraper.scrape_url.return_value = {"url": "https://example.com", "status": "ok"}
        self.scraper.schedule_scrape.return_value = {"job_id": "job-1"}
        self.scraper.run_due_jobs.return_value = {"ran": 0, "errors": 0, "total_due": 0}
        self.deps = ScraperRouteDependencies(
            scraper=self.scraper,
            first=lambda params, key: params.get(key, [None])[0],
            parse_bool=parse_bool,
            read_json=lambda: self.payload,
        )

    def test_every_declared_path_has_a_handler(self) -> None:
        self.payload = {"url": "https://example.com"}
        for route in contract()["routes"]:
            with self.subTest(route=route):
                result = (
                    handle_get(route["path"], "project=core", self.deps)
                    if route["method"] == "GET"
                    else handle_post(route["path"], self.deps)
                )
                self.assertIsNotNone(result)
        self.assertIsNone(handle_get("/api/chat", "", self.deps))
        self.assertIsNone(handle_post("/api/chat", self.deps))

    def test_get_preserves_project_default_and_response_shape(self) -> None:
        status = handle_get("/api/scraper/status", "project=core", self.deps)
        self.assertEqual(status.status, HTTPStatus.OK)
        self.assertEqual(status.payload, {"project": "core", "scheduled_jobs": 1})
        self.scraper.status.assert_called_once_with(project="core")

        jobs = handle_get("/api/scraper/jobs", "", self.deps)
        self.assertEqual(jobs.status, HTTPStatus.OK)
        self.assertEqual(jobs.payload, {"jobs": [{"id": "job-1"}]})
        self.scraper.list_scheduled.assert_called_once_with(project="default")

    def test_scrape_preserves_boolean_defaults_and_explicit_values(self) -> None:
        self.payload = {"url": " https://example.com ", "project": " core "}
        result = handle_post("/api/scraper/scrape", self.deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertEqual(result.payload["status"], "ok")
        self.scraper.scrape_url.assert_called_once_with(
            "https://example.com", project="core", respect_robots=True,
            ingest_to_knowledge=True, allow_external=False,
        )

        self.scraper.scrape_url.reset_mock()
        self.payload = {"url": "https://example.com", "respect_robots": "false", "ingest": False, "allow_external": "yes"}
        handle_post("/api/scraper/scrape", self.deps)
        self.scraper.scrape_url.assert_called_once_with(
            "https://example.com", project="default", respect_robots=False,
            ingest_to_knowledge=False, allow_external=True,
        )

    def test_schedule_preserves_created_contract_and_defaults(self) -> None:
        self.payload = {"url": " https://example.com ", "interval_hours": "12.5", "label": "Updates", "allow_external": True}
        result = handle_post("/api/scraper/schedule", self.deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        self.assertEqual(result.payload, {"job": {"job_id": "job-1"}})
        self.scraper.schedule_scrape.assert_called_once_with(
            "https://example.com", project="default", interval_hours=12.5,
            label="Updates", allow_external=True,
        )

    def test_run_due_preserves_project_default_and_result_shape(self) -> None:
        result = handle_post("/api/scraper/run-due", self.deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertEqual(result.payload, {"ran": 0, "errors": 0, "total_due": 0})
        self.scraper.run_due_jobs.assert_called_once_with(project="default")
        self.payload = {"project": " core "}
        handle_post("/api/scraper/run-due", self.deps)
        self.scraper.run_due_jobs.assert_called_with(project="core")

    def test_missing_url_and_scraper_errors_return_bad_request(self) -> None:
        for path in ("/api/scraper/scrape", "/api/scraper/schedule"):
            with self.subTest(path=path):
                self.payload = {"url": "  "}
                result = handle_post(path, self.deps)
                self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
                self.assertEqual(result.payload, {"error": "url is required"})
        self.scraper.scrape_url.assert_not_called()
        self.scraper.schedule_scrape.assert_not_called()

        self.payload = {"url": "https://example.com"}
        self.scraper.scrape_url.side_effect = PermissionError("approval required")
        self.scraper.schedule_scrape.side_effect = ValueError("invalid URL")
        self.assertEqual(handle_post("/api/scraper/scrape", self.deps).payload, {"error": "approval required"})
        self.assertEqual(handle_post("/api/scraper/schedule", self.deps).payload, {"error": "invalid URL"})
        self.assertEqual(handle_post("/api/scraper/scrape", self.deps).status, HTTPStatus.BAD_REQUEST)
        self.assertEqual(handle_post("/api/scraper/schedule", self.deps).status, HTTPStatus.BAD_REQUEST)

    def test_invalid_schedule_interval_returns_bad_request(self) -> None:
        self.payload = {"url": "https://example.com", "interval_hours": "not-a-number"}
        result = handle_post("/api/scraper/schedule", self.deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("could not convert string to float", result.payload["error"])
        self.scraper.schedule_scrape.assert_not_called()
