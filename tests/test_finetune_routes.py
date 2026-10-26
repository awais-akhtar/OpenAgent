from __future__ import annotations

import json
import threading
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock
from urllib.request import Request, urlopen

from openagent.routes.finetune_routes import FineTuneRouteDependencies, contract, handle_get, handle_post
from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore


class FineTuneRouteTests(TestCase):
    def test_http_dataset_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                request = Request(
                    f"{base}/api/finetune/datasets",
                    data=json.dumps({"project": "core", "name": "Example dataset"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlopen(request, timeout=10) as response:
                    self.assertEqual(response.status, HTTPStatus.CREATED)
                    created = json.load(response)["dataset"]
                with urlopen(f"{base}/api/finetune/datasets?project=core", timeout=10) as response:
                    listed = json.load(response)["datasets"]
                self.assertIn(created["id"], {item["id"] for item in listed})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_http_job_creation_cancellation_and_persistence(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                def post(path: str, payload: dict) -> tuple[int, dict]:
                    request = Request(
                        f"{base}{path}", data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"}, method="POST",
                    )
                    with urlopen(request, timeout=10) as response:
                        return response.status, json.load(response)

                _, dataset_response = post("/api/finetune/datasets", {
                    "project": "release-check", "name": "Local smoke dataset",
                })
                dataset_id = dataset_response["dataset"]["id"]
                status, job_response = post("/api/finetune/jobs", {
                    "project": "release-check", "dataset_id": dataset_id,
                    "base_model": "local-test-model", "method": "lora",
                })
                self.assertEqual(status, HTTPStatus.CREATED)
                job_id = job_response["job"]["id"]
                self.assertEqual(job_response["job"]["status"], "queued")

                status, cancel_response = post(f"/api/finetune/jobs/{job_id}/cancel", {
                    "project": "release-check",
                })
                self.assertEqual(status, HTTPStatus.OK)
                self.assertEqual(cancel_response["job"]["status"], "cancelled")
                with urlopen(f"{base}/api/finetune/jobs?project=release-check", timeout=10) as response:
                    jobs = json.load(response)["jobs"]
                self.assertEqual(next(job for job in jobs if job["id"] == job_id)["status"], "cancelled")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_runtime_has_no_inline_finetune_dispatch(self) -> None:
        runtime = (Path(__file__).resolve().parents[1] / "openagent" / "routes" / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn('if parsed.path == "/api/finetune/', runtime)

    def dependencies(self, payload: dict | None = None) -> tuple[Mock, Mock, FineTuneRouteDependencies]:
        manager = Mock()
        read_json = Mock(return_value={} if payload is None else payload)
        return manager, read_json, FineTuneRouteDependencies(manager, read_json)

    def test_contract_covers_every_declared_get_and_post(self) -> None:
        manager, read_json, deps = self.dependencies()
        routes = contract()["routes"]
        self.assertEqual(contract()["route_count"], 16)
        self.assertEqual({route["method"] for route in routes}, {"GET", "POST"})
        for route in routes:
            with self.subTest(route=route):
                result = (
                    handle_get(route["path"], "", deps)
                    if route["method"] == "GET"
                    else handle_post(route["path"], deps)
                )
                self.assertIsNotNone(result)
        read_json.reset_mock()
        self.assertIsNone(handle_get("/api/other", "", deps))
        self.assertIsNone(handle_post("/api/other", deps))
        read_json.assert_not_called()

    def test_get_routes_preserve_project_and_filter_contract(self) -> None:
        manager, read_json, deps = self.dependencies()
        manager.status.return_value = {"datasets": 1}
        manager.list_datasets.return_value = [{"id": "ds"}]
        manager.list_jobs.return_value = [{"id": "job"}]
        manager.list_adapters.return_value = [{"id": "adapter"}]
        manager.list_templates.return_value = [{"id": "template"}]

        result = handle_get("/api/finetune/status", "project=research", deps)
        self.assertEqual((result.payload, result.status), ({"datasets": 1}, HTTPStatus.OK))
        manager.status.assert_called_once_with(project="research")

        result = handle_get("/api/finetune/datasets", "", deps)
        self.assertEqual((result.payload, result.status), ({"datasets": [{"id": "ds"}]}, HTTPStatus.OK))
        manager.list_datasets.assert_called_once_with(project="default")

        result = handle_get("/api/finetune/jobs", "project=research", deps)
        self.assertEqual((result.payload, result.status), ({"jobs": [{"id": "job"}]}, HTTPStatus.OK))
        manager.list_jobs.assert_called_once_with(project="research")

        result = handle_get("/api/finetune/adapters", "project=research&base_model=qwen%2F3b", deps)
        self.assertEqual((result.payload, result.status), ({"adapters": [{"id": "adapter"}]}, HTTPStatus.OK))
        manager.list_adapters.assert_called_once_with(project="research", base_model="qwen/3b")

        result = handle_get("/api/finetune/templates", "project=ignored", deps)
        self.assertEqual((result.payload, result.status), ({"templates": [{"id": "template"}]}, HTTPStatus.OK))
        manager.list_templates.assert_called_once_with()
        read_json.assert_not_called()

    def test_create_dataset_preserves_values_and_validation_status(self) -> None:
        manager, read_json, deps = self.dependencies({
            "name": "  Training  ", "project": "  research  ",
            "description": 17, "format": "chatml",
        })
        manager.create_dataset.return_value = {"id": "ds"}
        result = handle_post("/api/finetune/datasets", deps)
        self.assertEqual((result.payload, result.status), ({"dataset": {"id": "ds"}}, HTTPStatus.CREATED))
        manager.create_dataset.assert_called_once_with(
            "Training", project="research", description="17", format="chatml",
        )
        read_json.assert_called_once_with()

        manager, _, deps = self.dependencies({"name": "  ", "project": "  "})
        manager.create_dataset.side_effect = ValueError("Dataset name is required.")
        result = handle_post("/api/finetune/datasets", deps)
        self.assertEqual(
            (result.payload, result.status),
            ({"error": "Dataset name is required."}, HTTPStatus.BAD_REQUEST),
        )
        manager.create_dataset.assert_called_once_with(
            "", project="default", description="", format="jsonl",
        )

    def test_add_example_requires_dataset_and_converts_quality_score(self) -> None:
        manager, _, deps = self.dependencies({"dataset_id": "  "})
        result = handle_post("/api/finetune/examples", deps)
        self.assertEqual((result.payload, result.status), ({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST))
        manager.add_example.assert_not_called()

        manager, _, deps = self.dependencies({
            "dataset_id": " ds ", "user_input": " input ", "assistant_output": " answer ",
            "project": " science ", "system_prompt": 3, "quality_score": "0.75",
        })
        manager.add_example.return_value = {"id": "ex"}
        result = handle_post("/api/finetune/examples", deps)
        self.assertEqual((result.payload, result.status), ({"example": {"id": "ex"}}, HTTPStatus.CREATED))
        manager.add_example.assert_called_once_with(
            "ds", "input", "answer", project="science", system_prompt="3", quality_score=0.75,
        )

        manager, _, deps = self.dependencies({"dataset_id": "ds", "quality_score": "bad"})
        result = handle_post("/api/finetune/examples", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("could not convert string to float", result.payload["error"])
        manager.add_example.assert_not_called()

        manager, _, deps = self.dependencies({"dataset_id": "missing"})
        manager.add_example.side_effect = KeyError("Dataset not found: missing")
        result = handle_post("/api/finetune/examples", deps)
        self.assertEqual((result.payload, result.status),
                         ({"error": "'Dataset not found: missing'"}, HTTPStatus.BAD_REQUEST))

    def test_create_job_preserves_config_and_value_errors(self) -> None:
        manager, _, deps = self.dependencies({
            "dataset_id": " ds ", "base_model": " qwen ", "project": " research ",
            "method": "qlora", "config": {"epochs": 2}, "template_id": 9,
        })
        manager.create_job.return_value = {"id": "job"}
        result = handle_post("/api/finetune/jobs", deps)
        self.assertEqual((result.payload, result.status), ({"job": {"id": "job"}}, HTTPStatus.CREATED))
        manager.create_job.assert_called_once_with(
            "ds", "qwen", project="research", method="qlora",
            config={"epochs": 2}, template_id="9",
        )

        manager, _, deps = self.dependencies({"config": ["ignored"]})
        manager.create_job.side_effect = ValueError("base_model is required.")
        result = handle_post("/api/finetune/jobs", deps)
        self.assertEqual((result.payload, result.status),
                         ({"error": "base_model is required."}, HTTPStatus.BAD_REQUEST))
        manager.create_job.assert_called_once_with(
            "", "", project="default", method="lora", config=None, template_id="",
        )

        manager, _, deps = self.dependencies({"dataset_id": "private-dataset", "base_model": "local", "project": "other"})
        manager.create_job.side_effect = KeyError("Dataset not found in project: private-dataset")
        result = handle_post("/api/finetune/jobs", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("Dataset not found in project", result.payload["error"])

    def test_cancel_job_is_project_scoped_and_reports_conflict(self) -> None:
        manager, _, deps = self.dependencies({"project": "research"})
        manager.cancel_queued_job.return_value = {"id": "job-1", "status": "cancelled"}
        result = handle_post("/api/finetune/jobs/job-1/cancel", deps)
        self.assertEqual((result.payload, result.status),
                         ({"job": {"id": "job-1", "status": "cancelled"}}, HTTPStatus.OK))
        manager.cancel_queued_job.assert_called_once_with("job-1", project="research")

        manager, _, deps = self.dependencies({"project": "research"})
        manager.cancel_queued_job.return_value = {"id": "job-1", "status": "cancelling"}
        result = handle_post("/api/finetune/jobs/job-1/cancel", deps)
        self.assertEqual((result.payload, result.status),
                         ({"job": {"id": "job-1", "status": "cancelling"}}, HTTPStatus.ACCEPTED))

        manager, _, deps = self.dependencies({"project": "other"})
        manager.cancel_queued_job.side_effect = KeyError("hidden cross-project job")
        result = handle_post("/api/finetune/jobs/job-1/cancel", deps)
        self.assertEqual((result.payload, result.status),
                         ({"error": "Job not found"}, HTTPStatus.NOT_FOUND))

        manager, _, deps = self.dependencies()
        manager.cancel_queued_job.side_effect = RuntimeError("Only queued jobs can be cancelled")
        result = handle_post("/api/finetune/jobs/job-1/cancel", deps)
        self.assertEqual(result.status, HTTPStatus.CONFLICT)

    def test_register_adapter_preserves_numeric_and_module_conversions(self) -> None:
        manager, _, deps = self.dependencies({
            "name": " adapter ", "base_model": " phi ", "project": " science ",
            "description": 3, "rank": "8", "alpha": "12.5", "target_modules": ["q_proj", 2],
        })
        manager.register_adapter.return_value = {"id": "adapter"}
        result = handle_post("/api/finetune/adapters", deps)
        self.assertEqual((result.payload, result.status), ({"adapter": {"id": "adapter"}}, HTTPStatus.CREATED))
        manager.register_adapter.assert_called_once_with(
            "adapter", "phi", project="science", description="3", rank=8,
            alpha=12.5, target_modules=["q_proj", "2"],
        )

        manager, _, deps = self.dependencies({"rank": "bad", "target_modules": "ignored"})
        result = handle_post("/api/finetune/adapters", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("invalid literal for int()", result.payload["error"])
        manager.register_adapter.assert_not_called()

        manager, _, deps = self.dependencies({"target_modules": "ignored"})
        manager.register_adapter.side_effect = ValueError("Adapter name is required.")
        result = handle_post("/api/finetune/adapters", deps)
        self.assertEqual((result.payload, result.status),
                         ({"error": "Adapter name is required."}, HTTPStatus.BAD_REQUEST))
        manager.register_adapter.assert_called_once_with(
            "", "", project="default", description="", rank=16, alpha=16.0, target_modules=None,
        )

    def test_activate_adapter_uses_ok_not_created_and_missing_is_not_found(self) -> None:
        manager, _, deps = self.dependencies({"adapter_id": " "})
        result = handle_post("/api/finetune/adapters/activate", deps)
        self.assertEqual((result.payload, result.status),
                         ({"error": "adapter_id is required"}, HTTPStatus.BAD_REQUEST))
        manager.activate_adapter.assert_not_called()

        manager, _, deps = self.dependencies({"adapter_id": " a1 ", "project": " science "})
        manager.activate_adapter.return_value = {"id": "a1", "active": True}
        result = handle_post("/api/finetune/adapters/activate", deps)
        self.assertEqual((result.payload, result.status),
                         ({"adapter": {"id": "a1", "active": True}}, HTTPStatus.OK))
        manager.activate_adapter.assert_called_once_with("a1", project="science")

        manager.activate_adapter.side_effect = KeyError("Adapter not found: a1")
        result = handle_post("/api/finetune/adapters/activate", deps)
        self.assertEqual((result.payload, result.status),
                         ({"error": "'Adapter not found: a1'"}, HTTPStatus.NOT_FOUND))

    def test_train_job_success(self) -> None:
        manager, _, deps = self.dependencies({
            "project": "default",
            "output_dir": "/models/output",
            "checkpoint_interval": 10,
        })
        manager.start_model_training.return_value = {
            "started": True,
            "job_id": "job-1",
            "status": "running",
            "output_dir": "/models/output",
        }
        result = handle_post("/api/finetune/jobs/job-1/train", deps)
        self.assertEqual(result.status, HTTPStatus.ACCEPTED)
        self.assertIn("training", result.payload)
        manager.start_model_training.assert_called_once()
        args, kwargs = manager.start_model_training.call_args
        self.assertEqual(args[0], "job-1")
        self.assertEqual(kwargs.get("project"), "default")
        self.assertEqual(kwargs.get("checkpoint_interval"), 10)

    def test_train_job_not_found(self) -> None:
        manager, _, deps = self.dependencies({"project": "default"})
        manager.start_model_training.side_effect = KeyError("Job not found")
        result = handle_post("/api/finetune/jobs/missing/train", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_train_job_conflict(self) -> None:
        manager, _, deps = self.dependencies({"project": "default"})
        manager.start_model_training.side_effect = RuntimeError("Job is not in a runnable state")
        result = handle_post("/api/finetune/jobs/job-1/train", deps)
        self.assertEqual(result.status, HTTPStatus.CONFLICT)

    def test_train_job_missing_id(self) -> None:
        manager, _, deps = self.dependencies({"project": "default"})
        result = handle_post("/api/finetune/jobs//train", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_train_job_with_slash_in_id(self) -> None:
        manager, _, deps = self.dependencies({"project": "default"})
        result = handle_post("/api/finetune/jobs/a/b/train", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
