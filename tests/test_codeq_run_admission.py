"""CodeQ reports a truthful admission gate for the installed model policy."""

from __future__ import annotations

import unittest
import json
import threading
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openagent.auth import AuthManager
from openagent.auth import required_scope_for_path
from openagent.codebase.service import CodebaseService
from openagent.model_gateway import ProviderConfigurationError
from openagent.routes.codebase_routes import ROUTES, handle_get
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class CodeQRunAdmissionTests(unittest.TestCase):
    @staticmethod
    def response(policy: str, provider: str = "auto", model: str = ""):
        service = object.__new__(CodebaseService)
        selected: list[tuple[str, str]] = []
        def select_provider(selected_provider: str, **_kwargs):
            selected.append((selected_provider, model))
            return SimpleNamespace(
                id=selected_provider,
                default_model="qwen3-coder:30b",
                available=lambda: True,
                installed_models=lambda: ["qwen3-coder:30b"] if selected_provider == "ollama" else [],
            )
        service.gateway = SimpleNamespace(
            routing_policy=policy,
            select_provider=select_provider,
        )
        deps = SimpleNamespace(query_owner=lambda _params: "local", codebase=service)
        query = f"provider={provider}&model={model}" if model else f"provider={provider}"
        result = handle_get("/api/codebase/run-admission", query, deps)
        return result, selected

    def test_local_first_admits_run_subject_to_execution_checks(self) -> None:
        result, _selected = self.response("local-first")
        self.assertEqual(result.payload["autonomous_run"], "eligible")
        self.assertTrue(result.payload["manual_editor"])
        self.assertTrue(result.payload["read_only_questions"])
        self.assertIn("checked during execution", result.payload["reason"])

    def test_other_policy_is_only_eligible_not_promised_success(self) -> None:
        result, _selected = self.response("local-first")
        self.assertEqual(result.payload["autonomous_run"], "eligible")
        self.assertIn("checked during execution", result.payload["reason"])

    def test_selected_local_provider_is_admitted_independently_of_auto_policy(self) -> None:
        result, selected = self.response("local-first", provider="ollama", model="qwen3-coder:30b")
        self.assertEqual(result.payload["autonomous_run"], "eligible")
        self.assertEqual(selected, [("ollama", "qwen3-coder:30b")])

    def test_uninstalled_selected_model_is_blocked_before_run_creation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            service = CodebaseService(
                OpenAgentStore(root / "data"),
                SimpleNamespace(select_provider=lambda *_args, **_kwargs: SimpleNamespace(
                    id="ollama", default_model="phi4-mini:latest", available=lambda: True,
                    installed_models=lambda: ["phi4-mini:latest"],
                )),  # type: ignore[arg-type]
            )
            project = service.create_project(name="Admission guard", source_path=str(source))
            result = service.run_admission(provider="ollama", model="qwen3-coder:30b")
            self.assertEqual(result["autonomous_run"], "blocked")
            self.assertIn("not installed", result["reason"])
            with self.assertRaisesRegex(ValueError, "not installed"):
                service.create_run(codebase_project_id=project["id"], prompt="Build a project", provider="ollama", model="qwen3-coder:30b")
            self.assertEqual(service.list_runs(), [], "a blocked model must not leave a persisted run record")

    def test_unavailable_explicit_provider_is_blocked(self) -> None:
        service = object.__new__(CodebaseService)
        service.gateway = SimpleNamespace(
            select_provider=lambda *_args, **_kwargs: SimpleNamespace(id="ollama", available=lambda: False),
        )
        result = service.run_admission(provider="ollama", model="phi4-mini:latest")
        self.assertEqual(result["autonomous_run"], "blocked")
        self.assertIn("not available", result["reason"])

    def test_unavailable_local_model_blocks_before_a_run_starts(self) -> None:
        service = object.__new__(CodebaseService)
        def unavailable(*_args, **_kwargs):
            raise ProviderConfigurationError("The bounded local CPU model is unavailable")
        service.gateway = SimpleNamespace(routing_policy="local-first", select_provider=unavailable)
        result = service.run_admission()
        self.assertEqual(result["autonomous_run"], "blocked")
        self.assertIn("unavailable", result["reason"])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            service.create_run(codebase_project_id="unused", prompt="edit the file")

    def test_route_is_declared_and_requires_read_scope(self) -> None:
        self.assertIn(("GET", "/api/codebase/run-admission"), ROUTES)
        self.assertEqual(required_scope_for_path("/api/codebase/run-admission", "GET"), "codebase:read")

    def test_actual_http_requires_read_scope(self) -> None:
        with TemporaryDirectory() as directory, patch.dict("os.environ", {"OPENAGENT_AUTH_ENFORCED": "1", "OPENAGENT_MODEL_ROUTING": "local-first"}):
            store = OpenAgentStore(directory)
            auth = AuthManager(store)
            owner = auth.register("codeq-admission", "codeq-admission@example.test", "strong-password-1")
            read_token = auth.create_api_token(owner["user_id"], "read", ["codebase:read"])["token"]
            unrelated = auth.create_api_token(owner["user_id"], "other", ["knowledge:read"])["token"]
            server = build_http_server("127.0.0.1", 0, store)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                url = f"http://127.0.0.1:{server.server_address[1]}/api/codebase/run-admission"

                def get(token: str = "", *, selected: bool = False):
                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    request_url = url + ("?provider=ollama&model=qwen3-coder%3A30b" if selected else "")
                    try:
                        with urlopen(Request(request_url, headers=headers), timeout=10) as response:
                            return response.status, json.loads(response.read())
                    except HTTPError as error:
                        with error:
                            return error.code, json.loads(error.read())

                self.assertEqual(get()[0], 401)
                self.assertEqual(get(unrelated)[0], 403)
                status, payload = get(read_token)
                self.assertEqual(status, 200)
                self.assertIn(payload.get("data", payload)["autonomous_run"], {"eligible", "blocked"})
                selected_status, selected_payload = get(read_token, selected=True)
                self.assertEqual(selected_status, 200)
                self.assertIn(selected_payload.get("data", selected_payload)["autonomous_run"], {"eligible", "blocked"})
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
