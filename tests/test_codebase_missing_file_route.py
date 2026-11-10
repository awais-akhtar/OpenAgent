from __future__ import annotations

import unittest
from unittest.mock import Mock

from openagent.model_gateway import ProviderConfigurationError
from openagent.routes.codebase_routes import CodebaseRouteDependencies, contract, handle_get, handle_post


class CodebaseMissingFileRouteTests(unittest.TestCase):
    def test_reviewable_edit_proposal_routes_owner_and_hash(self) -> None:
        self.assertIn(
            {"method": "POST", "path": "/api/codebase/projects/{id}/propose-edit"},
            contract()["routes"],
        )
        codebase = Mock()
        codebase.propose_file_edit.return_value = {
            "path": "hello.py", "proposed_content": "print('hi')\n", "diff": "preview",
        }
        payload = {
            "path": "hello.py", "instruction": "Add greeting", "expected_sha256": "abc",
        }
        deps = CodebaseRouteDependencies(
            codebase=codebase,
            first=lambda values, key: values.get(key, [""])[0],
            parse_bool=lambda value, default=False: default,
            parse_int=lambda value, default=0: default,
            query_owner=lambda params: "local",
            payload_owner=lambda body: "alice",
            read_json=lambda: payload,
        )
        result = handle_post("/api/codebase/projects/project-1/propose-edit", deps)
        self.assertEqual(result.status, 200)
        codebase.propose_file_edit.assert_called_once_with(
            "project-1", "hello.py", "Add greeting", owner="alice", expected_sha256="abc")

    def test_read_only_code_question_routes_bounded_files_and_owner(self) -> None:
        codebase = Mock()
        codebase.ask_project.return_value = {"answer": "hello", "provider": "openvino-direct"}
        payload = {"question": "Explain greet", "paths": ["hello.py"]}
        deps = CodebaseRouteDependencies(
            codebase=codebase,
            first=lambda values, key: values.get(key, [""])[0],
            parse_bool=lambda value, default=False: default,
            parse_int=lambda value, default=0: default,
            query_owner=lambda params: "local",
            payload_owner=lambda body: "alice",
            read_json=lambda: payload,
        )
        result = handle_post("/api/codebase/projects/project-1/ask", deps)
        self.assertEqual(result.status, 200)
        self.assertEqual(result.payload["provider"], "openvino-direct")
        codebase.ask_project.assert_called_once_with(
            "project-1", "Explain greet", paths=["hello.py"], owner="alice")
        codebase.ask_project.side_effect = ProviderConfigurationError("local CPU model unavailable")
        error = handle_post("/api/codebase/projects/project-1/ask", deps)
        self.assertEqual(error.status, 503)

    def test_missing_project_file_is_not_found(self) -> None:
        codebase = Mock()
        codebase.read_file.side_effect = FileNotFoundError("missing.txt")
        deps = CodebaseRouteDependencies(
            codebase=codebase,
            first=lambda values, key: values.get(key, [""])[0],
            parse_bool=lambda value, default=False: default,
            parse_int=lambda value, default=0: default,
            query_owner=lambda params: "local",
            payload_owner=lambda payload: "local",
            read_json=lambda: {},
        )

        result = handle_get(
            "/api/codebase/projects/project-1/file",
            "path=missing.txt",
            deps,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.status, 404)
        self.assertEqual(result.payload, {"error": "missing.txt"})


if __name__ == "__main__":
    unittest.main()
