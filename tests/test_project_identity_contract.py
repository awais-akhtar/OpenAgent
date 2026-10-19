from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from openagent.codebase.service import CodebaseService
from openagent.routes.codebase_routes import CodebaseRouteDependencies, contract, handle_get
from openagent.storage import OpenAgentStore


class ProjectIdentityContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = CodebaseService(OpenAgentStore(Path(self.temp_dir.name) / "data"), None)  # type: ignore[arg-type]
        with self.service.store.session() as connection:
            for project_id, namespace, owner in (
                ("alice-codeq-1", "default", "alice"),
                ("alice-codeq-2", "research-space", "alice"),
                ("bob-codeq-1", "default", "bob"),
            ):
                connection.execute(
                    "INSERT INTO codebase_projects "
                    "(id, project, owner, name, source_path, map_json, memory_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, '{}', '{}', 'now', 'now')",
                    (project_id, namespace, owner, project_id, "private/path"),
                )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_service_lists_only_exact_owner_codeq_ids_and_persisted_namespaces(self) -> None:
        result = self.service.list_project_identities(owner="alice")

        self.assertEqual(
            {(item["codebase_project_id"], item["codebase_namespace"]) for item in result["projects"]},
            {("alice-codeq-1", "default"), ("alice-codeq-2", "research-space")},
        )
        self.assertFalse(result["cross_service_mapping"]["available"])
        self.assertNotIn("bob-codeq-1", str(result))
        self.assertNotIn("private/path", str(result))
        self.assertNotIn("owner", result["projects"][0])

    def test_read_route_uses_resolved_current_owner_not_query_owner(self) -> None:
        routes = {(item["method"], item["path"]) for item in contract()["routes"]}
        self.assertIn(("GET", "/api/codebase/project-identities"), routes)

        # This models Runtime's auth-enforced query_owner: it resolves the
        # bearer identity and does not trust ?owner=bob.
        deps = CodebaseRouteDependencies(
            codebase=self.service,
            first=lambda params, key: (params.get(key) or [""])[0],
            parse_bool=lambda value, default=False: default,
            parse_int=lambda value, default=0: default,
            query_owner=lambda _params: "alice",
            payload_owner=lambda _payload, default="local": default,
            read_json=lambda: {},
        )
        response = handle_get("/api/codebase/project-identities", "owner=bob", deps)

        self.assertIsNotNone(response)
        self.assertEqual(
            {item["codebase_project_id"] for item in response.payload["projects"]},
            {"alice-codeq-1", "alice-codeq-2"},
        )
        self.assertNotIn("bob-codeq-1", str(response.payload))


if __name__ == "__main__":
    unittest.main()
