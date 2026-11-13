"""Phase 0: pin the CodeQ API route contract and the SQLite schema before further migrations.

A change to any listed route, table or required column must update this file deliberately.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openagent.codebase.service import CodebaseService
from openagent.model_gateway import ModelGateway
from openagent.routes.codebase_routes import contract
from openagent.storage import OpenAgentStore

EXPECTED_ROUTES = {
    ("GET", "/api/codebase/run-admission"),
    ("GET", "/api/codebase/frameworks"),
    ("GET", "/api/codebase/projects"),
    ("GET", "/api/codebase/project-identities"),
    ("GET", "/api/codebase/projects/{id}"),
    ("GET", "/api/codebase/projects/{id}/tree"),
    ("GET", "/api/codebase/projects/{id}/file"),
    ("GET", "/api/codebase/projects/{id}/graph"),
    ("GET", "/api/codebase/runs"),
    ("GET", "/api/codebase/runs/{id}"),
    ("GET", "/api/codebase/goals"),
    ("GET", "/api/codebase/goals/{id}"),
    ("GET", "/api/codebase/runs/{id}/events"),
    ("GET", "/api/codebase/runs/{id}/events/stream"),
    ("POST", "/api/codebase/projects"),
    ("POST", "/api/codebase/folder-picker"),
    ("POST", "/api/codebase/preflight"),
    ("POST", "/api/codebase/projects/{id}/scan"),
    ("POST", "/api/codebase/projects/{id}/ask"),
    ("POST", "/api/codebase/projects/{id}/propose-edit"),
    ("POST", "/api/codebase/projects/{id}/file"),
    ("POST", "/api/codebase/projects/{id}/paths"),
    ("POST", "/api/codebase/projects/{id}/tools"),
    ("POST", "/api/codebase/runs"),
    ("POST", "/api/codebase/goals"),
    ("POST", "/api/codebase/goals/{id}/update"),
    ("POST", "/api/codebase/runs/{id}/cancel"),
    ("POST", "/api/codebase/runs/{id}/resume"),
    ("POST", "/api/codebase/runs/{id}/tools"),
    ("POST", "/api/codebase/approvals/{id}/resolve"),
    ("POST", "/api/codebase/runs/{id}/snapshots/{snapshot_id}/restore"),
}

EXPECTED_TABLES = {
    "codebase_projects", "codebase_workspaces", "codebase_runs", "codebase_tasks", "codebase_tool_calls",
    "codebase_file_changes", "codebase_test_runs", "codebase_events", "codebase_approvals", "codebase_snapshots",
    "codebase_memories", "codebase_goals", "codebase_goal_criteria", "codebase_goal_history",
    "codebase_evidence_links", "codebase_index_cache", "codebase_graph_revisions", "codebase_graph_nodes",
    "codebase_graph_edges", "codebase_graph_schema",
}

REQUIRED_COLUMNS = {
    "codebase_tasks": {"id", "run_id", "status", "depends_on_json", "done_predicates_json", "declared_files_json", "code_paths_json", "criterion_ids_json"},
    "codebase_runs": {"id", "codebase_project_id", "goal_id", "owner", "status", "provider", "model", "completion_json"},
    "codebase_test_runs": {"id", "run_id", "kind", "exit_code", "status", "source_fingerprint", "task_ids_json"},
    "codebase_goals": {"id", "codebase_project_id", "owner", "revision", "status"},
    "codebase_graph_revisions": {"id", "codebase_project_id", "owner", "revision"},
}


class Phase0ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.service = CodebaseService(OpenAgentStore(Path(self._tmp.name) / "data"), ModelGateway())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_route_contract_matches_the_pinned_api(self) -> None:
        declared = {(route["method"], route["path"]) for route in contract()["routes"]}
        self.assertEqual(EXPECTED_ROUTES, declared, "route added or removed: update this contract deliberately")

    def test_every_pinned_table_exists_after_startup(self) -> None:
        with self.service.store.session() as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertTrue(EXPECTED_TABLES <= tables, sorted(EXPECTED_TABLES - tables))

    def test_required_columns_are_present(self) -> None:
        with self.service.store.session() as connection:
            for table, columns in REQUIRED_COLUMNS.items():
                present = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                self.assertTrue(columns <= present, f"{table} missing {sorted(columns - present)}")

    def test_schema_version_row_is_recorded_for_the_graph_store(self) -> None:
        from openagent.codebase import graph_store
        with self.service.store.session() as connection:
            graph_store.migrate(connection)
            self.assertEqual(graph_store.GRAPH_SCHEMA_VERSION, graph_store.schema_version(connection))


if __name__ == "__main__":
    unittest.main()
