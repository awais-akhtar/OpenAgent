"""Phase 6 backend: the Work Graph panel reads only persisted, owner-scoped graph revisions."""

from __future__ import annotations

import tempfile
import json
import threading
import unittest
from urllib.request import urlopen
from pathlib import Path

from openagent.codebase import graph_store
from openagent.codebase.service import CodebaseService
from openagent.codebase.work_graph import build_code_knowledge_graph
from openagent.model_gateway import ModelGateway
from openagent.storage import OpenAgentStore
from openagent.server import build_http_server


def _map(hashes: dict[str, str]) -> dict:
    return {
        "root": "/repo",
        "file_inventory": [
            {"path": p, "content_sha256": h, "index_status": "fresh", "language": "python", "size_bytes": 10}
            for p, h in sorted(hashes.items())
        ],
        "symbols": [],
    }


class WorkGraphPanelApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.source = base / "src"
        self.source.mkdir()
        (self.source / "a.py").write_bytes(b"x = 1\r\n")
        self.service = CodebaseService(OpenAgentStore(base / "data"), ModelGateway())
        self.project = self.service.create_project(
            name="Panel", source_path=str(self.source), project="panel-proj", owner="alice",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _save(self, owner: str, hashes: dict[str, str]) -> None:
        graph = build_code_knowledge_graph(_map(hashes))
        with self.service.store.session() as connection:
            graph_store.migrate(connection)
            graph_store.save_revision(connection, self.project["id"], owner, graph)

    def test_empty_when_no_revision_has_been_recorded(self) -> None:
        result = self.service.project_work_graph(self.project["id"], owner="alice")
        self.assertIsNone(result["revision"])
        self.assertEqual([], result["nodes"])
        self.assertFalse(result["truncated"])

    def test_returns_the_latest_persisted_revision_for_the_owner(self) -> None:
        self._save("alice", {"a.py": "h1", "b.py": "h2"})
        result = self.service.project_work_graph(self.project["id"], owner="alice")
        self.assertIsNotNone(result["revision"])
        paths = {node.get("path") for node in result["nodes"] if node.get("kind") == "file"}
        self.assertEqual({"a.py", "b.py"}, paths)
        self.assertEqual(len(result["nodes"]), result["node_count"])

    def test_another_owner_cannot_read_the_graph(self) -> None:
        self._save("alice", {"a.py": "h1"})
        with self.assertRaises(KeyError):
            self.service.project_work_graph(self.project["id"], owner="mallory")

    def test_large_graphs_are_bounded_and_flagged(self) -> None:
        self._save("alice", {f"m{i}.py": f"h{i}" for i in range(30)})
        result = self.service.project_work_graph(self.project["id"], owner="alice", limit=5)
        self.assertEqual(5, len(result["nodes"]))
        self.assertTrue(result["truncated"])
        kept = {node["id"] for node in result["nodes"]}
        for edge in result["edges"]:
            self.assertIn(edge["source"], kept)
            self.assertIn(edge["target"], kept)


class TaskCheckEvidenceTests(unittest.TestCase):
    @staticmethod
    def _call(tool: str, command, output: dict, status: str = "completed", error: str = "") -> dict:
        import json as _json
        return {
            "id": f"call-{tool}", "task_id": "run:T2", "tool": tool,
            "input_json": _json.dumps({"command": command}), "output_json": _json.dumps(output),
            "status": status, "error": error,
        }

    def test_failed_test_command_inside_a_task_is_evidence(self) -> None:
        row = self._call("run_shell", ["python", "-m", "unittest", "discover"], {"exit_code": 1})
        check = CodebaseService._task_check_row(row)
        self.assertIsNotNone(check)
        self.assertEqual("failed", check["status"])
        self.assertEqual(["run:T2"], check["task_ids"])

    def test_passing_run_tests_call_is_verified_evidence(self) -> None:
        check = CodebaseService._task_check_row(self._call("run_tests", [], {"exit_code": 0}))
        self.assertEqual("passed", check["status"])

    def test_ordinary_shell_calls_are_not_checks(self) -> None:
        self.assertIsNone(CodebaseService._task_check_row(self._call("run_shell", ["ls", "-la"], {"exit_code": 0})))
        self.assertIsNone(CodebaseService._task_check_row(self._call("read_file", "", {})))


class WorkGraphPanelHttpIntegrationTests(unittest.TestCase):
    """Exercise the exact read-only HTTP records consumed by the React panel."""

    def test_project_goal_graph_run_history_source_and_events_over_local_http(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "app.py").write_bytes(b"def save_account():\r\n    return True\r\n")
            store = OpenAgentStore(base / "data")
            service = CodebaseService(store, ModelGateway())
            project = service.create_project(name="Graph HTTP", source_path=str(source), project="default", owner="local")
            goal = service.create_goal(
                codebase_project_id=project["id"], owner="local", goal="Keep the account save path healthy",
                acceptance_criteria=["The save function remains present"], milestones=["Core flow"],
            )
            project_map = service.refresh_project_map(project["id"], owner="local")
            graph = build_code_knowledge_graph(project_map)
            with store.session() as connection:
                graph_store.migrate(connection)
                graph_store.save_revision(connection, project["id"], "local", graph)
            # This is a real local CodeQ operation. It creates the durable
            # interactive run and persisted event history without model calls.
            interactive = service.invoke_project_tool(project["id"], "read_file", {"path": "app.py"}, owner="local")
            run_id = interactive["run"]["id"]

            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                origin = f"http://127.0.0.1:{server.server_port}"

                def get_json(path: str) -> dict:
                    with urlopen(origin + path, timeout=5) as response:
                        self.assertEqual(200, response.status)
                        return json.loads(response.read().decode("utf-8"))

                graph_response = get_json(f"/api/codebase/projects/{project['id']}/graph?limit=2000")
                goals_response = get_json(f"/api/codebase/goals?codebase_project_id={project['id']}&limit=100")
                runs_response = get_json("/api/codebase/runs?project=default&limit=200")
                run_response = get_json(f"/api/codebase/runs/{run_id}")
                events_response = get_json(f"/api/codebase/runs/{run_id}/events?after=0")
                file_response = get_json(f"/api/codebase/projects/{project['id']}/file?path=app.py")

                graph_file = next(node for node in graph_response["nodes"] if node.get("kind") == "file" and node.get("path") == "app.py")
                self.assertEqual(graph_file["content_sha256"], file_response["file"]["content_sha256"])
                self.assertEqual(goal["id"], goals_response["goals"][0]["id"])
                self.assertIn(run_id, {row["id"] for row in runs_response["runs"]})
                self.assertEqual(project["id"], run_response["run"]["codebase_project_id"])
                self.assertIn("run.created", {event["type"] for event in events_response["events"]})
                self.assertIn("save_account", file_response["file"]["content"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
