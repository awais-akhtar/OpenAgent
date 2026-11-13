from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from openagent.codebase.service import CodebaseService
from openagent.routes.codebase_routes import CodebaseRouteDependencies, ROUTES, contract, handle_get, handle_post
from openagent.storage import OpenAgentStore


class DurableGoalGateway:
    def complete(self, messages, **kwargs):
        return type("Result", (), {"provider": "test", "model": kwargs.get("model", ""), "text": "{}", "usage": {}})()


class CodeQDurableGoalTests(unittest.TestCase):
    def _service(self, root: Path) -> tuple[CodebaseService, dict]:
        source = root / "source"
        source.mkdir()
        (source / "main.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")
        service = CodebaseService(OpenAgentStore(root / "data"), DurableGoalGateway())  # type: ignore[arg-type]
        project = service.create_project(name="goal project", source_path=str(source), owner="alice")
        return service, project

    def test_goal_crud_keeps_selected_model_criteria_and_edit_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, project = self._service(Path(directory))
            goal = service.create_goal(
                codebase_project_id=project["id"], owner="alice", goal="Finish the app",
                provider="openvino-direct", model="qwen-small", role_models={"tester": "test-model"},
                acceptance_criteria=["App starts", "Tests pass"],
                milestones=["Foundation", "Release"],
                criterion_milestones={"App starts": "Foundation", "Tests pass": "Release"},
                constraints={"network": "local-only"},
            )
            self.assertEqual(goal["provider"], "openvino-direct")
            self.assertEqual(goal["model"], "qwen-small")
            self.assertEqual([item["criterion"] for item in goal["criteria"]], ["App starts", "Tests pass"])
            self.assertTrue(all(item["origin"] == "user" for item in goal["criteria"]))
            self.assertEqual([item["milestone_title"] for item in goal["criteria"]], ["Foundation", "Release"])
            changed = service.update_goal(goal["id"], owner="alice", expected_revision=goal["revision"], goal="Finish the app safely", acceptance_criteria=["App starts", "Tests pass", "No regressions"])
            self.assertEqual(changed["history"][-1]["action"], "updated")
            self.assertEqual(changed["revision"], goal["revision"] + 1)
            self.assertEqual([item["status"] for item in changed["criteria"][:2]], ["superseded", "superseded"])
            self.assertEqual([item["criterion"] for item in changed["criteria"] if item["status"] == "open"], ["App starts", "Tests pass", "No regressions"])
            self.assertEqual([item["milestone_title"] for item in changed["criteria"] if item["status"] == "open"], ["Foundation", "Foundation", "Foundation"])
            self.assertTrue(all(item["linked_task_ids"] == [] and item["evidence_ids"] == [] for item in changed["criteria"]))
            with self.assertRaisesRegex(ValueError, "changed since it was loaded"):
                service.update_goal(goal["id"], owner="alice", expected_revision=goal["revision"], provider="other")
            self.assertEqual([item["id"] for item in service.list_goals(owner="alice", codebase_project_id=project["id"])], [goal["id"]])
            with self.assertRaises(KeyError):
                service.get_goal(goal["id"], owner="bob")
            self.assertEqual(service.list_goals(owner="bob"), [])
            reopened = CodebaseService(OpenAgentStore(Path(directory) / "data"), DurableGoalGateway())  # type: ignore[arg-type]
            reopened_goal = reopened.get_goal(goal["id"], owner="alice")
            self.assertEqual(reopened_goal["goal"], "Finish the app safely")
            self.assertEqual(reopened_goal["milestones"], goal["milestones"])

    def test_identical_goal_submission_is_reconciled_without_duplicate_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, project = self._service(Path(directory))
            payload = {
                "codebase_project_id": project["id"], "owner": "alice", "goal": "Finish the app",
                "provider": "openvino-direct", "model": "small", "acceptance_criteria": ["Tests pass"],
                "milestones": ["Verification"],
            }
            first = service.create_goal(**payload)
            second = service.create_goal(**payload)
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(len(second["history"]), 1)
            self.assertEqual(len(service.list_goals(owner="alice", codebase_project_id=project["id"])), 1)

    def test_concurrent_goal_edits_use_revision_conflict_instead_of_lost_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, project = self._service(Path(directory))
            goal = service.create_goal(codebase_project_id=project["id"], owner="alice", goal="Initial goal")
            barrier = threading.Barrier(2)

            def edit(text: str):
                barrier.wait(timeout=5)
                try:
                    return service.update_goal(goal["id"], owner="alice", expected_revision=goal["revision"], goal=text)
                except ValueError as exc:
                    return exc

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(edit, ["First edit", "Second edit"]))
            self.assertEqual(sum(isinstance(item, dict) for item in results), 1)
            self.assertEqual(sum(isinstance(item, ValueError) for item in results), 1)
            current = service.get_goal(goal["id"], owner="alice")
            self.assertEqual(current["revision"], 2)
            self.assertIn(current["goal"], {"First edit", "Second edit"})

    def test_run_links_durable_goal_and_reuses_saved_model_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, project = self._service(Path(directory))
            goal = service.create_goal(
                codebase_project_id=project["id"], owner="alice", goal="Finish the app",
                provider="openvino-direct", model="selected-small", role_models={"programmer": "selected-small"},
            )
            with patch.object(service, "_execute_run", return_value=None):
                run = service.create_run(codebase_project_id=project["id"], goal_id=goal["id"], owner="alice")
            self.assertEqual(run["goal_id"], goal["id"])
            self.assertEqual(run["provider"], "openvino-direct")
            self.assertEqual(run["model"], "selected-small")
            self.assertEqual(run["role_models"], {"programmer": "selected-small"})
            self.assertEqual(run["user_prompt"], "Finish the app")
            self.assertEqual(run["goal"]["id"], goal["id"])
            self.assertEqual(service.get_goal(goal["id"], owner="alice")["runs"][0]["id"], run["id"])
            with self.assertRaisesRegex(ValueError, "edit the saved goal"):
                service.create_run(codebase_project_id=project["id"], goal_id=goal["id"], owner="alice", model="other-model")
            with self.assertRaisesRegex(ValueError, "active run"):
                service.update_goal(goal["id"], owner="alice", provider="other")

    def test_run_without_explicit_goal_creates_a_durable_goal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, project = self._service(Path(directory))
            with patch.object(service, "_execute_run", return_value=None):
                run = service.create_run(
                    codebase_project_id=project["id"], prompt="Add a health endpoint", owner="alice",
                    provider="openvino-direct", model="small-model", acceptance_criteria=["Endpoint responds"],
                )
            self.assertTrue(run["goal_id"])
            goal = service.get_goal(run["goal_id"], owner="alice")
            self.assertEqual(goal["goal"], "Add a health endpoint")
            self.assertEqual(goal["model"], "small-model")
            self.assertEqual(goal["criteria"][0]["criterion"], "Endpoint responds")
            self.assertEqual(goal["criteria"][0]["milestone_title"], "Complete goal")

    def test_goal_fallback_criterion_is_distinguished_from_explicit_criteria(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, project = self._service(Path(directory))
            goal = service.create_goal(codebase_project_id=project["id"], owner="alice", goal="Keep this exact request")
            self.assertEqual(goal["goal"], "Keep this exact request")
            self.assertEqual(goal["criteria"][0]["criterion"], "Keep this exact request")
            self.assertEqual(goal["criteria"][0]["origin"], "user_goal")
            self.assertTrue(goal["created_at"])
            self.assertEqual(goal["status"], "active")

    def test_existing_runs_schema_migrates_goal_link_without_losing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "openagent.sqlite3"
            connection = sqlite3.connect(db)
            connection.execute("""CREATE TABLE codebase_runs (
                id TEXT PRIMARY KEY, codebase_project_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                project TEXT NOT NULL DEFAULT 'default', owner TEXT NOT NULL DEFAULT 'local',
                user_prompt TEXT NOT NULL, status TEXT NOT NULL, provider TEXT NOT NULL DEFAULT 'auto',
                model TEXT NOT NULL DEFAULT '', allow_external INTEGER NOT NULL DEFAULT 0,
                iteration INTEGER NOT NULL DEFAULT 0, max_iterations INTEGER NOT NULL DEFAULT 8,
                completion_json TEXT NOT NULL DEFAULT '{}', summary TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '', started_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                finished_at TEXT NOT NULL DEFAULT '')""")
            connection.execute("""CREATE TABLE codebase_goals (
                id TEXT PRIMARY KEY, codebase_project_id TEXT NOT NULL, project TEXT NOT NULL DEFAULT 'default',
                owner TEXT NOT NULL DEFAULT 'local', original_goal TEXT NOT NULL, provider TEXT NOT NULL DEFAULT 'auto',
                model TEXT NOT NULL DEFAULT '', role_models_json TEXT NOT NULL DEFAULT '{}', allow_external INTEGER NOT NULL DEFAULT 0,
                constraints_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE codebase_goal_criteria (
                id TEXT PRIMARY KEY, goal_id TEXT NOT NULL, criterion TEXT NOT NULL, origin TEXT NOT NULL DEFAULT 'user',
                status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE codebase_goal_history (
                id TEXT PRIMARY KEY, goal_id TEXT NOT NULL, action TEXT NOT NULL,
                before_json TEXT NOT NULL DEFAULT '{}', after_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL)""")
            connection.execute("INSERT INTO codebase_goals (id, codebase_project_id, original_goal, created_at, updated_at) VALUES ('old-goal','p','old goal','t','t')")
            connection.execute("INSERT INTO codebase_goal_criteria (id, goal_id, criterion, created_at, updated_at) VALUES ('old-criterion','old-goal','old criterion','t','t')")
            connection.execute("INSERT INTO codebase_runs (id, codebase_project_id, workspace_id, user_prompt, status, started_at, updated_at) VALUES ('old-run','p','w','old','completed','t','t')")
            connection.commit()
            connection.close()
            service = CodebaseService(OpenAgentStore(db), DurableGoalGateway())  # type: ignore[arg-type]
            with service.store.session() as migrated:
                columns = {row[1] for row in migrated.execute("PRAGMA table_info(codebase_runs)")}
                row = migrated.execute("SELECT id, goal_id FROM codebase_runs WHERE id='old-run'").fetchone()
                old_goal = migrated.execute("SELECT milestones_json, revision FROM codebase_goals WHERE id='old-goal'").fetchone()
                old_criterion = migrated.execute("SELECT milestone_id, linked_task_ids_json, evidence_ids_json FROM codebase_goal_criteria WHERE id='old-criterion'").fetchone()
            self.assertIn("goal_id", columns)
            self.assertEqual(tuple(row), ("old-run", ""))
            self.assertEqual(json.loads(old_goal["milestones_json"])[0]["title"], "Complete goal")
            self.assertEqual(old_goal["revision"], 1)
            self.assertTrue(old_criterion["milestone_id"])
            self.assertEqual(old_criterion["linked_task_ids_json"], "[]")
            self.assertEqual(old_criterion["evidence_ids_json"], "[]")

    def test_goal_routes_validate_contract_and_keep_owner_scope(self) -> None:
        self.assertTrue({("GET", "/api/codebase/goals"), ("GET", "/api/codebase/goals/{id}"), ("POST", "/api/codebase/goals"), ("POST", "/api/codebase/goals/{id}/update")} <= {(item["method"], item["path"]) for item in contract()["routes"]})
        self.assertIn(("GET", "/api/codebase/goals"), ROUTES)
        codebase = Mock()
        payload = {"codebase_project_id": "p1", "goal": "Make it work", "acceptance_criteria": ["It runs"]}
        deps = CodebaseRouteDependencies(
            codebase=codebase,
            first=lambda values, key: values.get(key, [""])[0],
            parse_bool=lambda value, default=False: bool(value) if value is not None else default,
            parse_int=lambda value, default=0: int(value) if value else default,
            query_owner=lambda _params: "alice",
            payload_owner=lambda _body: "alice",
            read_json=lambda: payload,
        )
        codebase.create_goal.return_value = {"id": "g1", "owner": "alice"}
        created = handle_post("/api/codebase/goals", deps)
        self.assertEqual(created.status, 201)
        codebase.create_goal.assert_called_once_with(
            codebase_project_id="p1", goal="Make it work", owner="alice", provider="auto", model="",
            role_models=None, allow_external=False, acceptance_criteria=["It runs"], milestones=None,
            criterion_milestones=None, constraints=None,
        )
        codebase.get_goal.return_value = {"id": "g1", "owner": "alice"}
        fetched = handle_get("/api/codebase/goals/g1", "", deps)
        self.assertEqual(fetched.status, 200)
        codebase.get_goal.assert_called_once_with("g1", owner="alice")
        invalid = handle_post("/api/codebase/goals", CodebaseRouteDependencies(**{**deps.__dict__, "read_json": lambda: {**payload, "acceptance_criteria": "not-a-list"}}))
        self.assertEqual(invalid.status, 400)


if __name__ == "__main__":
    unittest.main()
