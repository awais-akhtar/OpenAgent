"""Persisted task/criterion/test links used by the Phase 4 completion gate."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openagent.codebase.evidence import source_fingerprint
from openagent.codebase.service import CodebaseService
from openagent.storage import OpenAgentStore


class _Gateway:
    def complete(self, messages, **kwargs):
        return type("Result", (), {"provider": "test", "model": "test", "text": "{}", "usage": {}})()


class Phase4CriterionEvidenceTests(unittest.TestCase):
    def _service(self, root: Path) -> tuple[CodebaseService, dict]:
        source = root / "source"
        source.mkdir()
        (source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        service = CodebaseService(OpenAgentStore(root / "data"), _Gateway())  # type: ignore[arg-type]
        project = service.create_project(name="criteria", source_path=str(source), owner="local")
        return service, project

    def _queued_run(self, service: CodebaseService, project: dict, goal_id: str) -> dict:
        with patch.object(service, "_execute_run", return_value=None):
            return service.create_run(
                codebase_project_id=project["id"], goal_id=goal_id, owner="local", max_iterations=20,
                completion={"runtime": {"enabled": True, "command": [sys.executable, "-c", "print('verified')"]}},
            )

    def test_plan_tasks_link_active_criteria_to_persisted_check_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, project = self._service(Path(directory))
            goal = service.create_goal(
                codebase_project_id=project["id"], owner="local", goal="Finish the feature",
                acceptance_criteria=["Feature is implemented", "Runtime check passes"],
            )
            run = self._queued_run(service, project, goal["id"])
            plan = [
                {"id": "T1", "title": "Implement the feature", "assigned_agent": "programmer", "depends_on": [], "criterion_ids": [goal["criteria"][0]["id"]]},
                {"id": "T2", "title": "Run final checks", "assigned_agent": "tester", "depends_on": ["T1"], "criterion_ids": [goal["criteria"][1]["id"]]},
            ]
            plan = service._attach_goal_criteria_to_tasks(run["id"], plan)
            service._store_tasks(run["id"], plan)
            links_before = [item["linked_task_ids"] for item in service.get_goal(goal["id"])["criteria"]]
            service._store_tasks(run["id"], plan)
            self.assertEqual([item["linked_task_ids"] for item in service.get_goal(goal["id"])["criteria"]], links_before)
            persisted = service.get_run(run["id"])["tasks"]
            self.assertEqual(set(persisted[0]["criterion_ids"]), {goal["criteria"][0]["id"]})
            self.assertEqual(set(persisted[1]["criterion_ids"]), {goal["criteria"][1]["id"]})
            self.assertEqual(set(service.get_goal(goal["id"])["criteria"][0]["linked_task_ids"]), {persisted[0]["id"]})
            with self.assertRaisesRegex(ValueError, "must belong to its goal"):
                service.link_task_criteria(run["id"], persisted[0]["id"], ["not-a-criterion"])

            for task in persisted:
                service._set_task(task["id"], "completed", {"summary": "work recorded"})
            fresh_run = service.get_run(run["id"])
            validation = service._validate(fresh_run)
            self.assertTrue(validation["passed"], validation["summary"])
            self.assertTrue(all(item.get("evidence_id") for item in validation["checks"]))
            self.assertEqual(service._evaluate_goal_criteria(fresh_run, validation), [])

            verified = service.get_goal(goal["id"])
            for criterion in verified["criteria"]:
                self.assertEqual(criterion["status"], "satisfied")
                self.assertTrue(criterion["evidence_ids"])
            test_rows = {row["id"] for row in service.get_run(run["id"])["validation"]}
            links = [item for item in service.evidence_links(run["id"]) if item["source_kind"] == "criterion"]
            self.assertTrue(links)
            self.assertTrue(all(item["source_id"] in test_rows for item in links))

            # Durable evidence survives reopening the goal while the source
            # is identical, and is invalidated as soon as it changes.
            resumed = self._queued_run(service, project, goal["id"])
            self.assertEqual(service._evaluate_goal_criteria(resumed, {"checks": []}), [])
            resumed_workspace = service._workspace(resumed["workspace_id"], resumed["owner"])
            (Path(resumed_workspace["workspace_path"]) / "app.py").write_text("VALUE = 99\n", encoding="utf-8")
            unresolved = service._evaluate_goal_criteria(resumed, {"checks": []})
            self.assertEqual(len(unresolved), 2)
            self.assertTrue(all(item["status"] == "open" for item in service.get_goal(goal["id"])["criteria"]))

    def test_missing_task_mapping_is_filled_and_superseded_criteria_stay_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, project = self._service(Path(directory))
            goal = service.create_goal(
                codebase_project_id=project["id"], owner="local", goal="Finish the feature",
                acceptance_criteria=["Old criterion"],
            )
            goal = service.update_goal(
                goal["id"], owner="local", expected_revision=goal["revision"],
                acceptance_criteria=["Current criterion"],
            )
            old, active = goal["criteria"]
            self.assertEqual(old["status"], "superseded")
            run = self._queued_run(service, project, goal["id"])
            plan = service._attach_goal_criteria_to_tasks(run["id"], [
                {"id": "T1", "title": "Implement and verify", "assigned_agent": "tester", "depends_on": [], "criterion_ids": ["invented-id"]},
            ])
            self.assertEqual(plan[0]["criterion_ids"], [active["id"]])
            service._store_tasks(run["id"], plan)
            result = service._evaluate_goal_criteria(service.get_run(run["id"]), {"checks": []})
            self.assertEqual([item["criterion_id"] for item in result], [active["id"]])
            statuses = {item["id"]: item["status"] for item in service.get_goal(goal["id"])["criteria"]}
            self.assertEqual(statuses[old["id"]], "superseded")
            self.assertEqual(statuses[active["id"]], "open")

    def test_publish_refuses_workspace_changes_after_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, project = self._service(Path(directory))
            goal = service.create_goal(codebase_project_id=project["id"], goal="Change the feature")
            run = self._queued_run(service, project, goal["id"])
            workspace = service._workspace(run["workspace_id"], run["owner"])
            workspace_root = Path(workspace["workspace_path"])
            source_root = Path(workspace["source_path"])
            verified_fingerprint = source_fingerprint(workspace_root)
            (workspace_root / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
            with service.store.session() as connection:
                connection.execute(
                    "INSERT INTO codebase_file_changes (id, run_id, filepath, old_hash, new_hash, patch, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("change-1", run["id"], "app.py", "old", "new", "", "now"),
                )

            result = service._publish_verified_changes(run, expected_fingerprint=verified_fingerprint)
            self.assertTrue(result["stale"])
            self.assertEqual(result["files"], [])
            self.assertEqual((source_root / "app.py").read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertIn("source.publish.blocked", {item["type"] for item in service.events(run["id"])})


if __name__ == "__main__":
    unittest.main()
