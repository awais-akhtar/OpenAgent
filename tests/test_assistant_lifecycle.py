from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from openagent.assistant import AssistantManager
from openagent.storage import OpenAgentStore


class AssistantLifecycleTests(unittest.TestCase):
    def test_task_crud_validates_and_is_project_scoped(self) -> None:
        with TemporaryDirectory() as directory:
            assistant = AssistantManager(OpenAgentStore(directory))
            goal = assistant.create_goal(title="Release", project="personal")
            foreign_goal = assistant.create_goal(title="Private goal", project="other")
            with self.assertRaisesRegex(ValueError, "does not exist in this project"):
                assistant.create_task(title="Cross-project link", project="personal", goal_id=foreign_goal["id"])
            self.assertEqual(assistant.tasks(project="personal"), [])
            task = assistant.create_task(title="Write notes", project="personal", goal_id=goal["id"])
            with self.assertRaises(KeyError):
                assistant.update_task(task["id"], project="other", updates={"title": "forged"})
            with self.assertRaises(ValueError):
                assistant.update_task(task["id"], project="personal", updates={"title": " "})
            updated = assistant.update_task(task["id"], project="personal", updates={"title": "Publish notes", "priority": "high"})
            self.assertEqual(updated["title"], "Publish notes")
            self.assertEqual(updated["priority"], "high")
            self.assertEqual(assistant.tasks(project="other"), [])
            self.assertTrue(assistant.delete_task(task["id"], project="personal"))
            self.assertFalse(assistant.delete_task(task["id"], project="personal"))

    def test_goal_crud_unlinks_project_tasks_and_keeps_other_projects(self) -> None:
        with TemporaryDirectory() as directory:
            assistant = AssistantManager(OpenAgentStore(directory))
            goal = assistant.create_goal(title="Research", project="personal")
            task = assistant.create_task(title="Find sources", project="personal", goal_id=goal["id"])
            assistant.create_goal(title="Other workspace", project="other")
            with self.assertRaises(KeyError):
                assistant.update_goal(goal["id"], project="other", updates={"title": "forged"})
            updated = assistant.update_goal(goal["id"], project="personal", updates={"title": "Research plan", "horizon": "long"})
            self.assertEqual(updated["title"], "Research plan")
            self.assertEqual(updated["horizon"], "long")
            self.assertTrue(assistant.delete_goal(goal["id"], project="personal"))
            self.assertEqual(assistant.get_task_for_project(task["id"], "personal")["goal_id"], "")
            self.assertEqual([item["title"] for item in assistant.goals(project="other")], ["Other workspace"])


if __name__ == "__main__":
    unittest.main()
