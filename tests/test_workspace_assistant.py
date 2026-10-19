from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.assistant import AssistantManager
from openagent.storage import OpenAgentStore
from openagent.workspace import WorkspaceManager


class WorkspaceAssistantTests(unittest.TestCase):
    def test_workspace_items_templates_dashboard_and_canvas(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            workspace = WorkspaceManager(store)

            folder = workspace.create_item(kind="folder", title="Research", project="core")
            wiki = workspace.create_item(
                kind="wiki_page",
                title="OpenAgent wiki",
                project="core",
                parent_id=folder["id"],
                content="OpenAgent is a local-first AI OS with memory and agents.",
                metadata={"pinned": True},
            )
            template = workspace.create_item(
                kind="template",
                title="Project brief",
                project="core",
                content="Project {{name}} has goal {{goal}}.",
            )
            document = workspace.instantiate_template(
                template["id"],
                title="OpenAgent brief",
                project="core",
                variables={"name": "OpenAgent", "goal": "private AI OS"},
            )
            canvas = workspace.create_item(
                kind="mind_map",
                title="Architecture map",
                project="core",
                metadata={"links": [wiki["id"], document["id"]]},
            )

            self.assertIn("private AI OS", document["content"])
            self.assertEqual(len(workspace.search("local-first", project="core")), 1)
            dashboard = workspace.dashboard(project="core")
            self.assertEqual(dashboard["by_kind"]["wiki_page"], 1)
            self.assertEqual(dashboard["pinned"][0]["id"], wiki["id"])
            canvas_model = workspace.canvas_model(project="core")
            self.assertEqual(canvas_model["nodes"][0]["id"], canvas["id"])
            self.assertEqual(len(canvas_model["edges"]), 2)
            self.assertGreaterEqual(store.health()["event_count"], 3)

    def test_assistant_tasks_goals_meetings_travel_and_crm(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            assistant = AssistantManager(store)

            profile = assistant.upsert_profile(
                user_id="owner",
                project="core",
                preferences={"tone": "direct"},
                facts={"timezone": "local"},
            )
            goal = assistant.create_goal(
                title="Launch OpenAgent",
                project="core",
                description="Ship the local-first AI OS foundation.",
                horizon="medium",
            )
            task = assistant.create_task(
                title="Prepare release checklist",
                project="core",
                priority="high",
                due_at="2026-06-30",
                goal_id=goal["id"],
            )
            brief = assistant.create_meeting_brief(
                title="OpenAgent roadmap",
                project="core",
                attendees=["owner@example.com"],
                notes="Review launch tasks.",
            )
            summary = assistant.summarize_meeting(
                title="Roadmap sync",
                project="core",
                transcript="Decision: ship stage 10.\nAction: update docs.\nFollow up next week.",
            )
            plan = assistant.create_travel_plan(
                destination="Berlin",
                project="core",
                starts_at="2026-07-01",
                ends_at="2026-07-05",
                preferences={"budget": "moderate"},
            )
            crm = assistant.upsert_crm_record(
                record_type="opportunity",
                name="Enterprise pilot",
                project="core",
                value=25000,
                contact="buyer@example.com",
                notes="Interested in self-hosted AI.",
            )

            self.assertEqual(profile["preferences"]["tone"], "direct")
            self.assertEqual(task["status"], "open")
            self.assertEqual(assistant.complete_task(task["id"], project="core")["status"], "completed")
            self.assertIn("agenda", brief["brief"])
            self.assertGreaterEqual(len(summary["summary"]["action_items"]), 2)
            self.assertEqual(len(summary["summary"]["decisions"]), 1)
            self.assertIn("flight_status", plan["plan"]["monitoring"])
            self.assertEqual(crm["record_type"], "opportunity")
            daily = assistant.daily_brief(project="core")
            self.assertEqual(len(daily["active_goals"]), 1)
            coaching = assistant.productivity_coaching(project="core")
            self.assertEqual(coaching["task_count"], 1)


if __name__ == "__main__":
    unittest.main()
