from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.models import ContextEvent
from openagent.storage import OpenAgentStore
from openagent.workflows import WorkflowEngine, WorkflowSafetyError


class WorkflowAutomationTests(unittest.TestCase):
    def test_file_actions_conditions_rollback_and_visual_model(self) -> None:
        with TemporaryDirectory() as tmp:
            engine = WorkflowEngine(OpenAgentStore(tmp))
            workflow = engine.create(
                name="File automation",
                project="core",
                steps=[
                    {
                        "action": "file_write",
                        "path": "notes/result.txt",
                        "content": "hello {{event.kind}}",
                        "if": {"path": "event.kind", "op": "equals", "value": "write"},
                    },
                    {"action": "file_read", "path": "notes/result.txt"},
                    {"action": "file_list", "path": "notes"},
                ],
            )

            dry_run = engine.run(workflow["id"], dry_run=True, event={"kind": "write"})
            self.assertEqual(dry_run["result"]["steps"][0]["status"], "planned")

            run = engine.run(workflow["id"], event={"kind": "write"})
            self.assertEqual(run["status"], "completed")
            self.assertIn("hello write", run["result"]["steps"][1]["content"])

            visual = engine.visual_model(workflow["id"])
            self.assertEqual(len(visual["nodes"]), 4)
            self.assertEqual(len(visual["edges"]), 3)

            schema = engine.builder_schema()
            self.assertIn("api_request", schema["actions"])
            self.assertIn("scheduled", schema["triggers"])

            layout = engine.save_layout(
                workflow["id"],
                {
                    "nodes": [
                        {"id": "step-2", "x": 80, "y": 40, "order": 1},
                        {"id": "trigger", "x": 40, "y": 40, "order": 0},
                        {"id": "step-1", "x": 120, "y": 40, "order": 2},
                    ]
                },
            )
            self.assertEqual(layout["layout"]["nodes"][0]["id"], "step-2")
            visual_with_layout = engine.visual_model(workflow["id"])
            self.assertEqual(visual_with_layout["layout"]["id"], layout["id"])
            audit_actions = {event["action"] for event in engine.store.list_audit_events(project="core")}
            self.assertIn("workflow.layout.save", audit_actions)

    def test_failed_workflow_executes_file_rollback(self) -> None:
        with TemporaryDirectory() as tmp:
            engine = WorkflowEngine(OpenAgentStore(tmp))
            workflow = engine.create(
                name="Rollback automation",
                project="core",
                steps=[
                    {"action": "file_write", "path": "rollback.txt", "content": "temporary"},
                    {"action": "browser_fetch", "url": "https://example.com"},
                ],
            )

            run = engine.run(workflow["id"])

            self.assertEqual(run["status"], "failed")
            self.assertFalse(engine.safe_file_path("rollback.txt").exists())
            self.assertIn("requires allow_network", run["result"]["steps"][-1]["error"])

    def test_event_and_scheduled_triggers(self) -> None:
        with TemporaryDirectory() as tmp:
            engine = WorkflowEngine(OpenAgentStore(tmp))
            event_workflow = engine.create(
                name="Event workflow",
                project="core",
                trigger_type="event",
                trigger={"event_name": "note.created"},
                steps=[
                    {
                        "action": "add_event",
                        "title": "Triggered",
                        "content": "Event {{event.name}}",
                    }
                ],
            )
            scheduled_workflow = engine.create(
                name="Scheduled workflow",
                project="core",
                trigger_type="scheduled",
                trigger={"interval_seconds": 0},
                steps=[{"action": "audit", "audit_action": "workflow.scheduled"}],
            )

            event_result = engine.trigger_event("note.created", {"name": "note.created"}, project="core")
            self.assertEqual(event_result["matched_runs"][0]["workflow_id"], event_workflow["id"])

            due_runs = engine.run_due_schedules()
            self.assertEqual(due_runs[0]["workflow_id"], scheduled_workflow["id"])

    def test_data_processing_database_spreadsheet_report_monitor_and_alert_actions(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            store.add_event(
                ContextEvent(
                    title="Metric",
                    content="Revenue reached 42.",
                    project="core",
                )
            )
            engine = WorkflowEngine(store)
            workflow = engine.create(
                name="Data processing",
                project="core",
                steps=[
                    {
                        "action": "spreadsheet_write",
                        "path": "sheets/report.csv",
                        "rows": [{"name": "OpenAgent", "score": 42}],
                    },
                    {"action": "spreadsheet_read", "path": "sheets/report.csv"},
                    {"action": "database_query", "query": "SELECT title, project FROM events"},
                    {
                        "action": "monitor_check",
                        "condition": {"path": "steps.2.row_count", "op": "equals", "value": 1},
                        "message": "Spreadsheet row count is healthy.",
                    },
                    {
                        "action": "report_generate",
                        "path": "reports/summary.md",
                        "title": "Automation Report",
                        "sections": [{"title": "Result", "content": "Generated locally."}],
                    },
                    {"action": "alert", "severity": "info", "message": "Report generated."},
                ],
            )

            run = engine.run(workflow["id"])

            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["result"]["steps"][1]["rows"][0]["name"], "OpenAgent")
            self.assertEqual(run["result"]["steps"][2]["rows"][0]["title"], "Metric")
            self.assertTrue(run["result"]["steps"][3]["passed"])
            self.assertTrue(engine.safe_file_path("reports/summary.md").exists())

    def test_network_actions_require_explicit_allow_network(self) -> None:
        with TemporaryDirectory() as tmp:
            engine = WorkflowEngine(OpenAgentStore(tmp))
            workflow = engine.create(
                name="Network automation",
                project="core",
                steps=[{"action": "api_request", "url": "https://example.com"}],
            )

            run = engine.run(workflow["id"])

            self.assertEqual(run["status"], "failed")
            with self.assertRaises(WorkflowSafetyError):
                engine._run_step(  # noqa: SLF001 - direct safety assertion for bootstrap engine
                    {"action": "api_request", "url": "https://example.com"},
                    project="core",
                    index=1,
                    context={},
                    allow_network=False,
                )


if __name__ == "__main__":
    unittest.main()
