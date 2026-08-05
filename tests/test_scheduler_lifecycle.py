from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.routes.scheduler_routes import ROUTES
from openagent.scheduler import Scheduler
from openagent.storage import OpenAgentStore


class SchedulerLifecycleTests(unittest.TestCase):
    def test_dependencies_revisions_history_and_delete(self) -> None:
        with TemporaryDirectory() as temp:
            scheduler = Scheduler(OpenAgentStore(temp))
            first = scheduler.create_task(title="First", project="core", schedule={"delay_seconds": 3600})
            second = scheduler.create_task(title="Second", project="core", schedule={"delay_seconds": 3600})
            self.assertEqual(scheduler.set_dependencies(second["id"], [first["id"]])["dependencies"], [first["id"]])
            scheduler.update_task(second["id"], patch={"title": "Second updated"})
            self.assertEqual(scheduler.revert(second["id"])["title"], "Second")
            self.assertEqual(scheduler.run_history(second["id"]), [])
            self.assertEqual(scheduler.stop(second["id"])["status"], "cancelled")
            self.assertTrue(scheduler.delete(second["id"])["deleted"])

    def test_natural_language_and_signed_webhook_token(self) -> None:
        with TemporaryDirectory() as temp:
            scheduler = Scheduler(OpenAgentStore(temp))
            task = scheduler.create_from_text("Review report in 15 minutes", project="core")
            self.assertEqual(task["schedule"]["delay_seconds"], 900)
            webhook = scheduler.create_webhook(task["id"], "https://example.invalid/openagent")
            self.assertTrue(webhook["token_displayed_once"])
            self.assertGreater(len(webhook["token"]), 32)

    def test_routes_cover_task_lifecycle(self) -> None:
        paths = {path for _, path in ROUTES}
        self.assertIn("/api/tasks/{id}/revert", paths)
        self.assertIn("/api/tasks/{id}/dependencies", paths)
        self.assertIn("/api/tasks/{id}/destinations", paths)
        self.assertIn("/api/tasks/parse", paths)

    def test_output_destinations_write_local_delivery_history(self) -> None:
        with TemporaryDirectory() as temp:
            scheduler = Scheduler(OpenAgentStore(temp))
            task = scheduler.create_task(title="Write report", schedule={"delay_seconds": 0})
            destination = scheduler.add_output_destination(
                task["id"], "local_file", config={"filename": "results.jsonl"}
            )
            self.assertEqual(destination["destination_type"], "local_file")
            result = scheduler.run_due()
            self.assertEqual(result["deliveries"][0]["status"], "completed")
            self.assertEqual(len(scheduler.delivery_history(task["id"])), 1)
            self.assertTrue((scheduler.store.data_dir / "task_outputs" / "results.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
