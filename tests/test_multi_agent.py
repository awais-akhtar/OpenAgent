from __future__ import annotations

from tempfile import TemporaryDirectory
import json
import threading
import time
import unittest

from openagent.multi_agent import ApprovalRequired, MultiAgentRuntime
from openagent.storage import OpenAgentStore


class MultiAgentRuntimeTests(unittest.TestCase):
    def test_durable_agent_job_completion_cancellation_and_retry(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = MultiAgentRuntime(OpenAgentStore(tmp))
            entered = threading.Event()
            release = threading.Event()

            def controlled_run(agent_id, goal, **kwargs):
                entered.set()
                release.wait(timeout=5)
                return {"run": {"id": 42, "agent": agent_id, "goal": goal, "response": "done"}, "verification": {"status": "passed"}}

            runtime.run_agent = controlled_run  # type: ignore[method-assign]
            job = runtime.start_job("planning", "Prepare release", project="core", owner="local")
            self.assertTrue(entered.wait(timeout=2))
            cancelling = runtime.cancel_job(job["id"], owner="local")
            self.assertEqual(cancelling["status"], "cancelling")
            release.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                cancelled = runtime.get_job(job["id"], owner="local")
                if cancelled and cancelled["status"] == "cancelled":
                    break
                time.sleep(0.02)
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertEqual(cancelled["result"], {})
            self.assertEqual(cancelled["current_step"], "Cancellation confirmed")

            retry = runtime.retry_job(job["id"], owner="local")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                completed = runtime.get_job(retry["id"], owner="local")
                if completed and completed["status"] == "completed":
                    break
                time.sleep(0.02)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["progress"], 100)
            self.assertEqual(completed["attempt"], 2)
            self.assertEqual(completed["retry_of"], job["id"])
            self.assertEqual(len(runtime.jobs(project="core", owner="local")), 2)
            with self.assertRaises(KeyError):
                runtime.cancel_job(job["id"], owner="someone-else")

    def test_cancelled_failure_and_restart_recovery_are_truthful(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            runtime = MultiAgentRuntime(store)
            entered = threading.Event()
            release = threading.Event()

            def failing_after_cancel(*_args, **_kwargs):
                entered.set()
                release.wait(timeout=5)
                raise RuntimeError("provider disconnected after cancellation")

            runtime.run_agent = failing_after_cancel  # type: ignore[method-assign]
            job = runtime.start_job("planning", "Cancel safely", project="core", owner="local")
            self.assertTrue(entered.wait(timeout=2))
            runtime.cancel_job(job["id"], owner="local")
            release.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                recovered = runtime.get_job(job["id"], owner="local")
                if recovered and recovered["status"] == "cancelled":
                    break
                time.sleep(0.02)
            self.assertEqual(recovered["status"], "cancelled")
            self.assertEqual(recovered["error"], "")

            now = "2026-01-01T00:00:00+00:00"
            with store.session() as connection:
                for job_id, status, cancelled in (("dead-running", "running", 0), ("dead-cancelling", "cancelling", 1)):
                    connection.execute(
                        """INSERT INTO agent_execution_jobs
                           (id, project, owner, agent, goal, status, progress, current_step,
                            cancel_requested, attempt, retry_of, result_json, error, created_at, updated_at, completed_at)
                           VALUES (?, 'core', 'local', 'planning', 'Recover', ?, 10, 'old', ?, 1, '', '{}', '', ?, ?, '')""",
                        (job_id, status, cancelled, now, now),
                    )
            restarted = MultiAgentRuntime(OpenAgentStore(tmp))
            interrupted = restarted.get_job("dead-running", owner="local")
            cancelled_job = restarted.get_job("dead-cancelling", owner="local")
            self.assertEqual(interrupted["status"], "failed")
            self.assertIn("restarted", interrupted["error"])
            self.assertEqual(cancelled_job["status"], "cancelled")
            self.assertEqual(cancelled_job["result"], {})

    def test_permissions_delegation_parallel_verification_and_marketplace(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = MultiAgentRuntime(OpenAgentStore(tmp))

            self.assertTrue(runtime.has_permission("executive", "agent.run", project="core"))
            marketplace = runtime.marketplace()
            self.assertIn("executive", {agent["id"] for agent in marketplace})

            checked = runtime.run_agent("planning", "Plan Stage 3", project="core")
            self.assertEqual(checked["run"]["agent"], "planning")
            self.assertIn(checked["verification"]["status"], {"passed", "warning"})

            delegated = runtime.delegate(
                "executive",
                "research",
                "Find local knowledge gaps",
                project="core",
            )
            self.assertEqual(delegated["from_agent"], "executive")
            self.assertEqual(delegated["to_agent"], "research")

            parallel = runtime.run_parallel(
                [
                    {"agent": "planning", "goal": "Plan memory work"},
                    {"agent": "security", "goal": "Review permissions"},
                ],
                project="core",
            )
            self.assertEqual(parallel["status"], "completed")
            self.assertEqual(len(parallel["results"]), 2)

    def test_human_approval_checkpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = MultiAgentRuntime(OpenAgentStore(tmp))

            with self.assertRaises(ApprovalRequired) as raised:
                runtime.run_agent(
                    "automation",
                    "Run a sensitive automation",
                    project="core",
                    require_approval=True,
                )
            approval = json.loads(str(raised.exception))
            resolved = runtime.resolve_approval(approval["id"], approved=True)
            self.assertEqual(resolved["status"], "approved")

            result = runtime.run_agent(
                "automation",
                "Run a sensitive automation",
                project="core",
                require_approval=True,
                approval_id=approval["id"],
            )
            self.assertEqual(result["run"]["agent"], "automation")


if __name__ == "__main__":
    unittest.main()

