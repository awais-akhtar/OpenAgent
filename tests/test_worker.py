from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.server import build_http_server
from openagent.storage import OpenAgentStore
from openagent.worker import WorkerRuntime


class WorkerRuntimeTests(unittest.TestCase):
    def test_worker_completes_echo_job_and_records_status(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            worker = WorkerRuntime(store, worker_id="test-worker")

            job = worker.enqueue("echo", {"message": "hello"}, project="core", priority=5)
            self.assertEqual(job["state"], "queued")
            self.assertEqual(store.worker_status(project="core")["queue_depth"], 1)

            result = worker.run_once()
            self.assertEqual(result["status"], "completed")
            completed = store.get_job(job["id"])
            self.assertIsNotNone(completed)
            self.assertEqual(completed["state"], "completed")
            self.assertEqual(completed["result"]["echo"]["message"], "hello")
            self.assertEqual(store.worker_status(project="core")["states"]["completed"], 1)

    def test_abandoned_lease_is_reclaimed(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            job = store.enqueue_job("echo", {"value": 1}, project="core")
            leased = store.lease_job("first-worker", lease_seconds=5)
            self.assertEqual(leased["id"], job["id"])

            with store.session() as connection:
                connection.execute(
                    "UPDATE job_locks SET lease_until = '2000-01-01T00:00:00+00:00' WHERE job_id = ?",
                    (job["id"],),
                )

            reclaimed = store.lease_job("second-worker", lease_seconds=5)
            self.assertIsNotNone(reclaimed)
            self.assertEqual(reclaimed["id"], job["id"])
            self.assertEqual(reclaimed["lease"]["worker_id"], "second-worker")
            self.assertEqual(reclaimed["attempts"], 2)

    def test_failed_unknown_job_dead_letters_after_max_attempts(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            worker = WorkerRuntime(store, worker_id="test-worker")
            job = worker.enqueue("missing.handler", {"x": 1}, project="core", max_attempts=1)

            result = worker.run_once(retry_delay_seconds=1)
            self.assertEqual(result["status"], "failed")
            failed = store.get_job(job["id"])
            self.assertIsNotNone(failed)
            self.assertEqual(failed["state"], "dead_lettered")
            dead_letters = store.list_dead_letter_jobs(project="core")
            self.assertEqual(len(dead_letters), 1)
            self.assertEqual(dead_letters[0]["job_id"], job["id"])


class WorkerRouteTests(unittest.TestCase):
    def test_worker_routes_enqueue_run_and_report(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                create = Request(
                    f"{base_url}/api/worker/jobs",
                    data=json.dumps(
                        {
                            "project": "core",
                            "job_type": "echo",
                            "payload": {"message": "from api"},
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                created = json.loads(urlopen(create, timeout=5).read().decode("utf-8"))
                self.assertEqual(created["job"]["state"], "queued")

                run = Request(f"{base_url}/api/worker/run-once", data=b"{}", method="POST")
                ran = json.loads(urlopen(run, timeout=5).read().decode("utf-8"))
                self.assertEqual(ran["status"], "completed")

                status = json.loads(urlopen(f"{base_url}/api/worker/status?project=core", timeout=5).read().decode("utf-8"))
                self.assertEqual(status["states"]["completed"], 1)
                jobs = json.loads(urlopen(f"{base_url}/api/worker/jobs?project=core", timeout=5).read().decode("utf-8"))
                self.assertEqual(jobs["jobs"][0]["result"]["echo"]["message"], "from api")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
