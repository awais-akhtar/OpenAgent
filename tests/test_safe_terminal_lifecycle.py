from __future__ import annotations

import json
import sys
import time
import unittest
from tempfile import TemporaryDirectory
import threading
from urllib.request import Request, urlopen

from openagent.safe_terminal import SafeTerminal
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


class SafeTerminalLifecycleTests(unittest.TestCase):
    def test_async_run_persists_incremental_redacted_output_and_completion(self) -> None:
        with TemporaryDirectory() as tmp:
            terminal = SafeTerminal(OpenAgentStore(tmp), workspace_root=tmp)
            script = "print('token=do-not-store', flush=True)"
            run = terminal.start(
                command=[sys.executable, "-u", "-c", script],
                cwd=tmp,
                project="core",
                owner="owner-1",
                approved=True,
            )
            self.assertIn(run["status"], {"running", "completed"})
            deadline = time.monotonic() + 10
            observed = terminal.get_run(run["id"], owner="owner-1")
            while observed["status"] == "running" and not observed["events"] and time.monotonic() < deadline:
                time.sleep(0.02)
                observed = terminal.get_run(run["id"], owner="owner-1")
            self.assertTrue(observed["events"])
            self.assertIn("token=<redacted>", "".join(item["chunk"] for item in observed["events"]))
            self.assertNotIn("do-not-store", "".join(item["chunk"] for item in observed["events"]))
            while observed["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.02)
                observed = terminal.get_run(run["id"], owner="owner-1")
            self.assertEqual(observed["status"], "completed")
            self.assertEqual(observed["streaming_mode"], "incremental-polling")
            with self.assertRaises(KeyError):
                terminal.get_run(run["id"], owner="owner-2")

    def test_cancel_is_owner_scoped_and_reports_actual_terminal_state(self) -> None:
        with TemporaryDirectory() as tmp:
            terminal = SafeTerminal(OpenAgentStore(tmp), workspace_root=tmp)
            script = "import time\nprint('ready', flush=True)\ntime.sleep(10)"
            run = terminal.start(
                command=[sys.executable, "-u", "-c", script],
                cwd=tmp,
                project="core",
                owner="owner-1",
                timeout_seconds=20,
                approved=True,
            )
            deadline = time.monotonic() + 5
            current = terminal.get_run(run["id"], owner="owner-1")
            while not current["events"] and current["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.02)
                current = terminal.get_run(run["id"], owner="owner-1")
            with self.assertRaises(KeyError):
                terminal.cancel(run["id"], owner="owner-2")
            requested = terminal.cancel(run["id"], owner="owner-1")
            self.assertTrue(requested["cancel_requested"])
            deadline = time.monotonic() + 5
            current = terminal.get_run(run["id"], owner="owner-1")
            while current["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.02)
                current = terminal.get_run(run["id"], owner="owner-1")
            self.assertEqual(current["status"], "cancelled")
            after_complete = terminal.cancel(run["id"], owner="owner-1")
            self.assertFalse(after_complete["cancel_requested"])
            self.assertEqual(after_complete["run"]["status"], "cancelled")

    def test_restart_marks_persisted_running_rows_interrupted(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            terminal = SafeTerminal(store, workspace_root=tmp)
            with store.session() as connection:
                connection.execute(
                    """INSERT INTO terminal_runs
                       (id, project, owner, cwd, command_json, status, started_at)
                       VALUES ('stale-run', 'core', 'owner-1', ?, '[\"python\"]', 'running', '2026-10-01T00:00:00+00:00')""",
                    (tmp,),
                )
            recovered = SafeTerminal(store, workspace_root=tmp).get_run("stale-run", owner="owner-1")
            self.assertEqual(recovered["status"], "interrupted")
            self.assertTrue(recovered["completed_at"])

    def test_clean_shutdown_terminates_supervised_child_and_closes_run(self) -> None:
        with TemporaryDirectory() as tmp:
            terminal = SafeTerminal(OpenAgentStore(tmp), workspace_root=tmp)
            run = terminal.start(
                command=[sys.executable, "-u", "-c", "import time\nprint('ready', flush=True)\ntime.sleep(30)"],
                cwd=tmp,
                project="core",
                owner="owner-1",
                timeout_seconds=60,
                approved=True,
            )
            terminal.shutdown()
            current = terminal.get_run(run["id"], owner="owner-1")
            self.assertEqual(current["status"], "cancelled")

    def test_start_keeps_command_approval_and_shell_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            terminal = SafeTerminal(OpenAgentStore(tmp), workspace_root=tmp)
            with self.assertRaisesRegex(ValueError, "approved=true"):
                terminal.start(command=["python", "--help"], cwd=tmp, owner="owner-1")
            with self.assertRaisesRegex(ValueError, "shell metacharacters"):
                terminal.start(command=["python", "-c", "print(1);print(2)"], cwd=tmp, approved=True)

    def test_http_async_terminal_start_poll_and_cancel(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                payload = {
                    "project": "core",
                    "command": [sys.executable, "-u", "-c", "import time\nprint('ready', flush=True)\ntime.sleep(10)"],
                    "cwd": tmp,
                    "timeout_seconds": 20,
                    "approved": True,
                }
                start_request = Request(
                    f"{base}/api/terminal/runs",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(start_request, timeout=5) as response:
                    self.assertEqual(response.status, 202)
                    run = json.loads(response.read().decode())["run"]
                self.assertEqual(run["status"], "running")
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    with urlopen(f"{base}/api/terminal/runs/{run['id']}", timeout=5) as response:
                        current = json.loads(response.read().decode())["run"]
                    if current["events"]:
                        break
                    time.sleep(0.03)
                self.assertTrue(current["events"])
                stop_request = Request(f"{base}/api/terminal/runs/{run['id']}", method="DELETE")
                with urlopen(stop_request, timeout=5) as response:
                    stop = json.loads(response.read().decode())
                self.assertTrue(stop["cancel_requested"])
                deadline = time.monotonic() + 5
                while current["status"] == "running" and time.monotonic() < deadline:
                    time.sleep(0.03)
                    with urlopen(f"{base}/api/terminal/runs/{run['id']}", timeout=5) as response:
                        current = json.loads(response.read().decode())["run"]
                self.assertEqual(current["status"], "cancelled")
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
