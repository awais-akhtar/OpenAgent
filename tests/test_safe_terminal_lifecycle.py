from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from urllib.request import Request, urlopen

from openagent.routes.terminal_routes import TerminalRouteDependencies, handle_delete, handle_get, handle_post
from openagent.safe_terminal import SHELL_MAX_SESSIONS, SafeTerminal, ShellSessionError, ShellSessionManager
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


def _wait_for(predicate, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class ShellSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name) / "project"
        (self.root / "sub").mkdir(parents=True)
        self.manager = ShellSessionManager()

    def tearDown(self) -> None:
        self.manager.close_all()
        self._tmp.cleanup()

    def _create(self, owner: str = "alice", **kwargs) -> dict:
        return self.manager.create(owner=owner, project_root=self.root, approved=True, **kwargs)

    def test_creating_a_session_requires_explicit_approval(self) -> None:
        with self.assertRaises(PermissionError):
            self.manager.create(owner="alice", project_root=self.root, approved=False)

    def test_working_directory_must_stay_inside_the_project(self) -> None:
        with self.assertRaises(PermissionError):
            self._create(cwd="..")
        self.assertTrue(self._create(cwd="sub")["cwd"].endswith("sub"))

    def test_streams_output_and_accepts_input(self) -> None:
        session = self._create()
        self.manager.write("alice", session["id"], "echo STREAM_7\r\n")
        self.assertTrue(_wait_for(lambda: "STREAM_7" in self.manager.read("alice", session["id"])["output"]))

    def test_resize_applies_to_the_live_terminal(self) -> None:
        session = self._create(cols=80, rows=24)
        resized = self.manager.resize("alice", session["id"], 100, 40)
        self.assertEqual((100, 40), (resized["cols"], resized["rows"]))

    def test_another_owner_cannot_see_write_read_resize_or_close(self) -> None:
        session = self._create(owner="alice")
        for action in (
            lambda: self.manager.read("mallory", session["id"]),
            lambda: self.manager.write("mallory", session["id"], "echo x\r\n"),
            lambda: self.manager.resize("mallory", session["id"], 80, 24),
            lambda: self.manager.close("mallory", session["id"]),
        ):
            with self.assertRaises(KeyError):
                action()
        self.assertTrue(self.manager.describe("alice", session["id"])["running"])

    def test_close_terminates_the_process_and_forgets_the_session(self) -> None:
        session = self._create()
        self.manager.close("alice", session["id"])
        with self.assertRaises(KeyError):
            self.manager.read("alice", session["id"])

    def test_exited_shell_is_reaped(self) -> None:
        session = self._create()
        self.manager.write("alice", session["id"], "exit\r\n")
        self.assertTrue(_wait_for(lambda: not self.manager.describe("alice", session["id"])["running"]))
        self.assertEqual(1, self.manager.reap_idle())

    def test_idle_session_is_reaped_after_the_timeout(self) -> None:
        now = [1000.0]
        manager = ShellSessionManager(clock=lambda: now[0])
        try:
            session = manager.create(owner="alice", project_root=self.root, approved=True)
            now[0] += 10_000
            self.assertEqual(1, manager.reap_idle())
            with self.assertRaises(KeyError):
                manager.describe("alice", session["id"])
        finally:
            manager.close_all()

    def test_session_count_is_capped_per_owner(self) -> None:
        for _ in range(SHELL_MAX_SESSIONS):
            self._create(owner="alice")
        with self.assertRaises(ShellSessionError):
            self._create(owner="alice")
        self.assertTrue(self._create(owner="bob")["id"])

    def test_oversized_input_is_refused(self) -> None:
        session = self._create()
        with self.assertRaises(ShellSessionError):
            self.manager.write("alice", session["id"], "x" * 5000)


class ShellSessionRouteTests(unittest.TestCase):
    """Session routes: approval, owner scope, input, output, resize and close over the route contract."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self._tmp.name)
        self.terminal = SafeTerminal(OpenAgentStore(root / "data"), workspace_root=root)

    def tearDown(self) -> None:
        self.terminal.shell_sessions.close_all()
        self._tmp.cleanup()

    def _deps(self, body: dict, owner: str = "alice") -> TerminalRouteDependencies:
        def parse_bool(value, default=False):
            return default if value is None else bool(value)

        def parse_int(value, default=0):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def first(params, key):
            values = params.get(key) or [""]
            return values[0]

        return TerminalRouteDependencies(
            terminal=self.terminal, first=first, parse_bool=parse_bool, parse_int=parse_int,
            query_owner=lambda params: owner, payload_owner=lambda payload: owner, read_json=lambda: body,
        )

    def test_session_routes_require_approval_and_keep_output_owner_scoped(self) -> None:
        refused = handle_post("/api/terminal/sessions", self._deps({"approved": False}))
        self.assertEqual(403, refused.status)
        created = handle_post("/api/terminal/sessions", self._deps({"approved": True}))
        self.assertEqual(201, created.status)
        session_id = created.payload["session"]["id"]

        handle_post(f"/api/terminal/sessions/{session_id}/input", self._deps({"text": "echo ROUTE_OK\r\n"}))
        self.assertTrue(_wait_for(lambda: "ROUTE_OK" in handle_get(f"/api/terminal/sessions/{session_id}", "", self._deps({})).payload["output"]["output"]))

        other = handle_get(f"/api/terminal/sessions/{session_id}", "", self._deps({}, owner="mallory"))
        self.assertEqual(404, other.status)

        resized = handle_post(f"/api/terminal/sessions/{session_id}/resize", self._deps({"cols": 100, "rows": 40}))
        self.assertEqual((100, 40), (resized.payload["session"]["cols"], resized.payload["session"]["rows"]))

        closed = handle_delete(f"/api/terminal/sessions/{session_id}", "", self._deps({}))
        self.assertFalse(closed.payload["running"])
        self.assertEqual(404, handle_get(f"/api/terminal/sessions/{session_id}", "", self._deps({})).status)
