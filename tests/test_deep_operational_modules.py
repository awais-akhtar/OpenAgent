from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.agent_bridge import AgentBridge
from openagent.document_processing import DocumentProcessing
from openagent.image_studio import ImageStudio
from openagent.mcp_runtime import MCPRuntime
from openagent.safe_terminal import SafeTerminal
from openagent.scheduler import Scheduler
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore


def tiny_png() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )


class DeepOperationalModuleTests(unittest.TestCase):
    def test_image_studio_upload_rotate_and_library_are_local(self) -> None:
        with TemporaryDirectory() as tmp:
            studio = ImageStudio(OpenAgentStore(tmp))
            asset = studio.upload_asset(filename="pixel.png", content=tiny_png(), project="core", owner="u1")

            self.assertEqual(asset["width"], 1)
            self.assertEqual(studio.library(project="core", owner="u1")["count"], 1)

            rotated = studio.rotate(asset["id"], degrees=90, owner="u1")
            self.assertEqual(rotated["operation"]["operation"], "rotate")
            self.assertTrue(rotated["output_path"].endswith(".png"))
            self.assertEqual(len(studio.operations(asset["id"], owner="u1")), 1)

    def test_scheduler_runs_due_tasks_and_notifications(self) -> None:
        with TemporaryDirectory() as tmp:
            scheduler = Scheduler(OpenAgentStore(tmp))
            run_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).replace(microsecond=0).isoformat()
            task = scheduler.create_task(
                title="Check local queue",
                project="core",
                owner="u1",
                schedule={"run_at": run_at},
            )

            result = scheduler.run_due(project="core", owner="u1")
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["runs"][0]["task_id"], task["id"])
            self.assertEqual(len(scheduler.notifications(project="core", owner="u1")), 1)

    def test_safe_terminal_records_buffered_stream_events(self) -> None:
        with TemporaryDirectory() as tmp:
            terminal = SafeTerminal(OpenAgentStore(tmp), workspace_root=tmp)
            run = terminal.run(command=["python", "--version"], cwd=tmp, project="core", owner="u1")

            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["exit_code"], 0)
            self.assertIn("events", run)
            self.assertEqual(terminal.get_run(run["id"], owner="u1")["id"], run["id"])
            with self.assertRaises(KeyError):
                terminal.get_run(run["id"], owner="u2")

    def test_mcp_runtime_registers_servers_tools_and_permissions(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = MCPRuntime(OpenAgentStore(tmp))
            server = runtime.register_server(
                name="local-tools",
                project="core",
                owner="u1",
                tools=[{"name": "search", "description": "Local search"}],
            )

            self.assertEqual(server["status"], "configured")
            self.assertEqual(len(runtime.tools(project="core", owner="u1")), 1)
            patched = runtime.patch_tool(server["tools"][0]["id"], enabled=False, reason="test disable")
            self.assertFalse(patched["enabled"])
            self.assertEqual(runtime.permissions(project="core", owner="u1")[0]["reason"], "test disable")
            self.assertEqual(runtime.reconnect(server["id"])["status"], "ready")

    def test_document_processing_and_agent_bridge_write_local_records(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            processing = DocumentProcessing(store)
            imported = processing.import_office_or_text(
                filename="note.txt",
                content=b"hello from local document processing",
                project="core",
                owner="u1",
            )
            self.assertEqual(imported["document"]["owner"], "u1")

            bridge = AgentBridge(store)
            memory = bridge.write_memory(title="Decision", content="Use local-first bridge.", project="core")
            self.assertEqual(memory["source"], "agent-bridge")
            document = bridge.create_document(title="Bridge doc", content="Agent bridge content", project="core", owner="u1")
            self.assertEqual(document["title"], "Bridge doc")
            self.assertGreaterEqual(bridge.documents_list(project="core", owner="u1")["count"], 2)


class DeepOperationalRouteTests(unittest.TestCase):
    def test_new_runtime_routes_are_callable(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                handlers = json.loads(urlopen(f"{base_url}/api/routes/handlers", timeout=5).read().decode("utf-8"))
                self.assertGreaterEqual(handlers["count"], 10)

                upload = Request(
                    f"{base_url}/api/gallery/upload",
                    data=json.dumps(
                        {
                            "project": "core",
                            "filename": "pixel.png",
                            "content_base64": base64.b64encode(tiny_png()).decode("ascii"),
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                asset = json.loads(urlopen(upload, timeout=5).read().decode("utf-8"))["asset"]
                self.assertEqual(asset["width"], 1)

                task_req = Request(
                    f"{base_url}/api/tasks",
                    data=json.dumps({"project": "core", "title": "Notify", "schedule": {"delay_seconds": 0}}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                task = json.loads(urlopen(task_req, timeout=5).read().decode("utf-8"))["task"]
                self.assertEqual(task["title"], "Notify")

                server_req = Request(
                    f"{base_url}/api/mcp/servers",
                    data=json.dumps({"project": "core", "name": "local", "tools": [{"name": "echo"}]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                mcp_server = json.loads(urlopen(server_req, timeout=5).read().decode("utf-8"))["server"]
                self.assertEqual(mcp_server["tools"][0]["name"], "echo")

                terminal_req = Request(
                    f"{base_url}/api/terminal/stream",
                    data=json.dumps({"project": "core", "command": ["python", "--version"]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                terminal_run = json.loads(urlopen(terminal_req, timeout=5).read().decode("utf-8"))["run"]
                self.assertEqual(terminal_run["status"], "completed")

                memory_req = Request(
                    f"{base_url}/api/agent-bridge/memory",
                    data=json.dumps({"project": "core", "title": "Bridge", "content": "Local bridge memory"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                event = json.loads(urlopen(memory_req, timeout=5).read().decode("utf-8"))["event"]
                self.assertEqual(event["source"], "agent-bridge")

                import_req = Request(
                    f"{base_url}/api/documents/import",
                    data=json.dumps(
                        {
                            "project": "core",
                            "title": "Imported note",
                            "filename": "note.txt",
                            "content_base64": base64.b64encode(b"route import document").decode("ascii"),
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                imported = json.loads(urlopen(import_req, timeout=5).read().decode("utf-8"))["document"]
                self.assertEqual(imported["title"], "Imported note")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
