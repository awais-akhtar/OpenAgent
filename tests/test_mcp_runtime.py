from __future__ import annotations

import os
import sys
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from openagent.mcp_runtime import MCPRuntime
from openagent.storage import OpenAgentStore


class MCPStdioBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.runtime = MCPRuntime(OpenAgentStore(self.workspace.name))

    def invoke(self, code: str, *, timeout: float = 1, limit: int = 1_000):
        server = self.runtime.register_server(
            name="stdio-test",
            command=[sys.executable, "-u", "-c", code],
            tools=[{"name": "echo", "approval_required": False}],
        )
        return self.runtime.invoke_tool(
            server["tools"][0]["id"], timeout_seconds=timeout, max_output_bytes=limit,
        )

    def test_json_rpc_exchange_and_filtered_environment(self) -> None:
        code = (
            "import json,os,sys\n"
            "for line in sys.stdin:\n"
            " msg=json.loads(line)\n"
            " if msg.get('id') == 1:\n"
            "  print(json.dumps({'jsonrpc':'2.0','id':1,'result':{}}),flush=True)\n"
            " elif msg.get('id') == 2:\n"
            "  print(json.dumps({'jsonrpc':'2.0','id':2,'result':{'secret':os.getenv('MCP_BOUNDARY_SECRET'),'name':msg['params']['name']}}),flush=True)\n"
        )
        with patch.dict(os.environ, {"MCP_BOUNDARY_SECRET": "hidden"}):
            result = self.invoke(code)
        self.assertEqual(result["response"], {"secret": None, "name": "echo"})

    def test_missing_executable_is_rejected(self) -> None:
        server = self.runtime.register_server(
            name="missing", command=["missing-openagent-mcp-executable-12345"],
            tools=[{"name": "echo", "approval_required": False}],
        )
        with self.assertRaisesRegex(FileNotFoundError, "executable not found"):
            self.runtime.invoke_tool(server["tools"][0]["id"])
        self.assertIn(
            "invoke_failed",
            {event["event_type"] for event in self.runtime.get_server(server["id"])["events"]},
        )

    def test_json_rpc_error_is_reported(self) -> None:
        code = (
            "import json,sys\n"
            "for line in sys.stdin:\n"
            " msg=json.loads(line)\n"
            " if msg.get('id'):\n"
            "  print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'error':{'code':-32000,'message':'failed'}}),flush=True)\n"
        )
        with self.assertRaisesRegex(ValueError, "failed"):
            self.invoke(code)

    def test_timeout_returns_promptly(self) -> None:
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "in time"):
            self.invoke("import time; time.sleep(10)", timeout=0.2)
        self.assertLess(time.monotonic() - started, 3)

    def test_stdout_and_stderr_are_capped_before_response(self) -> None:
        for stream in ("stdout", "stderr"):
            with self.subTest(stream=stream):
                code = f"import sys,time; sys.{stream}.write('x'*5000); sys.{stream}.flush(); time.sleep(10)"
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "output exceeded"):
                    self.invoke(code, timeout=2, limit=1_000)
                self.assertLess(time.monotonic() - started, 3)

    def test_requested_output_limit_has_a_hard_ceiling(self) -> None:
        code = "import sys,time; sys.stdout.write('x'*1100000); sys.stdout.flush(); time.sleep(10)"
        with self.assertRaisesRegex(RuntimeError, "output exceeded"):
            self.invoke(code, timeout=2, limit=100_000_000)

    def test_http_tool_uses_approved_local_egress(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                response = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {
                    "tool": body["params"]["name"], "arguments": body["params"]["arguments"],
                }}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            registered = self.runtime.register_server(
                name="local-http", transport="http",
                endpoint=f"http://127.0.0.1:{server.server_address[1]}/mcp",
                tools=[{"name": "echo", "approval_required": False}],
            )
            tool_id = registered["tools"][0]["id"]
            with self.assertRaisesRegex(PermissionError, "approval id"):
                self.runtime.invoke_tool(tool_id, arguments={"message": "hello"})
            result = self.runtime.invoke_tool(
                tool_id, arguments={"message": "hello"}, approval_id="test-approval",
            )
            self.assertEqual(result["response"], {"tool": "echo", "arguments": {"message": "hello"}})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
