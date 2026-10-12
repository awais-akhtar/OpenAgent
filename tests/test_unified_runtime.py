from __future__ import annotations

import json
import sys
import time
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.request import Request, urlopen

from openagent.compute_router import ComputeRequest, ComputeRouter
from openagent.mcp_runtime import MCPRuntime
from openagent.model_gateway import ModelGateway
from openagent.safe_terminal import SafeTerminal
from openagent.server import build_http_server
from openagent.storage import OpenAgentStore
from openagent.tools.runtime import ToolRuntime, ToolSpec


class UnifiedToolRuntimeTests(unittest.TestCase):
    def test_safe_terminal_adapter_returns_traceable_result(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            runtime = ToolRuntime(store)
            terminal = SafeTerminal(store, workspace_root=tmp)
            from openagent.tools.runtime import register_safe_terminal

            register_safe_terminal(runtime, terminal)
            result = runtime.execute_named("terminal.run", {"command": ["python", "--version"]}, project="core", owner="u1")
            self.assertTrue(result.success)
            self.assertEqual(result.status, "succeeded")
            self.assertTrue(result.trace_id)
            self.assertIn("terminal.run", [item["id"] for item in runtime.describe()["tools"]])

    def test_policy_requires_approval_for_external_tool(self) -> None:
        runtime = ToolRuntime()
        runtime.registry.register(
            ToolSpec("web.fetch", "Fetch one approved URL", risk=2, network="external"),
            lambda arguments, context: {"url": arguments.get("url")},
        )
        result = runtime.execute_named("web.fetch", {"url": "https://example.com"}, network="external")
        self.assertFalse(result.success)
        self.assertEqual(result.status, "needs_approval")

    def test_request_rejects_invalid_boundary_values(self) -> None:
        from openagent.tools.runtime import ToolRequest

        with self.assertRaisesRegex(ValueError, "network"):
            ToolRequest("terminal.run", network="internet")
        with self.assertRaisesRegex(ValueError, "timeout"):
            ToolRequest("terminal.run", timeout_seconds=0)

    def test_timeout_signals_a_cooperative_tool_to_stop(self) -> None:
        runtime = ToolRuntime()
        stopped = threading.Event()

        def wait_until_cancelled(_arguments, context):
            try:
                while True:
                    context.check()
                    time.sleep(0.005)
            finally:
                stopped.set()

        runtime.registry.register(ToolSpec("test.wait", "Wait for cancellation"), wait_until_cancelled)
        result = runtime.execute_named("test.wait", timeout_seconds=0.05)
        self.assertEqual(result.status, "timeout")
        self.assertTrue(stopped.wait(1))

    def test_compute_router_respects_local_only_policy(self) -> None:
        router = ComputeRouter(ModelGateway())
        result = router.plan(ComputeRequest(task="write code", allow_external=False))
        self.assertIn(result["status"], {"ready", "needs_setup"})
        self.assertTrue(all(item["local"] or item["score"] < 0 for item in result["candidates"]))


class LiveMCPInvocationTests(unittest.TestCase):
    def test_stdio_tool_call_is_bounded_and_audited(self) -> None:
        server_code = (
            "import json,sys\n"
            "for line in sys.stdin:\n"
            "    msg=json.loads(line)\n"
            "    if msg.get('id') == 1:\n"
            "        print(json.dumps({'jsonrpc':'2.0','id':1,'result':{'protocolVersion':'2024-11-05'}}), flush=True)\n"
            "    elif msg.get('method') == 'tools/call':\n"
            "        print(json.dumps({'jsonrpc':'2.0','id':2,'result':{'content':[{'type':'text','text':'ok'}]}}), flush=True)\n"
        )
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            runtime = MCPRuntime(store)
            server = runtime.register_server(
                name="test-server",
                project="core",
                owner="u1",
                command=[sys.executable, "-u", "-c", server_code],
                tools=[{"name": "echo", "description": "Echo test", "approval_required": False}],
            )
            result = runtime.invoke_tool(server["tools"][0]["id"], project="core", owner="u1", arguments={})
            self.assertEqual(result["response"]["content"][0]["text"], "ok")
            events = runtime.get_server(server["id"])["events"]
            self.assertIn("invoke_completed", [event["event_type"] for event in events])


class RuntimeRouteTests(unittest.TestCase):
    def test_shared_runtime_and_compute_routes_are_live(self) -> None:
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                runtime = json.loads(urlopen(f"{base_url}/api/tools/runtime", timeout=10).read().decode("utf-8"))
                self.assertIn("terminal.run", [item["id"] for item in runtime["tools"]])
                plan_request = Request(
                    f"{base_url}/api/compute/plan",
                    data=json.dumps({"task": "code review", "allow_external": False}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                plan = json.loads(urlopen(plan_request, timeout=10).read().decode("utf-8"))
                self.assertIn(plan["status"], {"ready", "needs_setup"})
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
