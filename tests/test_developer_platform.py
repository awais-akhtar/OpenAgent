from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.context import ContextEngine
from openagent.developer import DeveloperPlatform
from openagent.storage import OpenAgentStore


class DeveloperPlatformTests(unittest.TestCase):
    def test_developer_platform_lifecycle(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            context = ContextEngine(store)
            context.ingest(
                "Developer note",
                "OpenAgent developer platform keeps extension records local.",
                project="core",
            )
            platform = DeveloperPlatform(
                store,
                resolvers={
                    "health": lambda args: store.health(),
                    "context": lambda args: context.snapshot(project=str(args.get("project", "")).strip() or None),
                    "search_context": lambda query, project="default", limit=10: context.search(
                        query,
                        project=project,
                        limit=limit,
                    ),
                },
            )

            key = platform.create_api_key(label="Local SDK", scopes=["api.read", "tool.invoke"])
            self.assertTrue(key["key"].startswith("oa_dev_"))
            self.assertTrue(platform.verify_api_key(key["key"], scope="api.read")["valid"])
            self.assertFalse(platform.verify_api_key(key["key"], scope="admin")["valid"])

            graphql = platform.execute_graphql('{ health developerStatus context(project: "core") }')
            self.assertIn("health", graphql["data"])
            self.assertEqual(graphql["data"]["context"]["project"], "core")
            self.assertGreaterEqual(graphql["data"]["developerStatus"]["tools"], 3)

            webhook = platform.register_webhook(event="openagent.test", target_url="log://local")
            delivered = platform.dispatch_webhook(event="openagent.test", payload={"ok": True})
            self.assertEqual(delivered["deliveries"][0]["webhook_id"], webhook["id"])
            self.assertEqual(delivered["deliveries"][0]["status"], "delivered")

            platform.register_webhook(event="openagent.external", target_url="https://example.com/hook")
            blocked = platform.dispatch_webhook(event="openagent.external", payload={"ok": True})
            self.assertEqual(blocked["deliveries"][0]["status"], "blocked")

            plugin = platform.register_plugin(
                {
                    "id": "local-notes",
                    "name": "Local Notes",
                    "version": "0.1.0",
                    "entrypoint": "local_notes:plugin",
                    "permissions": ["context.read"],
                }
            )
            self.assertEqual(plugin["name"], "Local Notes")

            tool = platform.register_tool(
                tool_id="local.echo",
                name="Local Echo",
                handler="echo",
                schema={"message": "string"},
            )
            invocation = platform.invoke_tool(tool["id"], input_data={"message": "hello"})
            self.assertEqual(invocation["output"]["echo"]["message"], "hello")

            approval = platform.invoke_tool(
                "context.search",
                input_data={"query": "extension", "project": "core"},
            )
            self.assertEqual(approval["status"], "approval_required")
            search = platform.invoke_tool(
                "context.search",
                input_data={"query": "extension", "project": "core"},
                approved=True,
            )
            self.assertEqual(len(search["output"]["results"]), 1)

            mcp_server = platform.register_mcp_server(
                name="Local Context MCP",
                command="openagent-mcp",
                args=["context"],
            )
            self.assertEqual(mcp_server["name"], "Local Context MCP")
            self.assertIn("tools", platform.mcp_manifest())

            package = platform.publish_extension_package(
                package_type="plugin",
                name="local-notes",
                version="0.1.0",
                source="local",
                manifest={"plugin_id": plugin["id"]},
            )
            self.assertEqual(package["package_type"], "plugin")

            sdk = platform.generate_sdk(language="python")
            self.assertIn("openagent_client.py", sdk["files"])
            self.assertIn("agent_sdk.py", sdk["files"])
            self.assertIn("workflow_sdk.py", sdk["files"])
            self.assertIn("plugin_sdk.py", sdk["files"])
            self.assertEqual(len(platform.sdk_artifacts()), 1)

    def test_graphql_unknown_field_reports_error(self) -> None:
        with TemporaryDirectory() as tmp:
            platform = DeveloperPlatform(OpenAgentStore(tmp))

            result = platform.execute_graphql("{ missingField developerStatus }")
            self.assertIn("missingField", result["errors"][0]["field"])
            self.assertIn("developerStatus", result["data"])


if __name__ == "__main__":
    unittest.main()
