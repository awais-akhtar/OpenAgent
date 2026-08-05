"""Stage 8 developer platform and extensibility layer."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import re
import secrets
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class DeveloperPlatformError(RuntimeError):
    """Raised when a developer platform operation is invalid or unsafe."""


Resolver = Callable[..., Any]


class DeveloperPlatform:
    def __init__(self, store: OpenAgentStore, resolvers: dict[str, Resolver] | None = None):
        self.store = store
        self.resolvers = resolvers or {}
        self.initialize()
        self.seed_defaults()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS developer_api_keys (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    scopes_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS developer_webhooks (
                    id TEXT PRIMARY KEY,
                    event TEXT NOT NULL,
                    target_url TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    secret_hint TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_dev_webhooks_event_enabled
                    ON developer_webhooks(event, enabled);

                CREATE TABLE IF NOT EXISTS developer_webhook_deliveries (
                    id TEXT PRIMARY KEY,
                    webhook_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_dev_webhook_deliveries_created
                    ON developer_webhook_deliveries(created_at DESC);

                CREATE TABLE IF NOT EXISTS developer_plugins (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    entrypoint TEXT NOT NULL DEFAULT '',
                    permissions_json TEXT NOT NULL DEFAULT '[]',
                    manifest_json TEXT NOT NULL DEFAULT '{}',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    installed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS developer_tools (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    schema_json TEXT NOT NULL DEFAULT '{}',
                    handler TEXT NOT NULL DEFAULT 'registered',
                    permissions_json TEXT NOT NULL DEFAULT '[]',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS developer_tool_invocations (
                    id TEXT PRIMARY KEY,
                    tool_id TEXT NOT NULL,
                    input_json TEXT NOT NULL DEFAULT '{}',
                    output_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS developer_mcp_servers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    command TEXT NOT NULL,
                    args_json TEXT NOT NULL DEFAULT '[]',
                    env_json TEXT NOT NULL DEFAULT '{}',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS developer_extension_packages (
                    id TEXT PRIMARY KEY,
                    package_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'available',
                    manifest_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS developer_sdk_artifacts (
                    id TEXT PRIMARY KEY,
                    language TEXT NOT NULL,
                    name TEXT NOT NULL,
                    files_json TEXT NOT NULL DEFAULT '{}',
                    generated_at TEXT NOT NULL
                );
                """
            )

    def seed_defaults(self) -> None:
        self.register_tool(
            tool_id="echo",
            name="Echo",
            description="Return the input payload unchanged.",
            handler="echo",
            schema={"input": "object", "output": "object"},
            audit=False,
        )
        self.register_tool(
            tool_id="context.search",
            name="Context Search",
            description="Search local OpenAgent context without leaving the data directory.",
            handler="context.search",
            permissions=["context.read"],
            schema={"query": "string", "project": "string", "limit": "integer"},
            audit=False,
        )
        self.register_tool(
            tool_id="graphql.query",
            name="GraphQL Query",
            description="Run a local read-only GraphQL-style query against OpenAgent.",
            handler="graphql.query",
            permissions=["api.read"],
            schema={"query": "string", "variables": "object"},
            audit=False,
        )
        if not self.extension_packages():
            self.publish_extension_package(
                package_type="template",
                name="starter-plugin",
                version="0.1.0",
                source="builtin",
                manifest={
                    "description": "Starter plugin manifest for local extensions.",
                    "permissions": ["context.read"],
                },
                audit=False,
            )

    def status(self) -> dict[str, Any]:
        return {
            "api_keys": len(self.api_keys()),
            "webhooks": len(self.webhooks()),
            "plugins": len(self.plugins()),
            "tools": len(self.tools()),
            "mcp_servers": len(self.mcp_servers()),
            "extension_packages": len(self.extension_packages()),
            "sdk_artifacts": len(self.sdk_artifacts()),
            "graphql_fields": sorted(self.graphql_schema()["fields"]),
            "network_default": False,
        }

    def create_api_key(self, *, label: str, scopes: list[str] | None = None) -> dict[str, Any]:
        label = label.strip()
        if not label:
            raise ValueError("API key label is required.")
        key = f"oa_dev_{secrets.token_urlsafe(24)}"
        record_id = unique_id("api-key", label)
        created_at = utc_now()
        scopes = scopes or ["api.read"]
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO developer_api_keys (
                    id, label, key_hash, scopes_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (record_id, label, hash_secret(key), json.dumps(scopes, sort_keys=True), "active", created_at),
            )
        self._audit("developer.api_key.create", record_id, metadata={"label": label, "scopes": scopes})
        return {**(self.get_api_key(record_id) or {}), "key": key}

    def get_api_key(self, key_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM developer_api_keys WHERE id = ?", (key_id,)).fetchone()
        return api_key_row(row) if row else None

    def api_keys(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM developer_api_keys ORDER BY created_at DESC").fetchall()
        return [api_key_row(row) for row in rows]

    def verify_api_key(self, key: str, *, scope: str | None = None) -> dict[str, Any]:
        key_hash = hash_secret(key)
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT * FROM developer_api_keys WHERE key_hash = ? AND status = 'active'",
                (key_hash,),
            ).fetchone()
            if row:
                connection.execute(
                    "UPDATE developer_api_keys SET last_used_at = ? WHERE id = ?",
                    (utc_now(), row["id"]),
                )
        if not row:
            return {"valid": False, "reason": "unknown_key"}
        record = api_key_row(row)
        if scope and scope not in record["scopes"] and "*" not in record["scopes"]:
            return {"valid": False, "reason": "scope_denied", "record": record}
        return {"valid": True, "record": self.get_api_key(record["id"])}

    def graphql_schema(self) -> dict[str, Any]:
        fields = {
            "health": "OpenAgent health record",
            "context": "Unified context snapshot",
            "agents": "Built-in agent specs",
            "models": "Model provider specs",
            "workflows": "Workflow list",
            "workspace": "Workspace dashboard",
            "assistant": "Assistant daily brief",
            "semantic": "Semantic memory status",
            "media": "Media ingestion status",
            "voice": "Voice runtime status",
            "production": "Production runtime status",
            "developerStatus": "Developer platform status",
            "webhooks": "Registered webhooks",
            "plugins": "Registered plugins",
            "tools": "Registered tools",
            "mcp": "Local MCP manifest",
            "extensions": "Extension package catalog",
        }
        return {"fields": fields, "mode": "read_only", "network": False}

    def execute_graphql(
        self,
        query: str,
        *,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        variables = variables or {}
        fields = parse_graphql_root_fields(query)
        if not fields:
            raise ValueError("GraphQL query must select at least one root field.")
        data: dict[str, Any] = {}
        errors = []
        for alias, name, args in fields:
            merged_args = {**variables, **args}
            try:
                data[alias] = self.resolve_graphql_field(name, merged_args)
            except KeyError:
                errors.append({"field": name, "message": f"Unknown GraphQL field '{name}'."})
        self._audit(
            "developer.graphql.query",
            "graphql",
            metadata={"fields": [name for _, name, _ in fields], "error_count": len(errors)},
        )
        result: dict[str, Any] = {"data": data, "extensions": {"local": True, "network": False}}
        if errors:
            result["errors"] = errors
        return result

    def resolve_graphql_field(self, name: str, args: dict[str, Any]) -> Any:
        if name == "developerStatus":
            return self.status()
        if name == "webhooks":
            return self.webhooks()
        if name == "plugins":
            return self.plugins()
        if name == "tools":
            return self.tools()
        if name == "mcp":
            return self.mcp_manifest()
        if name == "extensions":
            return self.extension_packages()
        resolver = self.resolvers.get(name)
        if resolver is None:
            raise KeyError(name)
        return resolver(args)

    def register_webhook(
        self,
        *,
        event: str,
        target_url: str,
        enabled: bool = True,
        secret_hint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = event.strip() or "*"
        target_url = target_url.strip()
        parsed = urlparse(target_url)
        if parsed.scheme not in {"http", "https", "log"}:
            raise ValueError("Webhook target_url must use http, https, or log scheme.")
        webhook_id = stable_id("webhook", event, target_url)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO developer_webhooks (
                    id, event, target_url, enabled, secret_hint, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM developer_webhooks WHERE id = ?), ?), ?)
                """,
                (
                    webhook_id,
                    event,
                    target_url,
                    1 if enabled else 0,
                    secret_hint,
                    json.dumps(metadata or {}, sort_keys=True),
                    webhook_id,
                    now,
                    now,
                ),
            )
        self._audit("developer.webhook.register", webhook_id, metadata={"event": event, "target_url": target_url})
        return self.get_webhook(webhook_id) or {}

    def get_webhook(self, webhook_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM developer_webhooks WHERE id = ?", (webhook_id,)).fetchone()
        return webhook_row(row) if row else None

    def webhooks(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM developer_webhooks ORDER BY created_at DESC").fetchall()
        return [webhook_row(row) for row in rows]

    def dispatch_webhook(
        self,
        *,
        event: str,
        payload: dict[str, Any],
        allow_network: bool = False,
    ) -> dict[str, Any]:
        event = event.strip()
        if not event:
            raise ValueError("Webhook event is required.")
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM developer_webhooks
                WHERE enabled = 1 AND event IN (?, '*')
                ORDER BY created_at
                """,
                (event,),
            ).fetchall()
        deliveries = []
        for row in rows:
            webhook = webhook_row(row)
            deliveries.append(self._deliver_webhook(webhook, event, payload, allow_network=allow_network))
        self._audit(
            "developer.webhook.dispatch",
            event,
            metadata={"delivery_count": len(deliveries), "allow_network": allow_network},
        )
        return {"event": event, "deliveries": deliveries}

    def webhook_deliveries(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM developer_webhook_deliveries ORDER BY created_at DESC"
            ).fetchall()
        return [webhook_delivery_row(row) for row in rows]

    def _deliver_webhook(
        self,
        webhook: dict[str, Any],
        event: str,
        payload: dict[str, Any],
        *,
        allow_network: bool,
    ) -> dict[str, Any]:
        target_url = webhook["target_url"]
        parsed = urlparse(target_url)
        response: dict[str, Any]
        if parsed.scheme == "log":
            status = "delivered"
            response = {"mode": "local_log", "target_url": target_url}
        elif not allow_network:
            status = "blocked"
            response = {"reason": "network approval required", "target_url": target_url}
        else:
            try:
                request = Request(
                    target_url,
                    data=json.dumps({"event": event, "payload": payload}).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": "OpenAgent-Webhook/0.6"},
                    method="POST",
                )
                with urlopen(request, timeout=30) as result:
                    response = {"http_status": result.status, "body": result.read(1000).decode("utf-8", "replace")}
                status = "delivered"
            except Exception as exc:
                status = "failed"
                response = {"error": str(exc)}
        delivery = {
            "id": unique_id("delivery", webhook["id"], event),
            "webhook_id": webhook["id"],
            "event": event,
            "payload": payload,
            "status": status,
            "response": response,
            "created_at": utc_now(),
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO developer_webhook_deliveries (
                    id, webhook_id, event, payload_json, status, response_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delivery["id"],
                    webhook["id"],
                    event,
                    json.dumps(payload, sort_keys=True),
                    status,
                    json.dumps(response, sort_keys=True),
                    delivery["created_at"],
                ),
            )
        return delivery

    def register_plugin(self, manifest: dict[str, Any]) -> dict[str, Any]:
        plugin_id = str(manifest.get("id", "")).strip()
        name = str(manifest.get("name", "")).strip()
        version = str(manifest.get("version", "0.1.0")).strip() or "0.1.0"
        if not plugin_id:
            plugin_id = stable_id("plugin", name, version)
        if not name:
            raise ValueError("Plugin name is required.")
        permissions = [str(item) for item in manifest.get("permissions", [])] if isinstance(manifest.get("permissions"), list) else []
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO developer_plugins (
                    id, name, version, description, entrypoint, permissions_json,
                    manifest_json, enabled, installed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT installed_at FROM developer_plugins WHERE id = ?), ?), ?)
                """,
                (
                    plugin_id,
                    name,
                    version,
                    str(manifest.get("description", "")),
                    str(manifest.get("entrypoint", "")),
                    json.dumps(permissions, sort_keys=True),
                    json.dumps(manifest, sort_keys=True),
                    1 if manifest.get("enabled", True) else 0,
                    plugin_id,
                    now,
                    now,
                ),
            )
        self._audit("developer.plugin.register", plugin_id, metadata={"name": name, "version": version})
        return self.get_plugin(plugin_id) or {}

    def get_plugin(self, plugin_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM developer_plugins WHERE id = ?", (plugin_id,)).fetchone()
        return plugin_row(row) if row else None

    def plugins(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM developer_plugins ORDER BY name").fetchall()
        return [plugin_row(row) for row in rows]

    def register_tool(
        self,
        *,
        tool_id: str,
        name: str,
        description: str = "",
        schema: dict[str, Any] | None = None,
        handler: str = "registered",
        permissions: list[str] | None = None,
        enabled: bool = True,
        audit: bool = True,
    ) -> dict[str, Any]:
        tool_id = tool_id.strip()
        name = name.strip()
        if not tool_id:
            raise ValueError("Tool id is required.")
        if not name:
            raise ValueError("Tool name is required.")
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO developer_tools (
                    id, name, description, schema_json, handler, permissions_json,
                    enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM developer_tools WHERE id = ?), ?), ?)
                """,
                (
                    tool_id,
                    name,
                    description,
                    json.dumps(schema or {}, sort_keys=True),
                    handler,
                    json.dumps(permissions or [], sort_keys=True),
                    1 if enabled else 0,
                    tool_id,
                    now,
                    now,
                ),
            )
        if audit:
            self._audit("developer.tool.register", tool_id, metadata={"handler": handler})
        return self.get_tool(tool_id) or {}

    def get_tool(self, tool_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM developer_tools WHERE id = ?", (tool_id,)).fetchone()
        return tool_row(row) if row else None

    def tools(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM developer_tools ORDER BY id").fetchall()
        return [tool_row(row) for row in rows]

    def invoke_tool(
        self,
        tool_id: str,
        *,
        input_data: dict[str, Any] | None = None,
        approved: bool = False,
    ) -> dict[str, Any]:
        tool = self.get_tool(tool_id)
        if tool is None:
            raise ValueError(f"Unknown developer tool '{tool_id}'.")
        if not tool["enabled"]:
            raise DeveloperPlatformError(f"Developer tool '{tool_id}' is disabled.")
        input_data = input_data or {}
        if tool["permissions"] and not approved:
            output = {"approval_required": True, "permissions": tool["permissions"]}
            status = "approval_required"
        else:
            output = self._run_tool_handler(tool, input_data)
            status = "completed"
        invocation = {
            "id": unique_id("tool-invocation", tool_id),
            "tool_id": tool_id,
            "input": input_data,
            "output": output,
            "status": status,
            "created_at": utc_now(),
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO developer_tool_invocations (
                    id, tool_id, input_json, output_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    invocation["id"],
                    tool_id,
                    json.dumps(input_data, sort_keys=True),
                    json.dumps(output, sort_keys=True),
                    status,
                    invocation["created_at"],
                ),
            )
        self._audit("developer.tool.invoke", tool_id, metadata={"status": status})
        return invocation

    def _run_tool_handler(self, tool: dict[str, Any], input_data: dict[str, Any]) -> dict[str, Any]:
        handler = tool["handler"]
        if handler == "echo":
            return {"echo": input_data}
        if handler == "context.search":
            resolver = self.resolvers.get("search_context")
            if resolver is None:
                raise DeveloperPlatformError("No context.search resolver is configured.")
            return {
                "results": resolver(
                    str(input_data.get("query", "")),
                    project=str(input_data.get("project", "default")),
                    limit=int(input_data.get("limit", 10)),
                )
            }
        if handler == "graphql.query":
            return self.execute_graphql(
                str(input_data.get("query", "")),
                variables=input_data.get("variables") if isinstance(input_data.get("variables"), dict) else {},
            )
        return {"registered": True, "handler": handler, "input": input_data}

    def register_mcp_server(
        self,
        *,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = name.strip()
        command = command.strip()
        if not name:
            raise ValueError("MCP server name is required.")
        if not command:
            raise ValueError("MCP server command is required.")
        server_id = stable_id("mcp", name, command)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO developer_mcp_servers (
                    id, name, command, args_json, env_json, enabled, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM developer_mcp_servers WHERE id = ?), ?), ?)
                """,
                (
                    server_id,
                    name,
                    command,
                    json.dumps(args or [], sort_keys=True),
                    json.dumps(env or {}, sort_keys=True),
                    1 if enabled else 0,
                    json.dumps(metadata or {}, sort_keys=True),
                    server_id,
                    now,
                    now,
                ),
            )
        self._audit("developer.mcp.register", server_id, metadata={"name": name, "enabled": enabled})
        return self.get_mcp_server(server_id) or {}

    def get_mcp_server(self, server_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM developer_mcp_servers WHERE id = ?", (server_id,)).fetchone()
        return mcp_server_row(row) if row else None

    def mcp_servers(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM developer_mcp_servers ORDER BY name").fetchall()
        return [mcp_server_row(row) for row in rows]

    def mcp_manifest(self) -> dict[str, Any]:
        return {
            "protocol": "model-context-protocol",
            "server": "openagent-local",
            "transport": "stdio-or-http",
            "resources": [
                {"uri": "openagent://context", "description": "Local context snapshot"},
                {"uri": "openagent://knowledge", "description": "Local knowledge search"},
                {"uri": "openagent://workflows", "description": "Local workflow catalog"},
            ],
            "tools": self.tools(),
            "configured_servers": self.mcp_servers(),
            "network_default": False,
        }

    def publish_extension_package(
        self,
        *,
        package_type: str,
        name: str,
        version: str,
        source: str,
        manifest: dict[str, Any] | None = None,
        status: str = "available",
        audit: bool = True,
    ) -> dict[str, Any]:
        package_type = package_type.strip()
        name = name.strip()
        version = version.strip()
        source = source.strip() or "local"
        if not package_type:
            raise ValueError("Extension package type is required.")
        if not name:
            raise ValueError("Extension package name is required.")
        if not version:
            raise ValueError("Extension package version is required.")
        package_id = stable_id("extension", package_type, name, version, source)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO developer_extension_packages (
                    id, package_type, name, version, source, status, manifest_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM developer_extension_packages WHERE id = ?), ?), ?)
                """,
                (
                    package_id,
                    package_type,
                    name,
                    version,
                    source,
                    status,
                    json.dumps(manifest or {}, sort_keys=True),
                    package_id,
                    now,
                    now,
                ),
            )
        if audit:
            self._audit("developer.extension.publish", package_id, metadata={"name": name, "version": version})
        return self.get_extension_package(package_id) or {}

    def get_extension_package(self, package_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT * FROM developer_extension_packages WHERE id = ?",
                (package_id,),
            ).fetchone()
        return extension_package_row(row) if row else None

    def extension_packages(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM developer_extension_packages ORDER BY name").fetchall()
        return [extension_package_row(row) for row in rows]

    def generate_sdk(self, *, language: str = "python") -> dict[str, Any]:
        language = language.strip().lower() or "python"
        if language != "python":
            raise ValueError("Only the Python SDK artifact is implemented in Stage 8.")
        files = {
            "openagent_client.py": PYTHON_SDK,
            "agent_sdk.py": PYTHON_AGENT_SDK,
            "workflow_sdk.py": PYTHON_WORKFLOW_SDK,
            "plugin_sdk.py": PYTHON_PLUGIN_SDK,
            "README.md": SDK_README,
        }
        artifact_id = unique_id("sdk", language)
        generated_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO developer_sdk_artifacts (
                    id, language, name, files_json, generated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    language,
                    "OpenAgent Python SDK",
                    json.dumps(files, sort_keys=True),
                    generated_at,
                ),
            )
        self._audit("developer.sdk.generate", artifact_id, metadata={"language": language})
        return {"id": artifact_id, "language": language, "name": "OpenAgent Python SDK", "files": files, "generated_at": generated_at}

    def sdk_artifacts(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM developer_sdk_artifacts ORDER BY generated_at DESC").fetchall()
        return [sdk_artifact_row(row) for row in rows]

    def _audit(
        self,
        action: str,
        resource: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store.add_audit_event(
            AuditEvent(
                actor="developer-platform",
                action=action,
                resource=resource,
                project="developer",
                metadata=metadata or {},
            )
        )


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:32]


def unique_id(*parts: str) -> str:
    return stable_id(*parts, utc_now(), secrets.token_hex(8))


def hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_graphql_root_fields(query: str) -> list[tuple[str, str, dict[str, Any]]]:
    body = root_selection_body(query)
    fields: list[tuple[str, str, dict[str, Any]]] = []
    index = 0
    while index < len(body):
        index = skip_ws(body, index)
        first_name, index = read_name(body, index)
        if not first_name:
            index += 1
            continue
        alias = first_name
        name = first_name
        index = skip_ws(body, index)
        if index < len(body) and body[index] == ":":
            index = skip_ws(body, index + 1)
            name, index = read_name(body, index)
        index = skip_ws(body, index)
        args: dict[str, Any] = {}
        if index < len(body) and body[index] == "(":
            raw_args, index = read_balanced(body, index, "(", ")")
            args = parse_graphql_args(raw_args[1:-1])
        index = skip_ws(body, index)
        if index < len(body) and body[index] == "{":
            _, index = read_balanced(body, index, "{", "}")
        if name:
            fields.append((alias, name, args))
    return fields


def root_selection_body(query: str) -> str:
    start = query.find("{")
    if start == -1:
        return ""
    body, _ = read_balanced(query, start, "{", "}")
    return body[1:-1]


def skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def read_name(text: str, index: int) -> tuple[str, int]:
    match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[index:])
    if not match:
        return "", index
    name = match.group(0)
    return name, index + len(name)


def read_balanced(text: str, index: int, opener: str, closer: str) -> tuple[str, int]:
    depth = 0
    in_string = False
    escaped = False
    for cursor in range(index, len(text)):
        char = text[cursor]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[index : cursor + 1], cursor + 1
    return text[index:], len(text)


def parse_graphql_args(raw: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for item in split_top_level(raw, ","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        args[key.strip()] = parse_graphql_value(value.strip())
    return args


def split_top_level(text: str, delimiter: str) -> list[str]:
    items = []
    start = 0
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        elif char == delimiter and depth == 0:
            items.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def parse_graphql_value(raw: str) -> Any:
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw in {"true", "false"}:
        return raw == "true"
    if raw == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def api_key_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "label": row["label"],
        "scopes": json.loads(row["scopes_json"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "last_used_at": row["last_used_at"],
    }


def webhook_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "event": row["event"],
        "target_url": row["target_url"],
        "enabled": bool(row["enabled"]),
        "secret_hint": row["secret_hint"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def webhook_delivery_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "webhook_id": row["webhook_id"],
        "event": row["event"],
        "payload": json.loads(row["payload_json"]),
        "status": row["status"],
        "response": json.loads(row["response_json"]),
        "created_at": row["created_at"],
    }


def plugin_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "version": row["version"],
        "description": row["description"],
        "entrypoint": row["entrypoint"],
        "permissions": json.loads(row["permissions_json"]),
        "manifest": json.loads(row["manifest_json"]),
        "enabled": bool(row["enabled"]),
        "installed_at": row["installed_at"],
        "updated_at": row["updated_at"],
    }


def tool_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "schema": json.loads(row["schema_json"]),
        "handler": row["handler"],
        "permissions": json.loads(row["permissions_json"]),
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def mcp_server_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "command": row["command"],
        "args": json.loads(row["args_json"]),
        "env": json.loads(row["env_json"]),
        "enabled": bool(row["enabled"]),
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def extension_package_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "package_type": row["package_type"],
        "name": row["name"],
        "version": row["version"],
        "source": row["source"],
        "status": row["status"],
        "manifest": json.loads(row["manifest_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def sdk_artifact_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "language": row["language"],
        "name": row["name"],
        "files": json.loads(row["files_json"]),
        "generated_at": row["generated_at"],
    }


PYTHON_SDK = '''"""Tiny OpenAgent SDK generated by the local developer platform."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


class OpenAgentClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def get(self, path: str) -> dict:
        request = Request(self.base_url + path, headers=self._headers(), method="GET")
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers({"Content-Type": "application/json"}),
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def graphql(self, query: str, variables: dict | None = None) -> dict:
        return self.post("/api/graphql", {"query": query, "variables": variables or {}})

    def run_tool(self, tool_id: str, input_data: dict | None = None, approved: bool = False) -> dict:
        return self.post(
            "/api/developer/tools/invoke",
            {"tool_id": tool_id, "input": input_data or {}, "approved": approved},
        )

    def run_agent(self, agent: str, goal: str, project: str = "default") -> dict:
        return self.post("/api/agents/run", {"agent": agent, "goal": goal, "project": project})

    def create_workflow(self, name: str, steps: list[dict], project: str = "default") -> dict:
        return self.post("/api/workflows", {"name": name, "steps": steps, "project": project})

    def _headers(self, extra: dict | None = None) -> dict:
        headers = dict(extra or {})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
'''


PYTHON_AGENT_SDK = '''"""Agent extension helpers for OpenAgent plugins."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentManifest:
    id: str
    name: str
    description: str
    permissions: list[str] = field(default_factory=list)

    def to_manifest(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "permissions": self.permissions,
            "type": "agent",
        }
'''


PYTHON_WORKFLOW_SDK = '''"""Workflow extension helpers for OpenAgent plugins."""

from __future__ import annotations


def workflow_manifest(name: str, steps: list[dict], *, trigger_type: str = "manual") -> dict:
    return {
        "type": "workflow",
        "name": name,
        "trigger_type": trigger_type,
        "steps": steps,
    }


def tool_step(tool_id: str, input_data: dict | None = None, *, approved: bool = False) -> dict:
    return {
        "action": "api_request",
        "tool_id": tool_id,
        "input": input_data or {},
        "approved": approved,
    }
'''


PYTHON_PLUGIN_SDK = '''"""Plugin manifest helpers for OpenAgent."""

from __future__ import annotations


def plugin_manifest(
    plugin_id: str,
    name: str,
    version: str,
    *,
    entrypoint: str = "",
    permissions: list[str] | None = None,
) -> dict:
    return {
        "id": plugin_id,
        "name": name,
        "version": version,
        "entrypoint": entrypoint,
        "permissions": permissions or [],
    }
'''


SDK_README = """# OpenAgent Python SDK

This generated SDK uses only the Python standard library. It calls the local OpenAgent HTTP API and does not add telemetry or hidden network behavior.

```python
from openagent_client import OpenAgentClient

client = OpenAgentClient("http://127.0.0.1:8765")
print(client.graphql("{ health developerStatus }"))
```
"""
