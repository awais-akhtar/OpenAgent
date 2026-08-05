"""Local MCP server registry, tool permissions, and connection state."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class MCPRuntime:
    """Stores MCP runtime configuration without launching untrusted processes."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS mcp_servers (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    name TEXT NOT NULL,
                    transport TEXT NOT NULL DEFAULT 'stdio',
                    command_json TEXT NOT NULL DEFAULT '[]',
                    endpoint TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'configured',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_mcp_servers_project_owner
                    ON mcp_servers(project, owner, updated_at DESC);

                CREATE TABLE IF NOT EXISTS mcp_tools (
                    id TEXT PRIMARY KEY,
                    server_id TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    approval_required INTEGER NOT NULL DEFAULT 1,
                    schema_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_mcp_tools_server
                    ON mcp_tools(server_id, name);

                CREATE TABLE IF NOT EXISTS mcp_tool_permissions (
                    id TEXT PRIMARY KEY,
                    tool_id TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    approval_required INTEGER NOT NULL DEFAULT 1,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(tool_id) REFERENCES mcp_tools(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_mcp_tool_permissions_tool
                    ON mcp_tool_permissions(tool_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS mcp_connection_events (
                    id TEXT PRIMARY KEY,
                    server_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
                );
                """
            )

    def register_server(
        self,
        *,
        name: str,
        project: str = "default",
        owner: str = "local",
        transport: str = "stdio",
        command: list[str] | None = None,
        endpoint: str = "",
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("server name is required")
        server_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO mcp_servers (
                    id, project, owner, name, transport, command_json, endpoint,
                    status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'configured', ?, ?, ?)
                """,
                (
                    server_id,
                    project,
                    owner,
                    name.strip(),
                    transport or "stdio",
                    json.dumps([str(part) for part in (command or [])]),
                    endpoint,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                    now,
                ),
            )
            for tool in tools or []:
                self._insert_tool(connection, server_id, project, owner, tool)
            self._event(connection, server_id, "registered", {"tool_count": len(tools or [])})
        self.store.add_audit_event(
            AuditEvent(
                actor="mcp-runtime",
                action="mcp.server.register",
                resource=f"mcp:{server_id}",
                project=project,
                metadata={"owner": owner, "transport": transport},
            )
        )
        return self.get_server(server_id)

    def list_servers(self, *, project: str = "default", owner: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        sql = "SELECT * FROM mcp_servers WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
        return [server_row(row) for row in rows]

    def get_server(self, server_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM mcp_servers WHERE id = ?", (server_id,)).fetchone()
            if row is None:
                raise KeyError("MCP server not found")
            tools = connection.execute("SELECT * FROM mcp_tools WHERE server_id = ? ORDER BY name", (server_id,)).fetchall()
            events = connection.execute("SELECT * FROM mcp_connection_events WHERE server_id = ? ORDER BY created_at DESC LIMIT 20", (server_id,)).fetchall()
        result = server_row(row)
        result["tools"] = [tool_row(tool) for tool in tools]
        result["events"] = [event_row(event) for event in events]
        return result

    def reconnect(self, server_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM mcp_servers WHERE id = ?", (server_id,)).fetchone()
            if row is None:
                raise KeyError("MCP server not found")
            connection.execute("UPDATE mcp_servers SET status = 'ready', updated_at = ? WHERE id = ?", (now, server_id))
            self._event(connection, server_id, "reconnect", {"status": "ready"})
        return self.get_server(server_id)

    def tools(self, *, project: str = "default", owner: str = "", enabled: bool | None = None) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        if enabled is not None:
            clauses.append("enabled = ?")
            args.append(1 if enabled else 0)
        sql = "SELECT * FROM mcp_tools WHERE " + " AND ".join(clauses) + " ORDER BY name"
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
        return [tool_row(row) for row in rows]

    def permissions(self, *, project: str = "default", owner: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        sql = "SELECT * FROM mcp_tool_permissions WHERE " + " AND ".join(clauses) + " ORDER BY created_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
        return [permission_row(row) for row in rows]

    def patch_tool(
        self,
        tool_id: str,
        *,
        enabled: bool | None = None,
        approval_required: bool | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM mcp_tools WHERE id = ?", (tool_id,)).fetchone()
            if row is None:
                raise KeyError("MCP tool not found")
            new_enabled = bool(row["enabled"]) if enabled is None else bool(enabled)
            new_approval = bool(row["approval_required"]) if approval_required is None else bool(approval_required)
            connection.execute(
                "UPDATE mcp_tools SET enabled = ?, approval_required = ?, updated_at = ? WHERE id = ?",
                (1 if new_enabled else 0, 1 if new_approval else 0, utc_now(), tool_id),
            )
            connection.execute(
                """
                INSERT INTO mcp_tool_permissions (
                    id, tool_id, project, owner, enabled, approval_required, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    tool_id,
                    row["project"],
                    row["owner"],
                    1 if new_enabled else 0,
                    1 if new_approval else 0,
                    reason,
                    utc_now(),
                ),
            )
            updated = connection.execute("SELECT * FROM mcp_tools WHERE id = ?", (tool_id,)).fetchone()
        return tool_row(updated)

    def _insert_tool(self, connection: Any, server_id: str, project: str, owner: str, tool: dict[str, Any]) -> None:
        now = utc_now()
        connection.execute(
            """
            INSERT INTO mcp_tools (
                id, server_id, project, owner, name, description, enabled,
                approval_required, schema_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                server_id,
                project,
                owner,
                str(tool.get("name", "tool")).strip() or "tool",
                str(tool.get("description", "")),
                1 if bool(tool.get("enabled", True)) else 0,
                1 if bool(tool.get("approval_required", True)) else 0,
                json.dumps(tool.get("schema") if isinstance(tool.get("schema"), dict) else {}, sort_keys=True),
                now,
                now,
            ),
        )

    def _event(self, connection: Any, server_id: str, event_type: str, detail: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO mcp_connection_events (id, server_id, event_type, detail_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid4().hex, server_id, event_type, json.dumps(detail, sort_keys=True), utc_now()),
        )


def server_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "name": row["name"],
        "transport": row["transport"],
        "command": json.loads(row["command_json"] or "[]"),
        "endpoint": row["endpoint"],
        "status": row["status"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def tool_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "server_id": row["server_id"],
        "project": row["project"],
        "owner": row["owner"],
        "name": row["name"],
        "description": row["description"],
        "enabled": bool(row["enabled"]),
        "approval_required": bool(row["approval_required"]),
        "schema": json.loads(row["schema_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def event_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "server_id": row["server_id"],
        "event_type": row["event_type"],
        "detail": json.loads(row["detail_json"] or "{}"),
        "created_at": row["created_at"],
    }


def permission_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "tool_id": row["tool_id"],
        "project": row["project"],
        "owner": row["owner"],
        "enabled": bool(row["enabled"]),
        "approval_required": bool(row["approval_required"]),
        "reason": row["reason"],
        "created_at": row["created_at"],
    }
