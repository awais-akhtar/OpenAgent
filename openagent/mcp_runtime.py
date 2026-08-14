"""Local MCP server registry, tool permissions, and connection state."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any
from uuid import uuid4

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class MCPRuntime:
    """Persist MCP configuration and invoke explicitly approved local tools.

    Registration remains data-only. Invocation is a separate, auditable path:
    the stored tool and server are checked first, then a bounded JSON-RPC
    exchange is performed over stdio or HTTP.
    """

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

    def get_server(self, server_id: str, *, project: str = "", owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            clauses = ["id = ?"]
            params: list[Any] = [server_id]
            if project:
                clauses.append("project = ?")
                params.append(project)
            if owner:
                clauses.append("owner = ?")
                params.append(owner)
            row = connection.execute("SELECT * FROM mcp_servers WHERE " + " AND ".join(clauses), params).fetchone()
            if row is None:
                raise KeyError("MCP server not found")
            tools = connection.execute("SELECT * FROM mcp_tools WHERE server_id = ? ORDER BY name", (server_id,)).fetchall()
            events = connection.execute("SELECT * FROM mcp_connection_events WHERE server_id = ? ORDER BY created_at DESC LIMIT 20", (server_id,)).fetchall()
        result = server_row(row)
        result["tools"] = [tool_row(tool) for tool in tools]
        result["events"] = [event_row(event) for event in events]
        return result

    def reconnect(self, server_id: str, *, project: str = "", owner: str = "") -> dict[str, Any]:
        now = utc_now()
        with self.store.session() as connection:
            clauses = ["id = ?"]
            params: list[Any] = [server_id]
            if project:
                clauses.append("project = ?")
                params.append(project)
            if owner:
                clauses.append("owner = ?")
                params.append(owner)
            row = connection.execute("SELECT * FROM mcp_servers WHERE " + " AND ".join(clauses), params).fetchone()
            if row is None:
                raise KeyError("MCP server not found")
            connection.execute("UPDATE mcp_servers SET status = 'ready', updated_at = ? WHERE id = ?", (now, server_id))
            self._event(connection, server_id, "reconnect", {"status": "ready"})
        return self.get_server(server_id, project=project, owner=owner)

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
        project: str = "",
        owner: str = "",
        enabled: bool | None = None,
        approval_required: bool | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        with self.store.session() as connection:
            clauses = ["id = ?"]
            params: list[Any] = [tool_id]
            if project:
                clauses.append("project = ?")
                params.append(project)
            if owner:
                clauses.append("owner = ?")
                params.append(owner)
            row = connection.execute("SELECT * FROM mcp_tools WHERE " + " AND ".join(clauses), params).fetchone()
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

    def invoke_tool(
        self,
        tool_id: str,
        *,
        arguments: dict[str, Any] | None = None,
        project: str = "default",
        owner: str = "",
        approval_id: str = "",
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 100_000,
    ) -> dict[str, Any]:
        """Call one registered MCP tool through its configured server.

        The server process is short-lived for stdio calls. This keeps the
        first implementation deterministic and avoids leaving arbitrary child
        processes behind when the desktop runtime closes.
        """

        with self.store.session() as connection:
            clauses = ["t.id = ?"]
            params: list[Any] = [tool_id]
            if project:
                clauses.append("t.project = ?")
                params.append(project)
            if owner:
                clauses.append("t.owner = ?")
                params.append(owner)
            row = connection.execute(
                "SELECT t.*, s.transport, s.command_json, s.endpoint, s.status AS server_status, s.id AS parent_server_id "
                "FROM mcp_tools t JOIN mcp_servers s ON s.id = t.server_id WHERE " + " AND ".join(clauses),
                params,
            ).fetchone()
        if row is None:
            raise KeyError("MCP tool not found")
        if not bool(row["enabled"]):
            raise PermissionError("MCP tool is disabled")
        if bool(row["approval_required"]) and not approval_id:
            raise PermissionError("MCP tool invocation requires an approval id")
        schema = json.loads(row["schema_json"] or "{}")
        self._validate_arguments(schema, arguments or {})
        timeout = max(0.2, min(float(timeout_seconds), 300.0))
        transport = str(row["transport"] or "stdio").lower()
        try:
            if transport == "stdio":
                response = self._invoke_stdio(
                    json.loads(row["command_json"] or "[]"),
                    str(row["name"]),
                    arguments or {},
                    timeout,
                    max_output_bytes,
                )
            elif transport in {"http", "sse", "streamable-http"}:
                if not approval_id:
                    raise PermissionError("HTTP-based MCP transport requires an approval id")
                response = self._invoke_http(str(row["endpoint"]), str(row["name"]), arguments or {}, timeout, max_output_bytes)
            else:
                raise ValueError(f"unsupported MCP transport: {transport}")
        except Exception as exc:
            with self.store.session() as connection:
                self._event(connection, row["parent_server_id"], "invoke_failed", {"tool_id": tool_id, "error": str(exc)})
            self.store.add_audit_event(
                AuditEvent(
                    actor="mcp-runtime",
                    action="mcp.tool.invoke",
                    resource=f"mcp-tool:{tool_id}",
                    project=project,
                    status="failed",
                    metadata={"owner": owner, "approval_id": bool(approval_id), "error": str(exc)},
                )
            )
            raise
        with self.store.session() as connection:
            self._event(connection, row["parent_server_id"], "invoke_completed", {"tool_id": tool_id})
        self.store.add_audit_event(
            AuditEvent(
                actor="mcp-runtime",
                action="mcp.tool.invoke",
                resource=f"mcp-tool:{tool_id}",
                project=project,
                status="allowed",
                metadata={"owner": owner, "approval_id": bool(approval_id), "transport": transport},
            )
        )
        return {"tool": tool_row(row), "transport": transport, "response": response}

    @staticmethod
    def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ValueError("MCP tool arguments must be an object")
        required = schema.get("required", []) if isinstance(schema, dict) else []
        missing = [str(key) for key in required if str(key) not in arguments]
        if missing:
            raise ValueError(f"missing required MCP arguments: {', '.join(missing)}")

    @staticmethod
    def _bounded_json(value: Any, max_output_bytes: int) -> Any:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded.encode("utf-8")) <= max(1_000, int(max_output_bytes)):
            return value
        return {"truncated": True, "preview": encoded[: max(500, int(max_output_bytes) // 2)], "limit_bytes": int(max_output_bytes)}

    def _invoke_stdio(
        self,
        command: list[Any],
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float,
        max_output_bytes: int,
    ) -> Any:
        if not command:
            raise ValueError("MCP stdio server has no command")
        executable = [str(part) for part in command]
        process = subprocess.Popen(
            executable,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
        try:
            self._write_rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "OpenAgent", "version": "1"}}})
            initialize = self._read_rpc(process, timeout, expected_id=1)
            if isinstance(initialize, dict) and initialize.get("error"):
                raise ValueError(str(initialize["error"]))
            self._write_rpc(process, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
            self._write_rpc(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}})
            response = self._read_rpc(process, timeout, expected_id=2)
            if isinstance(response, dict) and response.get("error"):
                raise ValueError(str(response["error"]))
            return self._bounded_json(response.get("result", response), max_output_bytes) if isinstance(response, dict) else response
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
            try:
                process.terminate()
                process.wait(timeout=1)
            except Exception:
                process.kill()

    @staticmethod
    def _write_rpc(process: subprocess.Popen[str], payload: dict[str, Any]) -> None:
        if process.stdin is None:
            raise RuntimeError("MCP server stdin is unavailable")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()

    @staticmethod
    def _read_rpc(process: subprocess.Popen[str], timeout: float, *, expected_id: int | None = None) -> dict[str, Any]:
        if process.stdout is None:
            raise RuntimeError("MCP server stdout is unavailable")
        values: queue.Queue[str] = queue.Queue(maxsize=1)

        def read_line() -> None:
            try:
                values.put(process.stdout.readline())
            except Exception as exc:
                values.put(json.dumps({"error": str(exc)}))

        threading.Thread(target=read_line, daemon=True).start()
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(0.01, deadline - time.monotonic())
            try:
                line = values.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("MCP server did not return a response in time") from exc
            if not line:
                raise RuntimeError("MCP server closed stdout before returning a response")
            payload = json.loads(line)
            if expected_id is None or payload.get("id") == expected_id or payload.get("error"):
                return payload
            threading.Thread(target=read_line, daemon=True).start()

    def _invoke_http(self, endpoint: str, tool_name: str, arguments: dict[str, Any], timeout: float, max_output_bytes: int) -> Any:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("MCP HTTP endpoint must use http:// or https://")
        request = Request(
            endpoint,
            data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read(max(1_000, int(max_output_bytes)) + 1)
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"MCP HTTP request failed: {exc}") from exc
        if len(body) > max_output_bytes:
            return {"truncated": True, "preview": body[: max_output_bytes // 2].decode("utf-8", errors="replace"), "limit_bytes": max_output_bytes}
        payload = json.loads(body.decode("utf-8", errors="replace"))
        if isinstance(payload, dict) and payload.get("error"):
            raise ValueError(str(payload["error"]))
        return self._bounded_json(payload.get("result", payload) if isinstance(payload, dict) else payload, max_output_bytes)

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
