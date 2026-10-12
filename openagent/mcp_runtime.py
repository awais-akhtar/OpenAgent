"""Local MCP server registry, tool permissions, and connection state."""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import AuditEvent, utc_now
from .egress import EgressClient
from .process_policy import DEFAULT_ENV_KEYS
from .storage import OpenAgentStore


class MCPRuntime:
    """Persist MCP configuration and invoke explicitly approved local tools.

    Registration remains data-only. Invocation is a separate, auditable path:
    the stored tool and server are checked first, then a bounded JSON-RPC
    exchange is performed over stdio or HTTP.
    """

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.egress = EgressClient()
        self.initialize()

    def initialize(self) -> None:
        self.store.initialize()

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
        max_output_bytes = max(1_000, min(int(max_output_bytes), 1_000_000))
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
                response = self._invoke_http(
                    str(row["endpoint"]), str(row["name"]), arguments or {}, timeout,
                    max_output_bytes, approved=bool(approval_id),
                )
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
        program = executable[0].strip()
        resolved = shutil.which(program)
        if resolved is None and Path(program).is_file():
            resolved = str(Path(program).resolve())
        if not resolved or not Path(resolved).is_file() or (os.name != "nt" and not os.access(resolved, os.X_OK)):
            raise FileNotFoundError(f"MCP stdio executable not found: {program}")
        if os.name == "nt" and Path(resolved).suffix.lower() in {".bat", ".cmd"}:
            raise ValueError("MCP stdio server must be a direct executable")
        executable[0] = resolved
        child_env = {key: value for key, value in os.environ.items() if key.upper() in DEFAULT_ENV_KEYS}
        process = subprocess.Popen(
            executable,
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        limit = max(1_000, int(max_output_bytes))
        events: queue.Queue[bytes | BaseException | None] = queue.Queue()
        overflow = threading.Event()
        deadline = time.monotonic() + timeout

        def drain_stdout() -> None:
            pending = bytearray()
            total = 0
            try:
                while True:
                    chunk = os.read(process.stdout.fileno(), 4096)  # type: ignore[union-attr]
                    if not chunk:
                        events.put(None)
                        return
                    total += len(chunk)
                    if total > limit:
                        overflow.set()
                        return
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line, _, rest = pending.partition(b"\n")
                        events.put(bytes(line))
                        pending = bytearray(rest)
            except OSError as exc:
                events.put(exc)

        def drain_stderr() -> None:
            total = 0
            try:
                while True:
                    chunk = os.read(process.stderr.fileno(), 4096)  # type: ignore[union-attr]
                    if not chunk:
                        return
                    total += len(chunk)
                    if total > limit:
                        overflow.set()
                        return
            except OSError:
                pass

        readers = [threading.Thread(target=drain_stdout, daemon=True), threading.Thread(target=drain_stderr, daemon=True)]
        for reader in readers:
            reader.start()
        try:
            self._write_rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "OpenAgent", "version": "1"}}}, deadline, overflow)
            initialize = self._read_rpc(events, deadline, overflow, expected_id=1)
            if isinstance(initialize, dict) and initialize.get("error"):
                raise ValueError(str(initialize["error"]))
            self._write_rpc(process, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, deadline, overflow)
            self._write_rpc(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}, deadline, overflow)
            response = self._read_rpc(events, deadline, overflow, expected_id=2)
            if overflow.is_set():
                raise RuntimeError("MCP server output exceeded the configured limit")
            if isinstance(response, dict) and response.get("error"):
                raise ValueError(str(response["error"]))
            return self._bounded_json(response.get("result", response), max_output_bytes) if isinstance(response, dict) else response
        finally:
            self._terminate_stdio_tree(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
            for reader in readers:
                reader.join(timeout=0.5)

    @staticmethod
    def _terminate_stdio_tree(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            process.wait(timeout=0.5)
            return
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.kill()
            process.wait(timeout=1)

    @staticmethod
    def _write_rpc(process: subprocess.Popen[bytes], payload: dict[str, Any], deadline: float, overflow: threading.Event) -> None:
        if process.stdin is None:
            raise RuntimeError("MCP server stdin is unavailable")
        result: queue.Queue[BaseException | None] = queue.Queue(maxsize=1)

        def write() -> None:
            try:
                process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                process.stdin.flush()
                result.put(None)
            except (OSError, ValueError) as exc:
                result.put(exc)

        threading.Thread(target=write, daemon=True).start()
        while True:
            if overflow.is_set():
                raise RuntimeError("MCP server output exceeded the configured limit")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MCP server did not return a response in time")
            try:
                error = result.get(timeout=min(remaining, 0.02))
            except queue.Empty:
                continue
            if error is not None:
                raise RuntimeError("MCP server closed stdin") from error
            return

    @staticmethod
    def _read_rpc(events: queue.Queue[bytes | BaseException | None], deadline: float,
                  overflow: threading.Event, *, expected_id: int | None = None) -> dict[str, Any]:
        while True:
            if overflow.is_set():
                raise RuntimeError("MCP server output exceeded the configured limit")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MCP server did not return a response in time")
            try:
                line = events.get(timeout=min(remaining, 0.02))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("MCP server closed stdout before returning a response")
            if isinstance(line, BaseException):
                raise RuntimeError("MCP server stdout failed") from line
            payload = json.loads(line.decode("utf-8"))
            if expected_id is None or payload.get("id") == expected_id or payload.get("error"):
                return payload

    def _invoke_http(
        self, endpoint: str, tool_name: str, arguments: dict[str, Any], timeout: float,
        max_output_bytes: int, *, approved: bool,
    ) -> Any:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("MCP HTTP endpoint must use http:// or https://")
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}).encode("utf-8")
        response = self.egress.fetch(
            endpoint, approved=approved, allow_loopback=True, purpose="MCP tool invocation",
            body=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
            timeout_seconds=timeout, max_response_bytes=max(1_000, int(max_output_bytes)),
            allowed_content_types={"application/json"},
        )
        if not 200 <= response.status < 300:
            raise RuntimeError(f"MCP HTTP request returned HTTP {response.status}")
        payload = json.loads(response.body.decode("utf-8", errors="replace"))
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
