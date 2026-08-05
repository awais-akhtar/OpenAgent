"""MCP runtime route contracts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..mcp_runtime import MCPRuntime
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "mcp_runtime"
ROUTES = (
    ("GET", "/api/mcp/servers"),
    ("GET", "/api/mcp/tools"),
    ("GET", "/api/mcp/permissions"),
    ("POST", "/api/mcp/servers"),
    ("POST", "/api/mcp/servers/{id}/reconnect"),
    ("PATCH", "/api/mcp/tools/{id}"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class MCPRuntimeRouteDependencies:
    runtime: MCPRuntime
    first: Any
    parse_bool: Any
    parse_int: Any
    query_owner: Any
    payload_owner: Any
    read_json: Any


def handle_get(path: str, query: str, deps: MCPRuntimeRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/mcp"):
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    owner = deps.query_owner(params)
    if path == "/api/mcp/servers":
        return JsonRouteResult(
            {"servers": deps.runtime.list_servers(project=project, owner=owner, limit=deps.parse_int(deps.first(params, "limit"), default=100))}
        )
    if path == "/api/mcp/tools":
        enabled_raw = deps.first(params, "enabled")
        enabled = None if enabled_raw is None else deps.parse_bool(enabled_raw, default=True)
        return JsonRouteResult({"tools": deps.runtime.tools(project=project, owner=owner, enabled=enabled)})
    if path == "/api/mcp/permissions":
        return JsonRouteResult(
            {
                "permissions": deps.runtime.permissions(
                    project=project,
                    owner=owner,
                    limit=deps.parse_int(deps.first(params, "limit"), default=100),
                )
            }
        )
    return None


def handle_post(path: str, deps: MCPRuntimeRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/mcp"):
        return None
    payload = deps.read_json()
    owner = deps.payload_owner(payload)
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/mcp/servers":
            server = deps.runtime.register_server(
                name=str(payload.get("name", "")).strip(),
                project=project,
                owner=owner,
                transport=str(payload.get("transport", "stdio")).strip() or "stdio",
                command=[str(item) for item in payload.get("command", [])] if isinstance(payload.get("command"), list) else [],
                endpoint=str(payload.get("endpoint", "")),
                tools=payload.get("tools") if isinstance(payload.get("tools"), list) else [],
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"server": server}, HTTPStatus.CREATED)
        if path.startswith("/api/mcp/servers/") and path.endswith("/reconnect"):
            server_id = unquote(path.removeprefix("/api/mcp/servers/").removesuffix("/reconnect")).strip("/")
            return JsonRouteResult({"server": deps.runtime.reconnect(server_id)})
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None


def handle_patch(path: str, deps: MCPRuntimeRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/mcp/tools/"):
        return None
    payload = deps.read_json()
    tool_id = unquote(path.removeprefix("/api/mcp/tools/")).strip("/")
    try:
        tool = deps.runtime.patch_tool(
            tool_id,
            enabled=deps.parse_bool(payload.get("enabled"), default=True) if "enabled" in payload else None,
            approval_required=deps.parse_bool(payload.get("approval_required"), default=True) if "approval_required" in payload else None,
            reason=str(payload.get("reason", "")),
        )
        return JsonRouteResult({"tool": tool})
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
