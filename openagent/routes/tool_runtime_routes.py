"""HTTP contracts for the shared local tool runtime."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from uuid import uuid4

from ..tools.runtime import ToolRequest, ToolRuntime
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "tool_runtime"
ROUTES = (
    ("GET", "/api/tools/runtime"),
    ("POST", "/api/tools/runtime/execute"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ToolRuntimeRouteDependencies:
    runtime: ToolRuntime
    payload_owner: Any
    read_json: Any


def handle_get(path: str, deps: ToolRuntimeRouteDependencies) -> JsonRouteResult | None:
    if path != "/api/tools/runtime":
        return None
    return JsonRouteResult(deps.runtime.describe())


def handle_post(path: str, deps: ToolRuntimeRouteDependencies) -> JsonRouteResult | None:
    if path != "/api/tools/runtime/execute":
        return None
    payload = deps.read_json()
    arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
    try:
        request = ToolRequest(
            tool=str(payload.get("tool", "")).strip(),
            arguments=arguments,
            project=str(payload.get("project", "default")).strip() or "default",
            owner=deps.payload_owner(payload),
            workspace=str(payload.get("workspace", "")),
            network=str(payload.get("network", "none")).strip() or "none",
            timeout_seconds=max(0.1, min(_as_float(payload.get("timeout_seconds"), 30.0), 300)),
            approval_id=str(payload.get("approval_id", "")).strip(),
            trace_id=str(payload.get("trace_id", "")).strip() or uuid4().hex,
        )
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    result = deps.runtime.execute(request)
    status = HTTPStatus.OK
    if result.status == "denied":
        status = HTTPStatus.FORBIDDEN
    elif result.status == "needs_approval":
        status = HTTPStatus.PRECONDITION_REQUIRED
    elif result.status in {"failed", "timeout", "cancelled"}:
        status = HTTPStatus.BAD_REQUEST
    return JsonRouteResult({"result": result.to_dict()}, status)


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
