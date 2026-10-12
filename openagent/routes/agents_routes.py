"""Agent and delegation HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from ..multi_agent import ApprovalRequired, PermissionDenied
from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "agents"
ROUTES = (
    ("GET", "/api/agents"),
    ("POST", "/api/agents/run"),
    ("POST", "/api/agents/run-checked"),
    ("POST", "/api/agents/delegate"),
    ("POST", "/api/agents/parallel"),
    ("GET", "/api/agents/marketplace"),
    ("GET", "/api/agents/permissions"),
    ("GET", "/api/agents/approvals"),
    ("POST", "/api/agents/permissions/grant"),
    ("POST", "/api/agents/approvals/request"),
    ("POST", "/api/agents/approvals/resolve"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class AgentRouteDependencies:
    agents: Any
    multi_agent: Any
    first: Callable[..., str | None]
    parse_bool: Callable[..., bool]
    read_json: Callable[[], dict[str, Any]]


def handle_get(path: str, query: str, deps: AgentRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/agents":
        return JsonRouteResult({"agents": deps.agents.specs()})
    if path == "/api/agents/marketplace":
        return JsonRouteResult({"agents": deps.multi_agent.marketplace()})
    if path not in {"/api/agents/permissions", "/api/agents/approvals"}:
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    if path == "/api/agents/permissions":
        return JsonRouteResult({"permissions": deps.multi_agent.list_permissions(project=project)})
    return JsonRouteResult({"approvals": deps.multi_agent.list_approvals(
        project=project, status=deps.first(params, "status"),
    )})


def handle_post(path: str, deps: AgentRouteDependencies) -> JsonRouteResult | None:
    if path not in {route for method, route in ROUTES if method == "POST"}:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    multi_agent = deps.multi_agent
    try:
        if path == "/api/agents/run":
            goal = str(payload.get("goal", "")).strip()
            if not goal:
                return JsonRouteResult({"error": "A goal is required."}, HTTPStatus.BAD_REQUEST)
            run = deps.agents.run(
                str(payload.get("agent", "executive")).strip() or "executive",
                goal, project=project,
            )
            return JsonRouteResult({"run": run}, HTTPStatus.CREATED)
        if path == "/api/agents/run-checked":
            result = multi_agent.run_agent(
                str(payload.get("agent", "executive")).strip() or "executive",
                str(payload.get("goal", "")).strip(), project=project,
                require_approval=deps.parse_bool(payload.get("require_approval"), default=False),
                approval_id=str(payload.get("approval_id", "")).strip() or None,
                verify=deps.parse_bool(payload.get("verify"), default=True),
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        if path == "/api/agents/delegate":
            result = multi_agent.delegate(
                str(payload.get("from_agent", "executive")).strip() or "executive",
                str(payload.get("to_agent", "planning")).strip() or "planning",
                str(payload.get("goal", "")).strip(), project=project,
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        if path == "/api/agents/parallel":
            tasks = payload.get("tasks", [])
            if not isinstance(tasks, list):
                return JsonRouteResult({"error": "tasks must be a list."}, HTTPStatus.BAD_REQUEST)
            result = multi_agent.run_parallel(
                [task for task in tasks if isinstance(task, dict)], project=project,
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        if path == "/api/agents/permissions/grant":
            permission = multi_agent.grant(
                str(payload.get("agent", "")).strip(),
                str(payload.get("scope", "")).strip(), project=project,
                granted_by=str(payload.get("granted_by", "user")).strip() or "user",
            )
            return JsonRouteResult({"permission": permission}, HTTPStatus.CREATED)
        if path == "/api/agents/approvals/request":
            approval = multi_agent.request_approval(
                requested_by=str(payload.get("requested_by", "user")).strip() or "user",
                action=str(payload.get("action", "")).strip(),
                reason=str(payload.get("reason", "")).strip(), project=project,
                payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
            )
            return JsonRouteResult({"approval": approval}, HTTPStatus.CREATED)
        approval = multi_agent.resolve_approval(
            str(payload.get("approval_id", "")).strip(),
            approved=deps.parse_bool(payload.get("approved"), default=False),
        )
        return JsonRouteResult({"approval": approval})
    except (ApprovalRequired, PermissionDenied, ValueError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
