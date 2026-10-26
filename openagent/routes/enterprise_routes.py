"""Enterprise administration route handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "enterprise"
ROUTES = (
    ("GET", "/api/enterprise/status"),
    ("GET", "/api/enterprise/users"),
    ("GET", "/api/enterprise/roles"),
    ("GET", "/api/enterprise/workspaces"),
    ("GET", "/api/enterprise/policies"),
    ("GET", "/api/enterprise/sso"),
    ("GET", "/api/enterprise/backups"),
    ("GET", "/api/enterprise/disaster-recovery"),
    ("GET", "/api/enterprise/workspace/members"),
    ("GET", "/api/enterprise/org-memory"),
    ("POST", "/api/enterprise/access/check"),
    ("POST", "/api/enterprise/audit/export"),
    ("POST", "/api/enterprise/policies/evaluate"),
    ("POST", "/api/enterprise/sso/validate"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class EnterpriseRouteDependencies:
    enterprise: Any
    enterprise_identity: Any
    auth: Any
    read_json: Callable[[], dict]
    auth_scope_owner: Callable[[], str]
    bearer_token: Callable[[], str]
    first: Callable
    parse_qs_param: Callable


def handle_get(path: str, query: str, deps: EnterpriseRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/enterprise/"):
        return None
    params = parse_qs(query)
    first = deps.first
    enterprise = deps.enterprise

    if path == "/api/enterprise/status":
        return JsonRouteResult(enterprise.status())

    if path == "/api/enterprise/users":
        return JsonRouteResult({"users": enterprise.users()})

    if path == "/api/enterprise/roles":
        return JsonRouteResult({"roles": enterprise.roles()})

    if path == "/api/enterprise/workspaces":
        return JsonRouteResult({"workspaces": enterprise.workspaces()})

    if path == "/api/enterprise/policies":
        return JsonRouteResult({"policies": enterprise.policies()})

    if path == "/api/enterprise/sso":
        return JsonRouteResult({"sso_configs": enterprise.sso_configs()})

    if path == "/api/enterprise/backups":
        return JsonRouteResult({"backups": enterprise.backups()})

    if path == "/api/enterprise/disaster-recovery":
        return JsonRouteResult({"plans": enterprise.disaster_recovery_plans()})

    if path == "/api/enterprise/workspace/members":
        workspace_id = first(params, "workspace_id") or ""
        if not workspace_id:
            return JsonRouteResult({"error": "workspace_id query param required"}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"members": enterprise.members(workspace_id)})

    if path == "/api/enterprise/org-memory":
        workspace_id = first(params, "workspace_id") or ""
        if not workspace_id:
            return JsonRouteResult({"error": "workspace_id query param required"}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"org_memory": enterprise.org_memory(workspace_id)})

    return None


_ENTERPRISE_POST_PATHS = {
    "/api/enterprise/access/check",
    "/api/enterprise/audit/export",
    "/api/enterprise/policies/evaluate",
    "/api/enterprise/sso/validate",
}


def handle_post(path: str, deps: EnterpriseRouteDependencies) -> JsonRouteResult | None:
    if path not in _ENTERPRISE_POST_PATHS:
        return None
    enterprise = deps.enterprise

    try:
        payload = deps.read_json()

        if path == "/api/enterprise/access/check":
            user_id = str(payload.get("user_id", "")).strip()
            workspace_id = str(payload.get("workspace_id", "")).strip()
            permission = str(payload.get("permission", "")).strip()
            if not user_id or not workspace_id or not permission:
                return JsonRouteResult(
                    {"error": "user_id, workspace_id, and permission are required"},
                    HTTPStatus.BAD_REQUEST,
                )
            return JsonRouteResult(enterprise.check_access(
                user_id=user_id,
                workspace_id=workspace_id,
                permission=permission,
            ))

        if path == "/api/enterprise/audit/export":
            project = str(payload.get("project", "")).strip() or None
            return JsonRouteResult(enterprise.export_audit(project=project))

        if path == "/api/enterprise/policies/evaluate":
            scope = str(payload.get("scope", "")).strip()
            if not scope:
                return JsonRouteResult({"error": "scope is required"}, HTTPStatus.BAD_REQUEST)
            context = payload.get("context")
            if not isinstance(context, dict):
                context = {}
            return JsonRouteResult(enterprise.evaluate_policy(scope=scope, context=context))

        if path == "/api/enterprise/sso/validate":
            token = str(payload.get("token", "")).strip()
            if not token:
                return JsonRouteResult({"error": "token is required"}, HTTPStatus.BAD_REQUEST)
            config_id = str(payload.get("config_id", "")).strip()
            provider = str(payload.get("provider", "")).strip()
            audience = str(payload.get("audience", "")).strip()
            return JsonRouteResult(enterprise.validate_sso_token(
                token=token,
                config_id=config_id,
                provider=provider,
                audience=audience,
            ))

    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return None
