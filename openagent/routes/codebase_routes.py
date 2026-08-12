"""Autonomous codebase workspace API contracts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..codebase import CodebaseService
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "codebase"
ROUTES = (
    ("GET", "/api/codebase/projects"),
    ("GET", "/api/codebase/projects/{id}"),
    ("GET", "/api/codebase/projects/{id}/tree"),
    ("GET", "/api/codebase/projects/{id}/file"),
    ("GET", "/api/codebase/runs"),
    ("GET", "/api/codebase/runs/{id}"),
    ("GET", "/api/codebase/runs/{id}/events"),
    ("GET", "/api/codebase/runs/{id}/events/stream"),
    ("POST", "/api/codebase/projects"),
    ("POST", "/api/codebase/folder-picker"),
    ("POST", "/api/codebase/projects/{id}/scan"),
    ("POST", "/api/codebase/projects/{id}/file"),
    ("POST", "/api/codebase/projects/{id}/tools"),
    ("POST", "/api/codebase/runs"),
    ("POST", "/api/codebase/runs/{id}/cancel"),
    ("POST", "/api/codebase/runs/{id}/resume"),
    ("POST", "/api/codebase/runs/{id}/tools"),
    ("POST", "/api/codebase/approvals/{id}/resolve"),
    ("POST", "/api/codebase/runs/{id}/snapshots/{snapshot_id}/restore"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class CodebaseRouteDependencies:
    codebase: CodebaseService
    first: Any
    parse_bool: Any
    parse_int: Any
    query_owner: Any
    payload_owner: Any
    read_json: Any


def handle_get(path: str, query: str, deps: CodebaseRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/codebase"):
        return None
    params = parse_qs(query)
    owner = deps.query_owner(params) or "local"
    try:
        if path == "/api/codebase/projects":
            return JsonRouteResult({"projects": deps.codebase.list_projects(project=deps.first(params, "project") or "default", owner=owner)})
        if path == "/api/codebase/runs":
            return JsonRouteResult({"runs": deps.codebase.list_runs(project=deps.first(params, "project") or "default", owner=owner, limit=deps.parse_int(deps.first(params, "limit"), default=50))})
        if path.startswith("/api/codebase/projects/"):
            remainder = unquote(path.removeprefix("/api/codebase/projects/")).strip("/")
            if remainder.endswith("/tree"):
                return JsonRouteResult(deps.codebase.file_tree(remainder.removesuffix("/tree"), owner=owner))
            if remainder.endswith("/file"):
                return JsonRouteResult({"file": deps.codebase.read_file(remainder.removesuffix("/file"), deps.first(params, "path"), owner=owner)})
            return JsonRouteResult({"project": deps.codebase.get_project(remainder, owner=owner)})
        if path.startswith("/api/codebase/runs/"):
            remainder = unquote(path.removeprefix("/api/codebase/runs/")).strip("/")
            if remainder.endswith("/events/stream"):
                return None
            if remainder.endswith("/events"):
                run_id = remainder.removesuffix("/events")
                return JsonRouteResult({"events": deps.codebase.events(run_id, owner=owner, after=deps.parse_int(deps.first(params, "after"), default=0))})
            return JsonRouteResult({"run": deps.codebase.get_run(remainder, owner=owner)})
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except (ValueError, OSError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None


def handle_post(path: str, deps: CodebaseRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/codebase"):
        return None
    payload = deps.read_json()
    owner = deps.payload_owner(payload)
    try:
        if path == "/api/codebase/folder-picker":
            return JsonRouteResult({"path": deps.codebase.pick_source_folder()})
        if path == "/api/codebase/projects":
            project = deps.codebase.create_project(
                name=str(payload.get("name", "")).strip(),
                source_path=str(payload.get("source_path", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner,
                backend=str(payload.get("backend", "local")).strip() or "local",
            )
            return JsonRouteResult({"project": project}, HTTPStatus.CREATED)
        if path == "/api/codebase/runs":
            run = deps.codebase.create_run(
                codebase_project_id=str(payload.get("codebase_project_id", "")).strip(),
                prompt=str(payload.get("prompt", "")).strip(),
                owner=owner,
                provider=str(payload.get("provider", "auto")).strip() or "auto",
                model=str(payload.get("model", "")).strip(),
                role_models=payload.get("role_models") if isinstance(payload.get("role_models"), dict) else {},
                allow_external=deps.parse_bool(payload.get("allow_external"), default=False),
                max_iterations=deps.parse_int(str(payload.get("max_iterations", "0")), default=0),
                completion=payload.get("completion") if isinstance(payload.get("completion"), dict) else {},
            )
            return JsonRouteResult({"run": run}, HTTPStatus.ACCEPTED)
        if path.startswith("/api/codebase/projects/") and path.endswith("/scan"):
            project_id = unquote(path.removeprefix("/api/codebase/projects/").removesuffix("/scan")).strip("/")
            return JsonRouteResult({"project_map": deps.codebase.refresh_project_map(project_id, owner=owner)})
        if path.startswith("/api/codebase/projects/") and path.endswith("/file"):
            project_id = unquote(path.removeprefix("/api/codebase/projects/").removesuffix("/file")).strip("/")
            return JsonRouteResult({"change": deps.codebase.write_file(project_id, str(payload.get("path", "")).strip(), str(payload.get("content", "")), owner=owner, expected_sha256=str(payload.get("expected_sha256", "")).strip())})
        if path.startswith("/api/codebase/projects/") and path.endswith("/tools"):
            project_id = unquote(path.removeprefix("/api/codebase/projects/").removesuffix("/tools")).strip("/")
            return JsonRouteResult(deps.codebase.invoke_project_tool(project_id, str(payload.get("tool", "")).strip(), payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}, owner=owner, approved=deps.parse_bool(payload.get("approved"), default=False)))
        if path.startswith("/api/codebase/approvals/") and path.endswith("/resolve"):
            approval_id = unquote(path.removeprefix("/api/codebase/approvals/").removesuffix("/resolve")).strip("/")
            return JsonRouteResult({"approval": deps.codebase.resolve_approval(approval_id, approved=deps.parse_bool(payload.get("approved"), default=False), owner=owner)})
        if path.startswith("/api/codebase/runs/"):
            remainder = unquote(path.removeprefix("/api/codebase/runs/")).strip("/")
            if remainder.endswith("/cancel"):
                return JsonRouteResult({"run": deps.codebase.cancel_run(remainder.removesuffix("/cancel"), owner=owner)})
            if remainder.endswith("/resume"):
                return JsonRouteResult({"run": deps.codebase.resume_run(remainder.removesuffix("/resume"), owner=owner)}, HTTPStatus.ACCEPTED)
            if remainder.endswith("/tools"):
                return JsonRouteResult({"result": deps.codebase.invoke_tool(remainder.removesuffix("/tools"), str(payload.get("tool", "")).strip(), payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}, owner=owner, approved=deps.parse_bool(payload.get("approved"), default=False))})
            marker = "/snapshots/"
            if marker in remainder and remainder.endswith("/restore"):
                run_id, snapshot_part = remainder.split(marker, 1)
                snapshot_id = snapshot_part.removesuffix("/restore").strip("/")
                return JsonRouteResult(deps.codebase.restore_run_snapshot(run_id, snapshot_id, owner=owner))
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except (ValueError, PermissionError, OSError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    except Exception as exc:
        return JsonRouteResult({"error": f"CodeQ request failed: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
    return None
