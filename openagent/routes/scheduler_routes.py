"""Scheduler and notification route contracts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..scheduler import Scheduler
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "scheduler"
ROUTES = (
    ("GET", "/api/tasks"),
    ("GET", "/api/tasks/notifications"),
    ("GET", "/api/tasks/{id}"),
    ("GET", "/api/tasks/{id}/runs"),
    ("GET", "/api/tasks/{id}/destinations"),
    ("GET", "/api/tasks/{id}/deliveries"),
    ("POST", "/api/tasks"),
    ("POST", "/api/tasks/parse"),
    ("POST", "/api/tasks/run-due"),
    ("POST", "/api/tasks/notifications"),
    ("POST", "/api/tasks/notifications/{id}/read"),
    ("POST", "/api/tasks/{id}/pause"),
    ("POST", "/api/tasks/{id}/resume"),
    ("POST", "/api/tasks/{id}/stop"),
    ("POST", "/api/tasks/{id}/revert"),
    ("POST", "/api/tasks/{id}/clear-cache"),
    ("POST", "/api/tasks/{id}/dependencies"),
    ("POST", "/api/tasks/{id}/webhook"),
    ("POST", "/api/tasks/{id}/destinations"),
    ("POST", "/api/tasks/{id}/delete"),
    ("PATCH", "/api/tasks/{id}"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class SchedulerRouteDependencies:
    scheduler: Scheduler
    first: Any
    parse_bool: Any
    parse_int: Any
    query_owner: Any
    payload_owner: Any
    read_json: Any


def handle_get(path: str, query: str, deps: SchedulerRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/tasks"):
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    owner = deps.query_owner(params)
    if path == "/api/tasks":
        return JsonRouteResult(
            {
                "tasks": deps.scheduler.list_tasks(
                    project=project,
                    owner=owner,
                    status=deps.first(params, "status") or "",
                    limit=deps.parse_int(deps.first(params, "limit"), default=100),
                )
            }
        )
    if path == "/api/tasks/notifications":
        return JsonRouteResult(
            {
                "notifications": deps.scheduler.notifications(
                    project=project,
                    owner=owner,
                    unread_only=deps.parse_bool(deps.first(params, "unread"), default=False),
                    limit=deps.parse_int(deps.first(params, "limit"), default=100),
                )
            }
        )
    if path.startswith("/api/tasks/"):
        suffix = unquote(path.removeprefix("/api/tasks/")).strip("/")
        if suffix.endswith("/runs"):
            task_id = suffix.removesuffix("/runs").strip("/")
            try:
                return JsonRouteResult({"runs": deps.scheduler.run_history(task_id, owner=owner, limit=deps.parse_int(deps.first(params, "limit"), default=100))})
            except KeyError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        if suffix.endswith("/destinations"):
            task_id = suffix.removesuffix("/destinations").strip("/")
            try:
                return JsonRouteResult({"destinations": deps.scheduler.output_destinations(task_id, owner=owner)})
            except KeyError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        if suffix.endswith("/deliveries"):
            task_id = suffix.removesuffix("/deliveries").strip("/")
            try:
                return JsonRouteResult({
                    "deliveries": deps.scheduler.delivery_history(
                        task_id, owner=owner, limit=deps.parse_int(deps.first(params, "limit"), default=100)
                    )
                })
            except KeyError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        try:
            return JsonRouteResult({"task": deps.scheduler.get_task(suffix, owner=owner)})
        except KeyError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return None


def handle_post(path: str, deps: SchedulerRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/tasks"):
        return None
    payload = deps.read_json()
    owner = deps.payload_owner(payload)
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/tasks":
            task = deps.scheduler.create_task(
                title=str(payload.get("title", "")).strip(),
                project=project,
                owner=owner,
                task_type=str(payload.get("task_type", "reminder")).strip() or "reminder",
                schedule=payload.get("schedule") if isinstance(payload.get("schedule"), dict) else {},
                payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
            )
            return JsonRouteResult({"task": task}, HTTPStatus.CREATED)
        if path == "/api/tasks/parse":
            return JsonRouteResult(
                {"task": deps.scheduler.create_from_text(str(payload.get("text", "")), project=project, owner=owner)},
                HTTPStatus.CREATED,
            )
        if path == "/api/tasks/run-due":
            return JsonRouteResult(
                deps.scheduler.run_due(
                    project=project,
                    owner=owner if owner != "local" else "",
                    limit=deps.parse_int(str(payload.get("limit", "25")), default=25),
                )
            )
        if path == "/api/tasks/notifications":
            notification = deps.scheduler.add_notification(
                title=str(payload.get("title", "")).strip() or "OpenAgent notification",
                body=str(payload.get("body", "")),
                project=project,
                owner=owner,
                task_id=str(payload.get("task_id", "")),
                level=str(payload.get("level", "info")).strip() or "info",
            )
            return JsonRouteResult({"notification": notification}, HTTPStatus.CREATED)
        if path.startswith("/api/tasks/notifications/") and path.endswith("/read"):
            notification_id = unquote(path.removeprefix("/api/tasks/notifications/").removesuffix("/read")).strip("/")
            return JsonRouteResult({"notification": deps.scheduler.mark_notification_read(
                notification_id, project=project, owner=owner if owner != "local" else "",
            )})
        if path.startswith("/api/tasks/") and path.endswith("/pause"):
            task_id = unquote(path.removeprefix("/api/tasks/").removesuffix("/pause")).strip("/")
            return JsonRouteResult({"task": deps.scheduler.pause(task_id, owner=owner if owner != "local" else "")})
        if path.startswith("/api/tasks/") and path.endswith("/resume"):
            task_id = unquote(path.removeprefix("/api/tasks/").removesuffix("/resume")).strip("/")
            return JsonRouteResult({"task": deps.scheduler.resume(task_id, owner=owner if owner != "local" else "")})
        if path.startswith("/api/tasks/"):
            suffix = unquote(path.removeprefix("/api/tasks/")).strip("/")
            action = suffix.rsplit("/", 1)[-1] if "/" in suffix else ""
            task_id = suffix.rsplit("/", 1)[0] if action else suffix
            scope_owner = owner if owner != "local" else ""
            if action == "stop":
                return JsonRouteResult({"task": deps.scheduler.stop(task_id, owner=scope_owner)})
            if action == "revert":
                return JsonRouteResult({"task": deps.scheduler.revert(task_id, owner=scope_owner)})
            if action == "clear-cache":
                return JsonRouteResult(deps.scheduler.clear_cache(task_id, owner=scope_owner))
            if action == "dependencies":
                values = payload.get("dependency_ids")
                return JsonRouteResult(deps.scheduler.set_dependencies(task_id, [str(item) for item in values] if isinstance(values, list) else [], owner=scope_owner))
            if action == "webhook":
                return JsonRouteResult(
                    {"webhook": deps.scheduler.create_webhook(task_id, str(payload.get("target_url", "")), owner=scope_owner)},
                    HTTPStatus.CREATED,
                )
            if action == "destinations":
                destination = deps.scheduler.add_output_destination(
                    task_id,
                    str(payload.get("destination_type", "run_history")),
                    config=payload.get("config") if isinstance(payload.get("config"), dict) else {},
                    owner=scope_owner,
                    allow_external=bool(payload.get("allow_external")),
                )
                return JsonRouteResult({"destination": destination}, HTTPStatus.CREATED)
            if action == "delete":
                return JsonRouteResult(deps.scheduler.delete(task_id, owner=scope_owner))
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None


def handle_patch(path: str, deps: SchedulerRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/tasks/"):
        return None
    payload = deps.read_json()
    owner = deps.payload_owner(payload, default="")
    task_id = unquote(path.removeprefix("/api/tasks/")).strip("/")
    try:
        return JsonRouteResult({"task": deps.scheduler.update_task(task_id, patch=payload, owner=owner)})
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
