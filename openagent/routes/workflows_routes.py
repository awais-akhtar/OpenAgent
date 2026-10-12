"""Workflow HTTP contracts and request handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from ..workflows import WorkflowSafetyError
from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "workflows"
ROUTES = (
    ("GET", "/api/workflows"),
    ("POST", "/api/workflows"),
    ("PUT", "/api/workflows/{id}"),
    ("DELETE", "/api/workflows/{id}"),
    ("GET", "/api/workflows/builder"),
    ("GET", "/api/workflows/visual"),
    ("POST", "/api/workflows/run"),
    ("POST", "/api/workflows/{id}/run"),
    ("POST", "/api/workflows/run-due"),
    ("POST", "/api/workflows/event"),
    ("GET", "/api/workflows/layout"),
    ("POST", "/api/workflows/layout"),
    ("GET", "/api/workflow-runs"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class WorkflowRouteDependencies:
    store: Any
    workflows: Any
    read_json: Callable[[], dict[str, Any]]
    first: Callable[[dict[str, list[str]], str], str | None]
    parse_int: Callable[..., int]
    parse_bool: Callable[..., bool]


def _workflow_id(path: str) -> str | None:
    prefix = "/api/workflows/"
    if not path.startswith(prefix):
        return None
    suffix = path.removeprefix(prefix).strip("/")
    if not suffix or "/" in suffix or suffix in {"builder", "visual", "layout", "run", "run-due", "event"}:
        return None
    decoded = unquote(suffix)
    return decoded if "/" not in decoded and "\\" not in decoded else None


def handle_get(path: str, query: str, deps: WorkflowRouteDependencies) -> JsonRouteResult | None:
    params = parse_qs(query)
    if path == "/api/workflows":
        records = deps.store.list_workflows(
            project=deps.first(params, "project"),
            limit=deps.parse_int(deps.first(params, "limit"), default=100),
        )
        return JsonRouteResult({"workflows": [
            {**workflow, "trigger": deps.workflows.get_trigger(workflow["id"])}
            for workflow in records
        ]})
    if path == "/api/workflows/builder":
        return JsonRouteResult(deps.workflows.builder_schema())
    if path == "/api/workflows/visual":
        workflow_id = deps.first(params, "workflow_id") or ""
        records = deps.store.list_workflows(limit=200)
        workflow = next((item for item in records if item.get("id") == workflow_id), None)
        if not workflow:
            return JsonRouteResult({"nodes": [], "edges": []})
        steps = workflow.get("steps") or []
        nodes = [{"id": "trigger", "label": workflow.get("trigger_type", "Manual trigger"), "x": 0, "y": 20}]
        nodes.extend(
            {"id": f"s{index}", "label": str(step.get("action", step.get("type", "step"))),
             "x": (index + 1) * 130, "y": 20}
            for index, step in enumerate(steps)
        )
        edges = [{"source": nodes[index]["id"], "target": nodes[index + 1]["id"]}
                 for index in range(len(nodes) - 1)]
        return JsonRouteResult({"nodes": nodes, "edges": edges, "workflow": workflow})
    if path == "/api/workflows/layout":
        workflow_id = deps.first(params, "workflow_id")
        if not workflow_id:
            return JsonRouteResult({"error": "workflow_id is required."}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"layout": deps.workflows.get_layout(workflow_id)})
    if path == "/api/workflow-runs":
        return JsonRouteResult({"runs": deps.store.list_workflow_runs(
            project=deps.first(params, "project"),
            limit=deps.parse_int(deps.first(params, "limit"), default=100),
        )})
    return None


def handle_post(path: str, deps: WorkflowRouteDependencies) -> JsonRouteResult | None:
    if path not in {route for method, route in ROUTES if method == "POST"} and not (
        path.startswith("/api/workflows/") and path.endswith("/run")
    ):
        return None
    payload = deps.read_json()
    if path == "/api/workflows":
        name = str(payload.get("name", "")).strip()
        steps = payload.get("steps", [])
        if not name:
            return JsonRouteResult({"error": "A workflow name is required."}, HTTPStatus.BAD_REQUEST)
        if not isinstance(steps, list):
            return JsonRouteResult({"error": "Workflow steps must be a list."}, HTTPStatus.BAD_REQUEST)
        try:
            workflow = deps.workflows.create(
                name=name,
                steps=[step for step in steps if isinstance(step, dict)],
                project=str(payload.get("project", "default")).strip() or "default",
                description=str(payload.get("description", "")),
                trigger_type=str(payload.get("trigger_type", "manual")),
                trigger=payload.get("trigger") if isinstance(payload.get("trigger"), dict) else {},
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"workflow": workflow}, HTTPStatus.CREATED)
    if path == "/api/workflows/run" or (path.startswith("/api/workflows/") and path.endswith("/run")):
        workflow_id = str(payload.get("workflow_id", "")).strip()
        if not workflow_id and path.endswith("/run"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "workflows"]:
                workflow_id = unquote(parts[2])
        if not workflow_id:
            return JsonRouteResult({"error": "workflow_id is required."}, HTTPStatus.BAD_REQUEST)
        try:
            run = deps.workflows.run(
                workflow_id,
                allow_network=deps.parse_bool(payload.get("allow_network"), default=False),
                dry_run=deps.parse_bool(payload.get("dry_run"), default=False),
            )
        except (ValueError, WorkflowSafetyError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"run": run}, HTTPStatus.CREATED)
    if path == "/api/workflows/run-due":
        runs = deps.workflows.run_due_schedules(
            allow_network=deps.parse_bool(payload.get("allow_network"), default=False)
        )
        return JsonRouteResult({"runs": runs}, HTTPStatus.CREATED)
    if path == "/api/workflows/event":
        event_name = str(payload.get("event_name", "")).strip()
        if not event_name:
            return JsonRouteResult({"error": "event_name is required."}, HTTPStatus.BAD_REQUEST)
        try:
            result = deps.workflows.trigger_event(
                event_name,
                payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                project=str(payload.get("project", "default")).strip() or "default",
                allow_network=deps.parse_bool(payload.get("allow_network"), default=False),
            )
        except (ValueError, WorkflowSafetyError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result, HTTPStatus.CREATED)
    if path == "/api/workflows/layout":
        try:
            layout = deps.workflows.save_layout(
                str(payload.get("workflow_id", "")).strip(),
                payload.get("layout") if isinstance(payload.get("layout"), dict) else {},
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"layout": layout}, HTTPStatus.CREATED)
    return None


def handle_put(path: str, deps: WorkflowRouteDependencies) -> JsonRouteResult | None:
    workflow_id = _workflow_id(path)
    if workflow_id is None:
        return None
    payload = deps.read_json()
    try:
        result = deps.workflows.update(
            workflow_id,
            name=str(payload["name"]) if "name" in payload else None,
            description=str(payload["description"]) if "description" in payload else None,
            steps=payload["steps"] if isinstance(payload.get("steps"), list) else None,
            trigger_type=str(payload["trigger_type"]) if "trigger_type" in payload else None,
            trigger=payload["trigger"] if isinstance(payload.get("trigger"), dict) else None,
            enabled=deps.parse_bool(payload.get("enabled"), default=False) if "enabled" in payload else None,
        )
    except KeyError:
        return JsonRouteResult({"error": "Workflow not found."}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return JsonRouteResult({"workflow": result})


def handle_delete(path: str, deps: WorkflowRouteDependencies) -> JsonRouteResult | None:
    workflow_id = _workflow_id(path)
    if workflow_id is None:
        return None
    try:
        result = deps.workflows.delete(workflow_id)
    except KeyError:
        return JsonRouteResult({"error": "Workflow not found."}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult(result)
