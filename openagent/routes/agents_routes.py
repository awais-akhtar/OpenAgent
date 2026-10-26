"""Agent and delegation HTTP routes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from ..multi_agent import ApprovalRequired, PermissionDenied
from . import JsonRouteResult, StreamRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "agents"
ROUTES = (
    ("GET", "/api/agents"),
    ("GET", "/api/agent-runs"),
    ("GET", "/api/agents/jobs"),
    ("GET", "/api/agents/jobs/{id}"),
    ("GET", "/api/agents/jobs/{id}/events"),
    ("GET", "/api/agents/jobs/{id}/events/stream"),
    ("POST", "/api/agents/run"),
    ("POST", "/api/agents/run-checked"),
    ("POST", "/api/agents/jobs"),
    ("POST", "/api/agents/jobs/{id}/cancel"),
    ("POST", "/api/agents/jobs/{id}/retry"),
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
    store: Any
    owner: str
    first: Callable[..., str | None]
    parse_bool: Callable[..., bool]
    read_json: Callable[[], dict[str, Any]]
    auth_enforced: bool = False


def handle_get(path: str, query: str, deps: AgentRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/agents":
        return JsonRouteResult({"agents": deps.agents.specs()})
    if path == "/api/agent-runs":
        params = parse_qs(query, keep_blank_values=True)
        project = deps.first(params, "project")
        if not project or not project.strip():
            return JsonRouteResult({"error": "project is required."}, HTTPStatus.BAD_REQUEST)
        project = project.strip()

        def bounded_integer(key: str, default: int, minimum: int, maximum: int) -> int:
            raw = deps.first(params, key)
            try:
                value = default if raw is None else int(raw)
            except (TypeError, ValueError):
                value = default
            return max(minimum, min(value, maximum))

        limit = bounded_integer("limit", 20, 1, 100)
        offset = bounded_integer("offset", 0, 0, 2**31 - 1)
        owner = deps.owner if deps.auth_enforced else None
        if deps.auth_enforced and (
            not owner or not deps.store.agent_run_project_is_owned(project, owner)
        ):
            return JsonRouteResult({"error": "Project not found."}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult({
            "runs": deps.store.list_agent_runs(
                limit=limit, project=project, owner=owner, offset=offset,
            ),
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": deps.store.count_agent_runs(project, owner=owner),
            },
        })
    if path == "/api/agents/jobs" or path.startswith("/api/agents/jobs/"):
        params = parse_qs(query, keep_blank_values=True)
        project = (deps.first(params, "project") or "").strip()
        if path == "/api/agents/jobs":
            if not project:
                return JsonRouteResult({"error": "project is required."}, HTTPStatus.BAD_REQUEST)
            limit_raw = deps.first(params, "limit")
            try:
                limit = int(limit_raw) if limit_raw is not None else 50
            except (TypeError, ValueError):
                limit = 50
            return JsonRouteResult({"jobs": deps.multi_agent.jobs(project=project, owner=deps.owner, limit=limit)})
        remainder = path.removeprefix("/api/agents/jobs/").strip("/")
        if remainder.endswith("/events/stream"):
            job_id = remainder.removesuffix("/events/stream").strip("/")
            if not job_id or "/" in job_id:
                return JsonRouteResult({"error": "job id is required."}, HTTPStatus.BAD_REQUEST)
            try:
                # Support cursor-based reconnection with ?after=<cursor> parameter
                after = (deps.first(params, "after") or "").strip()

                # Support event filtering by type: ?type=started,progress
                event_type_filter = (deps.first(params, "type") or "").strip()

                # Support filtering by job status: ?status=running
                status_filter = (deps.first(params, "status") or "").strip()

                # Get limit from query param (default 1000 for streaming, max 1000)
                limit_raw = deps.first(params, "limit")
                try:
                    stream_limit = int(limit_raw) if limit_raw is not None else 1000
                    stream_limit = max(1, min(stream_limit, 1000))
                except (TypeError, ValueError):
                    stream_limit = 1000

                # Retrieve events with filters
                events = deps.multi_agent.job_events(
                    job_id,
                    owner=deps.owner,
                    after=after,
                    limit=stream_limit,
                    event_type=event_type_filter if event_type_filter else None,
                    status_filter=status_filter if status_filter else None,
                )

                def stream_events():
                    """Stream events in SSE format with proper error handling and cleanup."""
                    from . import ServerSentEvent
                    try:
                        # Send stream start event
                        yield ServerSentEvent(
                            data={
                                "event": "stream_started",
                                "job_id": job_id,
                                "cursor": after,
                                "filters": {
                                    "type": event_type_filter if event_type_filter else None,
                                    "status": status_filter if status_filter else None,
                                },
                            },
                            event="message",
                        )

                        # Stream all events (deduplicate by tracking IDs)
                        seen_ids = set()
                        for evt in events.get("events", []):
                            evt_id = evt.get("id")
                            # Skip duplicate events (already seen)
                            if evt_id and evt_id in seen_ids:
                                continue
                            if evt_id:
                                seen_ids.add(evt_id)
                            yield ServerSentEvent(data=evt, event="message")

                        # Send stream end event with next cursor for pagination
                        next_cursor = events.get("next") or events.get("cursor") or ""
                        yield ServerSentEvent(
                            data={
                                "event": "stream_ended",
                                "job_id": job_id,
                                "next_cursor": next_cursor,
                                "has_more": bool(events.get("next")),
                                "event_count": len(events.get("events", [])),
                            },
                            event="message",
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Send error event before closing
                        yield ServerSentEvent(
                            data={
                                "event": "stream_error",
                                "error": str(exc),
                                "job_id": job_id,
                            },
                            event="error",
                        )

                from . import StreamRouteResult
                return StreamRouteResult(
                    stream_events(),
                    {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
                )
            except KeyError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        if remainder.endswith("/events"):
            job_id = remainder.removesuffix("/events").strip("/")
            if not job_id or "/" in job_id:
                return JsonRouteResult({"error": "job id is required."}, HTTPStatus.BAD_REQUEST)
            after = (deps.first(params, "after") or "").strip()
            limit_raw = deps.first(params, "limit")
            try:
                limit = int(limit_raw) if limit_raw is not None else 100
            except (TypeError, ValueError):
                limit = 100
            try:
                return JsonRouteResult(deps.multi_agent.job_events(job_id, owner=deps.owner, after=after, limit=limit))
            except KeyError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        job_id = remainder
        job = deps.multi_agent.get_job(job_id, owner=deps.owner)
        return JsonRouteResult({"job": job} if job else {"error": "Agent job not found."}, HTTPStatus.OK if job else HTTPStatus.NOT_FOUND)
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
    if path not in {route for method, route in ROUTES if method == "POST"} and not path.startswith("/api/agents/jobs/"):
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    multi_agent = deps.multi_agent
    try:
        if path == "/api/agents/run":
            goal = str(payload.get("goal", "")).strip()
            if not goal:
                return JsonRouteResult({"error": "A goal is required."}, HTTPStatus.BAD_REQUEST)
            project = str(payload.get("project", "default")).strip() or "default"
            if not deps.store.claim_agent_run_project(project, deps.owner):
                return JsonRouteResult({"error": "Project not found."}, HTTPStatus.NOT_FOUND)
            run = deps.agents.run(
                str(payload.get("agent", "executive")).strip() or "executive",
                goal, project=project, owner=deps.owner,
            )
            return JsonRouteResult({"run": run}, HTTPStatus.CREATED)
        if path == "/api/agents/run-checked":
            goal = str(payload.get("goal", "")).strip()
            if not goal:
                return JsonRouteResult({"error": "A goal is required."}, HTTPStatus.BAD_REQUEST)
            if not deps.store.claim_agent_run_project(project, deps.owner):
                return JsonRouteResult({"error": "Project not found."}, HTTPStatus.NOT_FOUND)
            result = multi_agent.run_agent(
                str(payload.get("agent", "executive")).strip() or "executive",
                goal, project=project, owner=deps.owner,
                require_approval=deps.parse_bool(payload.get("require_approval"), default=False),
                approval_id=str(payload.get("approval_id", "")).strip() or None,
                verify=deps.parse_bool(payload.get("verify"), default=True),
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        if path == "/api/agents/jobs":
            if deps.parse_bool(payload.get("require_approval"), default=False) and not multi_agent.approval_is_approved(
                str(payload.get("approval_id", "")).strip() or None
            ):
                return JsonRouteResult({"error": "An approved agent run approval is required."}, HTTPStatus.BAD_REQUEST)
            job = multi_agent.start_job(
                str(payload.get("agent", "executive")).strip() or "executive",
                str(payload.get("goal", "")).strip(), project=project, owner=deps.owner,
            )
            return JsonRouteResult({"job": job}, HTTPStatus.ACCEPTED)
        if path.startswith("/api/agents/jobs/"):
            remainder = path.removeprefix("/api/agents/jobs/").strip("/")
            if remainder.endswith("/cancel"):
                job = multi_agent.cancel_job(remainder.removesuffix("/cancel").strip("/"), owner=deps.owner)
                return JsonRouteResult({"job": job})
            if remainder.endswith("/retry"):
                job = multi_agent.retry_job(remainder.removesuffix("/retry").strip("/"), owner=deps.owner)
                return JsonRouteResult({"job": job}, HTTPStatus.ACCEPTED)
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
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc).strip("'")}, HTTPStatus.NOT_FOUND)
    except (ApprovalRequired, PermissionDenied, ValueError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
