"""Durable worker queue route contracts and handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "worker"
ROUTES = (
    ("GET", "/api/worker/status"),
    ("GET", "/api/worker/jobs"),
    ("GET", "/api/worker/jobs/{id}"),
    ("GET", "/api/worker/dead-letters"),
    ("POST", "/api/worker/jobs"),
    ("POST", "/api/worker/run-once"),
    ("POST", "/api/worker/jobs/{id}/cancel"),
    ("POST", "/api/worker/jobs/{id}/retry"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class WorkerRouteDependencies:
    status: Callable[[str], dict[str, Any]]
    list_jobs: Callable[[str, str, int], list[dict[str, Any]]]
    get_job: Callable[[str], dict[str, Any] | None]
    dead_letters: Callable[[str, int], list[dict[str, Any]]]
    enqueue: Callable[[str, dict[str, Any], str, int, int], dict[str, Any]]
    run_once: Callable[[], dict[str, Any]]
    cancel: Callable[[str, str], dict[str, Any] | None]
    retry: Callable[[str], dict[str, Any] | None]
    read_json: Callable[[], dict[str, Any]]


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def handle_get(path: str, query: str, deps: WorkerRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/worker/status":
        params = parse_qs(query)
        return JsonRouteResult(deps.status(_first(params, "project") or "default"))
    if path == "/api/worker/jobs":
        params = parse_qs(query)
        return JsonRouteResult(
            {
                "jobs": deps.list_jobs(
                    _first(params, "project") or "default",
                    _first(params, "state"),
                    _parse_int(_first(params, "limit"), 50),
                )
            }
        )
    if path == "/api/worker/dead-letters":
        params = parse_qs(query)
        return JsonRouteResult(
            {
                "dead_letters": deps.dead_letters(
                    _first(params, "project") or "default",
                    _parse_int(_first(params, "limit"), 50),
                )
            }
        )
    if path.startswith("/api/worker/jobs/"):
        job_id = unquote(path.removeprefix("/api/worker/jobs/")).strip()
        if "/" in job_id:
            return None
        job = deps.get_job(job_id)
        if job is None:
            return JsonRouteResult({"error": "Worker job not found."}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult({"job": job})
    return None


def handle_post(path: str, deps: WorkerRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/worker/jobs":
        payload = deps.read_json()
        job_type = str(payload.get("job_type", "")).strip()
        if not job_type:
            return JsonRouteResult({"error": "job_type is required."}, HTTPStatus.BAD_REQUEST)
        job_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        job = deps.enqueue(
            job_type,
            job_payload,
            str(payload.get("project", "default")).strip() or "default",
            _parse_int(str(payload.get("priority", "0")), 0),
            _parse_int(str(payload.get("max_attempts", "3")), 3),
        )
        return JsonRouteResult({"job": job}, HTTPStatus.CREATED)
    if path == "/api/worker/run-once":
        return JsonRouteResult(deps.run_once())
    if path.startswith("/api/worker/jobs/") and path.endswith("/cancel"):
        job_id = unquote(path.removeprefix("/api/worker/jobs/").removesuffix("/cancel")).strip("/")
        payload = deps.read_json()
        job = deps.cancel(job_id, str(payload.get("reason", "cancelled")).strip() or "cancelled")
        if job is None:
            return JsonRouteResult({"error": "Worker job not found."}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult({"job": job})
    if path.startswith("/api/worker/jobs/") and path.endswith("/retry"):
        job_id = unquote(path.removeprefix("/api/worker/jobs/").removesuffix("/retry")).strip("/")
        job = deps.retry(job_id)
        if job is None:
            return JsonRouteResult({"error": "Worker job not found."}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult({"job": job})
    return None
