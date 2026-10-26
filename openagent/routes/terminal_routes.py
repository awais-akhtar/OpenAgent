"""Safe terminal route contracts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..safe_terminal import SafeTerminal
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "safe_terminal"
ROUTES = (
    ("GET", "/api/terminal/runs"),
    ("GET", "/api/terminal/runs/{id}"),
    ("POST", "/api/terminal/runs"),
    ("POST", "/api/terminal/stream"),
    ("DELETE", "/api/terminal/runs/{id}"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class TerminalRouteDependencies:
    terminal: SafeTerminal
    first: Any
    parse_bool: Any
    parse_int: Any
    query_owner: Any
    payload_owner: Any
    read_json: Any


def handle_get(path: str, query: str, deps: TerminalRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/terminal"):
        return None
    params = parse_qs(query)
    try:
        if path == "/api/terminal/runs":
            return JsonRouteResult(
                {
                    "runs": deps.terminal.runs(
                        project=deps.first(params, "project") or "default",
                        owner=deps.query_owner(params),
                        limit=deps.parse_int(deps.first(params, "limit"), default=50),
                    )
                }
            )
        if path.startswith("/api/terminal/runs/"):
            run_id = unquote(path.removeprefix("/api/terminal/runs/")).strip("/")
            return JsonRouteResult({"run": deps.terminal.get_run(run_id, owner=deps.query_owner(params))})
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return None


def handle_post(path: str, deps: TerminalRouteDependencies) -> JsonRouteResult | None:
    if path not in {"/api/terminal/stream", "/api/terminal/runs"}:
        return None
    payload = deps.read_json()
    command = payload.get("command")
    if isinstance(command, str):
        command = command.split()
    if not isinstance(command, list):
        return JsonRouteResult({"error": "command must be a list of command arguments"}, HTTPStatus.BAD_REQUEST)
    try:
        operation = deps.terminal.start if path == "/api/terminal/runs" else deps.terminal.run
        run = operation(
            command=[str(part) for part in command],
            cwd=str(payload.get("cwd", "")),
            project=str(payload.get("project", "default")).strip() or "default",
            owner=deps.payload_owner(payload),
            timeout_seconds=deps.parse_int(str(payload.get("timeout_seconds", "30")), default=30),
            approved=deps.parse_bool(payload.get("approved"), default=False),
        )
        return JsonRouteResult({"run": run}, HTTPStatus.ACCEPTED if path == "/api/terminal/runs" else HTTPStatus.CREATED)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def handle_delete(path: str, query: str, deps: TerminalRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/terminal/runs/"):
        return None
    run_id = unquote(path.removeprefix("/api/terminal/runs/")).strip("/")
    if not run_id:
        return JsonRouteResult({"error": "run ID is required"}, HTTPStatus.BAD_REQUEST)
    try:
        return JsonRouteResult(deps.terminal.cancel(run_id, owner=deps.query_owner(parse_qs(query))))
    except KeyError:
        return JsonRouteResult({"error": "terminal run not found"}, HTTPStatus.NOT_FOUND)
