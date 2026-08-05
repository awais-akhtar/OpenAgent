"""Model cookbook and safe serve route contracts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..model_cookbook import ModelCookbook
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "model_cookbook"
ROUTES = (
    ("GET", "/api/model-cookbook/hosts"),
    ("POST", "/api/model-cookbook/hosts"),
    ("GET", "/api/model-cookbook/cached"),
    ("POST", "/api/model-cookbook/serve"),
    ("POST", "/api/model-cookbook/adopt"),
    ("POST", "/api/model-cookbook/stop/{id}"),
    ("GET", "/api/model-cookbook/logs/{id}"),
    ("GET", "/api/model-cookbook/hardware-fit"),
    ("GET", "/api/model-cookbook/certifications"),
    ("POST", "/api/model-cookbook/certify"),
    ("POST", "/api/model-cookbook/certifications/{id}/revoke"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ModelCookbookRouteDependencies:
    cookbook: ModelCookbook
    read_json: Any


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def _parse_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def handle_get(path: str, query: str, deps: ModelCookbookRouteDependencies) -> JsonRouteResult | None:
    params = parse_qs(query)
    project = _first(params, "project") or "default"
    if path == "/api/model-cookbook/hosts":
        return JsonRouteResult({"hosts": deps.cookbook.hosts(project=project)})
    if path == "/api/model-cookbook/cached":
        return JsonRouteResult({"files": deps.cookbook.cached_files(project=project, root=_first(params, "root"))})
    if path == "/api/model-cookbook/hardware-fit":
        return JsonRouteResult(
            deps.cookbook.hardware_fit(
                project=project,
                model_ref=_first(params, "model_ref") or "local-model",
                parameter_count_b=_parse_float(_first(params, "parameter_count_b"), 3.0),
            )
        )
    if path == "/api/model-cookbook/certifications":
        return JsonRouteResult({"certifications": deps.cookbook.certifications(project=project)})
    if path.startswith("/api/model-cookbook/logs/"):
        session_id = unquote(path.removeprefix("/api/model-cookbook/logs/")).strip()
        try:
            return JsonRouteResult({"logs": deps.cookbook.logs(session_id)})
        except KeyError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return None


def handle_post(path: str, deps: ModelCookbookRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/model-cookbook/"):
        return None
    payload = deps.read_json()
    try:
        if path == "/api/model-cookbook/hosts":
            host = deps.cookbook.add_host(
                label=str(payload.get("label", "Local host")).strip() or "Local host",
                project=str(payload.get("project", "default")).strip() or "default",
                host_type=str(payload.get("host_type", "local")).strip() or "local",
                address=str(payload.get("address", "127.0.0.1")).strip() or "127.0.0.1",
                config=payload.get("config") if isinstance(payload.get("config"), dict) else {},
            )
            return JsonRouteResult({"host": host}, HTTPStatus.CREATED)
        if path == "/api/model-cookbook/serve":
            session = deps.cookbook.serve(
                project=str(payload.get("project", "default")).strip() or "default",
                host_id=str(payload.get("host_id", "")).strip(),
                model_ref=str(payload.get("model_ref", "")).strip(),
                runner=str(payload.get("runner", "")).strip(),
                args=[str(item) for item in payload.get("args", [])] if isinstance(payload.get("args"), list) else [],
                allow_process=bool(payload.get("allow_process")),
            )
            return JsonRouteResult({"session": session}, HTTPStatus.CREATED)
        if path == "/api/model-cookbook/adopt":
            session = deps.cookbook.adopt(
                project=str(payload.get("project", "default")).strip() or "default",
                host_id=str(payload.get("host_id", "")).strip(),
                model_ref=str(payload.get("model_ref", "")).strip(),
                runner=str(payload.get("runner", "")).strip(),
                pid=int(payload.get("pid", 0) or 0),
                command=[str(item) for item in payload.get("command", [])] if isinstance(payload.get("command"), list) else None,
            )
            return JsonRouteResult({"session": session}, HTTPStatus.CREATED)
        if path == "/api/model-cookbook/certify":
            certification = deps.cookbook.certify_endpoint(
                endpoint=str(payload.get("endpoint", "")),
                project=str(payload.get("project", "default")).strip() or "default",
                model_ref=str(payload.get("model_ref", "")).strip(),
                allow_external=bool(payload.get("allow_external")),
                api_key=str(payload.get("api_key", "")),
                completion_probe=bool(payload.get("completion_probe", True)),
            )
            return JsonRouteResult({"certification": certification}, HTTPStatus.CREATED)
        if path.startswith("/api/model-cookbook/certifications/") and path.endswith("/revoke"):
            certification_id = unquote(path.removeprefix("/api/model-cookbook/certifications/").removesuffix("/revoke")).strip("/")
            return JsonRouteResult({"certification": deps.cookbook.revoke_certification(certification_id)})
        if path.startswith("/api/model-cookbook/stop/"):
            session_id = unquote(path.removeprefix("/api/model-cookbook/stop/")).strip()
            return JsonRouteResult({"session": deps.cookbook.stop(session_id)})
    except (ValueError, KeyError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None
