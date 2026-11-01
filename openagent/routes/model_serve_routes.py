"""Routes for G20 (Model Serve) and G22 (Ultra Model) with real model downloads."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..model_downloader import ModelDownloader
from ..model_serve_manager import ModelServeManager
from ..ultra_model_service import UltraModelService
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "model_serve"
ROUTES = (
    # G20: Model Serve Routes
    ("POST", "/api/model-serve/plans"),
    ("POST", "/api/model-serve/sessions/{id}/start"),
    ("POST", "/api/model-serve/sessions/{id}/stop"),
    ("GET", "/api/model-serve/sessions/{id}"),
    ("GET", "/api/model-serve/sessions"),
    # G22: Ultra Model Routes
    ("GET", "/api/ultra-models/models"),
    ("POST", "/api/ultra-models/models/{id}/download"),
    ("POST", "/api/ultra-models/sessions"),
    ("POST", "/api/ultra-models/sessions/{id}/launch"),
    ("POST", "/api/ultra-models/sessions/{id}/stop"),
    ("POST", "/api/ultra-models/sessions/{id}/reload"),
    ("GET", "/api/ultra-models/sessions/{id}"),
    ("GET", "/api/ultra-models/sessions"),
    ("GET", "/api/ultra-models/readiness"),
    ("GET", "/api/ultra-models/models/{id}/calibrate"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ModelServeRouteDependencies:
    serve_manager: ModelServeManager
    ultra_model_service: UltraModelService
    read_json: Any


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def handle_get(path: str, query: str, deps: ModelServeRouteDependencies) -> JsonRouteResult | None:
    """Handle GET requests for model serve endpoints."""
    params = parse_qs(query)

    # G20 routes
    if path == "/api/model-serve/sessions":
        sessions = deps.serve_manager.list_sessions()
        return JsonRouteResult({"sessions": sessions})

    if path.startswith("/api/model-serve/sessions/") and path.endswith(""):
        session_id = unquote(path.removeprefix("/api/model-serve/sessions/")).strip()
        session = deps.serve_manager.get_session(session_id)
        if not session:
            return JsonRouteResult({"error": f"Session {session_id} not found"}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult(session)

    # G22 routes
    if path == "/api/ultra-models/models":
        models = deps.ultra_model_service.get_available_models()
        return JsonRouteResult({"models": models})

    if path == "/api/ultra-models/sessions":
        sessions = deps.ultra_model_service.list_sessions()
        return JsonRouteResult({"sessions": sessions})

    if path == "/api/ultra-models/readiness":
        status = deps.ultra_model_service.get_readiness_status()
        return JsonRouteResult(status)

    if path.startswith("/api/ultra-models/sessions/") and path.endswith(""):
        session_id = unquote(path.removeprefix("/api/ultra-models/sessions/")).strip()
        session = deps.ultra_model_service.get_session(session_id)
        if not session:
            return JsonRouteResult({"error": f"Session {session_id} not found"}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult(session)

    if path.startswith("/api/ultra-models/models/") and path.endswith("/calibrate"):
        model_id = unquote(path.removeprefix("/api/ultra-models/models/").removesuffix("/calibrate")).strip()
        calibration = deps.ultra_model_service.calibrate_for_hardware(model_id)
        if "error" in calibration:
            return JsonRouteResult(calibration, HTTPStatus.NOT_FOUND)
        return JsonRouteResult(calibration)

    return None


def handle_post(path: str, deps: ModelServeRouteDependencies) -> JsonRouteResult | None:
    """Handle POST requests for model serve endpoints."""
    if not path.startswith(("/api/model-serve/", "/api/ultra-models/")):
        return None

    payload = deps.read_json()

    # G20: Model Serve routes
    if path == "/api/model-serve/plans":
        model_id = str(payload.get("model_id", "")).strip()
        model_path = str(payload.get("model_path", "")).strip()
        if not model_id or not model_path:
            return JsonRouteResult({"error": "model_id and model_path required"}, HTTPStatus.BAD_REQUEST)

        plan = deps.serve_manager.create_serve_plan(
            model_id=model_id,
            model_path=model_path,
            host=str(payload.get("host", "127.0.0.1")).strip(),
            port=int(payload.get("port", 8000)),
        )
        return JsonRouteResult(plan, HTTPStatus.CREATED)

    if path.startswith("/api/model-serve/sessions/") and path.endswith("/start"):
        session_id = unquote(path.removeprefix("/api/model-serve/sessions/").removesuffix("/start")).strip()
        result = deps.serve_manager.start_serve(session_id)
        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)

    if path.startswith("/api/model-serve/sessions/") and path.endswith("/stop"):
        session_id = unquote(path.removeprefix("/api/model-serve/sessions/").removesuffix("/stop")).strip()
        result = deps.serve_manager.stop_serve(session_id)
        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)

    # G22: Ultra Model routes
    if path.startswith("/api/ultra-models/models/") and path.endswith("/download"):
        model_id = unquote(path.removeprefix("/api/ultra-models/models/").removesuffix("/download")).strip()
        result = deps.ultra_model_service.download_model(model_id)
        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)

    if path == "/api/ultra-models/sessions":
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id:
            return JsonRouteResult({"error": "model_id required"}, HTTPStatus.BAD_REQUEST)

        result = deps.ultra_model_service.create_session(
            model_id=model_id,
            session_name=str(payload.get("name", "")).strip() or None,
        )
        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result, HTTPStatus.CREATED)

    if path.startswith("/api/ultra-models/sessions/") and path.endswith("/launch"):
        session_id = unquote(path.removeprefix("/api/ultra-models/sessions/").removesuffix("/launch")).strip()
        result = deps.ultra_model_service.launch_session(session_id)
        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)

    if path.startswith("/api/ultra-models/sessions/") and path.endswith("/stop"):
        session_id = unquote(path.removeprefix("/api/ultra-models/sessions/").removesuffix("/stop")).strip()
        result = deps.ultra_model_service.stop_session(session_id)
        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)

    if path.startswith("/api/ultra-models/sessions/") and path.endswith("/reload"):
        session_id = unquote(path.removeprefix("/api/ultra-models/sessions/").removesuffix("/reload")).strip()
        result = deps.ultra_model_service.reload_session(session_id)
        if "error" in result:
            return JsonRouteResult(result, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)

    return None
