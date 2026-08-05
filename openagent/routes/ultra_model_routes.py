"""Ultra-scale local model engine route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..ultra_model_engine import LaunchApprovalRequired, UltraModelEngine, UltraModelError
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "ultra_model"
ROUTES = (
    ("GET", "/api/ultra-model/status"),
    ("GET", "/api/ultra-model/manifests"),
    ("GET", "/api/ultra-model/plans"),
    ("GET", "/api/ultra-model/sessions"),
    ("POST", "/api/ultra-model/manifests"),
    ("POST", "/api/ultra-model/plan"),
    ("POST", "/api/ultra-model/route"),
    ("POST", "/api/ultra-model/calibrate"),
    ("POST", "/api/ultra-model/launch"),
    ("POST", "/api/ultra-model/warmup"),
    ("POST", "/api/ultra-model/verify"),
    ("POST", "/api/ultra-model/sessions/{id}/stop"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class UltraModelRouteDependencies:
    engine: UltraModelEngine
    read_json: Any


def handle_get(path: str, query: str, deps: UltraModelRouteDependencies) -> JsonRouteResult | None:
    if path not in {
        "/api/ultra-model/status",
        "/api/ultra-model/manifests",
        "/api/ultra-model/plans",
        "/api/ultra-model/sessions",
    }:
        return None
    params = parse_qs(query)
    project = _first(params, "project") or "default"
    limit = _as_int(_first(params, "limit"), 50)
    if path == "/api/ultra-model/status":
        return JsonRouteResult(deps.engine.status(project=project))
    if path == "/api/ultra-model/manifests":
        return JsonRouteResult({"manifests": deps.engine.manifests(project=project, limit=limit)})
    if path == "/api/ultra-model/plans":
        return JsonRouteResult({"plans": deps.engine.plans(project=project, limit=limit)})
    return JsonRouteResult({"sessions": deps.engine.sessions(project=project, limit=limit)})


def handle_post(path: str, deps: UltraModelRouteDependencies) -> JsonRouteResult | None:
    is_stop = path.startswith("/api/ultra-model/sessions/") and path.endswith("/stop")
    if path not in {
        "/api/ultra-model/manifests",
        "/api/ultra-model/plan",
        "/api/ultra-model/route",
        "/api/ultra-model/calibrate",
        "/api/ultra-model/launch",
        "/api/ultra-model/warmup",
        "/api/ultra-model/verify",
    } and not is_stop:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/ultra-model/manifests":
            manifest = deps.engine.register_manifest(
                name=str(payload.get("name", "")).strip(),
                shard_paths=[str(item) for item in payload.get("shard_paths", [])]
                if isinstance(payload.get("shard_paths"), list)
                else [str(payload.get("shard_path", ""))],
                parameter_count_b=float(payload.get("parameter_count_b", 0)),
                active_parameter_count_b=_optional_float(payload.get("active_parameter_count_b")),
                quantization_bits=float(payload.get("quantization_bits", 4)),
                architecture=str(payload.get("architecture", "dense")),
                draft_model_path=str(payload.get("draft_model_path", "")),
                project=project,
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"manifest": manifest}, HTTPStatus.CREATED)
        if path == "/api/ultra-model/plan":
            specification = payload.get("specification") if isinstance(payload.get("specification"), dict) else {}
            plan = deps.engine.plan(
                manifest_id=str(payload.get("manifest_id", "")).strip(),
                specification=specification,
                project=project,
                context_tokens=_as_int(payload.get("context_tokens"), 4096),
                expected_output_tokens=_as_int(payload.get("expected_output_tokens"), 256),
                hardware=payload.get("hardware") if isinstance(payload.get("hardware"), dict) else None,
                trusted_nodes=[item for item in payload.get("trusted_nodes", []) if isinstance(item, dict)]
                if isinstance(payload.get("trusted_nodes"), list)
                else [],
            )
            return JsonRouteResult({"plan": plan}, HTTPStatus.CREATED)
        if path == "/api/ultra-model/route":
            result = deps.engine.route_request(
                str(payload.get("prompt", "")),
                high_assurance=_as_bool(payload.get("high_assurance"), False),
            )
            return JsonRouteResult(result)
        if path == "/api/ultra-model/calibrate":
            calibration = deps.engine.calibrate(
                project=project,
                size_mb=_as_int(payload.get("size_mb"), 32),
            )
            return JsonRouteResult({"calibration": calibration}, HTTPStatus.CREATED)
        if path == "/api/ultra-model/launch":
            result = deps.engine.launch(
                str(payload.get("manifest_id", "")).strip(),
                approved=_as_bool(payload.get("approved"), False),
                project=project,
                executable=str(payload.get("executable", "")).strip() or None,
                host=str(payload.get("host", "127.0.0.1")).strip() or "127.0.0.1",
                port=_as_int(payload.get("port"), 8081),
                allow_distributed=_as_bool(payload.get("allow_distributed"), False),
                keep_warm=_as_bool(payload.get("keep_warm"), True),
            )
            return JsonRouteResult(result, HTTPStatus.ACCEPTED)
        if path == "/api/ultra-model/warmup":
            result = deps.engine.warmup(
                str(payload.get("session_id", "")).strip(),
                timeout=float(payload.get("timeout", 30)),
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        if path == "/api/ultra-model/verify":
            raw_evidence = payload.get("evidence", [])
            evidence = raw_evidence if isinstance(raw_evidence, list) else []
            report = deps.engine.verify_output(
                str(payload.get("answer", "")),
                evidence=evidence,
                project=project,
                require_citations=_as_bool(payload.get("require_citations"), True),
                expected_json=_as_bool(payload.get("expected_json"), False),
            )
            return JsonRouteResult({"verification": report}, HTTPStatus.CREATED)
        session_id = unquote(path.removeprefix("/api/ultra-model/sessions/").removesuffix("/stop")).strip("/")
        stopped = deps.engine.stop(session_id, reason=str(payload.get("reason", "user requested stop")))
        return JsonRouteResult({"session": stopped})
    except LaunchApprovalRequired as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.FORBIDDEN)
    except (UltraModelError, ValueError, TypeError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)
