"""Adaptive CPU inference engine route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from ..model_gateway import ModelGateway, ProviderConfigurationError, ProviderUnavailableError
from ..models import AuditEvent
from ..storage import OpenAgentStore
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "inference"
ROUTES = (
    ("GET", "/api/inference/status"),
    ("POST", "/api/inference/benchmark"),
    ("POST", "/api/inference/cache/clear"),
    ("POST", "/api/inference/warmup"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class InferenceRouteDependencies:
    gateway: ModelGateway
    store: OpenAgentStore
    read_json: Any


def handle_get(path: str, query: str, deps: InferenceRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/inference/status":
        return JsonRouteResult(deps.gateway.inference_status())
    return None


def handle_post(path: str, deps: InferenceRouteDependencies) -> JsonRouteResult | None:
    if path not in {
        "/api/inference/benchmark",
        "/api/inference/cache/clear",
        "/api/inference/warmup",
    }:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/inference/benchmark":
            result = deps.gateway.benchmark_inference_engine(
                iterations=_as_int(payload.get("iterations"), 100)
            )
            action = "inference.benchmark"
        elif path == "/api/inference/cache/clear":
            result = deps.gateway.clear_inference_cache()
            action = "inference.cache.clear"
        else:
            result = deps.gateway.warmup(
                provider_id=str(payload.get("provider", "auto")).strip() or "auto",
                model=str(payload.get("model", "")).strip() or None,
            )
            action = "inference.warmup"
    except (ValueError, ProviderConfigurationError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    except ProviderUnavailableError as exc:
        return JsonRouteResult(
            {
                "error": str(exc),
                "status": "unavailable",
                "privacy_boundary": "The failed warmup did not cross the local data boundary.",
            },
            HTTPStatus.SERVICE_UNAVAILABLE,
        )
    deps.store.add_audit_event(
        AuditEvent(
            actor="inference-engine",
            action=action,
            resource="inference:local-runtime",
            project=project,
            metadata={"provider": result.get("provider", "local"), "status": result.get("status", "ok")},
        )
    )
    return JsonRouteResult(result, HTTPStatus.CREATED)


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
