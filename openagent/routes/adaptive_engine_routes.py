"""Adaptive heterogeneous compiler and runtime HTTP contracts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs

from ..adaptive_engine.runtime import AdaptiveExecutionEngine, AdaptiveRuntimeError
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "adaptive_engine"
ROUTES = (
    ("GET", "/api/adaptive-engine/status"),
    ("GET", "/api/adaptive-engine/hardware"),
    ("GET", "/api/adaptive-engine/plans"),
    ("GET", "/api/adaptive-engine/benchmarks"),
    ("GET", "/api/adaptive-engine/tuning"),
    ("GET", "/api/adaptive-engine/direct/status"),
    ("GET", "/api/adaptive-engine/direct/calibrations"),
    ("GET", "/api/adaptive-engine/role-options"),
    ("GET", "/api/adaptive-engine/roles"),
    ("GET", "/api/adaptive-engine/orchestrations"),
    ("POST", "/api/adaptive-engine/inspect"),
    ("POST", "/api/adaptive-engine/compile"),
    ("POST", "/api/adaptive-engine/benchmark"),
    ("POST", "/api/adaptive-engine/autotune"),
    ("POST", "/api/adaptive-engine/kv/session"),
    ("POST", "/api/adaptive-engine/kv/append"),
    ("POST", "/api/adaptive-engine/kv/release"),
    ("POST", "/api/adaptive-engine/scheduler/queue"),
    ("POST", "/api/adaptive-engine/scheduler/next"),
    ("POST", "/api/adaptive-engine/direct/load"),
    ("POST", "/api/adaptive-engine/direct/load-llm"),
    ("POST", "/api/adaptive-engine/direct/infer"),
    ("POST", "/api/adaptive-engine/direct/generate"),
    ("POST", "/api/adaptive-engine/direct/benchmark"),
    ("POST", "/api/adaptive-engine/direct/calibrate"),
    ("POST", "/api/adaptive-engine/direct/unload"),
    ("POST", "/api/adaptive-engine/direct/stop"),
    ("POST", "/api/adaptive-engine/roles"),
    ("POST", "/api/adaptive-engine/orchestrate"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class AdaptiveEngineRouteDependencies:
    engine: AdaptiveExecutionEngine
    read_json: Any


def handle_get(path: str, query: str, deps: AdaptiveEngineRouteDependencies) -> JsonRouteResult | None:
    if path not in {
        "/api/adaptive-engine/status",
        "/api/adaptive-engine/hardware",
        "/api/adaptive-engine/plans",
        "/api/adaptive-engine/benchmarks",
        "/api/adaptive-engine/tuning",
        "/api/adaptive-engine/direct/status",
        "/api/adaptive-engine/direct/calibrations",
        "/api/adaptive-engine/role-options",
        "/api/adaptive-engine/roles",
        "/api/adaptive-engine/orchestrations",
    }:
        return None
    params = parse_qs(query)
    project = _first(params, "project") or "default"
    limit = _as_int(_first(params, "limit"), 30)
    refresh = _as_bool(_first(params, "refresh"), False)
    try:
        if path == "/api/adaptive-engine/status":
            return JsonRouteResult(deps.engine.status(project=project, refresh_hardware=refresh))
        if path == "/api/adaptive-engine/hardware":
            return JsonRouteResult(deps.engine.hardware(refresh=refresh))
        if path == "/api/adaptive-engine/plans":
            return JsonRouteResult({"plans": deps.engine.plans(project=project, limit=limit)})
        if path == "/api/adaptive-engine/benchmarks":
            return JsonRouteResult({"benchmarks": deps.engine.benchmarks(project=project, limit=limit)})
        if path == "/api/adaptive-engine/direct/status":
            return JsonRouteResult(deps.engine.direct_status(start=_as_bool(_first(params, "start"), False)))
        if path == "/api/adaptive-engine/direct/calibrations":
            return JsonRouteResult({"calibrations": deps.engine.direct_calibrations(project=project, limit=limit)})
        if path == "/api/adaptive-engine/role-options":
            return JsonRouteResult(deps.engine.model_role_options())
        if path == "/api/adaptive-engine/roles":
            return JsonRouteResult({"roles": deps.engine.model_roles(project=project)})
        if path == "/api/adaptive-engine/orchestrations":
            return JsonRouteResult({"runs": deps.engine.orchestration_runs(project=project, limit=limit)})
        return JsonRouteResult({"tuning": deps.engine.autotuner.results(project=project, limit=limit)})
    except AdaptiveRuntimeError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def handle_post(path: str, deps: AdaptiveEngineRouteDependencies) -> JsonRouteResult | None:
    if path not in {item[1] for item in ROUTES if item[0] == "POST"}:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/adaptive-engine/inspect":
            result = deps.engine.inspect_model(
                str(payload.get("path", "")),
                project=project,
                parameter_count_hint=_as_int(payload.get("parameter_count_hint"), 0),
                quantization_bits_hint=_as_float(payload.get("quantization_bits_hint"), 0.0),
                include_tensors=_as_bool(payload.get("include_tensors"), False),
            )
        elif path == "/api/adaptive-engine/compile":
            memory_budget = payload.get("memory_budget_gb")
            result = deps.engine.compile_model(
                str(payload.get("path", "")),
                project=project,
                prompt_tokens=_as_int(payload.get("prompt_tokens"), 512),
                context_tokens=_as_int(payload.get("context_tokens"), 4096),
                memory_budget_gb=None if memory_budget in {None, ""} else float(memory_budget),
                quality_loss_limit=_as_float(payload.get("quality_loss_limit"), 0.08),
                power_mode=str(payload.get("power_mode", "balanced")),
                max_device_transitions=_as_int(payload.get("max_device_transitions"), 4),
                backend_compile=_as_bool(payload.get("backend_compile"), True),
                parameter_count_hint=_as_int(payload.get("parameter_count_hint"), 0),
                quantization_bits_hint=_as_float(payload.get("quantization_bits_hint"), 0.0),
            )
        elif path == "/api/adaptive-engine/benchmark":
            result = deps.engine.benchmark_device(
                str(payload.get("device_id", "")),
                project=project,
                iterations=_as_int(payload.get("iterations"), 20),
                dimension=_as_int(payload.get("dimension"), 256),
            )
        elif path == "/api/adaptive-engine/autotune":
            result = deps.engine.autotune_device(
                str(payload.get("device_id", "")),
                project=project,
                phase=str(payload.get("phase", "decode")),
                iterations=_as_int(payload.get("iterations"), 12),
            )
        elif path == "/api/adaptive-engine/direct/load":
            result = deps.engine.load_direct_model(
                str(payload.get("path", "")),
                device_id=str(payload.get("device_id", "")),
                project=project,
                model_id=str(payload.get("model_id", "")),
                config=_dict(payload.get("config")),
            )
        elif path == "/api/adaptive-engine/direct/load-llm":
            result = deps.engine.load_direct_llm(
                str(payload.get("path", "")),
                device_id=str(payload.get("device_id", "")),
                project=project,
                model_id=str(payload.get("model_id", "openagent-direct-chat")),
                config=_dict(payload.get("config")),
                scheduler=_dict(payload.get("scheduler")),
            )
        elif path == "/api/adaptive-engine/direct/infer":
            inputs = payload.get("inputs")
            result = deps.engine.infer_direct(
                str(payload.get("model_id", "")),
                inputs=inputs if isinstance(inputs, (dict, str)) else None,
                output_mode=str(payload.get("output_mode", "summary")),
                project=project,
            )
        elif path == "/api/adaptive-engine/direct/generate":
            result = deps.engine.generate_direct(
                str(payload.get("model_id", "openagent-direct-chat")),
                messages=_messages(payload.get("messages")),
                prompt=str(payload.get("prompt", "")),
                generation=_dict(payload.get("generation")),
                project=project,
            )
        elif path == "/api/adaptive-engine/direct/benchmark":
            inputs = payload.get("inputs")
            result = deps.engine.benchmark_direct_model(
                str(payload.get("model_id", "")),
                project=project,
                iterations=_as_int(payload.get("iterations"), 20),
                warmup=_as_int(payload.get("warmup"), 2),
                inputs=inputs if isinstance(inputs, (dict, str)) else None,
            )
        elif path == "/api/adaptive-engine/direct/calibrate":
            requested_devices = payload.get("devices")
            result = deps.engine.calibrate_direct_model(
                str(payload.get("path", "")),
                project=project,
                devices=[str(item) for item in requested_devices] if isinstance(requested_devices, list) else None,
                iterations=_as_int(payload.get("iterations"), 10),
                absolute_tolerance=_as_float(payload.get("absolute_tolerance"), 0.03),
                keep_resident=_as_bool(payload.get("keep_resident"), True),
                model_id=str(payload.get("model_id", "")),
            )
        elif path == "/api/adaptive-engine/direct/unload":
            result = deps.engine.unload_direct_model(str(payload.get("model_id", "")), project=project)
        elif path == "/api/adaptive-engine/direct/stop":
            deps.engine.close()
            result = {"status": "stopped"}
        elif path == "/api/adaptive-engine/roles":
            result = deps.engine.upsert_model_role(
                str(payload.get("role", "")),
                project=project,
                provider_id=str(payload.get("provider_id", "auto")),
                model_id=str(payload.get("model_id", "")),
                model_path=str(payload.get("model_path", "")),
                device_id=str(payload.get("device_id", "auto")),
                system_prompt=str(payload.get("system_prompt", "")),
                enabled=_as_bool(payload.get("enabled"), True),
                config=_dict(payload.get("config")),
            )
        elif path == "/api/adaptive-engine/orchestrate":
            requested_roles = payload.get("roles")
            result = deps.engine.orchestrate_models(
                str(payload.get("task", "")),
                project=project,
                mode=str(payload.get("mode", "supervised-code")),
                roles=[str(item) for item in requested_roles] if isinstance(requested_roles, list) else None,
                max_rounds=_as_int(payload.get("max_rounds"), 2),
                allow_external=_as_bool(payload.get("allow_external"), False),
            )
        elif path == "/api/adaptive-engine/kv/session":
            result = deps.engine.create_kv_session(
                str(payload.get("session_id", "")),
                prefix_tokens=_int_list(payload.get("prefix_tokens")),
                priority=_as_int(payload.get("priority"), 50),
            )
        elif path == "/api/adaptive-engine/kv/append":
            result = deps.engine.append_kv(str(payload.get("session_id", "")), _int_list(payload.get("token_ids")))
        elif path == "/api/adaptive-engine/kv/release":
            result = {"released": deps.engine.release_kv(str(payload.get("session_id", "")))}
        elif path == "/api/adaptive-engine/scheduler/queue":
            result = deps.engine.queue_request(
                request_id=str(payload.get("request_id", "")),
                model_id=str(payload.get("model_id", "")),
                phase=str(payload.get("phase", "decode")),
                remaining_tokens=_as_int(payload.get("remaining_tokens"), 1),
                priority=_as_int(payload.get("priority"), 80),
                deadline_seconds=_as_float(payload.get("deadline_seconds"), 0.0),
                session_id=str(payload.get("session_id", "")),
            )
        else:
            result = {"batch": deps.engine.next_batch(power_mode=str(payload.get("power_mode", "balanced")))}
    except (AdaptiveRuntimeError, KeyError, MemoryError, RuntimeError, TypeError, ValueError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return JsonRouteResult(result, HTTPStatus.CREATED)


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_list(value: object) -> list[int]:
    return [int(item) for item in value] if isinstance(value, list) else []


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _messages(value: object) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    result = []
    for item in value:
        if isinstance(item, dict):
            result.append({"role": str(item.get("role", "user")), "content": str(item.get("content", ""))})
    return result or None
