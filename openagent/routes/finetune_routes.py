"""Fine-tuning and adapter route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "finetune"
ROUTES = (
    ("GET", "/api/finetune/status"),
    ("GET", "/api/finetune/datasets"),
    ("POST", "/api/finetune/datasets"),
    ("POST", "/api/finetune/examples"),
    ("GET", "/api/finetune/jobs"),
    ("POST", "/api/finetune/jobs"),
    ("GET", "/api/finetune/adapters"),
    ("POST", "/api/finetune/adapters"),
    ("POST", "/api/finetune/adapters/activate"),
    ("GET", "/api/finetune/templates"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class FineTuneRouteDependencies:
    fine_tuning: Any
    read_json: Callable[[], dict[str, Any]]


def _project(value: Any) -> str:
    return str(value).strip() or "default"


def handle_get(path: str, query: str, deps: FineTuneRouteDependencies) -> JsonRouteResult | None:
    if path not in {route for method, route in ROUTES if method == "GET"}:
        return None
    params = parse_qs(query)
    project = params.get("project", [None])[0] or "default"
    if path == "/api/finetune/status":
        return JsonRouteResult(deps.fine_tuning.status(project=project))
    if path == "/api/finetune/datasets":
        return JsonRouteResult({"datasets": deps.fine_tuning.list_datasets(project=project)})
    if path == "/api/finetune/jobs":
        return JsonRouteResult({"jobs": deps.fine_tuning.list_jobs(project=project)})
    if path == "/api/finetune/adapters":
        base_model = params.get("base_model", [None])[0] or ""
        return JsonRouteResult({"adapters": deps.fine_tuning.list_adapters(project=project, base_model=base_model)})
    return JsonRouteResult({"templates": deps.fine_tuning.list_templates()})


def handle_post(path: str, deps: FineTuneRouteDependencies) -> JsonRouteResult | None:
    if path not in {route for method, route in ROUTES if method == "POST"}:
        return None
    payload = deps.read_json()
    project = _project(payload.get("project", "default"))
    if path == "/api/finetune/datasets":
        try:
            dataset = deps.fine_tuning.create_dataset(
                str(payload.get("name", "")).strip(),
                project=project,
                description=str(payload.get("description", "")),
                format=str(payload.get("format", "jsonl")),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"dataset": dataset}, HTTPStatus.CREATED)
    if path == "/api/finetune/examples":
        dataset_id = str(payload.get("dataset_id", "")).strip()
        if not dataset_id:
            return JsonRouteResult({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
        try:
            example = deps.fine_tuning.add_example(
                dataset_id,
                str(payload.get("user_input", "")).strip(),
                str(payload.get("assistant_output", "")).strip(),
                project=project,
                system_prompt=str(payload.get("system_prompt", "")),
                quality_score=float(payload.get("quality_score", 1.0)),
            )
        except (ValueError, KeyError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"example": example}, HTTPStatus.CREATED)
    if path == "/api/finetune/jobs":
        try:
            job = deps.fine_tuning.create_job(
                str(payload.get("dataset_id", "")).strip(),
                str(payload.get("base_model", "")).strip(),
                project=project,
                method=str(payload.get("method", "lora")),
                config=payload.get("config") if isinstance(payload.get("config"), dict) else None,
                template_id=str(payload.get("template_id", "")),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"job": job}, HTTPStatus.CREATED)
    if path == "/api/finetune/adapters":
        try:
            adapter = deps.fine_tuning.register_adapter(
                str(payload.get("name", "")).strip(),
                str(payload.get("base_model", "")).strip(),
                project=project,
                description=str(payload.get("description", "")),
                rank=int(payload.get("rank", 16)),
                alpha=float(payload.get("alpha", 16.0)),
                target_modules=[str(module) for module in payload.get("target_modules", [])]
                if isinstance(payload.get("target_modules"), list) else None,
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"adapter": adapter}, HTTPStatus.CREATED)
    adapter_id = str(payload.get("adapter_id", "")).strip()
    if not adapter_id:
        return JsonRouteResult({"error": "adapter_id is required"}, HTTPStatus.BAD_REQUEST)
    try:
        adapter = deps.fine_tuning.activate_adapter(adapter_id, project=project)
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult({"adapter": adapter})
