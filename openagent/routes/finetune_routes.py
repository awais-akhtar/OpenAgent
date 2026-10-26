"""Fine-tuning and adapter route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

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
    ("POST", "/api/finetune/jobs/{job_id}/cancel"),
    ("POST", "/api/finetune/jobs/{job_id}/train"),
    ("POST", "/api/finetune/jobs/{job_id}/run-cpu"),
    ("GET", "/api/finetune/jobs/{job_id}/stream-status"),
    ("GET", "/api/finetune/checkpoints"),
    ("GET", "/api/finetune/checkpoints/{checkpoint_id}"),
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
    checkpoint_detail_prefix = "/api/finetune/checkpoints/"
    is_checkpoint_detail = path.startswith(checkpoint_detail_prefix) and path != "/api/finetune/checkpoints/"
    _stream_status_prefix = "/api/finetune/jobs/"
    is_stream_status = (
        path.startswith(_stream_status_prefix)
        and path.endswith("/stream-status")
        and path != "/api/finetune/jobs//stream-status"
    )
    if path not in {route for method, route in ROUTES if method == "GET"} and not is_checkpoint_detail and not is_stream_status:
        return None
    params = parse_qs(query)
    project = params.get("project", [None])[0] or "default"
    if is_checkpoint_detail:
        checkpoint_id = unquote(path.removeprefix(checkpoint_detail_prefix).strip("/"))
        if not checkpoint_id or "/" in checkpoint_id:
            return JsonRouteResult({"error": "checkpoint_id is required."}, HTTPStatus.BAD_REQUEST)
        try:
            return JsonRouteResult({"checkpoint": deps.fine_tuning.get_checkpoint(checkpoint_id, project=project)})
        except KeyError:
            return JsonRouteResult({"error": "Checkpoint not found."}, HTTPStatus.NOT_FOUND)
    if is_stream_status:
        job_id = path.removeprefix(_stream_status_prefix).removesuffix("/stream-status").strip("/")
        if not job_id or "/" in job_id:
            return JsonRouteResult({"error": "job_id is required"}, HTTPStatus.BAD_REQUEST)
        try:
            job = deps.fine_tuning.get_job(job_id, project=project)
        except KeyError:
            return JsonRouteResult({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
        checkpoints = deps.fine_tuning.list_checkpoints(project=project, job_id=job_id, limit=100)
        return JsonRouteResult({
            "job_id": job_id,
            "status": job.get("status"),
            "progress": job.get("progress", 0.0),
            "current_loss": job.get("current_loss", 0.0),
            "checkpoints": checkpoints,
        })
    if path == "/api/finetune/status":
        return JsonRouteResult(deps.fine_tuning.status(project=project))
    if path == "/api/finetune/datasets":
        return JsonRouteResult({"datasets": deps.fine_tuning.list_datasets(project=project)})
    if path == "/api/finetune/jobs":
        return JsonRouteResult({"jobs": deps.fine_tuning.list_jobs(project=project)})
    if path == "/api/finetune/checkpoints":
        job_id = params.get("job_id", [None])[0] or ""
        limit = int(params.get("limit", ["50"])[0] or 50)
        return JsonRouteResult({"checkpoints": deps.fine_tuning.list_checkpoints(project=project, job_id=job_id, limit=limit)})
    if path == "/api/finetune/adapters":
        base_model = params.get("base_model", [None])[0] or ""
        return JsonRouteResult({"adapters": deps.fine_tuning.list_adapters(project=project, base_model=base_model)})
    return JsonRouteResult({"templates": deps.fine_tuning.list_templates()})


def handle_post(path: str, deps: FineTuneRouteDependencies) -> JsonRouteResult | None:
    is_cancel = path.startswith("/api/finetune/jobs/") and path.endswith("/cancel")
    is_train = path.startswith("/api/finetune/jobs/") and path.endswith("/train")
    is_run_cpu = path.startswith("/api/finetune/jobs/") and path.endswith("/run-cpu")
    if path not in {route for method, route in ROUTES if method == "POST"} and not is_cancel and not is_train and not is_run_cpu:
        return None
    payload = deps.read_json()
    project = _project(payload.get("project", "default"))
    if is_run_cpu:
        job_id = path.removeprefix("/api/finetune/jobs/").removesuffix("/run-cpu").strip("/")
        if not job_id or "/" in job_id:
            return JsonRouteResult({"error": "job_id is required"}, HTTPStatus.BAD_REQUEST)
        model_dir = str(payload.get("model_dir", "")).strip()
        if not model_dir:
            return JsonRouteResult({"error": "model_dir is required"}, HTTPStatus.BAD_REQUEST)
        output_dir = str(payload.get("output_dir", "")).strip()
        try:
            result = deps.fine_tuning.run_job_cpu_streaming(
                job_id, model_dir, project=project, output_dir=output_dir
            )
        except KeyError:
            return JsonRouteResult({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
        except RuntimeError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.CONFLICT)
        return JsonRouteResult(result)
    if is_train:
        job_id = path.removeprefix("/api/finetune/jobs/").removesuffix("/train").strip("/")
        if not job_id or "/" in job_id:
            return JsonRouteResult({"error": "job_id is required"}, HTTPStatus.BAD_REQUEST)
        try:
            result = deps.fine_tuning.start_model_training(
                job_id,
                project=project,
                output_dir=str(payload.get("output_dir", "")).strip() or None,
                checkpoint_interval=int(payload.get("checkpoint_interval", 10)),
            )
        except KeyError:
            return JsonRouteResult({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
        except RuntimeError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.CONFLICT)
        return JsonRouteResult({"training": result}, HTTPStatus.ACCEPTED)
    if is_cancel:
        job_id = path.removeprefix("/api/finetune/jobs/").removesuffix("/cancel").strip("/")
        if not job_id or "/" in job_id:
            return JsonRouteResult({"error": "job_id is required"}, HTTPStatus.BAD_REQUEST)
        try:
            job = deps.fine_tuning.cancel_queued_job(job_id, project=project)
        except KeyError:
            return JsonRouteResult({"error": "Job not found"}, HTTPStatus.NOT_FOUND)
        except RuntimeError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.CONFLICT)
        status = HTTPStatus.ACCEPTED if job.get("status") == "cancelling" else HTTPStatus.OK
        return JsonRouteResult({"job": job}, status)
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
        except (ValueError, KeyError) as exc:
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
