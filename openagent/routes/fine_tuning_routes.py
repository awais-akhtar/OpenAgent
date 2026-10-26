"""Fine-tuning API routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable

from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "fine_tuning"
ROUTES = (
    ("GET", "/api/fine-tuning/models"),
    ("POST", "/api/fine-tuning/jobs"),
    ("GET", "/api/fine-tuning/jobs"),
    ("GET", "/api/fine-tuning/jobs/{id}"),
    ("POST", "/api/fine-tuning/jobs/{id}/start"),
    ("POST", "/api/fine-tuning/jobs/{id}/finalize"),
    ("POST", "/api/fine-tuning/jobs/{id}/validate"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class FineTuningRouteDependencies:
    fine_tuning: Any
    read_json: Callable[[], dict[str, Any]]


def handle_get(path: str, deps: FineTuningRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/fine-tuning/models":
        models = deps.fine_tuning.get_available_models()
        return JsonRouteResult(models)

    if path == "/api/fine-tuning/jobs":
        jobs = deps.fine_tuning.list_jobs()
        return JsonRouteResult({"jobs": jobs})

    if path.startswith("/api/fine-tuning/jobs/"):
        job_id = path.removeprefix("/api/fine-tuning/jobs/").strip("/")
        if "/" in job_id or not job_id:
            return JsonRouteResult({"error": "job_id required"}, HTTPStatus.BAD_REQUEST)
        try:
            job = deps.fine_tuning.get_job_status(job_id)
            return JsonRouteResult({"job": job})
        except ValueError:
            return JsonRouteResult({"error": "Job not found"}, HTTPStatus.NOT_FOUND)

    return None


def handle_post(path: str, deps: FineTuningRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/fine-tuning/jobs":
        payload = deps.read_json()
        try:
            job_id = str(payload.get("job_id", "")).strip()
            if not job_id:
                return JsonRouteResult(
                    {"error": "job_id is required"}, HTTPStatus.BAD_REQUEST
                )

            job = deps.fine_tuning.create_finetuning_job(
                job_id=job_id,
                model=str(payload.get("model", "prajjwal1/bert-tiny")).strip(),
                training_data_path=str(payload.get("training_data_path", "")).strip(),
                learning_rate=float(payload.get("learning_rate", 2e-5)),
                num_epochs=int(payload.get("num_epochs", 3)),
                dtype=str(payload.get("dtype", "float32")).strip() or "float32",
            )
            return JsonRouteResult({"job": job}, HTTPStatus.CREATED)
        except (ValueError, TypeError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    if path.startswith("/api/fine-tuning/jobs/"):
        job_id = path.removeprefix("/api/fine-tuning/jobs/").removesuffix("/start").removesuffix("/finalize").removesuffix("/validate").strip("/")

        if path.endswith("/start"):
            try:
                result = deps.fine_tuning.start_training(job_id)
                return JsonRouteResult(result)
            except ValueError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)

        if path.endswith("/finalize"):
            try:
                result = deps.fine_tuning.finalize_checkpoint(job_id)
                return JsonRouteResult(result)
            except ValueError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)

        if path.endswith("/validate"):
            try:
                result = deps.fine_tuning.validate_finetuned_model(job_id)
                return JsonRouteResult(result)
            except ValueError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)

    return None
