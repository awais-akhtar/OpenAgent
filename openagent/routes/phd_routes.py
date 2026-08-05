"""PhD application workspace route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..phd_workspace import PhDWorkspace
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "phd"
ROUTES = (
    ("GET", "/api/phd/status"),
    ("GET", "/api/phd/profile"),
    ("POST", "/api/phd/profile"),
    ("GET", "/api/phd/documents"),
    ("GET", "/api/phd/documents/{id}"),
    ("POST", "/api/phd/documents"),
    ("DELETE", "/api/phd/documents/{id}"),
    ("GET", "/api/phd/applications"),
    ("GET", "/api/phd/applications/{id}"),
    ("POST", "/api/phd/applications"),
    ("PATCH", "/api/phd/applications/{id}"),
    ("DELETE", "/api/phd/applications/{id}"),
    ("POST", "/api/phd/applications/{id}/research"),
    ("GET", "/api/phd/applications/{id}/sources"),
    ("POST", "/api/phd/applications/{id}/generate/start"),
    ("POST", "/api/phd/applications/{id}/generate"),
    ("POST", "/api/phd/applications/{id}/export"),
    ("POST", "/api/phd/command"),
    ("GET", "/api/phd/generation-runs"),
    ("GET", "/api/phd/generation-runs/{id}"),
    ("GET", "/api/phd/artifacts"),
    ("GET", "/api/phd/artifacts/{id}"),
    ("PATCH", "/api/phd/artifacts/{id}"),
    ("DELETE", "/api/phd/artifacts/{id}"),
    ("POST", "/api/phd/artifacts/{id}/authenticity"),
    ("POST", "/api/phd/artifacts/{id}/export"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class PhDRouteDependencies:
    workspace: PhDWorkspace
    first: Any
    parse_bool: Any
    parse_int: Any
    query_owner: Any
    payload_owner: Any
    read_json: Any


def handle_get(path: str, query: str, deps: PhDRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/phd/"):
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    owner = deps.query_owner(params) or "local"
    try:
        if path == "/api/phd/status":
            return JsonRouteResult(deps.workspace.status(project=project, owner=owner))
        if path == "/api/phd/profile":
            return JsonRouteResult(deps.workspace.get_profile(project=project, owner=owner))
        if path == "/api/phd/documents":
            return JsonRouteResult(
                {
                    "documents": deps.workspace.list_documents(
                        project=project,
                        owner=owner,
                        category=deps.first(params, "category") or "",
                        query=deps.first(params, "q") or "",
                        limit=deps.parse_int(deps.first(params, "limit"), default=100),
                    )
                }
            )
        if path.startswith("/api/phd/documents/"):
            document_id = _single_id(path, "/api/phd/documents/")
            if document_id:
                return JsonRouteResult(
                    {
                        "document": deps.workspace.get_document(
                            document_id,
                            owner=owner,
                            include_text=deps.parse_bool(deps.first(params, "include_text"), default=False),
                        )
                    }
                )
        if path == "/api/phd/applications":
            return JsonRouteResult(
                {
                    "applications": deps.workspace.list_applications(
                        project=project,
                        owner=owner,
                        status=deps.first(params, "status") or "",
                        limit=deps.parse_int(deps.first(params, "limit"), default=100),
                    )
                }
            )
        if path == "/api/phd/generation-runs":
            return JsonRouteResult(
                {
                    "runs": deps.workspace.list_generation_runs(
                        project=project,
                        owner=owner,
                        application_id=deps.first(params, "application_id") or "",
                        limit=deps.parse_int(deps.first(params, "limit"), default=50),
                    )
                }
            )
        if path.startswith("/api/phd/generation-runs/"):
            run_id = _single_id(path, "/api/phd/generation-runs/")
            if run_id:
                return JsonRouteResult({"run": deps.workspace.get_generation_run(run_id, owner=owner)})
        if path.startswith("/api/phd/applications/") and path.endswith("/sources"):
            application_id = unquote(path.removeprefix("/api/phd/applications/").removesuffix("/sources")).strip("/")
            return JsonRouteResult({"sources": deps.workspace.list_research_sources(application_id, owner=owner)})
        if path.startswith("/api/phd/applications/"):
            application_id = _single_id(path, "/api/phd/applications/")
            if application_id:
                return JsonRouteResult({"application": deps.workspace.get_application(application_id, owner=owner)})
        if path == "/api/phd/artifacts":
            return JsonRouteResult(
                {
                    "artifacts": deps.workspace.list_artifacts(
                        project=project,
                        owner=owner,
                        application_id=deps.first(params, "application_id") or "",
                        latest_only=deps.parse_bool(deps.first(params, "latest_only"), default=False),
                        limit=deps.parse_int(deps.first(params, "limit"), default=200),
                    )
                }
            )
        if path.startswith("/api/phd/artifacts/"):
            artifact_id = _single_id(path, "/api/phd/artifacts/")
            if artifact_id:
                return JsonRouteResult({"artifact": deps.workspace.get_artifact(artifact_id, owner=owner)})
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None


def handle_post(path: str, deps: PhDRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/phd/"):
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    owner = deps.payload_owner(payload) or "local"
    try:
        if path == "/api/phd/profile":
            profile = payload.get("profile")
            if not isinstance(profile, dict):
                return JsonRouteResult({"error": "profile must be an object."}, HTTPStatus.BAD_REQUEST)
            return JsonRouteResult(deps.workspace.save_profile(profile, project=project, owner=owner))
        if path == "/api/phd/documents":
            metadata = payload.get("metadata")
            document = deps.workspace.upload_document(
                filename=str(payload.get("filename", "document.txt")),
                content=deps.workspace.decode_base64(str(payload.get("content_base64", ""))),
                category=str(payload.get("category", "other")),
                label=str(payload.get("label", "")),
                content_type=str(payload.get("content_type", "")),
                project=project,
                owner=owner,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
            return JsonRouteResult({"document": document}, HTTPStatus.CREATED)
        if path == "/api/phd/applications":
            fields = payload.get("application") if isinstance(payload.get("application"), dict) else payload
            return JsonRouteResult({"application": deps.workspace.create_application(fields, project=project, owner=owner)}, HTTPStatus.CREATED)
        if path == "/api/phd/command":
            artifact_types = payload.get("artifact_types")
            document_ids = payload.get("document_ids")
            result = deps.workspace.run_command(
                str(payload.get("application_id", "")),
                str(payload.get("command", "")),
                artifact_types=[str(item) for item in artifact_types] if isinstance(artifact_types, list) else None,
                document_ids=[str(item) for item in document_ids] if isinstance(document_ids, list) else None,
                provider=str(payload.get("provider", "auto")) or "auto",
                model=str(payload.get("model", "")).strip() or None,
                allow_external_model=deps.parse_bool(payload.get("allow_external_model"), default=False),
                research_before_generation=deps.parse_bool(payload.get("research_before_generation"), default=False),
                allow_external_research=deps.parse_bool(payload.get("allow_external_research"), default=False),
                owner=owner,
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        if path.startswith("/api/phd/applications/") and path.endswith("/research"):
            application_id = unquote(path.removeprefix("/api/phd/applications/").removesuffix("/research")).strip("/")
            urls = payload.get("urls")
            return JsonRouteResult(
                deps.workspace.research_application(
                    application_id,
                    query=str(payload.get("query", "")),
                    urls=[str(item) for item in urls] if isinstance(urls, list) else [],
                    allow_external=deps.parse_bool(payload.get("allow_external"), default=False),
                    max_sources=deps.parse_int(str(payload.get("max_sources", "6")), default=6),
                    owner=owner,
                )
            )
        if path.startswith("/api/phd/applications/") and path.endswith("/generate/start"):
            application_id = unquote(path.removeprefix("/api/phd/applications/").removesuffix("/generate/start")).strip("/")
            artifact_types = payload.get("artifact_types")
            document_ids = payload.get("document_ids")
            return JsonRouteResult(
                {
                    "run": deps.workspace.start_generation(
                        application_id,
                        str(payload.get("command", "")),
                        artifact_types=[str(item) for item in artifact_types] if isinstance(artifact_types, list) else None,
                        document_ids=[str(item) for item in document_ids] if isinstance(document_ids, list) else None,
                        provider=str(payload.get("provider", "auto")) or "auto",
                        model=str(payload.get("model", "")).strip() or None,
                        allow_external_model=deps.parse_bool(payload.get("allow_external_model"), default=False),
                        research_before_generation=deps.parse_bool(payload.get("research_before_generation"), default=False),
                        allow_external_research=deps.parse_bool(payload.get("allow_external_research"), default=False),
                        owner=owner,
                    )
                },
                HTTPStatus.ACCEPTED,
            )
        if path.startswith("/api/phd/applications/") and path.endswith("/generate"):
            application_id = unquote(path.removeprefix("/api/phd/applications/").removesuffix("/generate")).strip("/")
            artifact_types = payload.get("artifact_types")
            document_ids = payload.get("document_ids")
            return JsonRouteResult(
                deps.workspace.run_command(
                    application_id,
                    str(payload.get("command", "")),
                    artifact_types=[str(item) for item in artifact_types] if isinstance(artifact_types, list) else None,
                    document_ids=[str(item) for item in document_ids] if isinstance(document_ids, list) else None,
                    provider=str(payload.get("provider", "auto")) or "auto",
                    model=str(payload.get("model", "")).strip() or None,
                    allow_external_model=deps.parse_bool(payload.get("allow_external_model"), default=False),
                    research_before_generation=deps.parse_bool(payload.get("research_before_generation"), default=False),
                    allow_external_research=deps.parse_bool(payload.get("allow_external_research"), default=False),
                    owner=owner,
                ),
                HTTPStatus.CREATED,
            )
        if path.startswith("/api/phd/applications/") and path.endswith("/export"):
            application_id = unquote(path.removeprefix("/api/phd/applications/").removesuffix("/export")).strip("/")
            document_ids = payload.get("document_ids")
            return JsonRouteResult(
                deps.workspace.export_application(
                    application_id,
                    owner=owner,
                    include_source_documents=deps.parse_bool(payload.get("include_source_documents"), default=False),
                    document_ids=[str(item) for item in document_ids] if isinstance(document_ids, list) else None,
                )
            )
        if path.startswith("/api/phd/artifacts/") and path.endswith("/authenticity"):
            artifact_id = unquote(path.removeprefix("/api/phd/artifacts/").removesuffix("/authenticity")).strip("/")
            return JsonRouteResult(deps.workspace.review_artifact(artifact_id, owner=owner))
        if path.startswith("/api/phd/artifacts/") and path.endswith("/export"):
            artifact_id = unquote(path.removeprefix("/api/phd/artifacts/").removesuffix("/export")).strip("/")
            return JsonRouteResult(
                deps.workspace.export_artifact(
                    artifact_id,
                    export_format=str(payload.get("format", "docx")),
                    owner=owner,
                )
            )
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None


def handle_patch(path: str, deps: PhDRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/phd/"):
        return None
    payload = deps.read_json()
    owner = deps.payload_owner(payload) or "local"
    try:
        if path.startswith("/api/phd/applications/"):
            application_id = _single_id(path, "/api/phd/applications/")
            if application_id:
                fields = payload.get("application") if isinstance(payload.get("application"), dict) else payload
                return JsonRouteResult({"application": deps.workspace.update_application(application_id, fields, owner=owner)})
        if path.startswith("/api/phd/artifacts/"):
            artifact_id = _single_id(path, "/api/phd/artifacts/")
            if artifact_id:
                return JsonRouteResult(
                    {
                        "artifact": deps.workspace.revise_artifact(
                            artifact_id,
                            content=str(payload.get("content", "")),
                            title=str(payload.get("title", "")),
                            owner=owner,
                        )
                    },
                    HTTPStatus.CREATED,
                )
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None


def handle_delete(path: str, deps: PhDRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/phd/"):
        return None
    owner = deps.payload_owner({}) or "local"
    try:
        if path.startswith("/api/phd/documents/"):
            document_id = _single_id(path, "/api/phd/documents/")
            if document_id:
                return JsonRouteResult(deps.workspace.delete_document(document_id, owner=owner))
        if path.startswith("/api/phd/applications/"):
            application_id = _single_id(path, "/api/phd/applications/")
            if application_id:
                return JsonRouteResult(deps.workspace.delete_application(application_id, owner=owner))
        if path.startswith("/api/phd/artifacts/"):
            artifact_id = _single_id(path, "/api/phd/artifacts/")
            if artifact_id:
                return JsonRouteResult(deps.workspace.delete_artifact(artifact_id, owner=owner))
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return None


def _single_id(path: str, prefix: str) -> str:
    value = unquote(path.removeprefix(prefix)).strip("/")
    return "" if not value or "/" in value else value
