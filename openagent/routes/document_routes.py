"""Document lifecycle route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from ..documents import DocumentError, DocumentManager
from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "document"
ROUTES = (
    ("POST", "/api/document"),
    ("GET", "/api/documents/library"),
    ("GET", "/api/document/{doc_id}"),
    ("PUT", "/api/document/{doc_id}"),
    ("PATCH", "/api/document/{doc_id}"),
    ("POST", "/api/document/{doc_id}/archive"),
    ("DELETE", "/api/document/{doc_id}"),
    ("GET", "/api/document/{doc_id}/versions"),
    ("GET", "/api/document/{doc_id}/version/{num}"),
    ("POST", "/api/document/{doc_id}/restore/{num}"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class DocumentRouteDependencies:
    documents: DocumentManager
    read_json: Callable[[], dict[str, Any]]
    auth_scope_owner: Callable[[], str]
    document_write_owner: Callable[[dict[str, Any]], str]
    first: Callable[[dict[str, list[str]], str], str | None]
    parse_int: Callable[..., int]
    parse_bool: Callable[..., bool]


def _document_id(path: str) -> str | None:
    prefix = "/api/document/"
    return unquote(path.removeprefix(prefix)).strip() if path.startswith(prefix) else None


def handle_get(path: str, query: str, deps: DocumentRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/documents/library":
        params = parse_qs(query)
        return JsonRouteResult(deps.documents.list_library(
            project=deps.first(params, "project") or "default",
            query=deps.first(params, "q") or "",
            tag=deps.first(params, "tag") or "",
            folder=deps.first(params, "folder") or "",
            status=deps.first(params, "status") or "",
            owner=deps.auth_scope_owner(),
            limit=deps.parse_int(deps.first(params, "limit"), default=50),
        ))
    doc_id = _document_id(path)
    if doc_id is None:
        return None
    try:
        if path.endswith("/versions"):
            doc_id = _document_id(path.removesuffix("/versions"))
            return JsonRouteResult({"versions": deps.documents.list_versions(doc_id, owner=deps.auth_scope_owner())})
        if "/version/" in path:
            raw_id, _, version_text = path.removeprefix("/api/document/").partition("/version/")
            try:
                version_number = int(version_text)
            except ValueError:
                return JsonRouteResult({"error": "invalid version number"}, HTTPStatus.BAD_REQUEST)
            return JsonRouteResult({"version": deps.documents.get_version(
                unquote(raw_id).strip(), version_number, owner=deps.auth_scope_owner(),
            )})
        if doc_id:
            return JsonRouteResult({"document": deps.documents.get_document(doc_id, owner=deps.auth_scope_owner())})
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return None


def handle_post(path: str, deps: DocumentRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/document":
        payload = deps.read_json()
        try:
            result = deps.documents.create_document(
                str(payload.get("title", "")),
                str(payload.get("content", "")),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=deps.document_write_owner(payload),
                doc_format=str(payload.get("format", "markdown")).strip() or "markdown",
                tags=[str(t) for t in payload.get("tags", [])] if isinstance(payload.get("tags"), list) else None,
                folder=str(payload.get("folder", "")),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"document": result}, HTTPStatus.CREATED)
    if not path.startswith("/api/document/"):
        return None
    try:
        if path.endswith("/archive"):
            doc_id = _document_id(path.removesuffix("/archive"))
            payload = deps.read_json()
            archived = deps.parse_bool(payload.get("archived"), default=True)
            return JsonRouteResult(deps.documents.archive_document(doc_id, archived, owner=deps.auth_scope_owner()))
        if "/restore/" in path:
            raw_id, _, version_text = path.removeprefix("/api/document/").partition("/restore/")
            try:
                version_number = int(version_text)
            except ValueError:
                return JsonRouteResult({"error": "invalid version number"}, HTTPStatus.BAD_REQUEST)
            return JsonRouteResult(deps.documents.restore_version(
                unquote(raw_id).strip(), version_number, owner=deps.auth_scope_owner(),
            ))
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return None


def handle_put(path: str, deps: DocumentRouteDependencies) -> JsonRouteResult | None:
    doc_id = _document_id(path)
    if doc_id is None:
        return None
    payload = deps.read_json()
    try:
        result = deps.documents.update_document(
            doc_id,
            title=payload["title"] if "title" in payload else None,
            content=payload["content"] if "content" in payload else None,
            if_version=int(payload["if_version"]) if "if_version" in payload else None,
            owner=deps.auth_scope_owner(),
        )
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except DocumentError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.CONFLICT)
    except (ValueError, TypeError):
        return JsonRouteResult({"error": "invalid version number"}, HTTPStatus.BAD_REQUEST)
    return JsonRouteResult({"document": result})


def handle_patch(path: str, deps: DocumentRouteDependencies) -> JsonRouteResult | None:
    doc_id = _document_id(path)
    if doc_id is None:
        return None
    payload = deps.read_json()
    try:
        result = deps.documents.patch_document(
            doc_id,
            tags=[str(t) for t in payload["tags"]] if isinstance(payload.get("tags"), list) else None,
            folder=payload["folder"] if "folder" in payload else None,
            title=payload["title"] if "title" in payload else None,
            owner=deps.auth_scope_owner(),
        )
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult({"document": result})


def handle_delete(path: str, deps: DocumentRouteDependencies) -> JsonRouteResult | None:
    doc_id = _document_id(path)
    if doc_id is None:
        return None
    try:
        return JsonRouteResult(deps.documents.delete_document(doc_id, owner=deps.auth_scope_owner()))
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
