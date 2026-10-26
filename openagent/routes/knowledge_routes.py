"""Knowledge and RAG route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from ..knowledge import KnowledgeEngine
from ..media import MediaManager
from ..model_gateway import ModelGateway, ProviderConfigurationError
from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "knowledge"
ROUTES = (
    ("GET", "/api/knowledge/status"),
    ("GET", "/api/knowledge/documents"),
    ("GET", "/api/knowledge/documents/{document_id}"),
    ("GET", "/api/knowledge/search"),
    ("GET", "/api/knowledge/graph"),
    ("GET", "/api/knowledge/tags"),
    ("GET", "/api/knowledge/timeline"),
    ("GET", "/api/knowledge/collections"),
    ("POST", "/api/knowledge/ingest"),
    ("POST", "/api/knowledge/upload"),
    ("POST", "/api/knowledge/collections/{collection_name}/rename"),
    ("PUT", "/api/knowledge/documents/{document_id}"),
    ("DELETE", "/api/knowledge/documents/{document_id}"),
    ("DELETE", "/api/knowledge/collections/{collection_name}"),
    ("POST", "/api/knowledge/semantic-search"),
    ("POST", "/api/knowledge/hybrid-search"),
    ("POST", "/api/knowledge/synthesize"),
    ("POST", "/api/knowledge/compress"),
    ("POST", "/api/knowledge/replay"),
    ("POST", "/api/embeddings"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class KnowledgeRouteDependencies:
    knowledge: KnowledgeEngine
    gateway: ModelGateway
    media: MediaManager
    read_json: Any
    headers: Any
    read_body: Any
    upload_limit_bytes: int
    auth_scope_owner: Callable[[], str] = lambda: ""


def _first(params: dict[str, list[str]], key: str) -> str | None:
    value = params.get(key, [None])[0]
    return value or None


def _parse_int(value: str | None, *, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _parse_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _knowledge_document_id(path: str) -> str | None:
    prefix = "/api/knowledge/documents/"
    if not path.startswith(prefix):
        return None
    document_id = unquote(path.removeprefix(prefix)).strip()
    return document_id if document_id and "/" not in document_id else None


def handle_get(path: str, query: str, deps: KnowledgeRouteDependencies) -> JsonRouteResult | None:
    if path not in {item for method, item in ROUTES if method == "GET"} and _knowledge_document_id(path) is None:
        return None
    params = parse_qs(query)
    project = _first(params, "project") or "default"
    if path == "/api/knowledge/status":
        return JsonRouteResult(deps.knowledge.status(project=project))
    if path == "/api/knowledge/documents":
        limit = _parse_int(_first(params, "limit"), default=100)
        owner = deps.auth_scope_owner()
        return JsonRouteResult({"documents": deps.knowledge.list_documents(project=project, limit=limit, owner=owner)})
    document_id = _knowledge_document_id(path)
    if document_id is not None:
        owner = deps.auth_scope_owner()
        try:
            return JsonRouteResult({"document": deps.knowledge.get_document(document_id, project=project, owner=owner)})
        except KeyError:
            return JsonRouteResult({"error": "Knowledge document not found."}, HTTPStatus.NOT_FOUND)
    if path == "/api/knowledge/search":
        search_query = _first(params, "q") or ""
        mode = _first(params, "mode") or "fts"
        owner = deps.auth_scope_owner()
        results = (
            deps.knowledge.semantic_search(search_query, project=project, owner=owner)
            if mode == "semantic"
            else deps.knowledge.search(search_query, project=project, owner=owner)
        )
        return JsonRouteResult({"results": results})
    if path == "/api/knowledge/graph":
        return JsonRouteResult(deps.knowledge.graph(project=project))
    if path == "/api/knowledge/collections":
        return JsonRouteResult({"collections": deps.knowledge.list_collections(project=project, owner=deps.auth_scope_owner())})
    limit = _parse_int(_first(params, "limit"), default=100)
    if path == "/api/knowledge/tags":
        return JsonRouteResult({"tags": deps.knowledge.tags(project=project, limit=limit)})
    return JsonRouteResult({"timeline": deps.knowledge.timeline(project=project, limit=limit)})


def _is_collection_rename(path: str) -> bool:
    return (
        path.startswith("/api/knowledge/collections/")
        and path.endswith("/rename")
        and path != "/api/knowledge/collections//rename"
    )


def handle_post(path: str, deps: KnowledgeRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/knowledge/upload":
        return _handle_upload(deps)
    if path not in {item for method, item in ROUTES if method == "POST"} and not _is_collection_rename(path):
        return None
    payload = deps.read_json()
    if path == "/api/embeddings":
        raw_texts = payload.get("texts", [])
        if isinstance(raw_texts, str):
            texts = [raw_texts]
        elif isinstance(raw_texts, list):
            texts = [str(item) for item in raw_texts]
        else:
            texts = []
        provider = str(payload.get("provider", "local")).strip() or "local"
        model = str(payload.get("model", "")).strip() or None
        allow_external = _parse_bool(payload.get("allow_external"), default=False)
        try:
            result = deps.gateway.embed(texts, provider_id=provider, model=model, allow_external=allow_external)
        except (ProviderConfigurationError, ValueError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({
            "provider": result.provider, "model": result.model, "dimensions": result.dimensions,
            "embeddings": result.embeddings, "metadata": result.metadata,
        }, HTTPStatus.CREATED)
    project = str(payload.get("project", "default")).strip() or "default"
    if path == "/api/knowledge/ingest":
        try:
            document = deps.knowledge.ingest_text(
                str(payload.get("title", "")).strip(), str(payload.get("content", "")).strip(),
                project=project, source=str(payload.get("source", "manual")).strip() or "manual",
                source_path=str(payload.get("source_path", "")).strip(),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                embedding_provider=str(payload.get("embedding_provider", "")).strip(),
                owner=deps.auth_scope_owner(),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"document": document}, HTTPStatus.CREATED)
    if path in {"/api/knowledge/semantic-search", "/api/knowledge/hybrid-search"}:
        method = deps.knowledge.semantic_search if path.endswith("/semantic-search") else deps.knowledge.hybrid_search
        results = method(
            str(payload.get("query", "")).strip(), project=project,
            limit=_parse_int(str(payload.get("limit", "10")), default=10),
            embedding_provider=str(payload.get("embedding_provider", "")).strip(),
            owner=deps.auth_scope_owner(),
        )
        return JsonRouteResult({"results": results})
    if path == "/api/knowledge/synthesize":
        result = deps.knowledge.synthesize(
            str(payload.get("query", "")).strip(), project=project,
            limit=_parse_int(str(payload.get("limit", "8")), default=8),
            embedding_provider=str(payload.get("embedding_provider", "")).strip(),
        )
        return JsonRouteResult(result, HTTPStatus.CREATED)
    if path == "/api/knowledge/compress":
        return JsonRouteResult(deps.knowledge.compress_memory(project=project), HTTPStatus.CREATED)
    # collection rename
    rename_prefix = "/api/knowledge/collections/"
    rename_suffix = "/rename"
    if path.startswith(rename_prefix) and path.endswith(rename_suffix):
        cname = unquote(path.removeprefix(rename_prefix).removesuffix(rename_suffix)).strip()
        if not cname or "/" in cname:
            return JsonRouteResult({"error": "collection name is required."}, HTTPStatus.BAD_REQUEST)
        project = str(payload.get("project", "default")).strip() or "default"
        new_name = str(payload.get("new_name", "")).strip()
        try:
            return JsonRouteResult(deps.knowledge.update_collection(cname, project=project, owner=deps.auth_scope_owner(), new_name=new_name))
        except KeyError:
            return JsonRouteResult({"error": "Collection not found."}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    replay = deps.knowledge.create_replay(
        project=project, label=str(payload.get("label", "Replay")).strip() or "Replay",
        query=str(payload.get("query", "")).strip(),
    )
    return JsonRouteResult({"replay": replay}, HTTPStatus.CREATED)


def handle_delete(path: str, query: str, deps: KnowledgeRouteDependencies) -> JsonRouteResult | None:
    # collection delete
    coll_prefix = "/api/knowledge/collections/"
    if path.startswith(coll_prefix):
        cname = unquote(path.removeprefix(coll_prefix)).strip().strip("/")
        if not cname or "/" in cname:
            return JsonRouteResult({"error": "collection name is required."}, HTTPStatus.BAD_REQUEST)
        project = _first(parse_qs(query), "project") or "default"
        try:
            return JsonRouteResult(deps.knowledge.delete_collection(cname, project=project, owner=deps.auth_scope_owner()))
        except KeyError:
            return JsonRouteResult({"error": "Collection not found."}, HTTPStatus.NOT_FOUND)
    document_id = _knowledge_document_id(path)
    if document_id is None:
        return None
    project = _first(parse_qs(query), "project") or "default"
    result = deps.knowledge.delete_document(document_id, project=project, owner=deps.auth_scope_owner())
    if not result.get("deleted"):
        return JsonRouteResult({"error": "Knowledge document not found."}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult(result)


def handle_put(path: str, deps: KnowledgeRouteDependencies) -> JsonRouteResult | None:
    document_id = _knowledge_document_id(path)
    if document_id is None:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        document = deps.knowledge.update_document(
            document_id,
            str(payload.get("title", "")).strip(),
            str(payload.get("content", "")),
            project=project,
            embedding_provider=str(payload.get("embedding_provider", "")).strip(),
            owner=deps.auth_scope_owner(),
        )
    except KeyError:
        return JsonRouteResult({"error": "Knowledge document not found."}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return JsonRouteResult({"document": document})


def _parse_multipart_parts(body: bytes, boundary: bytes) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    delimiter = b"--" + boundary
    for raw_part in body.split(delimiter):
        raw_part = raw_part.strip(b"\r\n")
        if not raw_part or raw_part == b"--":
            continue
        if raw_part.endswith(b"--"):
            raw_part = raw_part[:-2].rstrip(b"\r\n")
        header_end = raw_part.find(b"\r\n\r\n")
        offset = 4
        if header_end == -1:
            header_end = raw_part.find(b"\n\n")
            offset = 2
        if header_end == -1:
            continue
        headers_raw = raw_part[:header_end].decode("utf-8", errors="replace")
        content = raw_part[header_end + offset:]
        headers: dict[str, str] = {}
        for line in headers_raw.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        disposition = headers.get("content-disposition", "")
        name_match = re.search(r'name="([^"]*)"', disposition)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        parts.append({
            "headers": headers, "name": name_match.group(1) if name_match else "",
            "filename": filename_match.group(1) if filename_match else "",
            "content_type": headers.get("content-type", ""), "content": content.rstrip(b"\r\n"),
        })
    return parts


def _handle_upload(deps: KnowledgeRouteDependencies) -> JsonRouteResult:
    content_type = deps.headers.get("Content-Type", "")
    content_length = _parse_int(deps.headers.get("Content-Length"), default=0)
    if "multipart/form-data" not in content_type:
        return JsonRouteResult({"error": "multipart/form-data required"}, HTTPStatus.BAD_REQUEST)
    if content_length <= 0 or content_length > deps.upload_limit_bytes:
        return JsonRouteResult({"error": "Content-Length missing or exceeds the configured upload limit"}, HTTPStatus.BAD_REQUEST)
    raw = deps.read_body(content_length)
    boundary = b""
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[9:].strip('"').encode("utf-8")
            break
    if not boundary:
        return JsonRouteResult({"error": "No boundary in multipart form"}, HTTPStatus.BAD_REQUEST)
    parts = _parse_multipart_parts(raw, boundary)
    file_part = next(
        (part for part in parts if str(part.get("filename", "")).strip() and isinstance(part.get("content"), bytes)),
        None,
    )
    if file_part is None:
        return JsonRouteResult({"error": "No file was included in the upload."}, HTTPStatus.BAD_REQUEST)
    project_part = next((part for part in parts if part.get("name") == "project"), None)
    embedding_part = next((part for part in parts if part.get("name") == "embedding_provider"), None)
    project = (
        bytes(project_part.get("content", b"")).decode("utf-8", errors="replace").strip()
        if project_part else "default"
    ) or "default"
    embedding_provider = (
        bytes(embedding_part.get("content", b"")).decode("utf-8", errors="replace").strip()
        if embedding_part else ""
    )
    filename = str(file_part.get("filename", "")).strip() or "attachment.bin"
    content = bytes(file_part.get("content", b""))
    content_type_hint = str(file_part.get("content_type", "")).strip()
    asset = deps.media.ingest_bytes(
        content, filename=filename, project=project, source="chat-attachment",
        source_path=filename, metadata={"mime_type": content_type_hint} if content_type_hint else {},
        owner=deps.auth_scope_owner(),
    )
    analysis = asset.get("analysis") if isinstance(asset.get("analysis"), dict) else {}
    text_content = str(analysis.get("extracted_text", "")).strip()
    if not text_content and Path(filename).suffix.lower() in {".docx", ".odt", ".rtf", ".html", ".htm", ".csv", ".md"}:
        from ..document_processing import extract_office_text

        text_content = extract_office_text(filename, content).strip()
    document = None
    if text_content:
        document = deps.knowledge.ingest_text(
            title=filename, content=text_content, project=project, source="chat-attachment",
            source_path=filename,
            metadata={"asset_id": asset.get("id", ""), "mime_type": asset.get("mime_type", "")},
            embedding_provider=embedding_provider,
            owner=deps.auth_scope_owner(),
        )
    attachment = {
        "id": asset.get("id", ""), "name": filename, "filename": filename,
        "media_type": asset.get("media_type", "document"),
        "mime_type": asset.get("mime_type", content_type_hint),
        "size_bytes": asset.get("size_bytes", len(content)),
        "summary": analysis.get("summary", ""), "extracted_text": text_content[:50_000],
        "findings": analysis.get("findings", {}),
        "document_id": (document or {}).get("id", "") if isinstance(document, dict) else "",
    }
    return JsonRouteResult(
        {"attachment": attachment, "asset": asset, "document": document,
         "filename": filename, "chars_extracted": len(text_content)},
        HTTPStatus.CREATED,
    )
