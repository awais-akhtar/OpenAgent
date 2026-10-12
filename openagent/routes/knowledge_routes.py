"""Knowledge and RAG route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs

from ..knowledge import KnowledgeEngine
from ..media import MediaManager
from ..model_gateway import ModelGateway, ProviderConfigurationError
from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "knowledge"
ROUTES = (
    ("GET", "/api/knowledge/status"),
    ("GET", "/api/knowledge/documents"),
    ("GET", "/api/knowledge/search"),
    ("GET", "/api/knowledge/graph"),
    ("GET", "/api/knowledge/tags"),
    ("GET", "/api/knowledge/timeline"),
    ("POST", "/api/knowledge/ingest"),
    ("POST", "/api/knowledge/upload"),
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


def handle_get(path: str, query: str, deps: KnowledgeRouteDependencies) -> JsonRouteResult | None:
    if path not in {item for method, item in ROUTES if method == "GET"}:
        return None
    params = parse_qs(query)
    project = _first(params, "project") or "default"
    if path == "/api/knowledge/status":
        return JsonRouteResult(deps.knowledge.status(project=project))
    if path == "/api/knowledge/documents":
        limit = _parse_int(_first(params, "limit"), default=100)
        return JsonRouteResult({"documents": deps.knowledge.list_documents(project=project, limit=limit)})
    if path == "/api/knowledge/search":
        search_query = _first(params, "q") or ""
        mode = _first(params, "mode") or "fts"
        results = (
            deps.knowledge.semantic_search(search_query, project=project)
            if mode == "semantic"
            else deps.knowledge.search(search_query, project=project)
        )
        return JsonRouteResult({"results": results})
    if path == "/api/knowledge/graph":
        return JsonRouteResult(deps.knowledge.graph(project=project))
    limit = _parse_int(_first(params, "limit"), default=100)
    if path == "/api/knowledge/tags":
        return JsonRouteResult({"tags": deps.knowledge.tags(project=project, limit=limit)})
    return JsonRouteResult({"timeline": deps.knowledge.timeline(project=project, limit=limit)})


def handle_post(path: str, deps: KnowledgeRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/knowledge/upload":
        return _handle_upload(deps)
    if path not in {item for method, item in ROUTES if method == "POST"}:
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
    replay = deps.knowledge.create_replay(
        project=project, label=str(payload.get("label", "Replay")).strip() or "Replay",
        query=str(payload.get("query", "")).strip(),
    )
    return JsonRouteResult({"replay": replay}, HTTPStatus.CREATED)


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
