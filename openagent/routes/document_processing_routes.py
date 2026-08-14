"""Deep document processing route contracts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import unquote

from ..document_processing import DocumentProcessing
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "document_processing"
ROUTES = (
    ("POST", "/api/documents/import-pdf"),
    ("POST", "/api/documents/import"),
    ("POST", "/api/document/{id}/extract-pdf-text"),
    ("POST", "/api/document/{id}/render-preview"),
    ("POST", "/api/document/{id}/render-page"),
    ("POST", "/api/document/{id}/annotations"),
    ("POST", "/api/document/{id}/ocr-overlay"),
    ("POST", "/api/document/{id}/redact"),
    ("POST", "/api/document/{id}/export-pdf"),
    ("POST", "/api/document/{id}/export-docx"),
    ("POST", "/api/document/{id}/export-pptx"),
    ("POST", "/api/document/{id}/export-xlsx"),
    ("POST", "/api/documents/compare"),
    ("POST", "/api/documents/export-zip"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class DocumentProcessingRouteDependencies:
    processing: DocumentProcessing
    payload_owner: Any
    read_json: Any


def handle_post(path: str, deps: DocumentProcessingRouteDependencies) -> JsonRouteResult | None:
    is_known_path = (
        path in {"/api/documents/import-pdf", "/api/documents/import", "/api/documents/export-zip"}
        or (path.startswith("/api/document/") and path.endswith("/extract-pdf-text"))
        or (path.startswith("/api/document/") and path.endswith("/render-preview"))
        or (path.startswith("/api/document/") and any(path.endswith(suffix) for suffix in ("/render-page", "/annotations", "/ocr-overlay", "/redact", "/export-pdf", "/export-docx", "/export-pptx", "/export-xlsx")))
        or path == "/api/documents/compare"
    )
    if not is_known_path:
        return None
    payload = deps.read_json()
    owner = deps.payload_owner(payload, "local")
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/documents/import-pdf":
            result = deps.processing.import_pdf(
                filename=str(payload.get("filename", "document.pdf")),
                content=deps.processing.decode_base64(str(payload.get("content_base64", ""))),
                project=project,
                owner=owner,
                title=str(payload.get("title", "")),
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        if path == "/api/documents/import":
            result = deps.processing.import_office_or_text(
                filename=str(payload.get("filename", "document.txt")),
                content=deps.processing.decode_base64(str(payload.get("content_base64", ""))),
                project=project,
                owner=owner,
                title=str(payload.get("title", "")),
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        if path.startswith("/api/document/") and path.endswith("/extract-pdf-text"):
            doc_id = unquote(path.removeprefix("/api/document/").removesuffix("/extract-pdf-text")).strip("/")
            return JsonRouteResult(deps.processing.extract_document_text(doc_id, owner=owner if owner != "local" else ""))
        if path.startswith("/api/document/") and path.endswith("/render-preview"):
            doc_id = unquote(path.removeprefix("/api/document/").removesuffix("/render-preview")).strip("/")
            return JsonRouteResult(deps.processing.render_preview(doc_id, owner=owner if owner != "local" else ""))
        if path.startswith("/api/document/") and path.endswith("/render-page"):
            doc_id = unquote(path.removeprefix("/api/document/").removesuffix("/render-page")).strip("/")
            return JsonRouteResult(deps.processing.render_pdf_page(doc_id, page=int(payload.get("page", 0)), scale=float(payload.get("scale", 1.5)), owner=owner if owner != "local" else ""))
        if path.startswith("/api/document/") and path.endswith("/annotations"):
            doc_id = unquote(path.removeprefix("/api/document/").removesuffix("/annotations")).strip("/")
            return JsonRouteResult({"annotation": deps.processing.add_annotation(
                doc_id, page=int(payload.get("page", 0)), kind=str(payload.get("kind", "note")),
                geometry=payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {},
                text=str(payload.get("text", "")), metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                owner=owner if owner != "local" else "",
            )}, HTTPStatus.CREATED)
        if path.startswith("/api/document/") and path.endswith("/ocr-overlay"):
            doc_id = unquote(path.removeprefix("/api/document/").removesuffix("/ocr-overlay")).strip("/")
            blocks = payload.get("blocks")
            return JsonRouteResult({"overlay": deps.processing.add_ocr_overlay(doc_id, page=int(payload.get("page", 0)), text=str(payload.get("text", "")), blocks=[item for item in blocks if isinstance(item, dict)] if isinstance(blocks, list) else [], owner=owner if owner != "local" else "")}, HTTPStatus.CREATED)
        if path.startswith("/api/document/") and path.endswith("/redact"):
            doc_id = unquote(path.removeprefix("/api/document/").removesuffix("/redact")).strip("/")
            patterns = payload.get("patterns")
            return JsonRouteResult(deps.processing.redact_text(doc_id, [str(item) for item in patterns] if isinstance(patterns, list) else [], replacement=str(payload.get("replacement", "[REDACTED]")), owner=owner if owner != "local" else ""))
        if path.startswith("/api/document/") and path.endswith("/export-pdf"):
            doc_id = unquote(path.removeprefix("/api/document/").removesuffix("/export-pdf")).strip("/")
            return JsonRouteResult(deps.processing.export_pdf(doc_id, owner=owner if owner != "local" else "", signing_key=str(payload.get("signing_key", ""))))
        if path.startswith("/api/document/") and path.endswith("/export-docx"):
            doc_id = unquote(path.removeprefix("/api/document/").removesuffix("/export-docx")).strip("/")
            return JsonRouteResult(deps.processing.export_docx(doc_id, owner=owner if owner != "local" else ""))
        if path.startswith("/api/document/") and path.endswith("/export-pptx"):
            doc_id = unquote(path.removeprefix("/api/document/").removesuffix("/export-pptx")).strip("/")
            return JsonRouteResult(deps.processing.export_pptx(doc_id, owner=owner if owner != "local" else ""))
        if path.startswith("/api/document/") and path.endswith("/export-xlsx"):
            doc_id = unquote(path.removeprefix("/api/document/").removesuffix("/export-xlsx")).strip("/")
            return JsonRouteResult(deps.processing.export_xlsx(doc_id, owner=owner if owner != "local" else ""))
        if path == "/api/documents/compare":
            return JsonRouteResult(deps.processing.compare(str(payload.get("left_doc_id", "")), str(payload.get("right_doc_id", "")), owner=owner if owner != "local" else ""))
        if path == "/api/documents/export-zip":
            doc_ids = [str(item) for item in payload.get("doc_ids", [])] if isinstance(payload.get("doc_ids"), list) else []
            return JsonRouteResult(deps.processing.export_zip(doc_ids=doc_ids, project=project, owner=owner if owner != "local" else ""))
    except (ValueError, KeyError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None
