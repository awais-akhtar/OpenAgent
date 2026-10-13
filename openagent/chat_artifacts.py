"""Local file rendering for documents requested from a chat conversation."""

from __future__ import annotations

import base64
import hashlib
import html
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import AuditEvent
from .document_rendering import render_document_docx, render_document_pdf
from .storage import OpenAgentStore


CONTENT_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "md": "text/markdown; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
    "html": "text/html; charset=utf-8",
}


class ChatArtifactService:
    """Render a model response into a user-owned file and keep a local copy."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.root = store.data_dir / "chat_artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def render(self, *, title: str, content: str, file_format: str, project: str = "default") -> dict[str, Any]:
        title = (title or "OpenAgent document").strip()[:120]
        content = (content or "").strip()
        file_format = (file_format or "pdf").strip().lower().lstrip(".")
        if file_format not in CONTENT_TYPES:
            raise ValueError("Supported document formats are PDF, DOCX, Markdown, text, and HTML.")
        if not content:
            raise ValueError("Document content is required.")

        if file_format == "pdf":
            body = render_document_pdf(title, content)
        elif file_format == "docx":
            body = render_document_docx(title, content)
        elif file_format == "html":
            paragraphs = "\n".join(f"<p>{html.escape(line)}</p>" for line in content.splitlines() if line.strip())
            body = f"<!doctype html><html><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title></head><body><main><h1>{html.escape(title)}</h1>{paragraphs}</main></body></html>".encode("utf-8")
        else:
            body = content.encode("utf-8")

        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:56] or "openagent-document"
        filename = f"{slug}-{uuid4().hex[:8]}.{file_format}"
        folder = self.root / re.sub(r"[^A-Za-z0-9_.-]+", "_", project or "default")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        path.write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        self.store.add_audit_event(AuditEvent(actor="chat", action="chat.artifact.render", resource=f"artifact:{filename}", project=project, metadata={"format": file_format, "size_bytes": len(body), "sha256": digest}))
        return {
            "filename": filename,
            "format": file_format,
            "content_type": CONTENT_TYPES[file_format],
            "size_bytes": len(body),
            "sha256": digest,
            "local_path": str(path),
            "base64": base64.b64encode(body).decode("ascii"),
        }
