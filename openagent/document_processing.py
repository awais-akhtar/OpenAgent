"""Deep local document processing for PDFs, Office archives, and export bundles."""

from __future__ import annotations

import base64
import binascii
import difflib
import hashlib
import hmac
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from .documents import DocumentManager
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class DocumentProcessing:
    """Converts local files into versioned OpenAgent documents."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.documents = DocumentManager(store)
        self.artifact_dir = self.store.data_dir / "document_processing"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_artifacts (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_document_artifacts_doc ON document_artifacts(doc_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS document_annotations (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    geometry_json TEXT NOT NULL DEFAULT '{}',
                    text TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_document_annotations_doc ON document_annotations(doc_id, page, created_at);
                """
            )

    def import_pdf(
        self,
        *,
        filename: str,
        content: bytes,
        project: str = "default",
        owner: str = "local",
        title: str = "",
    ) -> dict[str, Any]:
        text = extract_pdf_text(content)
        if not text.strip():
            text = f"PDF imported: {Path(filename).name}. No extractable text found with the local fallback extractor."
        doc = self.documents.create_document(
            title=title or Path(filename).stem or "Imported PDF",
            content=text,
            project=project,
            owner=owner,
            doc_format="markdown",
            tags=["pdf", "imported"],
            folder="imports/pdf",
        )
        artifact = self._write_artifact(filename, content, project=project)
        self._register_artifact(doc["doc_id"], artifact, project=project, kind="source-pdf")
        self.store.add_audit_event(
            AuditEvent(
                actor="document-processing",
                action="document.import_pdf",
                resource=f"document:{doc['doc_id']}",
                project=project,
                metadata={"filename": filename, "artifact": str(artifact), "chars": len(text)},
            )
        )
        return {"document": doc, "chars_extracted": len(text), "artifact_path": str(artifact)}

    def import_office_or_text(
        self,
        *,
        filename: str,
        content: bytes,
        project: str = "default",
        owner: str = "local",
        title: str = "",
    ) -> dict[str, Any]:
        text = extract_office_text(filename, content)
        if not text.strip():
            text = content.decode("utf-8", errors="replace")
        doc = self.documents.create_document(
            title=title or Path(filename).stem or "Imported document",
            content=text,
            project=project,
            owner=owner,
            doc_format="markdown",
            tags=["imported"],
            folder="imports/documents",
        )
        artifact = self._write_artifact(filename, content, project=project)
        self._register_artifact(doc["doc_id"], artifact, project=project, kind="source-document")
        self.store.add_audit_event(
            AuditEvent(
                actor="document-processing",
                action="document.import",
                resource=f"document:{doc['doc_id']}",
                project=project,
                metadata={"filename": filename, "artifact": str(artifact), "chars": len(text)},
            )
        )
        return {"document": doc, "chars_extracted": len(text), "artifact_path": str(artifact)}

    def extract_document_text(self, doc_id: str, *, owner: str = "") -> dict[str, Any]:
        doc = self.documents.get_document(doc_id, owner=owner)
        content = str(doc.get("content", ""))
        return {
            "doc_id": doc_id,
            "title": doc["title"],
            "chars": len(content),
            "text": content,
            "privacy_boundary": "Document text is read from local SQLite editor storage.",
        }

    def render_preview(self, doc_id: str, *, owner: str = "", max_lines: int = 12) -> dict[str, Any]:
        doc = self.documents.get_document(doc_id, owner=owner)
        lines = str(doc.get("content", "")).splitlines()[: max(1, min(max_lines, 100))]
        return {
            "doc_id": doc_id,
            "title": doc["title"],
            "version": doc["version"],
            "preview_lines": lines,
            "line_count": len(str(doc.get("content", "")).splitlines()),
            "render_mode": "local-text-preview",
        }

    def render_pdf_page(self, doc_id: str, *, page: int = 0, scale: float = 1.5, owner: str = "") -> dict[str, Any]:
        self.documents.get_document(doc_id, owner=owner)
        source = self._latest_artifact(doc_id, kind="source-pdf")
        if source is None:
            raise ValueError("Document has no imported PDF artifact.")
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(source["local_path"])
        page_object = None
        bitmap = None
        try:
            page_count = len(pdf)
            if page < 0 or page >= page_count:
                raise ValueError("PDF page is out of range.")
            page_object = pdf[page]
            bitmap = page_object.render(scale=max(0.5, min(float(scale), 4.0)))
            output = self.artifact_dir / f"{doc_id}-page-{page + 1}-{uuid4().hex[:8]}.png"
            bitmap.to_pil().save(output, format="PNG", optimize=True)
        finally:
            if bitmap is not None:
                bitmap.close()
            if page_object is not None:
                page_object.close()
            pdf.close()
        artifact = self._register_artifact(doc_id, output, project=source["project"], kind="page-render", metadata={"page": page})
        return {"doc_id": doc_id, "page": page, "page_count": page_count, "artifact": artifact}

    def add_annotation(
        self,
        doc_id: str,
        *,
        page: int,
        kind: str = "note",
        geometry: dict[str, Any] | None = None,
        text: str = "",
        metadata: dict[str, Any] | None = None,
        owner: str = "",
    ) -> dict[str, Any]:
        self.documents.get_document(doc_id, owner=owner)
        annotation_id = uuid4().hex
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO document_annotations (id, doc_id, page, kind, geometry_json, text, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (annotation_id, doc_id, max(0, int(page)), kind, json.dumps(geometry or {}, sort_keys=True), text, json.dumps(metadata or {}, sort_keys=True), utc_now()),
            )
        return {"id": annotation_id, "doc_id": doc_id, "page": max(0, int(page)), "kind": kind, "geometry": geometry or {}, "text": text, "metadata": metadata or {}}

    def annotations(self, doc_id: str, *, owner: str = "") -> list[dict[str, Any]]:
        self.documents.get_document(doc_id, owner=owner)
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM document_annotations WHERE doc_id = ? ORDER BY page, created_at", (doc_id,)).fetchall()
        return [
            {"id": row["id"], "doc_id": row["doc_id"], "page": row["page"], "kind": row["kind"], "geometry": json.loads(row["geometry_json"]), "text": row["text"], "metadata": json.loads(row["metadata_json"]), "created_at": row["created_at"]}
            for row in rows
        ]

    def add_ocr_overlay(self, doc_id: str, *, page: int, text: str, blocks: list[dict[str, Any]] | None = None, owner: str = "") -> dict[str, Any]:
        return self.add_annotation(doc_id, page=page, kind="ocr-overlay", text=text, metadata={"blocks": blocks or [], "engine": "local-submitted"}, owner=owner)

    def redact_text(self, doc_id: str, patterns: list[str], *, replacement: str = "[REDACTED]", owner: str = "") -> dict[str, Any]:
        document = self.documents.get_document(doc_id, owner=owner)
        content = str(document.get("content", ""))
        replacements = 0
        for pattern in patterns:
            if not pattern:
                continue
            content, count = re.subn(re.escape(pattern), replacement, content, flags=re.IGNORECASE)
            replacements += count
        updated = self.documents.update_document(doc_id, content=content, if_version=document["version"], owner=owner)
        self.store.add_audit_event(
            AuditEvent(actor="document-processing", action="document.redact", resource=f"document:{doc_id}", project=document["project"], metadata={"patterns": len(patterns), "replacements": replacements})
        )
        return {"document": updated, "replacements": replacements, "redaction_mode": "versioned-text-layer"}

    def export_pdf(self, doc_id: str, *, owner: str = "", signing_key: str = "") -> dict[str, Any]:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen.canvas import Canvas

        document = self.documents.get_document(doc_id, owner=owner)
        path = self.artifact_dir / f"{doc_id}-v{document['version']}.pdf"
        canvas = Canvas(str(path), pagesize=A4, invariant=1, pageCompression=1)
        width, height = A4
        text = canvas.beginText(42, height - 54)
        text.setFont("Helvetica", 10)
        for line in str(document.get("content", "")).splitlines() or [""]:
            for chunk in [line[index:index + 95] for index in range(0, max(1, len(line)), 95)]:
                if text.getY() < 54:
                    canvas.drawText(text)
                    canvas.showPage()
                    text = canvas.beginText(42, height - 54)
                    text.setFont("Helvetica", 10)
                text.textLine(chunk)
        canvas.drawText(text)
        canvas.save()
        artifact = self._register_artifact(doc_id, path, project=document["project"], kind="pdf-export", metadata={"version": document["version"], "deterministic": True})
        signature = hmac.new(signing_key.encode("utf-8"), path.read_bytes(), hashlib.sha256).hexdigest() if signing_key else ""
        return {"document": document, "artifact": artifact, "signature": signature, "signature_algorithm": "HMAC-SHA256" if signature else "", "deterministic": True}

    def compare(self, left_doc_id: str, right_doc_id: str, *, owner: str = "") -> dict[str, Any]:
        left = self.documents.get_document(left_doc_id, owner=owner)
        right = self.documents.get_document(right_doc_id, owner=owner)
        diff = list(difflib.unified_diff(str(left.get("content", "")).splitlines(), str(right.get("content", "")).splitlines(), fromfile=left["title"], tofile=right["title"], lineterm=""))
        ratio = difflib.SequenceMatcher(None, str(left.get("content", "")), str(right.get("content", ""))).ratio()
        return {"left": {"doc_id": left_doc_id, "version": left["version"]}, "right": {"doc_id": right_doc_id, "version": right["version"]}, "similarity": round(ratio, 6), "diff": diff}

    def export_zip(self, *, doc_ids: list[str], project: str = "default", owner: str = "") -> dict[str, Any]:
        export_id = uuid4().hex
        path = self.artifact_dir / f"document-export-{export_id}.zip"
        exported = []
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            manifest = {"project": project, "created_at": utc_now(), "documents": []}
            for doc_id in doc_ids:
                doc = self.documents.get_document(doc_id, owner=owner)
                safe_title = re.sub(r"[^A-Za-z0-9_.-]+", "_", doc["title"]).strip("_") or doc_id
                name = f"{safe_title}-v{doc['version']}.md"
                archive.writestr(name, str(doc.get("content", "")))
                manifest["documents"].append({"doc_id": doc_id, "title": doc["title"], "file": name, "version": doc["version"]})
                exported.append(doc_id)
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        self.store.add_audit_event(
            AuditEvent(
                actor="document-processing",
                action="document.export_zip",
                resource=f"document_export:{export_id}",
                project=project,
                metadata={"count": len(exported), "path": str(path)},
            )
        )
        return {"path": str(path), "count": len(exported), "documents": exported}

    def decode_base64(self, value: str) -> bytes:
        try:
            return base64.b64decode(value.encode("ascii"), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("content_base64 must be valid base64.") from exc

    def _write_artifact(self, filename: str, content: bytes, *, project: str) -> Path:
        folder = self.artifact_dir / project
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{uuid4().hex}-{Path(filename or 'document.bin').name}"
        path.write_bytes(content)
        return path

    def _register_artifact(self, doc_id: str, path: Path, *, project: str, kind: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        artifact_id = uuid4().hex
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        created = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO document_artifacts (id, doc_id, project, kind, local_path, sha256, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (artifact_id, doc_id, project, kind, str(path), digest, json.dumps(metadata or {}, sort_keys=True), created),
            )
        return {"id": artifact_id, "doc_id": doc_id, "project": project, "kind": kind, "local_path": str(path), "sha256": digest, "metadata": metadata or {}, "created_at": created}

    def _latest_artifact(self, doc_id: str, *, kind: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM document_artifacts WHERE doc_id = ? AND kind = ? ORDER BY rowid DESC LIMIT 1", (doc_id, kind)).fetchone()
        return dict(row) if row else None


def extract_pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        extracted = "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
        if extracted:
            return extracted
    except Exception:
        pass
    pieces = []
    for block in re.findall(rb"BT(.*?)ET", content, re.DOTALL):
        for text in re.findall(rb"\((.*?)\)", block):
            decoded = text.decode("latin-1", errors="replace").strip()
            if decoded:
                pieces.append(decoded)
    if not pieces:
        for text in re.findall(rb"\((.*?)\)", content):
            decoded = text.decode("latin-1", errors="replace").strip()
            if decoded:
                pieces.append(decoded)
    return " ".join(pieces)


def extract_office_text(filename: str, content: bytes) -> str:
    lowered = filename.lower()
    if lowered.endswith((".docx", ".pptx", ".xlsx")):
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                xml_text = []
                for name in archive.namelist():
                    if name.endswith(".xml") and any(part in name for part in ("word/", "ppt/", "xl/")):
                        raw = archive.read(name).decode("utf-8", errors="replace")
                        cleaned = re.sub(r"<[^>]+>", " ", raw)
                        xml_text.append(re.sub(r"\s+", " ", cleaned).strip())
                return "\n".join(part for part in xml_text if part)
        except zipfile.BadZipFile:
            return ""
    return content.decode("utf-8", errors="replace")
