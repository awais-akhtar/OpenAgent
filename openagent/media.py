"""Stage 11 local media ingestion and understanding."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Any
from uuid import uuid4

from .document_processing import extract_pdf_text as extract_document_pdf_text
from .models import AuditEvent, ContextEvent, utc_now
from .process_policy import ProcessPolicy, run_process
from .storage import OpenAgentStore


TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log", ".html", ".htm", ".xml"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


class MediaManager:
    """Stores local media assets and deterministic local analysis records."""

    def __init__(self, store: OpenAgentStore):
        self.store = store
        self.assets_root = (self.store.data_dir / "media_assets").resolve()
        self.assets_root.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        self.store.initialize()

    def ingest_file(
        self,
        path: str | Path,
        *,
        project: str = "default",
        title: str = "",
        source: str = "file",
        metadata: dict[str, Any] | None = None,
        copy_to_store: bool = True,
    ) -> dict[str, Any]:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists() or not file_path.is_file():
            raise ValueError(f"Media file does not exist: {file_path}")
        data = file_path.read_bytes()
        return self.ingest_bytes(
            data,
            filename=file_path.name,
            project=project,
            title=title or file_path.stem,
            source=source,
            source_path=str(file_path),
            metadata=metadata,
            copy_to_store=copy_to_store,
        )

    def ingest_bytes(
        self,
        data: bytes,
        *,
        filename: str,
        project: str = "default",
        title: str = "",
        source: str = "upload",
        source_path: str = "",
        metadata: dict[str, Any] | None = None,
        copy_to_store: bool = True,
    ) -> dict[str, Any]:
        if not data:
            raise ValueError("Media data is required.")
        project = project.strip() or "default"
        suffix = Path(filename).suffix.lower()
        metadata = metadata or {}
        hinted_mime_type = str(metadata.get("mime_type", "")).strip()
        mime_type = hinted_mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        media_type = classify_media_type(suffix)
        if mime_type.startswith("audio/"):
            media_type = "audio"
        elif mime_type.startswith("video/"):
            media_type = "video"
        elif mime_type.startswith("image/"):
            media_type = "image"
        content_hash = hashlib.sha256(data).hexdigest()
        now = utc_now()
        asset_id = stable_id("media-asset", project, filename, content_hash)
        stored_path = ""
        if copy_to_store:
            safe_name = f"{asset_id}{suffix or '.bin'}"
            destination = (self.assets_root / safe_name).resolve()
            if self.assets_root not in destination.parents and destination != self.assets_root:
                raise ValueError("Resolved media path escaped the media asset directory.")
            destination.write_bytes(data)
            stored_path = str(destination)
        asset = {
            "id": asset_id,
            "project": project,
            "title": title.strip() or Path(filename).stem or "Untitled media",
            "media_type": media_type,
            "mime_type": mime_type,
            "source": source.strip() or "upload",
            "source_path": source_path,
            "stored_path": stored_path,
            "size_bytes": len(data),
            "content_hash": content_hash,
            "metadata": metadata or {},
            "created_at": now,
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO media_assets (
                    id, project, title, media_type, mime_type, source, source_path,
                    stored_path, size_bytes, content_hash, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset["id"],
                    asset["project"],
                    asset["title"],
                    asset["media_type"],
                    asset["mime_type"],
                    asset["source"],
                    asset["source_path"],
                    asset["stored_path"],
                    asset["size_bytes"],
                    asset["content_hash"],
                    json.dumps(asset["metadata"], sort_keys=True),
                    asset["created_at"],
                ),
            )
        analysis = self.analyze_bytes(asset, data, suffix=suffix)
        self.store.add_event(
            ContextEvent(
                title=f"Media ingested: {asset['title']}",
                content=analysis["summary"],
                type=f"media:{media_type}",
                source=f"media:{asset['source']}",
                project=project,
                metadata={"asset_id": asset_id, "analysis_id": analysis["id"], "mime_type": mime_type},
            )
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="media",
                action="media.ingest",
                resource=f"media:{asset_id}",
                project=project,
                metadata={"media_type": media_type, "size_bytes": len(data), "source": asset["source"]},
            )
        )
        return {**asset, "analysis": analysis}

    def analyze_asset(self, asset_id: str) -> dict[str, Any]:
        asset = self.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"Unknown media asset '{asset_id}'.")
        stored_path = asset.get("stored_path", "")
        if not stored_path:
            raise ValueError("Asset bytes were not copied into the local media store.")
        data = Path(stored_path).read_bytes()
        return self.analyze_bytes(asset, data, suffix=Path(stored_path).suffix.lower())

    def analyze_bytes(self, asset: dict[str, Any], data: bytes, *, suffix: str) -> dict[str, Any]:
        findings: dict[str, Any] = {
            "privacy_boundary": "Local deterministic metadata and text extraction only.",
            "content_hash": asset["content_hash"],
        }
        extracted_text = ""
        analyzer = "local-media-metadata"
        if suffix in TEXT_SUFFIXES:
            extracted_text = decode_text(data)
            findings["characters"] = len(extracted_text)
            analyzer = "local-text-reader"
        elif suffix == ".pdf":
            extracted_text = extract_pdf_text(data)
            findings["pdf_text_scan"] = bool(extracted_text)
            findings["text_quality"] = text_quality(extracted_text)
            analyzer = "local-pdf-text-scan"
        elif suffix in IMAGE_SUFFIXES:
            findings.update(image_findings(data, suffix=suffix))
            analyzer = "local-image-header-analysis"
        elif suffix in AUDIO_SUFFIXES:
            findings.update({"audio_format": suffix.lstrip("."), "transcript_available": False})
            analyzer = "local-audio-metadata"
        elif suffix in VIDEO_SUFFIXES:
            findings.update({"video_format": suffix.lstrip("."), "transcript_available": False})
            analyzer = "local-video-metadata"
        summary = summarize_asset(asset, findings, extracted_text)
        analysis_id = stable_id("media-analysis", asset["id"], analyzer, utc_now(), uuid4().hex)
        created_at = utc_now()
        analysis = {
            "id": analysis_id,
            "asset_id": asset["id"],
            "project": asset["project"],
            "analyzer": analyzer,
            "summary": summary,
            "extracted_text": extracted_text,
            "findings": findings,
            "created_at": created_at,
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO media_analyses (
                    id, asset_id, project, analyzer, summary, extracted_text, findings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis["id"],
                    analysis["asset_id"],
                    analysis["project"],
                    analysis["analyzer"],
                    analysis["summary"],
                    analysis["extracted_text"],
                    json.dumps(analysis["findings"], sort_keys=True),
                    analysis["created_at"],
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="media",
                action="media.analyze",
                resource=f"media:{asset['id']}",
                project=asset["project"],
                metadata={"analyzer": analyzer},
            )
        )
        return analysis

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM media_assets WHERE id = ?", (asset_id,)).fetchone()
        return asset_row(row) if row else None

    def assets(self, *, project: str = "default", media_type: str = "", limit: int = 50) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if media_type:
            clauses.append("media_type = ?")
            params.append(media_type)
        params.append(max(1, min(limit, 200)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM media_assets WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [asset_row(row) for row in rows]

    def analyses(self, *, project: str = "default", asset_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if asset_id:
            clauses.append("asset_id = ?")
            params.append(asset_id)
        params.append(max(1, min(limit, 200)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM media_analyses WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [analysis_row(row) for row in rows]

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT media_type, count(*) AS count FROM media_assets WHERE project = ? GROUP BY media_type",
                (project,),
            ).fetchall()
            analyses = connection.execute(
                "SELECT count(*) FROM media_analyses WHERE project = ?",
                (project,),
            ).fetchone()[0]
        return {
            "project": project,
            "assets_by_type": {row["media_type"]: row["count"] for row in rows},
            "analyses": analyses,
            "privacy_boundary": "Files are read locally and optionally copied into the OpenAgent data directory.",
        }


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def classify_media_type(suffix: str) -> str:
    if suffix in TEXT_SUFFIXES or suffix == ".pdf":
        return "document"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "binary"


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return ""


def extract_pdf_text(data: bytes) -> str:
    text = extract_document_pdf_text(data).strip()
    return text if text_quality(text)["usable"] else ""


def text_quality(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if not value:
        return {"usable": False, "characters": 0, "printable_ratio": 0.0, "word_count": 0}
    printable = sum(character.isprintable() or character in "\n\r\t" for character in value)
    letters = sum(character.isalpha() for character in value)
    controls = sum(ord(character) < 32 and character not in "\n\r\t" for character in value)
    words = re.findall(r"\b[\w'-]{2,}\b", value, flags=re.UNICODE)
    printable_ratio = printable / len(value)
    letter_ratio = letters / len(value)
    usable = printable_ratio >= 0.92 and letter_ratio >= 0.20 and controls == 0 and len(words) >= 3
    return {
        "usable": usable,
        "characters": len(value),
        "printable_ratio": round(printable_ratio, 4),
        "letter_ratio": round(letter_ratio, 4),
        "word_count": len(words),
    }


def image_findings(data: bytes, *, suffix: str) -> dict[str, Any]:
    width = height = 0
    image_format = suffix.lstrip(".")
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        image_format = "png"
    elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        if len(data) >= 10:
            width, height = struct.unpack("<HH", data[6:10])
        image_format = "gif"
    elif data.startswith(b"\xff\xd8"):
        width, height = jpeg_dimensions(data)
        image_format = "jpeg"
    # Attempt OCR via tesseract (if installed) or pytesseract
    ocr_text, ocr_available = _try_ocr(data, suffix)
    return {
        "image_format": image_format,
        "width": width,
        "height": height,
        "ocr_available": ocr_available,
        "ocr_text": ocr_text[:2000] if ocr_text else "",
        "vision_summary": (
            f"OCR extracted {len(ocr_text.split())} words from image."
            if ocr_text else
            "Header-level local image understanding. Install tesseract for OCR."
        ),
    }


def _try_ocr(data: bytes, suffix: str) -> tuple[str, bool]:
    """Attempt OCR using tesseract CLI or pytesseract (stdlib-safe fallback)."""

    # Try tesseract CLI (most common installation)
    tesseract = shutil.which("tesseract")
    if not tesseract:
        try:
            import pytesseract  # type: ignore

            configured = str(pytesseract.pytesseract.tesseract_cmd or "").strip()
            candidate = Path(configured).expanduser() if configured else None
            tesseract = str(candidate.resolve()) if candidate and candidate.is_file() else shutil.which(configured)
        except (ImportError, AttributeError, OSError):
            tesseract = None
    if tesseract:
        try:
            with tempfile.TemporaryDirectory(prefix="openagent-ocr-") as workspace_value:
                workspace = Path(workspace_value).resolve()
                input_path = workspace / f"input{suffix or '.png'}"
                input_path.write_bytes(data)
                result = run_process(
                    [tesseract, str(input_path), "stdout", "--psm", "3"],
                    cwd=workspace,
                    policy=ProcessPolicy(
                        timeout_seconds=30,
                        max_output_bytes=64_000,
                        allowed_cwd_roots=(workspace,),
                        allowed_executables=(tesseract,),
                    ),
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip(), True
        except (FileNotFoundError, OSError, PermissionError, ValueError):
            pass

    # Fallback: extract text-like sequences from raw bytes (ASCII art / embedded text)
    try:
        ascii_text = re.findall(rb"[ -~]{4,}", data)
        readable = [t.decode("ascii", errors="replace") for t in ascii_text if len(t) >= 4]
        joined = " ".join(readable[:50]).strip()
        if joined and len(joined) > 10:
            return joined[:500], False
    except Exception:
        pass

    return "", False


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        length = int.from_bytes(data[index + 2 : index + 4], "big")
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return width, height
        index += max(length + 2, 2)
    return 0, 0


def summarize_asset(asset: dict[str, Any], findings: dict[str, Any], extracted_text: str) -> str:
    if asset["media_type"] == "image":
        width = findings.get("width", 0)
        height = findings.get("height", 0)
        return f"Image asset analyzed locally: {findings.get('image_format')} {width}x{height}."
    if asset["media_type"] == "audio":
        return "Audio asset registered locally with metadata; no external transcription was called."
    if asset["media_type"] == "video":
        return "Video asset registered locally with metadata; no external video analysis was called."
    if extracted_text:
        return f"Document text extracted locally with {len(extracted_text)} characters."
    return f"{asset['media_type'].title()} asset registered locally with metadata."


def asset_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "title": row["title"],
        "media_type": row["media_type"],
        "mime_type": row["mime_type"],
        "source": row["source"],
        "source_path": row["source_path"],
        "stored_path": row["stored_path"],
        "size_bytes": row["size_bytes"],
        "content_hash": row["content_hash"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }


def analysis_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "asset_id": row["asset_id"],
        "project": row["project"],
        "analyzer": row["analyzer"],
        "summary": row["summary"],
        "extracted_text": row["extracted_text"],
        "findings": json.loads(row["findings_json"]),
        "created_at": row["created_at"],
    }
