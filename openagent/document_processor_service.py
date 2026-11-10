"""G25: Deep document processing - extraction, analysis, summarization."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .storage import OpenAgentStore

logger = logging.getLogger(__name__)


class DocumentProcessorService:
    """Process documents: extract info, analyze, summarize."""

    def __init__(self, store: OpenAgentStore | None = None):
        self.store = store
        self.supported_formats = [".txt", ".pdf", ".docx", ".md"]
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.documents: dict[tuple[str, str, str], dict[str, Any]] = {}

    def process_document(self, file_path: str, file_content: bytes, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Process uploaded document."""
        try:
            path = Path(file_path)

            # Uploads arrive as a filename plus bytes, so the content length is
            # authoritative; the name may not exist on disk.
            if len(file_content) > self.max_file_size:
                return {"error": f"File too large (max {self.max_file_size/1024/1024:.0f}MB)"}

            if path.suffix.lower() not in self.supported_formats:
                return {"error": f"Unsupported format: {path.suffix}"}

            # Extract text based on format
            text = self._extract_text(path, file_content)

            if not text:
                return {"error": "Could not extract text from document"}

            # Analyze document
            doc_id = str(uuid4())
            analysis = self._analyze_document(text)

            storage_key = (project, owner, doc_id)
            document = {
                "id": doc_id,
                "filename": path.name,
                "format": path.suffix,
                "text": text,
                "analysis": analysis,
                "project": project,
                "owner": owner,
            }
            if self.store is not None:
                with self.store.session() as connection:
                    connection.execute(
                        """INSERT INTO g25_documents
                           (id, project, owner, filename, format, text, analysis_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (doc_id, project, owner, path.name, path.suffix, text, json.dumps(analysis)),
                    )
            self.documents[storage_key] = document

            logger.info(f"Processed document: {path.name} (project={project}, owner={owner})")
            return {
                "document_id": doc_id,
                "filename": path.name,
                "analysis": analysis,
            }

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {"error": str(e)}

    def _extract_text(self, path: Path, file_content: bytes) -> str:
        """Extract text from various formats."""
        if path.suffix.lower() == ".txt":
            return file_content.decode("utf-8", errors="ignore")
        
        if path.suffix.lower() == ".md":
            return file_content.decode("utf-8", errors="ignore")
        
        if path.suffix.lower() == ".pdf":
            return self._extract_pdf_text(file_content)
        
        if path.suffix.lower() == ".docx":
            return self._extract_docx_text(file_content)
        
        return ""

    def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF."""
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(file_content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text
        except Exception as e:
            logger.warning("PDF extraction failed: %s", e)
            raise ValueError("Could not extract text from this PDF") from e

    def _extract_docx_text(self, file_content: bytes) -> str:
        """Extract text from DOCX."""
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(file_content))
            text = "\n".join(para.text for para in doc.paragraphs)
            return text
        except Exception as e:
            logger.warning("DOCX extraction failed: %s", e)
            raise ValueError("Could not extract text from this DOCX") from e

    def _analyze_document(self, text: str) -> dict[str, Any]:
        """Analyze document content."""
        lines = text.split("\n")
        words = text.split()
        
        return {
            "line_count": len([l for l in lines if l.strip()]),
            "word_count": len(words),
            "char_count": len(text),
            "avg_line_length": len(text) / max(len(lines), 1),
            "sentences": len(text.split(". ")) if ". " in text else 1,
            "paragraphs": len([l for l in lines if not l.strip()]),
            "key_terms": self._extract_key_terms(words),
        }

    def _extract_key_terms(self, words: list[str]) -> list[str]:
        """Extract key terms from text."""
        # Simple: return most common words (excluding common words)
        common = {"the", "a", "an", "and", "or", "in", "is", "to", "of", "for"}
        filtered = [w.lower() for w in words if w.lower() not in common and len(w) > 3]
        
        # Return top 10 most common
        from collections import Counter
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(10)]

    def get_document(self, doc_id: str, project: str = "default", owner: str = "default") -> dict[str, Any] | None:
        """Get document details."""
        storage_key = (project, owner, doc_id)
        if self.store is not None:
            with self.store.session() as connection:
                row = connection.execute(
                    "SELECT * FROM g25_documents WHERE id = ? AND project = ? AND owner = ?",
                    (doc_id, project, owner),
                ).fetchone()
            if row is None:
                return None
            return {
                "id": row["id"], "filename": row["filename"], "format": row["format"],
                "text": row["text"], "analysis": json.loads(row["analysis_json"]),
                "project": row["project"], "owner": row["owner"],
            }
        return self.documents.get(storage_key)

    def analyze_document(self, doc_id: str, project: str = "default", owner: str = "default") -> dict[str, Any] | None:
        """Analyze document (public method for routes)."""
        doc = self.get_document(doc_id, project, owner)
        if not doc:
            return None
        return doc.get("analysis", {})

    def list_documents(self, project: str = "default", owner: str = "default") -> list[dict[str, Any]]:
        """List all processed documents."""
        if self.store is not None:
            with self.store.session() as connection:
                records = connection.execute(
                    "SELECT id, filename, analysis_json FROM g25_documents WHERE project = ? AND owner = ? ORDER BY rowid DESC",
                    (project, owner),
                ).fetchall()
            return [{"id": row["id"], "filename": row["filename"], "analysis": json.loads(row["analysis_json"])} for row in records]
        return [
            {"id": d["id"], "filename": d["filename"], "analysis": d["analysis"]}
            for d in self.documents.values()
            if d.get("project") == project and d.get("owner") == owner
        ]

    def extract_information(self, doc_id: str, query: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Extract specific information from document."""
        doc = self.get_document(doc_id, project, owner)
        if not doc:
            return {"error": "Document not found"}

        text = doc["text"]
        # Simple: find matching lines
        matching_lines = [line for line in text.split("\n") if query.lower() in line.lower()]

        return {
            "query": query,
            "matches": len(matching_lines),
            "results": matching_lines[:5],  # First 5 matches
        }

    def summarize_document(self, doc_id: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Generate document summary."""
        doc = self.get_document(doc_id, project, owner)
        if not doc:
            return {"error": "Document not found"}

        text = doc["text"]

        # Simple summarization: extract first 3 sentences
        sentences = text.split(". ")
        summary = ". ".join(sentences[:3]) + "."

        return {
            "document_id": doc_id,
            "summary": summary,
            "word_count": len(summary.split()),
            "compression_ratio": len(summary) / len(text),
        }
