"""Local extraction for scientific text and PDF evidence."""

from __future__ import annotations

from io import BytesIO
from typing import Any


def summarise_document(data: bytes, *, name: str, media_type: str = "") -> dict[str, Any]:
    lower_name = name.lower()
    if lower_name.endswith(".pdf") or media_type == "application/pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return {"pages": len(reader.pages), "characters": len(text), "text": text[:50000], "extractor": "pypdf"}
        except (ImportError, OSError, ValueError):
            return {"pages": 0, "characters": 0, "text": "", "extractor": "metadata-only", "bytes": len(data)}
    text = data.decode("utf-8", errors="replace")
    return {"pages": 1, "characters": len(text), "text": text[:50000], "extractor": "utf-8"}
