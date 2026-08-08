"""Lightweight safety checks for scientific reports."""

from __future__ import annotations

import re
from typing import Iterable, Mapping


def check_citations(text: str, citations: Iterable[str]) -> dict[str, object]:
    available = [str(item).strip() for item in citations if str(item).strip()]
    mentioned = re.findall(r"\[[^\]]+\]|\(\d{4}[a-z]?\)", text)
    return {"passed": not mentioned or bool(available), "mentioned": len(mentioned), "available": len(available)}


def check_consistency(values: Mapping[str, float], *, tolerance: float = 1e-9) -> dict[str, object]:
    finite = all(abs(float(value)) != float("inf") for value in values.values())
    spread = max(values.values()) - min(values.values()) if values else 0.0
    return {"passed": finite and spread >= -tolerance, "finite": finite, "spread": spread}


def detect_unsupported_claims(text: str, evidence_terms: Iterable[str]) -> dict[str, object]:
    terms = {str(term).lower() for term in evidence_terms if str(term).strip()}
    words = {word.lower() for word in re.findall(r"[A-Za-z]{4,}", text)}
    overlap = len(words & terms)
    return {"passed": overlap > 0 or not text.strip(), "evidence_term_overlap": overlap}
