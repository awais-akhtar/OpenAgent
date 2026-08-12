"""Reviewable citation and consistency checks for scientific reports."""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterable, Mapping


CitationResolver = Callable[[Mapping[str, str]], Mapping[str, Any]]


def check_citations(
    text: str,
    citations: Iterable[str | Mapping[str, Any]],
    *,
    resolver: CitationResolver | None = None,
    allow_external: bool = False,
) -> dict[str, object]:
    """Validate citation structure and optionally resolve it through an approved caller."""

    records = [_citation_record(item, index) for index, item in enumerate(citations)]
    references = set(re.findall(r"\[([^\]]+)\]", text))
    known_ids = {record["id"] for record in records}
    missing = sorted(reference for reference in references if reference not in known_ids)
    malformed = [record["id"] for record in records if not (record["doi"] or record["url"] or record["title"])]
    resolutions: list[dict[str, Any]] = []
    if resolver is not None and allow_external:
        for record in records:
            try:
                resolved = dict(resolver(record))
                resolutions.append({"id": record["id"], "verified": bool(resolved.get("verified")), "details": resolved})
            except Exception as exc:
                resolutions.append({"id": record["id"], "verified": False, "error": str(exc)})
    external_checked = bool(resolver is not None and allow_external)
    unresolved = [item["id"] for item in resolutions if not item["verified"]]
    passed = not missing and not malformed and (not external_checked or not unresolved)
    return {
        "passed": passed,
        "references_in_text": sorted(references),
        "records": records,
        "missing_records": missing,
        "malformed_records": malformed,
        "external_checked": external_checked,
        "external_verified": external_checked and not unresolved,
        "resolutions": resolutions,
        "privacy_boundary": (
            "The caller approved external citation resolution."
            if external_checked
            else "No external citation lookup was performed."
        ),
    }


def check_consistency(values: Mapping[str, float], *, tolerance: float = 1e-9) -> dict[str, object]:
    numeric = {str(key): float(value) for key, value in values.items()}
    finite = all(math.isfinite(value) for value in numeric.values())
    spread = max(numeric.values()) - min(numeric.values()) if numeric else 0.0
    return {"passed": finite and spread >= -tolerance, "finite": finite, "spread": spread}


def detect_unsupported_claims(
    claims: str | Iterable[Mapping[str, Any]],
    evidence: Iterable[str] | Mapping[str, Any],
) -> dict[str, object]:
    """Require each structured claim to name available evidence.

    Plain text is retained only for backward compatibility and is marked for
    review because lexical overlap cannot establish scientific support.
    """

    available = set(evidence) if isinstance(evidence, Mapping) else {str(item) for item in evidence}
    if isinstance(claims, str):
        return {
            "passed": not claims.strip(),
            "claims_checked": 0,
            "unsupported": [] if not claims.strip() else [{"text": claims.strip(), "reason": "structured evidence_ids required"}],
            "method": "structured-provenance",
        }
    unsupported = []
    checked = 0
    for index, claim in enumerate(claims):
        checked += 1
        text = str(claim.get("text", "")).strip()
        references = claim.get("evidence_ids", [])
        references = [str(item) for item in references] if isinstance(references, (list, tuple)) else []
        missing = [item for item in references if item not in available]
        if not text or not references or missing:
            unsupported.append({"claim": index + 1, "text": text, "missing_evidence_ids": missing})
    return {"passed": not unsupported, "claims_checked": checked, "unsupported": unsupported, "method": "structured-provenance"}


def _citation_record(value: str | Mapping[str, Any], index: int) -> dict[str, str]:
    if isinstance(value, Mapping):
        identifier = str(value.get("id") or value.get("doi") or value.get("url") or f"citation-{index + 1}").strip()
        return {
            "id": identifier,
            "title": str(value.get("title", "")).strip(),
            "doi": _normalise_doi(str(value.get("doi", ""))),
            "url": str(value.get("url", "")).strip(),
        }
    raw = str(value).strip()
    doi = _normalise_doi(raw)
    url = raw if raw.startswith(("https://", "http://")) else ""
    return {"id": raw or f"citation-{index + 1}", "title": "" if doi or url else raw, "doi": doi, "url": url}


def _normalise_doi(value: str) -> str:
    candidate = value.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    return candidate if re.fullmatch(r"10\.\d{4,9}/\S+", candidate, flags=re.IGNORECASE) else ""
