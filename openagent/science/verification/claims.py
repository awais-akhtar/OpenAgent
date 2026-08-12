"""Trace scientific claims to explicit local results or cited sources."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..models import ScientificFinding


def check_claim_evidence(
    claims: Iterable[Mapping[str, Any]],
    evidence: Mapping[str, Any] | Iterable[str],
) -> ScientificFinding:
    records = [dict(item) for item in claims]
    available = set(evidence) if isinstance(evidence, Mapping) else {str(item) for item in evidence}
    unsupported: list[dict[str, Any]] = []
    for index, claim in enumerate(records):
        claim_id = str(claim.get("id") or f"claim:{index + 1}")
        text = str(claim.get("text", "")).strip()
        references = claim.get("evidence_ids", [])
        if not isinstance(references, (list, tuple)):
            references = []
        reference_ids = [str(item) for item in references if str(item).strip()]
        missing = [item for item in reference_ids if item not in available]
        if not text or not reference_ids or missing:
            unsupported.append({"claim_id": claim_id, "missing_evidence_ids": missing, "has_text": bool(text), "has_references": bool(reference_ids)})
    passed = not unsupported
    return ScientificFinding(
        check="claim_evidence_provenance",
        passed=passed,
        message="Every claim points to available evidence." if passed else "One or more claims lack traceable evidence.",
        severity="error" if not passed else "info",
        evidence={"claims_checked": len(records), "evidence_available": len(available), "unsupported": unsupported},
    )
