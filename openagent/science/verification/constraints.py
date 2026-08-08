"""Verification adapters for engineering constraint records."""

from __future__ import annotations

from typing import Iterable, Mapping

from ..models import ScientificFinding


def check_constraints(constraints: Iterable[Mapping[str, object]]) -> ScientificFinding:
    records = [dict(item) for item in constraints]
    failed = [item for item in records if not bool(item.get("passed"))]
    return ScientificFinding(
        check="design_constraints",
        passed=not failed,
        message="All design constraints pass." if not failed else "One or more design constraints fail.",
        severity="error" if failed else "info",
        evidence={"checked": len(records), "failed": failed},
    )
