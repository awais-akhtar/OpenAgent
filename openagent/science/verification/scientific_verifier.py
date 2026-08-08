"""Composable scientific verification for agent and tool results."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..models import ScientificFinding, ScientificVerification
from .constraints import check_constraints
from .equations import check_residuals, check_symbolic_equations
from .numeric import check_finite
from .units import check_units


class ScientificVerifier:
    def verify(
        self,
        result: Mapping[str, Any],
        *,
        values: Iterable[float] | None = None,
        residuals: Iterable[float] | None = None,
        units: Mapping[str, str] | None = None,
        constraints: Iterable[Mapping[str, object]] | None = None,
        equations: Iterable[Mapping[str, Any] | str] | None = None,
    ) -> ScientificVerification:
        findings: list[ScientificFinding] = []
        if values is None:
            values = _numeric_values(result)
        findings.append(check_finite(values))
        if residuals is not None:
            findings.append(check_residuals(residuals))
        if equations is not None:
            findings.append(check_symbolic_equations(equations))
        if units is not None:
            findings.append(check_units(units))
        if constraints is not None:
            findings.append(check_constraints(constraints))
        passed = all(item.passed for item in findings)
        score = sum(1.0 for item in findings if item.passed) / max(1, len(findings))
        return ScientificVerification(passed=passed, findings=tuple(findings), score=score)


def _numeric_values(value: Any) -> list[float]:
    values: list[float] = []
    if isinstance(value, Mapping):
        for child in value.values():
            values.extend(_numeric_values(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            values.extend(_numeric_values(child))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        values.append(float(value))
    return values
