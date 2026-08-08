"""Finite-value and numerical-stability checks."""

from __future__ import annotations

import math
from typing import Iterable

from ..models import ScientificFinding


def check_finite(values: Iterable[float], *, name: str = "numeric_values") -> ScientificFinding:
    numbers = [float(value) for value in values]
    invalid = [value for value in numbers if not math.isfinite(value)]
    return ScientificFinding(
        check="finite_values",
        passed=not invalid,
        message="All values are finite." if not invalid else "The result contains non-finite values.",
        severity="error" if invalid else "info",
        evidence={"name": name, "count": len(numbers), "invalid_count": len(invalid)},
    )
