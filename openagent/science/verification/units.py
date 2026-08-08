"""Conservative unit checks for common scalar engineering values."""

from __future__ import annotations

from typing import Mapping

from ..models import ScientificFinding


KNOWN_UNITS = {
    "", "m", "mm", "cm", "m2", "m3", "s", "min", "h", "kg", "g", "kg/m3",
    "N", "kN", "Pa", "kPa", "MPa", "GPa", "W", "kW", "W/mK", "K", "C",
    "J", "kJ", "mol", "A", "V", "Hz", "%",
}


def check_units(values: Mapping[str, str]) -> ScientificFinding:
    unknown = {str(key): str(unit) for key, unit in values.items() if str(unit) not in KNOWN_UNITS}
    return ScientificFinding(
        check="units",
        passed=not unknown,
        message="Units are recognised." if not unknown else "Some units need explicit review.",
        severity="error" if unknown else "info",
        evidence={"unknown": unknown, "checked": len(values)},
    )
