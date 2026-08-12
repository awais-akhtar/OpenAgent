"""Dimensional unit checks with an optional Pint backend."""

from __future__ import annotations

from typing import Any, Mapping

from ..models import ScientificFinding


KNOWN_UNITS = {
    "", "m", "mm", "cm", "m2", "m3", "s", "min", "h", "kg", "g", "kg/m3",
    "N", "kN", "Pa", "kPa", "MPa", "GPa", "W", "kW", "W/mK", "K", "C",
    "J", "kJ", "mol", "A", "V", "Hz", "%",
}


def check_units(values: Mapping[str, Any]) -> ScientificFinding:
    try:
        import pint
    except ImportError:
        pint = None
    unknown: dict[str, str] = {}
    mismatched: dict[str, dict[str, str]] = {}
    checked_dimensions = 0
    registry = pint.UnitRegistry() if pint is not None else None
    for key, value in values.items():
        unit, expected = _unit_pair(str(key), value)
        if registry is None:
            if unit not in KNOWN_UNITS:
                unknown[str(key)] = unit
            continue
        try:
            actual_quantity = registry.Unit(_pint_unit(unit))
            expected_quantity = registry.Unit(_pint_unit(expected)) if expected else None
            if expected_quantity is not None:
                checked_dimensions += 1
                if actual_quantity.dimensionality != expected_quantity.dimensionality:
                    mismatched[str(key)] = {"unit": unit, "expected": expected}
        except Exception:
            unknown[str(key)] = unit
    passed = not unknown and not mismatched
    return ScientificFinding(
        check="units",
        passed=passed,
        message="Units are dimensionally consistent." if passed and registry is not None else ("Units are recognised." if passed else "Some units need explicit review."),
        severity="error" if not passed else "info",
        evidence={"unknown": unknown, "mismatched": mismatched, "checked": len(values), "dimensional_checks": checked_dimensions, "engine": "pint" if registry is not None else "known-unit-fallback"},
    )


def _unit_pair(name: str, value: Any) -> tuple[str, str]:
    if isinstance(value, Mapping):
        return str(value.get("unit", "")), str(value.get("expected", ""))
    unit = str(value)
    lowered = name.lower()
    expectations = (
        (("stress", "pressure"), "Pa"),
        (("density",), "kg/m3"),
        (("mass",), "kg"),
        (("area",), "m2"),
        (("force", "load"), "N"),
        (("energy",), "J"),
        (("power",), "W"),
        (("temperature",), "K"),
        (("width", "height", "length", "span", "deflection"), "m"),
    )
    expected = next((dimension for names, dimension in expectations if any(item in lowered for item in names)), "")
    return unit, expected


def _pint_unit(unit: str) -> str:
    aliases = {"": "dimensionless", "m2": "meter**2", "m3": "meter**3", "kg/m3": "kilogram/meter**3", "W/mK": "watt/(meter*kelvin)", "C": "degC", "%": "percent"}
    if unit in aliases:
        return aliases[unit]
    return unit.replace("m3", "meter**3").replace("m2", "meter**2")
