"""Validation helpers for compact materials datasets."""

from __future__ import annotations

from typing import Any, Iterable


def validate_records(records: Iterable[dict[str, Any]], *, target: str) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or target not in record:
            continue
        numeric = {str(key): float(value) for key, value in record.items() if _number(value)}
        if numeric:
            validated.append({**record, **numeric})
    if not validated:
        raise ValueError(f"no usable records contain numeric target '{target}'")
    return validated


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
