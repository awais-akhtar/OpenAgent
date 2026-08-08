"""Flatten and summarise JSON-like simulation output."""

from __future__ import annotations

import math
from typing import Any


def numeric_leaves(value: Any, *, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(numeric_leaves(child, prefix=f"{prefix}{key}."))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(numeric_leaves(child, prefix=f"{prefix}{index}."))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        result[prefix.rstrip(".")] = float(value)
    return result
