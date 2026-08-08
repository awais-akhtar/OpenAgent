"""Small sensor-table helpers with no network or database side effects."""

from __future__ import annotations

import csv
from io import StringIO
import math
from statistics import fmean
from typing import Any, Iterable


def parse_sensor_csv(text: str) -> list[dict[str, float]]:
    reader = csv.DictReader(StringIO(text))
    if not reader.fieldnames:
        raise ValueError("sensor CSV needs a header row")
    rows: list[dict[str, float]] = []
    for row in reader:
        converted: dict[str, float] = {}
        for key, value in row.items():
            if key and value not in {None, ""}:
                try:
                    converted[key.strip()] = float(str(value).strip())
                except ValueError:
                    continue
        if converted:
            rows.append(converted)
    if not rows:
        raise ValueError("sensor CSV contains no numeric observations")
    return rows


def summarise_sensor_rows(rows: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if math.isfinite(float(value)):
                columns.setdefault(key, []).append(float(value))
    return {
        key: {
            "count": float(len(values)),
            "mean": fmean(values),
            "minimum": min(values),
            "maximum": max(values),
            "last": values[-1],
        }
        for key, values in columns.items()
        if values
    }
