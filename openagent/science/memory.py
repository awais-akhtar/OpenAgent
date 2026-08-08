"""Typed local memory for scientific assumptions and experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ScientificMemoryRecord:
    category: str
    name: str
    value: Any
    unit: str | None = None
    source: str = "user"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScientificMemory:
    """A reviewable memory collection for equations, units and experiments."""

    def __init__(self) -> None:
        self._records: list[ScientificMemoryRecord] = []

    def remember(self, category: str, name: str, value: Any, *, unit: str | None = None, source: str = "user") -> ScientificMemoryRecord:
        record = ScientificMemoryRecord(category.strip(), name.strip(), value, unit, source)
        self._records.append(record)
        return record

    def search(self, query: str = "") -> list[ScientificMemoryRecord]:
        terms = {term for term in query.lower().split() if term}
        if not terms:
            return list(self._records)
        return [record for record in self._records if terms & {record.category.lower(), record.name.lower(), str(record.value).lower()}]

    def context(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]
