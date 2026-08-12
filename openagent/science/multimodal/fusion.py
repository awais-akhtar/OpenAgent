"""A typed local evidence container for scientific agent workflows."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .image import analyse_image_bytes, observe_with_gateway, summarise_image
from .sensor import parse_sensor_csv, summarise_sensor_rows
from .simulation_output import numeric_leaves
from .document import summarise_document


class MultimodalScientificContext:
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def add_text(self, name: str, text: str) -> dict[str, Any]:
        item = {"kind": "text", "name": str(name), "text": str(text)}
        self._items.append(item)
        return item

    def add_sensor_csv(self, name: str, text: str) -> dict[str, Any]:
        rows = parse_sensor_csv(text)
        item = {"kind": "sensor", "name": str(name), "summary": summarise_sensor_rows(rows), "rows": len(rows)}
        self._items.append(item)
        return item

    def add_image(self, name: str, values: Iterable[float], *, width: int = 0, height: int = 0) -> dict[str, Any]:
        item = {"kind": "image", "name": str(name), "summary": summarise_image(values, width=width, height=height)}
        self._items.append(item)
        return item

    def add_image_bytes(
        self,
        name: str,
        data: bytes,
        *,
        media_type: str = "image/png",
        gateway: Any = None,
        vlm_model: str = "",
        provider_id: str = "auto",
        allow_external: bool = False,
    ) -> dict[str, Any]:
        local = analyse_image_bytes(data, name=name)
        vlm = observe_with_gateway(
            data,
            gateway=gateway,
            model=vlm_model,
            name=name,
            media_type=media_type,
            provider_id=provider_id,
            allow_external=allow_external,
        ) if vlm_model else {"status": "not-requested", "observations": [], "reason": "No vision model was selected."}
        item = {"kind": "image", "name": str(name), "local_analysis": local, "vlm": vlm}
        self._items.append(item)
        return item

    def add_simulation(self, name: str, value: Any) -> dict[str, Any]:
        parsed = json.loads(value) if isinstance(value, str) else value
        item = {"kind": "simulation", "name": str(name), "numeric_outputs": numeric_leaves(parsed)}
        self._items.append(item)
        return item

    def add_document(self, name: str, data: bytes, *, media_type: str = "") -> dict[str, Any]:
        item = {"kind": "document", "name": str(name), "summary": summarise_document(data, name=name, media_type=media_type)}
        self._items.append(item)
        return item

    def items(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._items]

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for item in self._items:
            by_kind[item["kind"]] = by_kind.get(item["kind"], 0) + 1
        return {
            "items": self.items(),
            "count": len(self._items),
            "by_kind": by_kind,
            "privacy_boundary": "Evidence stays in this local context object until the caller explicitly stores it.",
        }
