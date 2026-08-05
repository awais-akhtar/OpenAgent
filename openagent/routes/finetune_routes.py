"""Fine-tuning and adapter route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "finetune"
ROUTES = (
    ("GET", "/api/finetune/status"),
    ("GET", "/api/finetune/datasets"),
    ("POST", "/api/finetune/datasets"),
    ("POST", "/api/finetune/examples"),
    ("GET", "/api/finetune/jobs"),
    ("POST", "/api/finetune/jobs"),
    ("GET", "/api/finetune/adapters"),
    ("POST", "/api/finetune/adapters"),
    ("POST", "/api/finetune/adapters/activate"),
    ("GET", "/api/finetune/templates"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
