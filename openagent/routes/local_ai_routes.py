"""Local AI and model operations route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "local_ai"
ROUTES = (
    ("GET", "/api/models"),
    ("GET", "/api/model-policy"),
    ("GET", "/api/settings/runtime"),
    ("POST", "/api/settings/runtime"),
    ("POST", "/api/models/discover"),
    ("GET", "/api/local-ai/jobs"),
    ("POST", "/api/local-ai/jobs"),
    ("GET", "/api/local-ai/jobs/{id}"),
    ("GET", "/api/local-ai/status"),
    ("GET", "/api/local-ai/setup/status"),
    ("GET", "/api/local-ai/smooth-runtime"),
    ("GET", "/api/local-ai/models"),
    ("GET", "/api/local-ai/models/{id}/probe"),
    ("GET", "/api/local-ai/hardware"),
    ("GET", "/api/local-ai/adapters"),
    ("GET", "/api/local-ai/marketplace"),
    ("POST", "/api/local-ai/models/register"),
    ("POST", "/api/local-ai/models/download"),
    ("POST", "/api/local-ai/models/setup/job"),
    ("POST", "/api/local-ai/llamacpp/register"),
    ("POST", "/api/local-ai/models/auto-quantize"),
    ("POST", "/api/local-ai/models/delete"),
    ("POST", "/api/local-ai/models/{id}/dependencies"),
    ("POST", "/api/local-ai/models/{id}/tools"),
    ("POST", "/api/local-ai/benchmark"),
    ("POST", "/api/local-ai/cluster-plan"),
    ("POST", "/api/local-ai/models/quantize"),
    ("POST", "/api/local-ai/models/setup-chat"),
    ("POST", "/api/local-ai/ollama/pull"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
