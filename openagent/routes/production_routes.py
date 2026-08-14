"""Production runtime route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract


ROUTE_GROUP = "production"
ROUTES = (
    ("GET", "/api/production/status"),
    ("GET", "/api/production/policies"),
    ("GET", "/api/production/queue"),
    ("GET", "/api/production/audit-anchors"),
    ("GET", "/api/runtime-tools/status"),
    ("POST", "/api/production/policies"),
    ("POST", "/api/production/policies/evaluate"),
    ("POST", "/api/production/queue"),
    ("POST", "/api/production/queue/run-next"),
    ("POST", "/api/production/audit-anchor"),
    ("POST", "/api/production/vector/indexes"),
    ("POST", "/api/production/vector/upsert"),
    ("POST", "/api/production/vector/search"),
    ("POST", "/api/production/semantic/science"),
    ("POST", "/api/production/rag/rerank"),
    ("POST", "/api/production/oidc/validate"),
    ("POST", "/api/production/crypto/encrypt"),
    ("POST", "/api/production/crypto/decrypt"),
    ("POST", "/api/production/connectors/adapters"),
    ("POST", "/api/production/connectors/poll"),
    ("POST", "/api/production/automation/desktop"),
    ("POST", "/api/production/automation/browser"),
    ("POST", "/api/production/packages/app"),
    ("POST", "/api/production/marketplace/agents"),
    ("POST", "/api/production/gpu/plan"),
    ("POST", "/api/production/finetune/jobs"),
    ("POST", "/api/production/proactive/jobs"),
    ("POST", "/api/production/proactive/run"),
    ("POST", "/api/production/federated/rounds"),
    ("POST", "/api/production/sync/devices"),
    ("POST", "/api/production/sync/bundles"),
    ("POST", "/api/production/sdk/publish"),
    ("POST", "/api/production/sandbox/run"),
    ("POST", "/api/runtime-tools/browser-test"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
