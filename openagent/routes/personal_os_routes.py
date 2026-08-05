"""Personal OS route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract

ROUTE_GROUP = "personal_os"
ROUTES = (
    ("GET", "/api/personal-os/status"),
    ("GET", "/api/personal-os/presets"),
    ("POST", "/api/personal-os/presets"),
    ("GET", "/api/personal-os/signatures"),
    ("POST", "/api/personal-os/signatures"),
    ("GET", "/api/personal-os/drafts"),
    ("POST", "/api/personal-os/drafts"),
    ("GET", "/api/personal-os/drafts/{id}/versions"),
    ("GET", "/api/personal-os/notes"),
    ("POST", "/api/personal-os/notes"),
    ("GET", "/api/personal-os/gallery"),
    ("POST", "/api/personal-os/gallery"),
    ("POST", "/api/personal-os/gallery/favorite"),
    ("GET", "/api/personal-os/compare"),
    ("POST", "/api/personal-os/compare/start"),
    ("POST", "/api/personal-os/compare/vote"),
    ("GET", "/api/personal-os/export"),
    ("POST", "/api/personal-os/import"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
