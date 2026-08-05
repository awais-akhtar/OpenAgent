"""Local web scraper route contracts."""

from __future__ import annotations

from ._contracts import not_handled, route_contract


ROUTE_GROUP = "scraper"
ROUTES = (
    ("GET", "/api/scraper/status"),
    ("GET", "/api/scraper/jobs"),
    ("POST", "/api/scraper/scrape"),
    ("POST", "/api/scraper/schedule"),
    ("POST", "/api/scraper/run-due"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


handle_get = not_handled
handle_post = not_handled
