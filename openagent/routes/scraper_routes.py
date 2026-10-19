"""Local web scraper HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "scraper"
ROUTES = (
    ("GET", "/api/scraper/status"),
    ("GET", "/api/scraper/jobs"),
    ("GET", "/api/scraper/history"),
    ("POST", "/api/scraper/scrape"),
    ("POST", "/api/scraper/schedule"),
    ("POST", "/api/scraper/run-due"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ScraperRouteDependencies:
    scraper: Any
    first: Callable[[dict[str, list[str]], str], str | None]
    parse_bool: Callable[..., bool]
    read_json: Callable[[], dict[str, Any]]


def handle_get(path: str, query: str, deps: ScraperRouteDependencies) -> JsonRouteResult | None:
    if path not in {"/api/scraper/status", "/api/scraper/jobs", "/api/scraper/history"}:
        return None
    params = parse_qs(query)
    project = deps.first(params, "project") or "default"
    if path == "/api/scraper/status":
        return JsonRouteResult(deps.scraper.status(project=project))
    if path == "/api/scraper/history":
        try:
            limit = int(deps.first(params, "limit") or 100)
        except (TypeError, ValueError):
            limit = 100
        return JsonRouteResult({"results": deps.scraper.get_scrape_history(
            deps.first(params, "url") or "",
            project=project,
            limit=limit,
        )})
    return JsonRouteResult({"jobs": deps.scraper.list_scheduled(project=project)})


def handle_post(path: str, deps: ScraperRouteDependencies) -> JsonRouteResult | None:
    if path not in {"/api/scraper/scrape", "/api/scraper/schedule", "/api/scraper/run-due"}:
        return None
    payload = deps.read_json()
    if path == "/api/scraper/run-due":
        project = str(payload.get("project", "default")).strip() or "default"
        return JsonRouteResult(deps.scraper.run_due_jobs(project=project))

    url = str(payload.get("url", "")).strip()
    if not url:
        return JsonRouteResult({"error": "url is required"}, HTTPStatus.BAD_REQUEST)
    project = str(payload.get("project", "default")).strip() or "default"
    if path == "/api/scraper/scrape":
        try:
            result = deps.scraper.scrape_url(
                url, project=project,
                respect_robots=deps.parse_bool(payload.get("respect_robots"), default=True),
                ingest_to_knowledge=deps.parse_bool(payload.get("ingest"), default=True),
                allow_external=deps.parse_bool(payload.get("allow_external"), default=False),
            )
        except (ValueError, PermissionError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)

    try:
        job = deps.scraper.schedule_scrape(
            url, project=project,
            interval_hours=float(payload.get("interval_hours", 24.0)),
            label=str(payload.get("label", "")),
            allow_external=deps.parse_bool(payload.get("allow_external"), default=False),
        )
    except (ValueError, PermissionError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return JsonRouteResult({"job": job}, HTTPStatus.CREATED)
