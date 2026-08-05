"""Backup, restore, and disaster recovery route contracts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from ..backup_restore import BackupRestoreManager
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "backup_restore"
ROUTES = (
    ("GET", "/api/backups"),
    ("POST", "/api/backups/create"),
    ("POST", "/api/backups/restore-dry-run"),
    ("POST", "/api/backups/restore"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class BackupRouteDependencies:
    backups: BackupRestoreManager
    read_json: Any


def handle_get(path: str, deps: BackupRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/backups":
        return JsonRouteResult({"backups": deps.backups.backups()})
    return None


def handle_post(path: str, deps: BackupRouteDependencies) -> JsonRouteResult | None:
    if path not in {"/api/backups/create", "/api/backups/restore-dry-run", "/api/backups/restore"}:
        return None
    payload = deps.read_json()
    try:
        if path == "/api/backups/create":
            backup = deps.backups.create_backup(
                project=str(payload.get("project", "default")).strip() or "default",
                include_media=bool(payload.get("include_media", True)),
            )
            return JsonRouteResult({"backup": backup}, HTTPStatus.CREATED)
        if path == "/api/backups/restore-dry-run":
            result = deps.backups.restore_dry_run(str(payload.get("path", "")).strip())
            return JsonRouteResult(result)
        result = deps.backups.restore(
            str(payload.get("path", "")).strip(),
            confirm=bool(payload.get("confirm")),
        )
        return JsonRouteResult(result)
    except (FileNotFoundError, ValueError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
