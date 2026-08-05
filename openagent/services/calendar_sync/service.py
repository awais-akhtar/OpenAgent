"""Calendar provider sync service."""

from __future__ import annotations

from typing import Any

from ...calendar_sync import CalendarSyncManager


class CalendarSyncService:
    """Service facade for provider registration, sync jobs, and conflicts."""

    def __init__(self, calendar_sync: CalendarSyncManager) -> None:
        self.calendar_sync = calendar_sync

    def register_provider(
        self,
        *,
        provider: str,
        project: str = "default",
        owner: str = "local",
        base_url: str = "",
        account_ref: str = "",
        calendar_id: str = "",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.calendar_sync.register_provider(
            provider,
            project=project,
            owner=owner,
            base_url=base_url,
            account_ref=account_ref,
            calendar_id=calendar_id,
            config=config,
        )

    def providers(self, *, project: str = "default", owner: str = "") -> list[dict[str, Any]]:
        return self.calendar_sync.list_providers(project=project, owner=owner)

    def run_sync(
        self,
        provider_id: str,
        *,
        direction: str = "pull",
        conflict_policy: str = "latest_update",
        owner: str = "",
    ) -> dict[str, Any]:
        return self.calendar_sync.run_sync(
            provider_id,
            direction=direction,
            conflict_policy=conflict_policy,
            owner=owner,
        )

    def sync_job(self, sync_job_id: str, *, owner: str = "") -> dict[str, Any]:
        return self.calendar_sync.get_sync_job(sync_job_id, owner=owner)

    def conflicts(self, *, project: str = "default", status: str = "", owner: str = "") -> list[dict[str, Any]]:
        return self.calendar_sync.list_conflicts(project=project, status=status, owner=owner)

    def resolve_conflict(self, conflict_id: str, resolution: str, *, owner: str = "") -> dict[str, Any]:
        return self.calendar_sync.resolve_conflict(conflict_id, resolution, owner=owner)
