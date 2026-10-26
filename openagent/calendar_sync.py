"""External calendar synchronization hardening.

Independent implementation of a provider-based sync layer on top of
CalendarSystem: register a sync provider (a manual/local ICS feed or a real
HTTP(S) ICS URL), pull remote events into local calendar_events with a
stable UID mapping, detect genuine conflicts (both sides changed since the
last successful sync), and resolve them with an explicit policy.

This module never mutates CalendarSystem's own tables directly; it always
goes through CalendarSystem's public create_event/get_event/update_event/
list_events methods so calendar business rules (reminders, audit events,
etc.) stay centralized in one place.
"""

from __future__ import annotations

import base64
import hashlib
import json
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urljoin
from uuid import uuid4

from .calendar_system import CalendarSystem, event_to_ical, parse_ical
from .egress import EgressClient, EgressError
from .models import utc_now
from .storage import OpenAgentStore


class CalendarSyncError(RuntimeError):
    """Raised when a calendar sync operation is invalid or cannot proceed."""


SUPPORTED_PROVIDERS: tuple[str, ...] = ("manual", "caldav_http")
CONFLICT_RESOLUTIONS: tuple[str, ...] = ("server_wins", "client_wins", "latest_update")
CALDAV_SUCCESS_CODES: tuple[int, ...] = (200, 201, 204)


def _fingerprint(title: str, starts_at: str, ends_at: str, description: str = "", location: str = "") -> str:
    raw = "|".join([title or "", starts_at or "", ends_at or "", description or "", location or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _event_fingerprint(event: dict[str, Any]) -> str:
    return _fingerprint(
        event.get("title", ""),
        event.get("starts_at", ""),
        event.get("ends_at", ""),
        event.get("description", ""),
        event.get("location", ""),
    )


def _remote_identity(event: dict[str, Any]) -> str:
    uid = str(event.get("uid") or "").strip()
    recurrence_id = str(event.get("recurrence_id") or "").strip()
    return f"{uid}#{recurrence_id}" if uid and recurrence_id else uid


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


class CalendarSyncManager:
    """Owns sync provider registration, pull/push runs, and conflict resolution."""

    def __init__(self, store: OpenAgentStore, calendar_system: CalendarSystem) -> None:
        self.store = store
        self.calendar_system = calendar_system
        self.egress = EgressClient()
        self.initialize()

    def initialize(self) -> None:
        self.store.initialize()

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    def register_provider(
        self,
        provider: str,
        *,
        project: str = "default",
        owner: str = "local",
        base_url: str = "",
        account_ref: str = "",
        calendar_id: str = "",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider = (provider or "").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"provider must be one of {SUPPORTED_PROVIDERS}")
        provider_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO calendar_sync_providers
                    (id, project, provider, base_url, account_ref, calendar_id, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider_id,
                    project or "default",
                    provider,
                    base_url,
                    account_ref,
                    calendar_id,
                    json.dumps(config or {}),
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE calendar_sync_providers SET owner = ? WHERE id = ?",
                (owner or "local", provider_id),
            )
        return self.get_provider(provider_id)

    def _provider_row(self, connection: Any, provider_id: str, *, owner: str = "") -> Any:
        owner = (owner or "").strip()
        if owner:
            row = connection.execute(
                "SELECT * FROM calendar_sync_providers WHERE id = ? AND owner = ?",
                (provider_id, owner),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM calendar_sync_providers WHERE id = ?", (provider_id,)
            ).fetchone()
        if row is None:
            raise KeyError("provider not found")
        return row

    def _row_to_provider(self, row: Any) -> dict[str, Any]:
        return {
            "provider_id": row["id"],
            "project": row["project"],
            "owner": row["owner"],
            "provider": row["provider"],
            "base_url": row["base_url"],
            "account_ref": row["account_ref"],
            "calendar_id": row["calendar_id"],
            "sync_token": row["sync_token"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_provider(self, provider_id: str, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            return self._row_to_provider(self._provider_row(connection, provider_id, owner=owner))

    def list_providers(self, *, project: str = "default", owner: str = "") -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project or "default"]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM calendar_sync_providers WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at DESC, rowid DESC",
                args,
            ).fetchall()
            return [self._row_to_provider(row) for row in rows]

    def delete_provider(self, provider_id: str, *, owner: str = "") -> dict[str, Any]:
        """Remove a calendar provider registration and its sync state."""
        provider_id = str(provider_id or "").strip()
        if not provider_id:
            raise ValueError("provider_id is required.")
        with self.store.session() as connection:
            row = self._provider_row(connection, provider_id, owner=owner)
            snapshot = self._row_to_provider(row)
            connection.execute(
                "DELETE FROM calendar_sync_providers WHERE id = ?", (provider_id,)
            )
        return {"deleted": True, "provider_id": provider_id, "provider": snapshot["provider"]}

    def reauthenticate_provider(
        self,
        provider_id: str,
        *,
        owner: str = "",
        config: dict[str, Any] | None = None,
        account_ref: str = "",
    ) -> dict[str, Any]:
        """Update credentials / config for an existing provider (re-auth flow)."""
        provider_id = str(provider_id or "").strip()
        if not provider_id:
            raise ValueError("provider_id is required.")
        now = utc_now()
        with self.store.session() as connection:
            self._provider_row(connection, provider_id, owner=owner)
            sets: list[str] = ["updated_at = ?", "status = 'active'"]
            params: list[Any] = [now]
            if config is not None:
                sets.append("config_json = ?")
                params.append(json.dumps(config))
            if account_ref:
                sets.append("account_ref = ?")
                params.append(account_ref)
            # Reset the sync token so the next sync does a full pull with new credentials.
            sets.append("sync_token = ''")
            params.append(provider_id)
            connection.execute(
                f"UPDATE calendar_sync_providers SET {', '.join(sets)} WHERE id = ?",
                params,
            )
        return self.get_provider(provider_id, owner=owner)

    # ------------------------------------------------------------------
    # Sync execution
    # ------------------------------------------------------------------

    def run_sync(
        self,
        provider_id: str,
        *,
        direction: str = "pull",
        conflict_policy: str = "latest_update",
        owner: str = "",
        allow_external: bool = False,
        allow_local_transport: bool = False,
    ) -> dict[str, Any]:
        with self.store.session() as connection:
            provider_row = self._provider_row(connection, provider_id, owner=owner)
        project = provider_row["project"]
        job_id = uuid4().hex
        created_at = utc_now()
        pulled = pushed = conflicts = 0
        state = "completed"
        error = ""
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO calendar_sync_jobs
                    (id, provider_id, project, direction, conflict_policy, state, pulled, pushed, conflicts, error,
                     created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, '', ?, '')
                """,
                (job_id, provider_id, project, direction, conflict_policy, "running", created_at),
            )
        try:
            if provider_row["provider"] == "caldav_http" and not allow_external:
                raise CalendarSyncError("CalDAV network sync requires allow_external=true for this run")
            if direction in ("pull", "bidirectional"):
                pulled, conflicts = self._pull(
                    provider_row,
                    conflict_policy,
                    allow_external=allow_external,
                    allow_local_transport=allow_local_transport,
                )
            if direction in ("push", "bidirectional"):
                pushed = self._push(
                    provider_row,
                    allow_external=allow_external,
                    allow_local_transport=allow_local_transport,
                )
            if direction not in ("pull", "push", "bidirectional"):
                raise CalendarSyncError(f"unsupported direction '{direction}'")
        except CalendarSyncError as exc:
            state = "failed"
            error = str(exc)
        with self.store.session() as connection:
            connection.execute(
                "UPDATE calendar_sync_jobs SET state = ?, pulled = ?, pushed = ?, conflicts = ?, error = ?, completed_at = ? WHERE id = ?",
                (state, pulled, pushed, conflicts, error, utc_now(), job_id),
            )
            connection.execute(
                "UPDATE calendar_sync_providers SET status = ?, updated_at = ? WHERE id = ?",
                ("connected" if state == "completed" else "error", utc_now(), provider_id),
            )
        return {
            "sync_job_id": job_id,
            "state": state,
            "pulled": pulled,
            "pushed": pushed,
            "conflicts": conflicts,
            "error": error,
        }

    def get_sync_job(self, job_id: str, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            if owner:
                row = connection.execute(
                    """
                    SELECT j.* FROM calendar_sync_jobs j
                    JOIN calendar_sync_providers p ON p.id = j.provider_id
                    WHERE j.id = ? AND p.owner = ?
                    """,
                    (job_id, owner),
                ).fetchone()
            else:
                row = connection.execute("SELECT * FROM calendar_sync_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("sync job not found")
            return {
                "sync_job_id": row["id"],
                "provider_id": row["provider_id"],
                "state": row["state"],
                "direction": row["direction"],
                "pulled": row["pulled"],
                "pushed": row["pushed"],
                "conflicts": row["conflicts"],
                "error": row["error"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
            }

    def _pull(
        self,
        provider_row: Any,
        conflict_policy: str,
        *,
        allow_external: bool,
        allow_local_transport: bool,
    ) -> tuple[int, int]:
        remote_events, remote_meta = self._fetch_remote_events(
            provider_row,
            allow_external=allow_external,
            allow_local_transport=allow_local_transport,
        )
        project = provider_row["project"]
        calendar_id = provider_row["calendar_id"] or ""
        pulled = 0
        conflicts = 0
        seen_remote_uids: set[str] = set()
        for remote in remote_events:
            remote_uid = _remote_identity(remote)
            if not remote_uid:
                continue
            seen_remote_uids.add(remote_uid)
            remote_hash = _event_fingerprint(remote)
            with self.store.session() as connection:
                mapping = connection.execute(
                    "SELECT * FROM calendar_sync_uid_map WHERE provider_id = ? AND remote_uid = ?",
                    (provider_row["id"], remote_uid),
                ).fetchone()
            if mapping is None:
                pulled += self._materialize_new_remote_event(
                    provider_row,
                    project,
                    calendar_id,
                    remote,
                    remote_hash,
                    remote_etag=str(remote.get("remote_etag", "")),
                    remote_href=str(remote.get("remote_href", "")),
                )
                continue
            if mapping["remote_hash"] == remote_hash:
                self._update_remote_metadata(mapping["id"], str(remote.get("remote_etag", "")), str(remote.get("remote_href", "")))
                continue  # idempotent no-op: remote hasn't changed since the last sync
            applied, conflicted = self._apply_remote_change(
                provider_row,
                mapping,
                remote,
                remote_hash,
                conflict_policy,
                remote_etag=str(remote.get("remote_etag", "")),
                remote_href=str(remote.get("remote_href", "")),
            )
            pulled += applied
            conflicts += conflicted
        config = json.loads(provider_row["config_json"] or "{}")
        deleted = self._apply_remote_deletes(
            provider_row,
            seen_remote_uids,
            [str(item) for item in remote_meta.get("deleted_hrefs", []) if item],
            treat_missing=bool(remote_meta.get("report_based") and config.get("treat_missing_as_delete")),
        )
        pulled += deleted
        return pulled, conflicts

    def _fetch_remote_events(
        self,
        provider_row: Any,
        *,
        allow_external: bool,
        allow_local_transport: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        kind = provider_row["provider"]
        config = json.loads(provider_row["config_json"] or "{}")
        if kind == "manual":
            ics_text = str(config.get("ics_seed", ""))
            return parse_ical(ics_text) if ics_text else [], {}
        if kind == "caldav_http":
            if config.get("sync_method") == "report" or config.get("use_report"):
                return self._caldav_report_events(
                    provider_row,
                    allow_external=allow_external,
                    allow_local_transport=allow_local_transport,
                )
            ics_text = self._fetch_caldav_ics(
                provider_row,
                allow_external=allow_external,
                allow_local_transport=allow_local_transport,
            )
            return parse_ical(ics_text) if ics_text else [], {}
        raise CalendarSyncError(f"unsupported provider kind '{kind}'")

    def _fetch_caldav_ics(
        self,
        provider_row: Any,
        *,
        allow_external: bool,
        allow_local_transport: bool,
    ) -> str:
        config = json.loads(provider_row["config_json"] or "{}")
        base_url = str(provider_row["base_url"] or "").strip()
        if not base_url:
            raise CalendarSyncError("CalDAV provider has no base URL configured")
        try:
            response = self.egress.fetch(
                base_url,
                approved=allow_external,
                allow_loopback=allow_local_transport,
                purpose="CalDAV calendar pull",
                headers=self._caldav_headers(config, {"Accept": "text/calendar, application/octet-stream"}),
                timeout_seconds=float(config.get("timeout_seconds") or 20),
                max_response_bytes=int(config.get("max_response_bytes") or 20 * 1024 * 1024),
            )
        except (EgressError, PermissionError) as exc:
            raise CalendarSyncError(f"CalDAV pull blocked or failed: {exc}") from exc
        if not 200 <= response.status < 300:
            raise CalendarSyncError(f"CalDAV pull returned HTTP {response.status}")
        return response.body.decode("utf-8", errors="replace")

    def _caldav_report_events(
        self,
        provider_row: Any,
        *,
        allow_external: bool,
        allow_local_transport: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        config = json.loads(provider_row["config_json"] or "{}")
        collection_url = str(config.get("calendar_collection_url") or provider_row["base_url"]).rstrip("/")
        if not collection_url:
            raise CalendarSyncError("CalDAV provider has no collection URL configured")
        headers = self._caldav_headers(config, {"Depth": str(config.get("report_depth", "1")), "Content-Type": "application/xml; charset=utf-8"})
        sync_token = str(config.get("sync_token") or provider_row["sync_token"] or "").strip()
        if sync_token:
            body = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<D:sync-collection xmlns:D="DAV:">'
                f"<D:sync-token>{sync_token}</D:sync-token>"
                "<D:sync-level>1</D:sync-level>"
                "<D:prop><D:getetag/><C:calendar-data xmlns:C=\"urn:ietf:params:xml:ns:caldav\"/></D:prop>"
                "</D:sync-collection>"
            )
        else:
            body = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">'
                "<D:prop><D:getetag/><C:calendar-data/></D:prop>"
                '<C:filter><C:comp-filter name="VCALENDAR"><C:comp-filter name="VEVENT"/></C:comp-filter></C:filter>'
                "</C:calendar-query>"
            )
        try:
            response = self.egress.fetch(
                collection_url,
                approved=allow_external,
                allow_loopback=allow_local_transport,
                purpose="CalDAV collection REPORT",
                method="REPORT",
                body=body.encode("utf-8"),
                headers=headers,
                timeout_seconds=float(config.get("timeout_seconds") or 20),
                max_response_bytes=int(config.get("max_response_bytes") or 20 * 1024 * 1024),
            )
        except (EgressError, PermissionError) as exc:
            raise CalendarSyncError(f"CalDAV REPORT failed: {exc}") from exc
        if response.status not in {200, 207}:
            raise CalendarSyncError(f"CalDAV REPORT returned HTTP {response.status}")
        raw_xml = response.body.decode("utf-8", errors="replace")
        events: list[dict[str, Any]] = []
        deleted_hrefs: list[str] = []
        next_sync_token = ""
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as exc:
            raise CalendarSyncError(f"CalDAV REPORT returned invalid XML: {exc}") from exc
        for element in root.iter():
            if _xml_local_name(element.tag) == "sync-token" and element.text:
                next_sync_token = element.text.strip()
        for response in [el for el in root.iter() if _xml_local_name(el.tag) == "response"]:
            href = ""
            etag = ""
            calendar_data = ""
            status_text = ""
            for child in response.iter():
                name = _xml_local_name(child.tag)
                if name == "href" and child.text:
                    href = child.text.strip()
                elif name == "getetag" and child.text:
                    etag = child.text.strip().strip('"')
                elif name == "calendar-data" and child.text:
                    calendar_data = child.text
                elif name == "status" and child.text:
                    status_text = child.text
            if " 404 " in f" {status_text} " and href:
                deleted_hrefs.append(href)
                continue
            for event in parse_ical(calendar_data) if calendar_data else []:
                event["remote_href"] = href
                event["remote_etag"] = etag
                events.append(event)
        if next_sync_token:
            self._store_sync_token(provider_row["id"], next_sync_token)
        return events, {"report_based": True, "sync_token": next_sync_token, "deleted_hrefs": deleted_hrefs}

    def _store_sync_token(self, provider_id: str, sync_token: str) -> None:
        with self.store.session() as connection:
            connection.execute(
                "UPDATE calendar_sync_providers SET sync_token = ?, updated_at = ? WHERE id = ?",
                (sync_token, utc_now(), provider_id),
            )

    def _materialize_new_remote_event(
        self,
        provider_row: Any,
        project: str,
        calendar_id: str,
        remote: dict[str, Any],
        remote_hash: str,
        *,
        remote_etag: str = "",
        remote_href: str = "",
    ) -> int:
        created = self.calendar_system.create_event(
            remote.get("title") or "Untitled",
            remote.get("starts_at", ""),
            remote.get("ends_at", "") or remote.get("starts_at", ""),
            project=project,
            calendar_id=calendar_id,
            description=remote.get("description", ""),
            location=remote.get("location", ""),
            rrule=remote.get("rrule", ""),
            metadata={
                "sync_provider_id": provider_row["id"],
                "remote_uid": remote.get("uid", ""),
                "remote_identity": _remote_identity(remote),
                "remote_href": remote_href,
                "remote_etag": remote_etag,
                "recurrence_id": remote.get("recurrence_id", ""),
            },
        )
        local_hash = _event_fingerprint(created)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO calendar_sync_uid_map
                    (id, provider_id, local_event_id, remote_uid, remote_hash, local_hash, remote_etag, remote_href, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (uuid4().hex, provider_row["id"], created["id"], _remote_identity(remote), remote_hash, local_hash, remote_etag, remote_href, utc_now()),
            )
        return 1

    def _apply_remote_change(
        self,
        provider_row: Any,
        mapping: Any,
        remote: dict[str, Any],
        remote_hash: str,
        conflict_policy: str,
        *,
        remote_etag: str = "",
        remote_href: str = "",
    ) -> tuple[int, int]:
        try:
            local_event = self.calendar_system.get_event(mapping["local_event_id"])
        except KeyError:
            # The local event disappeared since the last sync. Drop the stale mapping
            # FIRST (there is a UNIQUE(provider_id, remote_uid) constraint) and then
            # recreate the event from the remote copy.
            with self.store.session() as connection:
                connection.execute("DELETE FROM calendar_sync_uid_map WHERE id = ?", (mapping["id"],))
            self._materialize_new_remote_event(
                provider_row,
                provider_row["project"],
                provider_row["calendar_id"] or "",
                remote,
                remote_hash,
                remote_etag=remote_etag,
                remote_href=remote_href,
            )
            return 1, 0
        current_local_hash = _event_fingerprint(local_event)
        if current_local_hash == mapping["local_hash"]:
            # Local side is untouched since the last sync: safe to apply the remote update directly.
            updated = self.calendar_system.update_event(mapping["local_event_id"], updates={
                "title": remote.get("title") or local_event["title"],
                "starts_at": remote.get("starts_at") or local_event["starts_at"],
                "ends_at": remote.get("ends_at") or local_event["ends_at"],
                "description": remote.get("description", ""),
                "location": remote.get("location", ""),
                "rrule": remote.get("rrule", ""),
                "status": remote.get("status", local_event.get("status", "confirmed")),
                "metadata": {
                    **(local_event.get("metadata") if isinstance(local_event.get("metadata"), dict) else {}),
                    "remote_href": remote_href,
                    "remote_etag": remote_etag,
                    "recurrence_id": remote.get("recurrence_id", ""),
                },
            })
            new_local_hash = _event_fingerprint(updated)
            with self.store.session() as connection:
                connection.execute(
                    "UPDATE calendar_sync_uid_map SET remote_hash = ?, local_hash = ?, remote_etag = ?, remote_href = ?, deleted_at = '', updated_at = ? WHERE id = ?",
                    (remote_hash, new_local_hash, remote_etag, remote_href, utc_now(), mapping["id"]),
                )
            return 1, 0
        # Both sides changed independently since the last sync: a genuine conflict.
        self._record_conflict(provider_row, mapping, local_event, remote, conflict_policy)
        return 0, 1

    def _update_remote_metadata(self, mapping_id: str, remote_etag: str, remote_href: str) -> None:
        if not remote_etag and not remote_href:
            return
        with self.store.session() as connection:
            connection.execute(
                "UPDATE calendar_sync_uid_map SET remote_etag = COALESCE(NULLIF(?, ''), remote_etag), "
                "remote_href = COALESCE(NULLIF(?, ''), remote_href), deleted_at = '', updated_at = ? WHERE id = ?",
                (remote_etag, remote_href, utc_now(), mapping_id),
            )

    def _apply_remote_deletes(
        self,
        provider_row: Any,
        seen_remote_uids: set[str],
        deleted_hrefs: list[str],
        *,
        treat_missing: bool = False,
    ) -> int:
        with self.store.session() as connection:
            rows: list[Any] = []
            if deleted_hrefs:
                placeholders = ",".join("?" for _ in deleted_hrefs)
                rows.extend(
                    connection.execute(
                        f"SELECT * FROM calendar_sync_uid_map WHERE provider_id = ? AND remote_href IN ({placeholders}) AND deleted_at = ''",
                        (provider_row["id"], *deleted_hrefs),
                    ).fetchall()
                )
            if treat_missing and seen_remote_uids:
                rows.extend(
                    connection.execute(
                        "SELECT * FROM calendar_sync_uid_map WHERE provider_id = ? AND deleted_at = ''",
                        (provider_row["id"],),
                    ).fetchall()
                )
            candidates = list({row["id"]: row for row in rows}.values())
        deleted_count = 0
        for row in candidates:
            if treat_missing and row["remote_uid"] in seen_remote_uids and row["remote_href"] not in deleted_hrefs:
                continue
            self.calendar_system.delete_event(row["local_event_id"])
            with self.store.session() as connection:
                connection.execute(
                    "UPDATE calendar_sync_uid_map SET deleted_at = ?, updated_at = ? WHERE id = ?",
                    (utc_now(), utc_now(), row["id"]),
                )
            deleted_count += 1
        return deleted_count

    def _record_conflict(
        self, provider_row: Any, mapping: Any, local_event: dict[str, Any], remote_event: dict[str, Any], conflict_policy: str
    ) -> None:
        conflict_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO calendar_sync_conflicts
                    (id, provider_id, project, local_event_id, remote_uid, local_payload_json, remote_payload_json,
                     local_updated_at, remote_seen_at, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
                """,
                (
                    conflict_id, provider_row["id"], provider_row["project"], mapping["local_event_id"],
                    remote_event.get("uid", ""), json.dumps(local_event), json.dumps(remote_event),
                    local_event["updated_at"], now, now,
                ),
            )
            if conflict_policy in CONFLICT_RESOLUTIONS:
                self._resolve_conflict_row(connection, conflict_id, conflict_policy)

    # ------------------------------------------------------------------
    # Push: manual adapters record local mappings; CalDAV adapters must PUT successfully.
    # ------------------------------------------------------------------

    def _push(
        self,
        provider_row: Any,
        *,
        allow_external: bool,
        allow_local_transport: bool,
    ) -> int:
        project = provider_row["project"]
        calendar_id = provider_row["calendar_id"] or ""
        local_events = self.calendar_system.list_events(project=project, calendar_id=calendar_id, limit=500)
        with self.store.session() as connection:
            mappings_by_local_id = {
                row["local_event_id"]: row
                for row in connection.execute(
                    "SELECT * FROM calendar_sync_uid_map WHERE provider_id = ?", (provider_row["id"],)
                ).fetchall()
            }
        pushed = 0
        for event in local_events:
            mapping = mappings_by_local_id.get(event["id"])
            local_hash = _event_fingerprint(event)
            if mapping is not None and mapping["local_hash"] == local_hash and not mapping["deleted_at"]:
                continue
            remote_uid = mapping["remote_uid"] if mapping is not None else f"openagent-{event['id']}@local-push"
            put_meta: dict[str, str] = {}
            if provider_row["provider"] == "caldav_http" and provider_row["base_url"]:
                put_meta = self._put_caldav_event(
                    provider_row,
                    event,
                    mapping=mapping,
                    remote_uid=remote_uid,
                    allow_external=allow_external,
                    allow_local_transport=allow_local_transport,
                )
            with self.store.session() as connection:
                if mapping is None:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO calendar_sync_uid_map
                            (id, provider_id, local_event_id, remote_uid, remote_hash, local_hash, remote_etag, remote_href, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uuid4().hex,
                            provider_row["id"],
                            event["id"],
                            remote_uid,
                            local_hash,
                            local_hash,
                            put_meta.get("remote_etag", ""),
                            put_meta.get("remote_href", ""),
                            utc_now(),
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE calendar_sync_uid_map SET remote_hash = ?, local_hash = ?, remote_etag = COALESCE(NULLIF(?, ''), remote_etag), "
                        "remote_href = COALESCE(NULLIF(?, ''), remote_href), deleted_at = '', updated_at = ? WHERE id = ?",
                        (local_hash, local_hash, put_meta.get("remote_etag", ""), put_meta.get("remote_href", ""), utc_now(), mapping["id"]),
                    )
            pushed += 1

        # Propagate local deletions to the CalDAV server.
        if provider_row["provider"] == "caldav_http" and provider_row["base_url"]:
            pushed += self._push_deletions(
                provider_row,
                allow_external=allow_external,
                allow_local_transport=allow_local_transport,
            )
        return pushed

    def _push_deletions(
        self,
        provider_row: Any,
        *,
        allow_external: bool,
        allow_local_transport: bool,
    ) -> int:
        """Issue CalDAV DELETE for each locally-deleted event that has a remote href."""
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM calendar_sync_uid_map WHERE provider_id = ? AND deleted_at != '' AND remote_href != ''",
                (provider_row["id"],),
            ).fetchall()
        config = json.loads(provider_row["config_json"] or "{}")
        base_url = str(config.get("calendar_collection_url") or provider_row["base_url"]).rstrip("/")
        deleted_count = 0
        for row in rows:
            remote_href = str(row["remote_href"])
            url = urljoin(base_url + "/", remote_href)
            del_headers = self._caldav_headers(config, {})
            if row["remote_etag"]:
                del_headers["If-Match"] = f'"{row["remote_etag"]}"'
            try:
                response = self.egress.fetch(
                    url,
                    approved=allow_external,
                    allow_loopback=allow_local_transport,
                    purpose="CalDAV event DELETE",
                    method="DELETE",
                    headers=del_headers,
                    timeout_seconds=float(config.get("timeout_seconds") or 20),
                    max_response_bytes=1024,
                )
            except (EgressError, PermissionError) as exc:
                raise CalendarSyncError(f"CalDAV DELETE failed for {remote_href}: {exc}") from exc
            if response.status not in {200, 204, 404}:
                raise CalendarSyncError(f"CalDAV DELETE returned HTTP {response.status} for {remote_href}")
            with self.store.session() as connection:
                connection.execute(
                    "UPDATE calendar_sync_uid_map SET remote_href = '', updated_at = ? WHERE id = ?",
                    (utc_now(), row["id"]),
                )
            deleted_count += 1
        return deleted_count

    def _put_caldav_event(
        self,
        provider_row: Any,
        event: dict[str, Any],
        *,
        mapping: Any = None,
        remote_uid: str = "",
        allow_external: bool,
        allow_local_transport: bool,
    ) -> dict[str, str]:
        config = json.loads(provider_row["config_json"] or "{}")
        base_url = str(config.get("calendar_collection_url") or provider_row["base_url"]).rstrip("/")
        if not base_url:
            raise CalendarSyncError("CalDAV provider has no collection URL configured")
        remote_href = str(mapping["remote_href"] or "") if mapping is not None else ""
        if remote_href:
            url = urljoin(base_url + "/", remote_href)
            returned_href = remote_href
        else:
            path_template = str(config.get("path_template") or "{event_id}.ics")
            relative_path = path_template.format(event_id=event["id"], uid=remote_uid or event.get("uid", event["id"])).lstrip("/")
            url = f"{base_url}/{relative_path}"
            returned_href = "/" + relative_path
        headers = self._caldav_headers(config, {
            "Content-Type": "text/calendar; charset=utf-8",
        })
        if mapping is not None and mapping["remote_etag"]:
            headers["If-Match"] = f'"{mapping["remote_etag"]}"'
        elif config.get("if_none_match", True):
            headers["If-None-Match"] = "*"
        try:
            response = self.egress.fetch(
                url,
                approved=allow_external,
                allow_loopback=allow_local_transport,
                purpose="CalDAV event PUT",
                method="PUT",
                body=self._caldav_ical_body(event).encode("utf-8"),
                headers=headers,
                timeout_seconds=float(config.get("timeout_seconds") or 20),
                max_response_bytes=1024 * 1024,
            )
        except (EgressError, PermissionError) as exc:
            raise CalendarSyncError(f"CalDAV PUT failed for {event['id']}: {exc}") from exc
        status = response.status
        etag = str(response.headers.get("etag", "")).strip().strip('"')
        if status == 412:
            # ETag mismatch: fetch the current ETag and retry once.
            try:
                head_resp = self.egress.fetch(
                    url,
                    approved=allow_external,
                    allow_loopback=allow_local_transport,
                    purpose="CalDAV event HEAD (ETag refresh)",
                    method="HEAD",
                    headers=self._caldav_headers(config, {}),
                    timeout_seconds=float(config.get("timeout_seconds") or 20),
                    max_response_bytes=1024,
                )
                fresh_etag = str(head_resp.headers.get("etag", "")).strip().strip('"')
            except (EgressError, PermissionError):
                fresh_etag = ""
            retry_headers = self._caldav_headers(config, {"Content-Type": "text/calendar; charset=utf-8"})
            if fresh_etag:
                retry_headers["If-Match"] = f'"{fresh_etag}"'
            try:
                response = self.egress.fetch(
                    url,
                    approved=allow_external,
                    allow_loopback=allow_local_transport,
                    purpose="CalDAV event PUT (retry after 412)",
                    method="PUT",
                    body=self._caldav_ical_body(event).encode("utf-8"),
                    headers=retry_headers,
                    timeout_seconds=float(config.get("timeout_seconds") or 20),
                    max_response_bytes=1024 * 1024,
                )
            except (EgressError, PermissionError) as exc:
                raise CalendarSyncError(f"CalDAV PUT retry failed for {event['id']}: {exc}") from exc
            status = response.status
            etag = str(response.headers.get("etag", "")).strip().strip('"')
        if status == 409:
            raise CalendarSyncError(f"CalDAV ETag/update conflict for {event['id']}: HTTP {status}")
        success_codes = tuple(int(code) for code in config.get("success_codes", CALDAV_SUCCESS_CODES))
        if status not in success_codes:
            raise CalendarSyncError(f"CalDAV PUT failed for {event['id']}: HTTP {status}")
        return {"remote_href": returned_href, "remote_etag": etag}

    def _caldav_headers(self, config: dict[str, Any], extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"User-Agent": "OpenAgent-CalDAV-Sync"}
        token = str(config.get("auth_token", "")).strip()
        username = str(config.get("username", "")).strip()
        password = str(config.get("password", "")).strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif username or password:
            raw = f"{username}:{password}".encode("utf-8")
            headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")
        if extra:
            headers.update(extra)
        return headers

    def _caldav_ical_body(self, event: dict[str, Any]) -> str:
        event_ical = event_to_ical(event).replace("\r\n", "\n").strip()
        if event_ical.startswith("BEGIN:VCALENDAR"):
            return event_ical.replace("\n", "\r\n")
        return "\r\n".join([
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//OpenAgent//Local Calendar Sync//EN",
            event_ical.replace("\n", "\r\n"),
            "END:VCALENDAR",
        ])

    # ------------------------------------------------------------------
    # Conflicts
    # ------------------------------------------------------------------

    def list_conflicts(self, *, project: str = "default", status: str = "", owner: str = "") -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project or "default"]
        if status:
            clauses.append("status = ?")
            args.append(status)
        if owner:
            clauses.append("provider_id IN (SELECT id FROM calendar_sync_providers WHERE owner = ?)")
            args.append(owner)
        sql = "SELECT * FROM calendar_sync_conflicts WHERE " + " AND ".join(clauses) + " ORDER BY created_at DESC, rowid DESC"
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
            return [self._row_to_conflict(row) for row in rows]

    def _row_to_conflict(self, row: Any) -> dict[str, Any]:
        return {
            "conflict_id": row["id"],
            "provider_id": row["provider_id"],
            "project": row["project"],
            "local_event_id": row["local_event_id"],
            "remote_uid": row["remote_uid"],
            "local_payload": json.loads(row["local_payload_json"] or "{}"),
            "remote_payload": json.loads(row["remote_payload_json"] or "{}"),
            "status": row["status"],
            "resolution": row["resolution"],
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
        }

    def resolve_conflict(self, conflict_id: str, resolution: str, *, owner: str = "") -> dict[str, Any]:
        if resolution not in CONFLICT_RESOLUTIONS:
            raise ValueError(f"resolution must be one of {CONFLICT_RESOLUTIONS}")
        with self.store.session() as connection:
            if owner:
                row = connection.execute(
                    """
                    SELECT c.id FROM calendar_sync_conflicts c
                    JOIN calendar_sync_providers p ON p.id = c.provider_id
                    WHERE c.id = ? AND p.owner = ?
                    """,
                    (conflict_id, owner),
                ).fetchone()
                if row is None:
                    raise KeyError("conflict not found")
            return self._resolve_conflict_row(connection, conflict_id, resolution)

    def _resolve_conflict_row(self, connection: Any, conflict_id: str, resolution: str) -> dict[str, Any]:
        row = connection.execute("SELECT * FROM calendar_sync_conflicts WHERE id = ?", (conflict_id,)).fetchone()
        if row is None:
            raise KeyError("conflict not found")
        if row["status"] != "open":
            return {"conflict_id": conflict_id, "status": row["status"], "resolution": row["resolution"]}
        local_payload = json.loads(row["local_payload_json"] or "{}")
        remote_payload = json.loads(row["remote_payload_json"] or "{}")
        chosen = resolution
        if resolution == "latest_update":
            chosen = "server_wins" if row["remote_seen_at"] >= row["local_updated_at"] else "client_wins"
        if chosen == "server_wins":
            updated = self.calendar_system.update_event(row["local_event_id"], updates={
                "title": remote_payload.get("title", ""),
                "starts_at": remote_payload.get("starts_at", ""),
                "ends_at": remote_payload.get("ends_at", ""),
                "description": remote_payload.get("description", ""),
                "location": remote_payload.get("location", ""),
            })
            new_local_hash = _event_fingerprint(updated)
        else:
            local_event = self.calendar_system.get_event(row["local_event_id"])
            new_local_hash = _event_fingerprint(local_event)
        remote_hash = _fingerprint(
            remote_payload.get("title", ""),
            remote_payload.get("starts_at", ""),
            remote_payload.get("ends_at", ""),
            remote_payload.get("description", ""),
            remote_payload.get("location", ""),
        )
        connection.execute(
            "UPDATE calendar_sync_uid_map SET remote_hash = ?, local_hash = ?, updated_at = ? "
            "WHERE provider_id = ? AND remote_uid = ?",
            (remote_hash, new_local_hash, utc_now(), row["provider_id"], row["remote_uid"]),
        )
        connection.execute(
            "UPDATE calendar_sync_conflicts SET status = 'resolved', resolution = ?, resolved_at = ? WHERE id = ?",
            (chosen, utc_now(), conflict_id),
        )
        return {"conflict_id": conflict_id, "status": "resolved", "resolution": chosen}
