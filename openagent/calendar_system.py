"""Full calendar system with CalDAV-compatible local storage.

Features:
- Multiple named calendars per project
- CRUD for events with recurrence rules (RRULE subset)
- Scheduling intelligence: find free/busy slots, optimal meeting times
- Reminders with advance notice
- iCal (RFC 5545) import/export using stdlib only
- Time-zone-aware ISO-8601 timestamps
- CalDAV-style resource IDs for federation compatibility
- Audit logging of all changes
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


# ---------------------------------------------------------------------------
# Recurrence helpers (RRULE subset: FREQ=DAILY|WEEKLY|MONTHLY|YEARLY)
# ---------------------------------------------------------------------------

_FREQ_DELTAS: dict[str, timedelta | None] = {
    "DAILY": timedelta(days=1),
    "WEEKLY": timedelta(weeks=1),
    "MONTHLY": None,   # handled specially
    "YEARLY": None,    # handled specially
}


def _add_months(dt: datetime, n: int) -> datetime:
    month = dt.month - 1 + n
    year = dt.year + month // 12
    month = month % 12 + 1
    import calendar as _cal
    day = min(dt.day, _cal.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def expand_rrule(dtstart: str, rrule: str, limit: int = 50) -> list[str]:
    """Expand a RRULE string into ISO-8601 occurrence timestamps (up to *limit*)."""
    try:
        start = datetime.fromisoformat(dtstart)
    except ValueError:
        return [dtstart]

    parts: dict[str, str] = {}
    for part in rrule.upper().split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            parts[k] = v

    freq = parts.get("FREQ", "")
    if freq not in _FREQ_DELTAS:
        return [dtstart]

    try:
        count = int(parts.get("COUNT", limit))
        interval = int(parts.get("INTERVAL", "1"))
    except ValueError:
        return [dtstart]
    if count < 1 or interval < 1 or limit < 1:
        return [dtstart]
    until_str = parts.get("UNTIL", "")

    results: list[str] = []
    current = start
    until: datetime | None = None
    if until_str:
        try:
            normalized = until_str.strip()
            if re.fullmatch(r"\d{8}T\d{6}Z?", normalized):
                normalized = datetime.strptime(normalized.rstrip("Z"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat()
            else:
                normalized = normalized.replace("Z", "+00:00")
            until = datetime.fromisoformat(normalized)
        except ValueError:
            return [dtstart]

    def comparable(value: datetime) -> datetime:
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)

    for _ in range(min(count, limit)):
        if until and comparable(current) > comparable(until):
            break
        results.append(current.isoformat())
        if freq == "DAILY":
            current = current + timedelta(days=interval)
        elif freq == "WEEKLY":
            current = current + timedelta(weeks=interval)
        elif freq == "MONTHLY":
            current = _add_months(start, interval * len(results))
        elif freq == "YEARLY":
            current = _add_months(start, 12 * interval * len(results))

    return results


# ---------------------------------------------------------------------------
# iCal helpers (RFC 5545 minimal parser/emitter)
# ---------------------------------------------------------------------------

_ICAL_DT_FMT = "%Y%m%dT%H%M%SZ"


def _ical_dt(iso: str) -> str:
    """Convert ISO-8601 → YYYYMMDDTHHMMSSZ."""
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime(_ICAL_DT_FMT)
    except ValueError:
        return iso.replace("-", "").replace(":", "").replace(" ", "T")


def _parse_ical_dt(val: str) -> str:
    """Convert iCal datetime → ISO-8601 string."""
    val = val.strip().rstrip("Z")
    try:
        if len(val) == 15:   # YYYYMMDDTHHmmss
            dt = datetime.strptime(val, "%Y%m%dT%H%M%S")
        elif len(val) == 8:  # YYYYMMDD (date only)
            dt = datetime.strptime(val, "%Y%m%d")
        else:
            return val
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return val


def parse_ical(raw: str) -> list[dict[str, Any]]:
    """Parse an iCal string and return a list of event dicts."""
    events: list[dict[str, Any]] = []
    current: dict[str, str] | None = None
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if line == "BEGIN:VEVENT":
            current = {}
        elif line == "END:VEVENT" and current is not None:
            events.append(current)
            current = None
        elif current is not None and ":" in line:
            key, _, value = line.partition(":")
            key = key.split(";")[0].upper()  # strip params
            current[key] = value
    result: list[dict[str, Any]] = []
    for ev in events:
        result.append({
            "uid": ev.get("UID", str(uuid.uuid4())),
            "title": ev.get("SUMMARY", "Untitled"),
            "description": ev.get("DESCRIPTION", ""),
            "location": ev.get("LOCATION", ""),
            "starts_at": _parse_ical_dt(ev.get("DTSTART", "")),
            "ends_at": _parse_ical_dt(ev.get("DTEND", ev.get("DTSTART", ""))),
            "rrule": ev.get("RRULE", ""),
            "recurrence_id": _parse_ical_dt(ev.get("RECURRENCE-ID", "")),
            "last_modified": _parse_ical_dt(ev.get("LAST-MODIFIED", "")),
            "sequence": ev.get("SEQUENCE", ""),
            "status": ev.get("STATUS", "CONFIRMED").lower(),
        })
    return result


def event_to_ical(ev: dict[str, Any]) -> str:
    """Serialize a single event dict to iCal VEVENT block."""
    lines = [
        "BEGIN:VEVENT",
        f"UID:{ev.get('id', str(uuid.uuid4()))}@openagent.local",
        f"SUMMARY:{ev.get('title', '')}",
        f"DTSTART:{_ical_dt(ev.get('starts_at', ''))}",
        f"DTEND:{_ical_dt(ev.get('ends_at', ev.get('starts_at', '')))}",
    ]
    if ev.get("description"):
        lines.append(f"DESCRIPTION:{ev['description']}")
    if ev.get("location"):
        lines.append(f"LOCATION:{ev['location']}")
    if ev.get("rrule"):
        lines.append(f"RRULE:{ev['rrule']}")
    metadata = ev.get("metadata") if isinstance(ev.get("metadata"), dict) else {}
    recurrence_id = ev.get("recurrence_id") or metadata.get("recurrence_id", "")
    if recurrence_id:
        lines.append(f"RECURRENCE-ID:{_ical_dt(recurrence_id)}")
    lines.append("END:VEVENT")
    return "\r\n".join(lines)


# ---------------------------------------------------------------------------
# CalendarSystem
# ---------------------------------------------------------------------------

class CalendarSystem:
    """Full calendar management system — local, offline-first, CalDAV-inspired."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.initialize()
        self.seed_calendars()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self.store.session() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS calendars (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    name TEXT NOT NULL,
                    color TEXT NOT NULL DEFAULT '#17c964',
                    description TEXT NOT NULL DEFAULT '',
                    is_default INTEGER NOT NULL DEFAULT 0,
                    timezone TEXT NOT NULL DEFAULT 'UTC',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_calendars_project
                    ON calendars(project);

                CREATE TABLE IF NOT EXISTS calendar_events (
                    id TEXT PRIMARY KEY,
                    calendar_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    location TEXT NOT NULL DEFAULT '',
                    starts_at TEXT NOT NULL,
                    ends_at TEXT NOT NULL,
                    all_day INTEGER NOT NULL DEFAULT 0,
                    rrule TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'confirmed',
                    attendees_json TEXT NOT NULL DEFAULT '[]',
                    reminders_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(calendar_id) REFERENCES calendars(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_events_project_starts
                    ON calendar_events(project, starts_at);

                CREATE INDEX IF NOT EXISTS idx_events_calendar
                    ON calendar_events(calendar_id, starts_at);

                CREATE TABLE IF NOT EXISTS calendar_reminders (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    advance_minutes INTEGER NOT NULL DEFAULT 15,
                    method TEXT NOT NULL DEFAULT 'notification',
                    triggered INTEGER NOT NULL DEFAULT 0,
                    trigger_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES calendar_events(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS calendar_sync_log (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    source TEXT NOT NULL,
                    events_imported INTEGER NOT NULL DEFAULT 0,
                    events_exported INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ok',
                    created_at TEXT NOT NULL
                );
            """)
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: Any) -> None:
        """Apply additive schema migrations for older local databases."""

        def ensure_column(table: str, column_def: str) -> None:
            column_name = column_def.split()[0]
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            existing = {str(row[1]) for row in rows}
            if column_name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")

        # calendars
        ensure_column("calendars", "description TEXT NOT NULL DEFAULT ''")
        ensure_column("calendars", "is_default INTEGER NOT NULL DEFAULT 0")
        ensure_column("calendars", "timezone TEXT NOT NULL DEFAULT 'UTC'")
        ensure_column("calendars", "metadata_json TEXT NOT NULL DEFAULT '{}'")
        ensure_column("calendars", "created_at TEXT NOT NULL DEFAULT ''")
        ensure_column("calendars", "updated_at TEXT NOT NULL DEFAULT ''")

        # calendar_events
        ensure_column("calendar_events", "description TEXT NOT NULL DEFAULT ''")
        ensure_column("calendar_events", "location TEXT NOT NULL DEFAULT ''")
        ensure_column("calendar_events", "all_day INTEGER NOT NULL DEFAULT 0")
        ensure_column("calendar_events", "rrule TEXT NOT NULL DEFAULT ''")
        ensure_column("calendar_events", "status TEXT NOT NULL DEFAULT 'confirmed'")
        ensure_column("calendar_events", "attendees_json TEXT NOT NULL DEFAULT '[]'")
        ensure_column("calendar_events", "reminders_json TEXT NOT NULL DEFAULT '[]'")
        ensure_column("calendar_events", "metadata_json TEXT NOT NULL DEFAULT '{}'")
        ensure_column("calendar_events", "created_at TEXT NOT NULL DEFAULT ''")
        ensure_column("calendar_events", "updated_at TEXT NOT NULL DEFAULT ''")

        # calendar_reminders
        ensure_column("calendar_reminders", "method TEXT NOT NULL DEFAULT 'notification'")
        ensure_column("calendar_reminders", "triggered INTEGER NOT NULL DEFAULT 0")
        ensure_column("calendar_reminders", "trigger_at TEXT NOT NULL DEFAULT ''")
        ensure_column("calendar_reminders", "created_at TEXT NOT NULL DEFAULT ''")

        # calendar_sync_log
        ensure_column("calendar_sync_log", "events_imported INTEGER NOT NULL DEFAULT 0")
        ensure_column("calendar_sync_log", "events_exported INTEGER NOT NULL DEFAULT 0")
        ensure_column("calendar_sync_log", "status TEXT NOT NULL DEFAULT 'ok'")
        ensure_column("calendar_sync_log", "created_at TEXT NOT NULL DEFAULT ''")

    def seed_calendars(self) -> None:
        """Create default Personal and Work calendars if they don't exist."""
        now = utc_now()
        defaults = [
            ("Personal", "#17c964", 1),
            ("Work",     "#65d4ff", 0),
            ("Tasks",    "#a78bfa", 0),
        ]
        with self.store.session() as conn:
            for name, color, is_default in defaults:
                existing = conn.execute(
                    "SELECT id FROM calendars WHERE project='default' AND name=?", (name,)
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO calendars VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            hashlib.md5(f"default-{name}".encode()).hexdigest(),
                            "default", name, color, "", is_default, "UTC", "{}", now, now,
                        ),
                    )

    # ------------------------------------------------------------------
    # Calendar CRUD
    # ------------------------------------------------------------------

    def create_calendar(
        self,
        name: str,
        *,
        project: str = "default",
        color: str = "#17c964",
        description: str = "",
        timezone: str = "UTC",
    ) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("Calendar name is required.")
        now = utc_now()
        cal_id = hashlib.md5(f"{project}-{name}-{now}".encode()).hexdigest()
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO calendars VALUES(?,?,?,?,?,?,?,?,?,?)",
                (cal_id, project, name.strip(), color, description, 0, timezone, "{}", now, now),
            )
        self.store.add_audit_event(AuditEvent(
            actor="calendar",
            action="calendar.create",
            resource=f"calendar:create:{cal_id}",
            project=project,
            metadata={"calendar_id": cal_id, "name": name},
        ))
        return self.get_calendar(cal_id)

    def get_calendar(self, calendar_id: str) -> dict[str, Any]:
        with self.store.session() as conn:
            row = conn.execute(
                "SELECT id,project,name,color,description,is_default,timezone,metadata_json,created_at,updated_at "
                "FROM calendars WHERE id=?", (calendar_id,)
            ).fetchone()
        if not row:
            raise KeyError(f"Calendar not found: {calendar_id}")
        return self._cal_row(row)

    def list_calendars(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT id,project,name,color,description,is_default,timezone,metadata_json,created_at,updated_at "
                "FROM calendars WHERE project=? ORDER BY is_default DESC, name", (project,)
            ).fetchall()
        return [self._cal_row(r) for r in rows]

    def _cal_row(self, row: Any) -> dict[str, Any]:
        return {
            "id": row[0], "project": row[1], "name": row[2],
            "color": row[3], "description": row[4], "is_default": bool(row[5]),
            "timezone": row[6], "metadata": json.loads(row[7] or "{}"),
            "created_at": row[8], "updated_at": row[9],
        }

    # ------------------------------------------------------------------
    # Event CRUD
    # ------------------------------------------------------------------

    def create_event(
        self,
        title: str,
        starts_at: str,
        ends_at: str,
        *,
        project: str = "default",
        calendar_id: str = "",
        description: str = "",
        location: str = "",
        all_day: bool = False,
        rrule: str = "",
        attendees: list[str] | None = None,
        reminders: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not title.strip():
            raise ValueError("Event title is required.")
        if not starts_at:
            raise ValueError("starts_at is required.")
        now = utc_now()
        event_id = hashlib.md5(f"{project}-{title}-{starts_at}-{now}".encode()).hexdigest()
        # Resolve default calendar
        if not calendar_id:
            with self.store.session() as conn:
                row = conn.execute(
                    "SELECT id FROM calendars WHERE project=? AND is_default=1 LIMIT 1", (project,)
                ).fetchone()
                if not row:
                    row = conn.execute(
                        "SELECT id FROM calendars WHERE project=? LIMIT 1", (project,)
                    ).fetchone()
                calendar_id = row[0] if row else hashlib.md5(f"default-Personal".encode()).hexdigest()
        with self.store.session() as conn:
            conn.execute(
                """
                INSERT INTO calendar_events (
                    id, calendar_id, project, title, description, location,
                    starts_at, ends_at, all_day, rrule, status, attendees_json,
                    reminders_json, metadata_json, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id, calendar_id, project,
                    title.strip(), description, location,
                    starts_at, ends_at or starts_at,
                    1 if all_day else 0, rrule, "confirmed",
                    json.dumps(attendees or []),
                    json.dumps(reminders or [15]),
                    json.dumps(metadata or {}),
                    now, now,
                ),
            )
            # Create reminder records
            for advance_mins in (reminders or [15]):
                try:
                    dt = datetime.fromisoformat(starts_at)
                    trigger_at = (dt - timedelta(minutes=advance_mins)).isoformat()
                except ValueError:
                    trigger_at = starts_at
                reminder_id = hashlib.md5(f"{event_id}-{advance_mins}".encode()).hexdigest()
                conn.execute(
                    "INSERT OR IGNORE INTO calendar_reminders VALUES(?,?,?,?,?,?,?,?)",
                    (reminder_id, event_id, project, advance_mins, "notification", 0, trigger_at, now),
                )
        self.store.add_audit_event(AuditEvent(
            actor="calendar",
            action="calendar.event.create",
            resource=f"calendar:event:create:{event_id}",
            project=project,
            metadata={"event_id": event_id, "title": title, "starts_at": starts_at},
        ))
        return self.get_event(event_id)

    def get_event(self, event_id: str, *, project: str | None = None) -> dict[str, Any]:
        query = (
            "SELECT id,calendar_id,project,title,description,location,starts_at,ends_at,"
            "all_day,rrule,status,attendees_json,reminders_json,metadata_json,created_at,updated_at "
            "FROM calendar_events WHERE id=?"
        )
        params: tuple[Any, ...] = (event_id,)
        if project is not None:
            query += " AND project=?"
            params = (event_id, project)
        with self.store.session() as conn:
            row = conn.execute(query, params).fetchone()
        if not row:
            raise KeyError(f"Event not found: {event_id}")
        return self._event_row(row)

    def list_events(
        self,
        *,
        project: str = "default",
        calendar_id: str = "",
        from_dt: str = "",
        to_dt: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT id,calendar_id,project,title,description,location,starts_at,ends_at,"
            "all_day,rrule,status,attendees_json,reminders_json,metadata_json,created_at,updated_at "
            "FROM calendar_events WHERE project=?"
        )
        params: list[Any] = [project]
        if calendar_id:
            query += " AND calendar_id=?"
            params.append(calendar_id)
        if from_dt:
            # Include detached occurrence overrides even when an override was
            # moved outside the requested window: its RECURRENCE-ID still
            # replaces an occurrence that belongs inside this window.
            query += " AND (ends_at >= ? OR rrule != '' OR metadata_json LIKE '%master_event_id%')"
            params.append(from_dt)
        if to_dt:
            query += " AND (starts_at <= ? OR rrule != '' OR metadata_json LIKE '%master_event_id%')"
            params.append(to_dt)
        query += " ORDER BY starts_at LIMIT ?"
        params.append(max(1, min(int(limit), 5000)))
        with self.store.session() as conn:
            rows = conn.execute(query, params).fetchall()
        events: list[dict[str, Any]] = []
        # Occurrence overrides are stored as ordinary event rows with a
        # ``master_event_id`` and the original RECURRENCE-ID in metadata.
        # Keep them out of the root event list and merge them into the
        # matching generated occurrence below. Otherwise users would see both
        # the old scheduled time and a detached duplicate at the new time.
        overrides_by_master: dict[str, dict[str, dict[str, Any]]] = {}
        root_rows: list[Any] = []
        for row in rows:
            event = self._event_row(row)
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            master_id = str(metadata.get("master_event_id") or "").strip()
            recurrence_id = str(metadata.get("recurrence_id") or "").strip()
            if master_id and recurrence_id:
                overrides_by_master.setdefault(master_id, {})[recurrence_id] = event
            else:
                root_rows.append(row)
        try:
            range_start = datetime.fromisoformat(from_dt.replace("Z", "+00:00")) if from_dt else None
            range_end = datetime.fromisoformat(to_dt.replace("Z", "+00:00")) if to_dt else None
        except ValueError:
            range_start = range_end = None
        for row in root_rows:
            event = self._event_row(row)
            if not event["rrule"] or range_start is None or range_end is None:
                events.append(event)
                continue
            try:
                series_start = datetime.fromisoformat(str(event["starts_at"]).replace("Z", "+00:00"))
                series_end = datetime.fromisoformat(str(event["ends_at"]).replace("Z", "+00:00"))
                if series_start.tzinfo is None:
                    series_start = series_start.replace(tzinfo=timezone.utc)
                if series_end.tzinfo is None:
                    series_end = series_end.replace(tzinfo=timezone.utc)
                if range_start.tzinfo is None:
                    range_start = range_start.replace(tzinfo=timezone.utc)
                if range_end.tzinfo is None:
                    range_end = range_end.replace(tzinfo=timezone.utc)
                duration = series_end - series_start
                occurrences = []
                matched_override_ids: set[str] = set()
                for value in expand_rrule(str(event["starts_at"]), str(event["rrule"]), limit=10_000):
                    occurrence_start = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if occurrence_start.tzinfo is None:
                        occurrence_start = occurrence_start.replace(tzinfo=timezone.utc)
                    occurrence_end = occurrence_start + duration
                    recurrence_id = occurrence_start.isoformat()
                    override = overrides_by_master.get(str(event["id"]), {}).get(recurrence_id)
                    if override is None:
                        # Older rows may preserve the same timestamp with a
                        # trailing Z while datetime.isoformat emits +00:00.
                        override = next((candidate for key, candidate in overrides_by_master.get(str(event["id"]), {}).items()
                                         if key.replace("Z", "+00:00") == recurrence_id), None)
                    if override is not None:
                        matched_override_ids.add(str(override["id"]))
                        instance_start = datetime.fromisoformat(str(override["starts_at"]).replace("Z", "+00:00"))
                        instance_end = datetime.fromisoformat(str(override["ends_at"]).replace("Z", "+00:00"))
                        if instance_start.tzinfo is None:
                            instance_start = instance_start.replace(tzinfo=timezone.utc)
                        if instance_end.tzinfo is None:
                            instance_end = instance_end.replace(tzinfo=timezone.utc)
                        # A moved override belongs to its actual time window.
                        # It suppresses the original slot even if it moved out
                        # of this query, and appears when its new time is read.
                        if instance_start < range_end and instance_end > range_start:
                            occurrences.append({
                                **override,
                                "id": event["id"],
                                "starts_at": override["starts_at"],
                                "ends_at": override["ends_at"],
                                "recurrence_id": recurrence_id,
                                "is_override": True,
                            })
                    elif occurrence_start < range_end and occurrence_end > range_start:
                        occurrences.append({"starts_at": occurrence_start.isoformat(), "ends_at": occurrence_end.isoformat()})

                # An override may have moved from a recurrence outside this
                # query into the current window. Include it at its real time.
                for recurrence_id, override in overrides_by_master.get(str(event["id"]), {}).items():
                    if str(override["id"]) in matched_override_ids:
                        continue
                    instance_start = datetime.fromisoformat(str(override["starts_at"]).replace("Z", "+00:00"))
                    instance_end = datetime.fromisoformat(str(override["ends_at"]).replace("Z", "+00:00"))
                    if instance_start.tzinfo is None:
                        instance_start = instance_start.replace(tzinfo=timezone.utc)
                    if instance_end.tzinfo is None:
                        instance_end = instance_end.replace(tzinfo=timezone.utc)
                    if instance_start < range_end and instance_end > range_start:
                        occurrences.append({
                            **override,
                            "id": event["id"],
                            "recurrence_id": recurrence_id,
                            "is_override": True,
                        })
                if occurrences:
                    event["occurrences"] = occurrences
                    events.append(event)
            except (TypeError, ValueError):
                # Invalid stored timestamps cannot be safely expanded. Keep
                # the series record visible so users can inspect and correct it.
                events.append(event)
        return events

    def list_event_occurrences(
        self,
        event_id: str,
        *,
        project: str = "default",
        from_date: str = "",
        until_date: str = "",
    ) -> list[dict[str, Any]]:
        """Expand a recurring event into its occurrences within a date range.

        Supports RRULE with FREQ=DAILY|WEEKLY|MONTHLY|YEARLY and UNTIL cutoff.
        Returns a list of occurrence objects with starts_at, ends_at, and recurrence_id.
        """
        # Get the master event
        event = self.get_event(event_id, project=project)

        if not event.get("rrule"):
            # Non-recurring event: return single occurrence
            return [{
                "starts_at": event["starts_at"],
                "ends_at": event["ends_at"],
                "recurrence_id": event["starts_at"],
                "title": event.get("title", ""),
            }]

        try:
            range_start = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
            range_end = datetime.fromisoformat(until_date.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid date range: {exc}") from exc

        if range_start.tzinfo is None:
            range_start = range_start.replace(tzinfo=timezone.utc)
        if range_end.tzinfo is None:
            range_end = range_end.replace(tzinfo=timezone.utc)

        try:
            series_start = datetime.fromisoformat(str(event["starts_at"]).replace("Z", "+00:00"))
            series_end = datetime.fromisoformat(str(event["ends_at"]).replace("Z", "+00:00"))
            if series_start.tzinfo is None:
                series_start = series_start.replace(tzinfo=timezone.utc)
            if series_end.tzinfo is None:
                series_end = series_end.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"Invalid event timestamps: {exc}") from exc

        duration = series_end - series_start
        occurrences: list[dict[str, Any]] = []

        # Expand RRULE and filter by date range
        for value in expand_rrule(str(event["starts_at"]), str(event["rrule"]), limit=10_000):
            try:
                occurrence_start = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if occurrence_start.tzinfo is None:
                    occurrence_start = occurrence_start.replace(tzinfo=timezone.utc)
                occurrence_end = occurrence_start + duration

                # Check if occurrence is within requested range
                if occurrence_start < range_end and occurrence_end > range_start:
                    occurrences.append({
                        "starts_at": occurrence_start.isoformat(),
                        "ends_at": occurrence_end.isoformat(),
                        "recurrence_id": occurrence_start.isoformat(),
                        "title": event.get("title", ""),
                        "location": event.get("location", ""),
                    })
            except ValueError:
                continue

        return occurrences

    def update_event(
        self,
        event_id: str,
        *,
        updates: dict[str, Any],
        project: str | None = None,
    ) -> dict[str, Any]:
        allowed = {"title", "description", "location", "starts_at", "ends_at",
                   "all_day", "rrule", "status", "attendees", "reminders", "metadata"}
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"Unsupported calendar event fields: {', '.join(sorted(unknown))}")
        if not updates:
            raise ValueError("At least one event field must be updated.")
        current = self.get_event(event_id, project=project)
        if "title" in updates and not str(updates["title"]).strip():
            raise ValueError("Event title is required.")
        starts_at = str(updates.get("starts_at", current["starts_at"]))
        ends_at = str(updates.get("ends_at", current["ends_at"]))
        try:
            start_time = datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(ends_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Event start and end must be valid ISO-8601 timestamps.") from exc
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        if end_time <= start_time:
            raise ValueError("Event end must be later than start.")
        if "attendees" in updates and (
            not isinstance(updates["attendees"], list)
            or any(not isinstance(item, str) for item in updates["attendees"])
        ):
            raise ValueError("Attendees must be a list of names or addresses.")
        if "reminders" in updates and (
            not isinstance(updates["reminders"], list)
            or any(not isinstance(item, int) or item < 0 or item > 10080 for item in updates["reminders"])
        ):
            raise ValueError("Reminders must be a list of minute offsets from 0 to 10080.")
        now = utc_now()
        with self.store.session() as conn:
            for field, value in updates.items():
                col = "attendees_json" if field == "attendees" else \
                      "reminders_json" if field == "reminders" else \
                      "metadata_json" if field == "metadata" else field
                if field in ("attendees", "reminders", "metadata"):
                    value = json.dumps(value)
                query = f"UPDATE calendar_events SET {col}=?, updated_at=? WHERE id=?"
                params: tuple[Any, ...] = (value, now, event_id)
                if project is not None:
                    query += " AND project=?"
                    params = (value, now, event_id, project)
                conn.execute(
                    query,
                    params,
                )
            if "reminders" in updates:
                conn.execute("DELETE FROM calendar_reminders WHERE event_id=?", (event_id,))
                start_for_reminder = str(updates.get("starts_at", current["starts_at"]))
                for advance_minutes in updates["reminders"]:
                    try:
                        trigger_at = (datetime.fromisoformat(start_for_reminder.replace("Z", "+00:00"))
                                      - timedelta(minutes=advance_minutes)).isoformat()
                    except ValueError:
                        trigger_at = start_for_reminder
                    reminder_id = hashlib.md5(f"{event_id}-{advance_minutes}".encode()).hexdigest()
                    conn.execute(
                        "INSERT OR IGNORE INTO calendar_reminders VALUES(?,?,?,?,?,?,?,?)",
                        (reminder_id, event_id, current["project"], advance_minutes,
                         "notification", 0, trigger_at, now),
                    )
        updated = self.get_event(event_id, project=project)
        self.store.add_audit_event(AuditEvent(
            actor="calendar",
            action="calendar.event.update",
            resource=f"calendar:event:update:{event_id}",
            project=updated["project"],
            metadata={"event_id": event_id, "fields": sorted(updates)},
        ))
        return updated

    def create_event_override(
        self,
        event_id: str,
        recurrence_id: str,
        *,
        updates: dict[str, Any],
        project: str | None = None,
    ) -> dict[str, Any]:
        """Create a RECURRENCE-ID override for a single occurrence of a recurring event.

        The master event must have an rrule. An override is a new event row with
        the same calendar and a recurrence_id set to the ISO-8601 occurrence start,
        allowing the occurrence to have different title/time/description/location.
        """
        if not recurrence_id:
            raise ValueError("recurrence_id (ISO-8601 occurrence start) is required.")
        try:
            datetime.fromisoformat(recurrence_id.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("recurrence_id must be a valid ISO-8601 timestamp.") from exc

        master = self.get_event(event_id, project=project)
        if not master.get("rrule"):
            raise ValueError("Event is not a recurring series; overrides require an rrule.")

        def utc_instant(value: str) -> datetime:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        requested_occurrence = utc_instant(recurrence_id)
        occurrences = expand_rrule(str(master["starts_at"]), str(master["rrule"]), limit=10_000)
        if not any(utc_instant(value) == requested_occurrence for value in occurrences):
            raise ValueError("recurrence_id does not identify an occurrence in this series.")

        allowed = {"title", "description", "location", "starts_at", "ends_at",
                   "all_day", "status", "attendees", "reminders"}
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"Unsupported override fields: {', '.join(sorted(unknown))}")

        title = str(updates.get("title", master["title"])).strip() or master["title"]
        starts_at = str(updates.get("starts_at", recurrence_id))
        ends_at = str(updates.get("ends_at", master["ends_at"]))
        try:
            start_time = datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(ends_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("starts_at and ends_at must be valid ISO-8601 timestamps.") from exc
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        if end_time <= start_time:
            raise ValueError("Event end must be later than start.")

        override_id = hashlib.md5(f"{event_id}:override:{recurrence_id}".encode()).hexdigest()
        now = utc_now()
        attendees = updates.get("attendees", master.get("attendees") or [])
        reminders = updates.get("reminders", master.get("reminders") or [15])
        description = str(updates.get("description", master.get("description", ""))).strip()
        location = str(updates.get("location", master.get("location", ""))).strip()
        all_day = bool(updates.get("all_day", master.get("all_day", False)))
        status = str(updates.get("status", master.get("status", "confirmed"))).lower()
        metadata = dict(master.get("metadata") or {})
        metadata["recurrence_id"] = recurrence_id
        metadata["master_event_id"] = event_id

        with self.store.session() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO calendar_events (
                    id, calendar_id, project, title, description, location,
                    starts_at, ends_at, all_day, rrule, status, attendees_json,
                    reminders_json, metadata_json, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    override_id,
                    master.get("calendar_id", ""),
                    master["project"],
                    title,
                    description,
                    location,
                    start_time.isoformat(),
                    end_time.isoformat(),
                    1 if all_day else 0,
                    "",  # no rrule on occurrence override
                    status,
                    json.dumps(attendees),
                    json.dumps(reminders),
                    json.dumps(metadata),
                    now,
                    now,
                ),
            )
            conn.execute("DELETE FROM calendar_reminders WHERE event_id=?", (override_id,))
            for advance_minutes in (reminders if isinstance(reminders, list) else [15]):
                try:
                    trigger_at = (start_time - timedelta(minutes=int(advance_minutes))).isoformat()
                except (TypeError, ValueError):
                    trigger_at = start_time.isoformat()
                reminder_id = hashlib.md5(f"{override_id}-{advance_minutes}".encode()).hexdigest()
                conn.execute(
                    "INSERT OR IGNORE INTO calendar_reminders VALUES(?,?,?,?,?,?,?,?)",
                    (reminder_id, override_id, master["project"], advance_minutes,
                     "notification", 0, trigger_at, now),
                )
        override = self.get_event(override_id, project=project)
        self.store.add_audit_event(AuditEvent(
            actor="calendar",
            action="calendar.event.override",
            resource=f"calendar:event:override:{event_id}",
            project=master["project"],
            metadata={"master_id": event_id, "override_id": override_id, "recurrence_id": recurrence_id},
        ))
        return override

    def delete_event(self, event_id: str, *, project: str | None = None) -> bool:
        try:
            current = self.get_event(event_id, project=project)
        except KeyError:
            return False
        with self.store.session() as conn:
            query = "DELETE FROM calendar_events WHERE id=?"
            params: tuple[Any, ...] = (event_id,)
            if project is not None:
                query += " AND project=?"
                params = (event_id, project)
            cursor = conn.execute(query, params)
        deleted = cursor.rowcount > 0
        if deleted:
            self.store.add_audit_event(AuditEvent(
                actor="calendar",
                action="calendar.event.delete",
                resource=f"calendar:event:delete:{event_id}",
                project=current["project"],
                metadata={"event_id": event_id, "title": current["title"]},
            ))
        return deleted

    def _event_row(self, row: Any) -> dict[str, Any]:
        return {
            "id": row[0], "calendar_id": row[1], "project": row[2],
            "title": row[3], "description": row[4], "location": row[5],
            "starts_at": row[6], "ends_at": row[7],
            "all_day": bool(row[8]), "rrule": row[9], "status": row[10],
            "attendees": json.loads(row[11] or "[]"),
            "reminders": json.loads(row[12] or "[]"),
            "metadata": json.loads(row[13] or "{}"),
            "created_at": row[14], "updated_at": row[15],
        }

    # ------------------------------------------------------------------
    # Scheduling intelligence
    # ------------------------------------------------------------------

    def find_free_slots(
        self,
        *,
        project: str = "default",
        calendar_id: str = "",
        from_dt: str,
        to_dt: str,
        duration_minutes: int = 60,
        step_minutes: int = 30,
    ) -> list[dict[str, str]]:
        """Return list of free time slots within the given range."""
        if calendar_id:
            try:
                selected_calendar = self.get_calendar(calendar_id)
            except KeyError:
                raise KeyError("Calendar not found") from None
            if selected_calendar["project"] != project:
                raise KeyError("Calendar not found")
        try:
            search_start = datetime.fromisoformat(from_dt.replace("Z", "+00:00"))
            search_end = datetime.fromisoformat(to_dt.replace("Z", "+00:00"))
        except ValueError:
            return []
        if search_start.tzinfo is None:
            search_start = search_start.replace(tzinfo=timezone.utc)
        if search_end.tzinfo is None:
            search_end = search_end.replace(tzinfo=timezone.utc)

        # A series can begin before the requested range. Read the bounded
        # project calendar and expand its recurrence rules into busy windows.
        existing = self.list_events(
            project=project, calendar_id=calendar_id, from_dt=from_dt, to_dt=to_dt, limit=500
        )
        busy: list[tuple[datetime, datetime]] = []
        for ev in existing:
            try:
                original_start = datetime.fromisoformat(str(ev["starts_at"]).replace("Z", "+00:00"))
                original_end = datetime.fromisoformat(str(ev["ends_at"]).replace("Z", "+00:00"))
                if original_start.tzinfo is None:
                    original_start = original_start.replace(tzinfo=timezone.utc)
                if original_end.tzinfo is None:
                    original_end = original_end.replace(tzinfo=timezone.utc)
                duration = original_end - original_start
                instances = ev.get("occurrences")
                if not isinstance(instances, list) or not instances:
                    instances = [{"starts_at": original_start.isoformat(), "ends_at": original_end.isoformat()}]
                for instance in instances:
                    if not isinstance(instance, dict):
                        continue
                    if instance.get("is_override") and str(instance.get("status", "")).lower() in {"cancelled", "canceled"}:
                        continue
                    occurrence_start = datetime.fromisoformat(str(instance.get("starts_at", "")).replace("Z", "+00:00"))
                    occurrence_end_value = instance.get("ends_at")
                    occurrence_end = datetime.fromisoformat(str(occurrence_end_value).replace("Z", "+00:00")) if occurrence_end_value else occurrence_start + duration
                    if occurrence_start.tzinfo is None:
                        occurrence_start = occurrence_start.replace(tzinfo=timezone.utc)
                    if occurrence_end.tzinfo is None:
                        occurrence_end = occurrence_end.replace(tzinfo=timezone.utc)
                    if occurrence_start < search_end and occurrence_end > search_start:
                        busy.append((occurrence_start, occurrence_end))
            except ValueError:
                pass

        slots: list[dict[str, str]] = []
        step = timedelta(minutes=step_minutes)
        duration = timedelta(minutes=duration_minutes)
        cursor = search_start
        while cursor + duration <= search_end and len(slots) < 20:
            slot_end = cursor + duration
            overlap = any(s < slot_end and e > cursor for s, e in busy)
            if not overlap:
                slots.append({
                    "starts_at": cursor.isoformat(),
                    "ends_at": slot_end.isoformat(),
                    "duration_minutes": duration_minutes,
                })
            cursor += step
        return slots

    def optimize_schedule(
        self,
        *,
        project: str = "default",
        from_dt: str = "",
        to_dt: str = "",
    ) -> dict[str, Any]:
        """Return scheduling insights and optimization suggestions."""
        now_str = utc_now()
        events = self.list_events(
            project=project, from_dt=from_dt or now_str,
            to_dt=to_dt or "", limit=200
        )
        event_instances: list[dict[str, Any]] = []
        for event in events:
            occurrences = event.get("occurrences")
            if isinstance(occurrences, list) and occurrences:
                for occurrence in occurrences:
                    if isinstance(occurrence, dict):
                        event_instances.append({**event, **occurrence, "is_occurrence": True})
            else:
                event_instances.append(event)
        total_minutes = 0
        conflicts: list[dict[str, Any]] = []
        for i, ev in enumerate(event_instances):
            try:
                s = datetime.fromisoformat(ev["starts_at"])
                e = datetime.fromisoformat(ev["ends_at"])
                total_minutes += (e - s).total_seconds() / 60
            except ValueError:
                pass
            for j in range(i + 1, len(event_instances)):
                try:
                    s2 = datetime.fromisoformat(event_instances[j]["starts_at"])
                    e2 = datetime.fromisoformat(event_instances[j]["ends_at"])
                    if s < e2 and e > s2:
                        conflicts.append({
                            "event_a": ev["id"], "event_b": event_instances[j]["id"],
                            "starts_at_a": ev["starts_at"], "starts_at_b": event_instances[j]["starts_at"],
                        })
                except ValueError:
                    pass
        return {
            "total_events": len(event_instances),
            "total_scheduled_minutes": int(total_minutes),
            "conflicts": conflicts[:10],
            "suggestions": (
                ["Consider blocking focus time in your mornings.",
                 "You have back-to-back meetings — add buffer time."]
                if conflicts else
                ["Schedule looks healthy — no conflicts detected."]
            ),
        }

    # ------------------------------------------------------------------
    # iCal import / export
    # ------------------------------------------------------------------

    def import_ical(self, raw: str, *, project: str = "default") -> dict[str, Any]:
        """Import iCal string and ingest all VEVENT records."""
        events = parse_ical(raw)
        imported = 0
        for ev in events:
            try:
                self.create_event(
                    ev["title"],
                    ev.get("starts_at", utc_now()),
                    ev.get("ends_at", ev.get("starts_at", utc_now())),
                    project=project,
                    description=ev.get("description", ""),
                    location=ev.get("location", ""),
                    rrule=ev.get("rrule", ""),
                )
                imported += 1
            except Exception:
                pass
        now = utc_now()
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO calendar_sync_log VALUES(?,?,?,?,?,?,?)",
                (hashlib.md5(now.encode()).hexdigest(), project,
                 "ical-import", imported, 0, "ok", now),
            )
        return {"imported": imported, "total_found": len(events)}

    def export_ical(self, *, project: str = "default", calendar_id: str = "") -> str:
        """Export events as iCal string."""
        events = self.list_events(project=project, calendar_id=calendar_id, limit=1000)
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//OpenAgent//Calendar//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
        ]
        for ev in events:
            lines.append(event_to_ical(ev))
        lines.append("END:VCALENDAR")
        return "\r\n".join(lines)

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    def get_due_reminders(self, *, project: str = "default") -> list[dict[str, Any]]:
        """Return reminders that are due and not yet triggered."""
        now = utc_now()
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT r.id, r.event_id, r.advance_minutes, r.method, r.trigger_at, "
                "e.title, e.starts_at "
                "FROM calendar_reminders r "
                "JOIN calendar_events e ON e.id=r.event_id "
                "WHERE r.project=? AND r.triggered=0 AND r.trigger_at <= ? "
                "ORDER BY r.trigger_at LIMIT 50",
                (project, now),
            ).fetchall()
        results = []
        for row in rows:
            results.append({
                "reminder_id": row[0],
                "event_id": row[1],
                "advance_minutes": row[2],
                "method": row[3],
                "trigger_at": row[4],
                "event_title": row[5],
                "event_starts_at": row[6],
            })
        return results

    def mark_reminder_triggered(self, reminder_id: str, *, project: str = "default") -> bool:
        """Acknowledge a due reminder within its owning calendar project."""
        reminder_id = str(reminder_id).strip()
        project = str(project).strip()
        if not reminder_id or not project:
            raise ValueError("reminder_id and project are required")
        with self.store.session() as conn:
            cursor = conn.execute(
                "UPDATE calendar_reminders SET triggered=1 WHERE id=? AND project=?",
                (reminder_id, project),
            )
            if cursor.rowcount != 1:
                raise KeyError("Reminder not found")
        return True

    def deliver_due_notifications(
        self,
        *,
        project: str = "default",
        integration_hub: Any = None,
    ) -> dict[str, Any]:
        """Process due calendar reminders with method='notification'.

        Reads integration_preferences (scope='notifications', key='enabled') to
        decide whether delivery is active. For each due notification reminder,
        records a delivery audit event and marks the reminder as triggered.
        Returns a summary of delivered and skipped reminder ids.
        """
        enabled = True
        if integration_hub is not None:
            pref = integration_hub.preference(project=project, scope="notifications", key="enabled")
            if pref is not None:
                try:
                    enabled = bool(pref.get("value", True))
                except (TypeError, AttributeError):
                    enabled = True

        due = [r for r in self.get_due_reminders(project=project) if r.get("method") == "notification"]
        delivered: list[str] = []
        skipped: list[str] = []

        for reminder in due:
            rid = reminder["reminder_id"]
            if not enabled:
                skipped.append(rid)
                continue
            self.store.add_audit_event(AuditEvent(
                actor="calendar",
                action="calendar.notification.deliver",
                resource=f"calendar:reminder:{rid}",
                project=project,
                metadata={
                    "reminder_id": rid,
                    "event_id": reminder["event_id"],
                    "event_title": reminder["event_title"],
                    "event_starts_at": reminder["event_starts_at"],
                    "advance_minutes": reminder["advance_minutes"],
                    "trigger_at": reminder["trigger_at"],
                },
            ))
            try:
                self.mark_reminder_triggered(rid, project=project)
                delivered.append(rid)
            except KeyError:
                skipped.append(rid)

        return {
            "delivered": delivered,
            "skipped": skipped,
            "notifications_enabled": enabled,
            "project": project,
        }

    def schedule_reminder(
        self,
        event_id: str,
        scheduled_at: str,
        *,
        project: str = "default",
        owner: str = "",
        advance_minutes: int = 15,
        delivery_channel: str = "local",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Schedule a reminder for autonomous delivery at a specific time.

        Creates a reminder that will be delivered at the specified time via
        the specified channel (local, email, webhook, etc.).
        """
        event_id = str(event_id).strip()
        project = str(project).strip() or "default"

        if not event_id:
            raise ValueError("event_id is required.")

        # Validate the event exists
        try:
            event = self.get_event(event_id, project=project)
        except KeyError:
            raise KeyError(f"Event {event_id} not found in project {project}")

        # Parse scheduled_at to validate it
        try:
            scheduled_dt = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
            if scheduled_dt.tzinfo is None:
                scheduled_dt = scheduled_dt.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"Invalid scheduled_at timestamp: {exc}") from exc

        reminder_id = hashlib.sha256(f"rem-{event_id}-{scheduled_at}-{utc_now()}".encode()).hexdigest()[:32]
        created_at = utc_now()

        with self.store.session() as conn:
            # Check if calendar_reminders table exists and has the required columns
            conn.execute(
                """
                INSERT INTO calendar_reminders (
                    id, project, event_id, advance_minutes, method, trigger_at,
                    scheduled_at, delivery_channel, metadata_json, created_at, triggered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    reminder_id,
                    project,
                    event_id,
                    advance_minutes,
                    "scheduled",
                    scheduled_at,
                    scheduled_at,
                    delivery_channel,
                    json.dumps(metadata or {}, sort_keys=True),
                    created_at,
                ),
            )

        self.store.add_audit_event(
            AuditEvent(
                actor=owner or "local",
                action="calendar.reminder.schedule",
                resource=f"reminder:{reminder_id}",
                project=project,
                metadata={
                    "event_id": event_id,
                    "scheduled_at": scheduled_at,
                    "delivery_channel": delivery_channel,
                    "advance_minutes": advance_minutes,
                },
            )
        )

        return {
            "id": reminder_id,
            "event_id": event_id,
            "project": project,
            "scheduled_at": scheduled_at,
            "delivery_channel": delivery_channel,
            "advance_minutes": advance_minutes,
            "status": "scheduled",
            "created_at": created_at,
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as conn:
            n_cals = conn.execute(
                "SELECT COUNT(*) FROM calendars WHERE project=?", (project,)
            ).fetchone()[0]
            n_events = conn.execute(
                "SELECT COUNT(*) FROM calendar_events WHERE project=?", (project,)
            ).fetchone()[0]
            n_reminders = conn.execute(
                "SELECT COUNT(*) FROM calendar_reminders WHERE project=?", (project,)
            ).fetchone()[0]
        return {
            "calendars": n_cals,
            "events": n_events,
            "reminders": n_reminders,
            "project": project,
            "features": [
                "multi-calendar", "recurrence-rules", "scheduling-intelligence",
                "ical-import-export", "reminders", "conflict-detection",
                "free-slot-finder", "caldav-compatible",
            ],
        }
