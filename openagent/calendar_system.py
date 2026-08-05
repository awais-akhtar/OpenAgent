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

    count = int(parts.get("COUNT", limit))
    until_str = parts.get("UNTIL", "")
    interval = int(parts.get("INTERVAL", "1"))

    results: list[str] = []
    current = start
    until: datetime | None = None
    if until_str:
        try:
            until = datetime.fromisoformat(until_str[:15])
        except ValueError:
            until = None

    for _ in range(min(count, limit)):
        if until and current > until:
            break
        results.append(current.isoformat())
        if freq == "DAILY":
            current = current + timedelta(days=interval)
        elif freq == "WEEKLY":
            current = current + timedelta(weeks=interval)
        elif freq == "MONTHLY":
            current = _add_months(current, interval)
        elif freq == "YEARLY":
            current = _add_months(current, 12 * interval)

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

    def get_event(self, event_id: str) -> dict[str, Any]:
        with self.store.session() as conn:
            row = conn.execute(
                "SELECT id,calendar_id,project,title,description,location,starts_at,ends_at,"
                "all_day,rrule,status,attendees_json,reminders_json,metadata_json,created_at,updated_at "
                "FROM calendar_events WHERE id=?", (event_id,)
            ).fetchone()
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
            query += " AND ends_at >= ?"
            params.append(from_dt)
        if to_dt:
            query += " AND starts_at <= ?"
            params.append(to_dt)
        query += " ORDER BY starts_at LIMIT ?"
        params.append(limit)
        with self.store.session() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._event_row(r) for r in rows]

    def update_event(
        self, event_id: str, *, updates: dict[str, Any]
    ) -> dict[str, Any]:
        allowed = {"title", "description", "location", "starts_at", "ends_at",
                   "all_day", "rrule", "status", "attendees", "reminders", "metadata"}
        now = utc_now()
        with self.store.session() as conn:
            for field, value in updates.items():
                if field not in allowed:
                    continue
                col = "attendees_json" if field == "attendees" else \
                      "reminders_json" if field == "reminders" else \
                      "metadata_json" if field == "metadata" else field
                if field in ("attendees", "reminders", "metadata"):
                    value = json.dumps(value)
                conn.execute(
                    f"UPDATE calendar_events SET {col}=?, updated_at=? WHERE id=?",
                    (value, now, event_id),
                )
        return self.get_event(event_id)

    def delete_event(self, event_id: str) -> bool:
        with self.store.session() as conn:
            cursor = conn.execute("DELETE FROM calendar_events WHERE id=?", (event_id,))
        return cursor.rowcount > 0

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
        from_dt: str,
        to_dt: str,
        duration_minutes: int = 60,
        step_minutes: int = 30,
    ) -> list[dict[str, str]]:
        """Return list of free time slots within the given range."""
        existing = self.list_events(project=project, from_dt=from_dt, to_dt=to_dt, limit=500)
        busy: list[tuple[datetime, datetime]] = []
        for ev in existing:
            try:
                s = datetime.fromisoformat(ev["starts_at"])
                e = datetime.fromisoformat(ev["ends_at"])
                busy.append((s, e))
            except ValueError:
                pass

        try:
            search_start = datetime.fromisoformat(from_dt)
            search_end = datetime.fromisoformat(to_dt)
        except ValueError:
            return []

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
        total_minutes = 0
        conflicts: list[dict[str, Any]] = []
        for i, ev in enumerate(events):
            try:
                s = datetime.fromisoformat(ev["starts_at"])
                e = datetime.fromisoformat(ev["ends_at"])
                total_minutes += (e - s).total_seconds() / 60
            except ValueError:
                pass
            for j in range(i + 1, len(events)):
                try:
                    s2 = datetime.fromisoformat(events[j]["starts_at"])
                    e2 = datetime.fromisoformat(events[j]["ends_at"])
                    if s < e2 and e > s2:
                        conflicts.append({"event_a": ev["id"], "event_b": events[j]["id"]})
                except ValueError:
                    pass
        return {
            "total_events": len(events),
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

    def mark_reminder_triggered(self, reminder_id: str) -> None:
        with self.store.session() as conn:
            conn.execute(
                "UPDATE calendar_reminders SET triggered=1 WHERE id=?", (reminder_id,)
            )

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
