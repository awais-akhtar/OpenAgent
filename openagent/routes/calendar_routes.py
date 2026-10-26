"""Calendar and external calendar-sync HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "calendar"
ROUTES = (
    ("GET", "/api/calendar/status"),
    ("GET", "/api/calendar/calendars"),
    ("GET", "/api/calendar/events"),
    ("GET", "/api/calendar/events/{id}/occurrences"),
    ("POST", "/api/calendar/events"),
    ("PATCH", "/api/calendar/events/{event_id}"),
    ("DELETE", "/api/calendar/events/{event_id}"),
    ("POST", "/api/calendar/ical-import"),
    ("POST", "/api/calendar/ical-export"),
    ("GET", "/api/calendar/free-slots"),
    ("GET", "/api/calendar/reminders"),
    ("POST", "/api/calendar/reminders/{reminder_id}/acknowledge"),
    ("POST", "/api/calendar/reminders/schedule"),
    ("GET", "/api/calendar/optimize"),
    ("GET", "/api/calendar/providers"),
    ("POST", "/api/calendar/providers/register"),
    ("DELETE", "/api/calendar/providers/{provider_id}"),
    ("POST", "/api/calendar/providers/{provider_id}/reauth"),
    ("GET", "/api/calendar/conflicts"),
    ("POST", "/api/calendar/conflicts/{conflict_id}/resolve"),
    ("POST", "/api/calendar/sync"),
    ("GET", "/api/calendar/sync/{sync_job_id}"),
    ("POST", "/api/calendar/events/{event_id}/override"),
    ("POST", "/api/calendar/reminders/deliver"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class CalendarRouteDependencies:
    calendar_system: Any
    calendar_sync: Any
    first: Callable[[dict[str, list[str]], str], str | None]
    parse_bool: Callable[[object, bool], bool]
    parse_int: Callable[[object, int], int]
    query_owner: Callable[[dict[str, list[str]]], str]
    payload_owner: Callable[[dict[str, object], str], str]
    auth_scope_owner: Callable[[], str]
    read_json: Callable[[], dict[str, object]]
    integration_hub: Any = None


def handle_get(path: str, query: str, deps: CalendarRouteDependencies) -> JsonRouteResult | None:
    params = parse_qs(query)
    first = deps.first
    calendar_system = deps.calendar_system
    calendar_sync = deps.calendar_sync

    if path == "/api/calendar/status":
        return JsonRouteResult(calendar_system.status(project=first(params, "project") or "default"))
    if path == "/api/calendar/calendars":
        return JsonRouteResult({"calendars": calendar_system.list_calendars(project=first(params, "project") or "default")})
    if path == "/api/calendar/events":
        return JsonRouteResult({
            "events": calendar_system.list_events(
                project=first(params, "project") or "default",
                from_dt=first(params, "from") or "",
                to_dt=first(params, "to") or "",
                calendar_id=first(params, "calendar_id") or "",
                limit=deps.parse_int(first(params, "limit"), default=100),
            )
        })
    if path.startswith("/api/calendar/events/") and path.endswith("/occurrences"):
        event_id = unquote(path.removeprefix("/api/calendar/events/").removesuffix("/occurrences")).strip()
        if not event_id or "/" in event_id:
            return JsonRouteResult({"error": "event_id is required."}, HTTPStatus.BAD_REQUEST)
        from_date = first(params, "from") or ""
        until_date = first(params, "until") or ""
        if not from_date or not until_date:
            return JsonRouteResult({"error": "from and until query params required"}, HTTPStatus.BAD_REQUEST)
        try:
            occurrences = calendar_system.list_event_occurrences(
                event_id,
                project=first(params, "project") or "default",
                from_date=from_date,
                until_date=until_date,
            )
        except (KeyError, ValueError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult({"occurrences": occurrences})
    if path == "/api/calendar/free-slots":
        from_dt = first(params, "from") or ""
        to_dt = first(params, "to") or ""
        if not from_dt or not to_dt:
            return JsonRouteResult({"error": "from and to query params required"}, HTTPStatus.BAD_REQUEST)
        try:
            slots = calendar_system.find_free_slots(
                project=first(params, "project") or "default",
                calendar_id=first(params, "calendar_id") or "",
                from_dt=from_dt,
                to_dt=to_dt,
                duration_minutes=deps.parse_int(first(params, "duration"), default=60),
            )
        except KeyError:
            return JsonRouteResult({"error": "Calendar not found"}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult({"slots": slots})
    if path == "/api/calendar/reminders":
        return JsonRouteResult({"reminders": calendar_system.get_due_reminders(project=first(params, "project") or "default")})
    if path == "/api/calendar/optimize":
        return JsonRouteResult(calendar_system.optimize_schedule(
            project=first(params, "project") or "default",
            from_dt=first(params, "from") or "",
            to_dt=first(params, "to") or "",
        ))
    if path == "/api/calendar/providers":
        return JsonRouteResult({
            "providers": calendar_sync.list_providers(
                project=first(params, "project") or "default",
                owner=deps.query_owner(params),
            )
        })
    if path == "/api/calendar/conflicts":
        return JsonRouteResult({
            "conflicts": calendar_sync.list_conflicts(
                project=first(params, "project") or "default",
                status=first(params, "status") or "",
                owner=deps.query_owner(params),
            )
        })
    if path.startswith("/api/calendar/sync/"):
        job_id = unquote(path.removeprefix("/api/calendar/sync/")).strip()
        try:
            return JsonRouteResult(calendar_sync.get_sync_job(job_id, owner=deps.auth_scope_owner()))
        except KeyError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return None


def handle_post(path: str, deps: CalendarRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/calendar/"):
        return None
    payload = deps.read_json()
    calendar_system = deps.calendar_system
    calendar_sync = deps.calendar_sync

    try:
        reminder_prefix = "/api/calendar/reminders/"
        if path.startswith(reminder_prefix) and path.endswith("/acknowledge"):
            reminder_id = unquote(path.removeprefix(reminder_prefix).removesuffix("/acknowledge")).strip()
            if not reminder_id or "/" in reminder_id:
                return JsonRouteResult({"error": "reminder_id is required"}, HTTPStatus.BAD_REQUEST)
            project = str(payload.get("project", "default")).strip()
            if not project:
                return JsonRouteResult({"error": "project is required"}, HTTPStatus.BAD_REQUEST)
            calendar_system.mark_reminder_triggered(reminder_id, project=project)
            return JsonRouteResult({"acknowledged": True, "reminder_id": reminder_id})
        if path == "/api/calendar/reminders/schedule":
            # Schedule a reminder for autonomous delivery
            event_id = str(payload.get("event_id", "")).strip()
            if not event_id:
                return JsonRouteResult({"error": "event_id is required"}, HTTPStatus.BAD_REQUEST)
            scheduled_at = str(payload.get("scheduled_at", "")).strip()
            if not scheduled_at:
                return JsonRouteResult({"error": "scheduled_at is required"}, HTTPStatus.BAD_REQUEST)
            project = str(payload.get("project", "default")).strip() or "default"

            reminder = calendar_system.schedule_reminder(
                event_id,
                scheduled_at,
                project=project,
                owner=deps.payload_owner(payload, ""),
                advance_minutes=int(payload.get("advance_minutes", 15)),
                delivery_channel=str(payload.get("delivery_channel", "local")).strip() or "local",
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"reminder": reminder}, HTTPStatus.CREATED)
        if path == "/api/calendar/calendars":
            cal = calendar_system.create_calendar(
                str(payload.get("name", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                color=str(payload.get("color", "#17c964")),
                description=str(payload.get("description", "")),
            )
            return JsonRouteResult({"calendar": cal}, HTTPStatus.CREATED)
        if path == "/api/calendar/events":
            ev = calendar_system.create_event(
                str(payload.get("title", "")).strip(),
                str(payload.get("starts_at", "")),
                str(payload.get("ends_at", payload.get("starts_at", ""))),
                project=str(payload.get("project", "default")).strip() or "default",
                calendar_id=str(payload.get("calendar_id", "")),
                description=str(payload.get("description", "")),
                location=str(payload.get("location", "")),
                all_day=deps.parse_bool(payload.get("all_day"), default=False),
                rrule=str(payload.get("rrule", "")),
                attendees=[str(a) for a in payload.get("attendees", [])] if isinstance(payload.get("attendees"), list) else None,
                reminders=[int(r) for r in payload.get("reminders", [15])] if isinstance(payload.get("reminders"), list) else None,
            )
            return JsonRouteResult({"event": ev}, HTTPStatus.CREATED)
        if path == "/api/calendar/ical-import":
            ical_data = str(payload.get("ical", "")).strip()
            if not ical_data:
                return JsonRouteResult({"error": "ical data is required"}, HTTPStatus.BAD_REQUEST)
            return JsonRouteResult(calendar_system.import_ical(
                ical_data,
                project=str(payload.get("project", "default")).strip() or "default",
            ))
        if path == "/api/calendar/ical-export":
            ical_str = calendar_system.export_ical(
                project=str(payload.get("project", "default")).strip() or "default",
                calendar_id=str(payload.get("calendar_id", "")),
            )
            return JsonRouteResult({"ical": ical_str})
        if path == "/api/calendar/providers/register":
            result = calendar_sync.register_provider(
                str(payload.get("provider", "")),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=deps.payload_owner(payload, "local"),
                base_url=str(payload.get("base_url", "")),
                account_ref=str(payload.get("account_ref", "")),
                calendar_id=str(payload.get("calendar_id", "")),
                config=payload.get("config") if isinstance(payload.get("config"), dict) else None,
            )
            return JsonRouteResult(result, HTTPStatus.CREATED)
        reauth_prefix = "/api/calendar/providers/"
        if path.startswith(reauth_prefix) and path.endswith("/reauth"):
            provider_id = unquote(path.removeprefix(reauth_prefix).removesuffix("/reauth")).strip()
            if not provider_id or "/" in provider_id:
                return JsonRouteResult({"error": "provider_id is required."}, HTTPStatus.BAD_REQUEST)
            try:
                result = calendar_sync.reauthenticate_provider(
                    provider_id,
                    owner=deps.auth_scope_owner(),
                    config=payload.get("config") if isinstance(payload.get("config"), dict) else None,
                    account_ref=str(payload.get("account_ref", "")),
                )
            except KeyError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return JsonRouteResult({"provider": result})
        if path == "/api/calendar/sync":
            try:
                result = calendar_sync.run_sync(
                    str(payload.get("provider_id", "")),
                    direction=str(payload.get("direction", "pull")),
                    conflict_policy=str(payload.get("conflict_policy", "latest_update")),
                    owner=deps.auth_scope_owner(),
                )
            except KeyError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return JsonRouteResult(result, HTTPStatus.ACCEPTED)
        if path.startswith("/api/calendar/conflicts/") and path.endswith("/resolve"):
            conflict_id = unquote(path.removeprefix("/api/calendar/conflicts/").removesuffix("/resolve")).strip()
            try:
                result = calendar_sync.resolve_conflict(
                    conflict_id,
                    str(payload.get("resolution", "")),
                    owner=deps.auth_scope_owner(),
                )
            except KeyError as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return JsonRouteResult(result)
        override_prefix = "/api/calendar/events/"
        if path.startswith(override_prefix) and path.endswith("/override"):
            event_id = unquote(path.removeprefix(override_prefix).removesuffix("/override")).strip()
            if not event_id or "/" in event_id:
                return JsonRouteResult({"error": "event_id is required."}, HTTPStatus.BAD_REQUEST)
            recurrence_id = str(payload.get("recurrence_id", "")).strip()
            if not recurrence_id:
                return JsonRouteResult({"error": "recurrence_id is required."}, HTTPStatus.BAD_REQUEST)
            project = str(payload.get("project", "default")).strip() or "default"
            updates = {k: v for k, v in payload.items() if k not in ("project", "recurrence_id")}
            try:
                override = calendar_system.create_event_override(
                    event_id, recurrence_id, updates=updates, project=project
                )
            except KeyError:
                return JsonRouteResult({"error": "Event not found."}, HTTPStatus.NOT_FOUND)
            return JsonRouteResult({"override": override}, HTTPStatus.CREATED)
        if path == "/api/calendar/reminders/deliver":
            project = str(payload.get("project", "default")).strip() or "default"
            result = calendar_system.deliver_due_notifications(
                project=project,
                integration_hub=deps.integration_hub,
            )
            return JsonRouteResult(result)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    except KeyError:
        return JsonRouteResult({"error": "Reminder not found."}, HTTPStatus.NOT_FOUND)
    return None


def handle_patch(path: str, query: str, deps: CalendarRouteDependencies) -> JsonRouteResult | None:
    prefix = "/api/calendar/events/"
    if not path.startswith(prefix):
        return None
    event_id = unquote(path.removeprefix(prefix)).strip()
    if not event_id or "/" in event_id:
        return JsonRouteResult({"error": "event_id is required"}, HTTPStatus.BAD_REQUEST)
    payload = deps.read_json()
    params = parse_qs(query)
    project = str(payload.get("project") or deps.first(params, "project") or "default").strip() or "default"
    updates = {key: value for key, value in payload.items() if key != "project"}
    if not updates:
        return JsonRouteResult({"error": "At least one event field must be updated."}, HTTPStatus.BAD_REQUEST)
    try:
        event = deps.calendar_system.update_event(event_id, updates=updates, project=project)
        return JsonRouteResult({"event": event})
    except KeyError:
        return JsonRouteResult({"error": "Event not found."}, HTTPStatus.NOT_FOUND)
    except (TypeError, ValueError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def handle_delete(path: str, query: str, deps: CalendarRouteDependencies) -> JsonRouteResult | None:
    provider_prefix = "/api/calendar/providers/"
    if path.startswith(provider_prefix):
        provider_id = unquote(path.removeprefix(provider_prefix).strip("/"))
        if not provider_id or "/" in provider_id:
            return JsonRouteResult({"error": "provider_id is required."}, HTTPStatus.BAD_REQUEST)
        try:
            result = deps.calendar_sync.delete_provider(provider_id, owner=deps.auth_scope_owner())
        except KeyError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)
    prefix = "/api/calendar/events/"
    if not path.startswith(prefix):
        return None
    event_id = unquote(path.removeprefix(prefix)).strip()
    if not event_id or "/" in event_id:
        return JsonRouteResult({"error": "event_id is required"}, HTTPStatus.BAD_REQUEST)
    params = parse_qs(query)
    project = str(deps.first(params, "project") or "default").strip() or "default"
    if not deps.calendar_system.delete_event(event_id, project=project):
        return JsonRouteResult({"error": "Event not found."}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult({"deleted": True, "id": event_id})
