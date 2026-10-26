from __future__ import annotations

from http import HTTPStatus
from unittest import TestCase
from unittest.mock import Mock

from openagent.routes.calendar_routes import (
    ROUTES, CalendarRouteDependencies, contract, handle_delete, handle_get, handle_post,
)


def _deps(payload=None, *, owner=""):
    cal_system = Mock()
    cal_sync = Mock()
    read_json = Mock(return_value=payload if payload is not None else {})
    first = lambda params, key: (params.get(key, [None])[0] or None)
    parse_bool = lambda value, default: default if value is None else bool(value)
    parse_int = lambda value, default: default if value is None else int(value)
    query_owner = lambda params: owner
    payload_owner = lambda p, default: owner or default
    return cal_system, cal_sync, CalendarRouteDependencies(
        calendar_system=cal_system,
        calendar_sync=cal_sync,
        first=first,
        parse_bool=parse_bool,
        parse_int=parse_int,
        query_owner=query_owner,
        payload_owner=payload_owner,
        auth_scope_owner=lambda: owner,
        read_json=read_json,
    )


class CalendarRouteContractTests(TestCase):
    def test_contract_has_provider_delete_and_reauth(self):
        self.assertIn(("DELETE", "/api/calendar/providers/{provider_id}"), ROUTES)
        self.assertIn(("POST", "/api/calendar/providers/{provider_id}/reauth"), ROUTES)
        self.assertEqual(contract()["group"], "calendar")


class CalendarProviderDeleteTests(TestCase):
    def test_delete_provider_success(self):
        cal_system, cal_sync, deps = _deps()
        cal_sync.delete_provider.return_value = {"deleted": True, "id": "prov-1"}
        result = handle_delete("/api/calendar/providers/prov-1", "", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertTrue(result.payload["deleted"])
        cal_sync.delete_provider.assert_called_once_with("prov-1", owner="")

    def test_delete_provider_not_found(self):
        cal_system, cal_sync, deps = _deps()
        cal_sync.delete_provider.side_effect = KeyError("not found")
        result = handle_delete("/api/calendar/providers/missing", "", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_delete_provider_bad_request_on_empty_id(self):
        cal_system, cal_sync, deps = _deps()
        result = handle_delete("/api/calendar/providers/", "", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_delete_provider_enforces_owner(self):
        cal_system, cal_sync, deps = _deps(owner="owner-A")
        cal_sync.delete_provider.return_value = {"deleted": True, "id": "prov-2"}
        handle_delete("/api/calendar/providers/prov-2", "", deps)
        cal_sync.delete_provider.assert_called_once_with("prov-2", owner="owner-A")

    def test_delete_event_returns_deleted(self):
        cal_system, cal_sync, deps = _deps()
        cal_system.delete_event.return_value = True
        result = handle_delete("/api/calendar/events/evt-1", "project=default", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertTrue(result.payload["deleted"])

    def test_delete_event_not_found(self):
        cal_system, cal_sync, deps = _deps()
        cal_system.delete_event.return_value = False
        result = handle_delete("/api/calendar/events/evt-missing", "project=default", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)


class CalendarProviderReauthTests(TestCase):
    def test_reauth_provider_success(self):
        cal_system, cal_sync, deps = _deps(payload={"account_ref": "token-xyz"})
        cal_sync.reauthenticate_provider.return_value = {"id": "prov-1", "status": "active"}
        result = handle_post("/api/calendar/providers/prov-1/reauth", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertIn("provider", result.payload)
        cal_sync.reauthenticate_provider.assert_called_once()

    def test_reauth_provider_not_found(self):
        cal_system, cal_sync, deps = _deps(payload={})
        cal_sync.reauthenticate_provider.side_effect = KeyError("not found")
        result = handle_post("/api/calendar/providers/missing/reauth", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_reauth_provider_empty_id_bad_request(self):
        cal_system, cal_sync, deps = _deps(payload={})
        result = handle_post("/api/calendar/providers//reauth", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_reauth_passes_owner_scope(self):
        cal_system, cal_sync, deps = _deps(payload={}, owner="owner-B")
        cal_sync.reauthenticate_provider.return_value = {"id": "prov-3", "status": "active"}
        handle_post("/api/calendar/providers/prov-3/reauth", deps)
        _, kwargs = cal_sync.reauthenticate_provider.call_args
        self.assertEqual(kwargs.get("owner"), "owner-B")


class CalendarEventOverrideTests(TestCase):
    def test_override_route_in_contract(self):
        routes = {(r["method"], r["path"]) for r in contract()["routes"]}
        self.assertIn(("POST", "/api/calendar/events/{event_id}/override"), routes)

    def test_override_creates_override(self):
        payload = {"recurrence_id": "2026-11-01T10:00:00", "title": "Special occurrence", "project": "default"}
        cal_system, cal_sync, deps = _deps(payload=payload)
        override_event = {"id": "ov-1", "title": "Special occurrence"}
        cal_system.create_event_override.return_value = override_event
        result = handle_post("/api/calendar/events/evt-1/override", deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        self.assertEqual(result.payload["override"]["id"], "ov-1")
        cal_system.create_event_override.assert_called_once_with(
            "evt-1", "2026-11-01T10:00:00",
            updates={"title": "Special occurrence"},
            project="default",
        )

    def test_override_missing_recurrence_id(self):
        payload = {"title": "No rid"}
        cal_system, cal_sync, deps = _deps(payload=payload)
        result = handle_post("/api/calendar/events/evt-1/override", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_override_event_not_found(self):
        payload = {"recurrence_id": "2026-11-01T10:00:00"}
        cal_system, cal_sync, deps = _deps(payload=payload)
        cal_system.create_event_override.side_effect = KeyError("not found")
        result = handle_post("/api/calendar/events/evt-1/override", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_override_value_error_returns_bad_request(self):
        payload = {"recurrence_id": "bad-timestamp"}
        cal_system, cal_sync, deps = _deps(payload=payload)
        cal_system.create_event_override.side_effect = ValueError("bad ts")
        result = handle_post("/api/calendar/events/evt-1/override", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)


class CalendarNotificationDeliveryTests(TestCase):
    def test_deliver_route_in_contract(self):
        routes = {(r["method"], r["path"]) for r in contract()["routes"]}
        self.assertIn(("POST", "/api/calendar/reminders/deliver"), routes)

    def test_deliver_calls_deliver_due_notifications(self):
        payload = {"project": "default"}
        cal_system, cal_sync, deps = _deps(payload=payload)
        cal_system.deliver_due_notifications.return_value = {
            "delivered": ["r1"], "skipped": [], "notifications_enabled": True, "project": "default"
        }
        result = handle_post("/api/calendar/reminders/deliver", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertEqual(result.payload["delivered"], ["r1"])
        cal_system.deliver_due_notifications.assert_called_once_with(
            project="default", integration_hub=None
        )

    def test_deliver_passes_integration_hub(self):
        from unittest.mock import Mock as M
        payload = {"project": "proj-x"}
        hub = M()
        cal_system, cal_sync, deps_base = _deps(payload=payload)
        # rebuild deps with integration_hub set
        from openagent.routes.calendar_routes import CalendarRouteDependencies
        import openagent.routes.calendar_routes as cr
        deps = CalendarRouteDependencies(
            calendar_system=cal_system, calendar_sync=cal_sync,
            first=deps_base.first, parse_bool=deps_base.parse_bool,
            parse_int=deps_base.parse_int, query_owner=deps_base.query_owner,
            payload_owner=deps_base.payload_owner, auth_scope_owner=deps_base.auth_scope_owner,
            read_json=deps_base.read_json, integration_hub=hub,
        )
        cal_system.deliver_due_notifications.return_value = {
            "delivered": [], "skipped": [], "notifications_enabled": True, "project": "proj-x"
        }
        handle_post("/api/calendar/reminders/deliver", deps)
        cal_system.deliver_due_notifications.assert_called_once_with(
            project="proj-x", integration_hub=hub
        )


class CalendarEventOccurrencesTests(TestCase):
    def test_occurrences_route_in_contract(self):
        routes = {(r["method"], r["path"]) for r in contract()["routes"]}
        self.assertIn(("GET", "/api/calendar/events/{id}/occurrences"), routes)

    def test_occurrences_success(self):
        cal_system, cal_sync, deps = _deps()
        occurrences = [
            {"starts_at": "2026-11-01T10:00:00+00:00", "ends_at": "2026-11-01T11:00:00+00:00", "recurrence_id": "2026-11-01T10:00:00+00:00"},
            {"starts_at": "2026-11-08T10:00:00+00:00", "ends_at": "2026-11-08T11:00:00+00:00", "recurrence_id": "2026-11-08T10:00:00+00:00"},
        ]
        cal_system.list_event_occurrences.return_value = occurrences
        result = handle_get("/api/calendar/events/evt-1/occurrences", "from=2026-11-01T00:00:00&until=2026-11-30T23:59:59", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertEqual(len(result.payload["occurrences"]), 2)
        cal_system.list_event_occurrences.assert_called_once()
        args, kwargs = cal_system.list_event_occurrences.call_args
        self.assertEqual(args[0], "evt-1")
        self.assertEqual(kwargs.get("from_date"), "2026-11-01T00:00:00")
        self.assertEqual(kwargs.get("until_date"), "2026-11-30T23:59:59")

    def test_occurrences_missing_from_date(self):
        cal_system, cal_sync, deps = _deps()
        result = handle_get("/api/calendar/events/evt-1/occurrences", "until=2026-11-30T23:59:59", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("error", result.payload)

    def test_occurrences_missing_until_date(self):
        cal_system, cal_sync, deps = _deps()
        result = handle_get("/api/calendar/events/evt-1/occurrences", "from=2026-11-01T00:00:00", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_occurrences_event_not_found(self):
        cal_system, cal_sync, deps = _deps()
        cal_system.list_event_occurrences.side_effect = KeyError("not found")
        result = handle_get("/api/calendar/events/missing-evt/occurrences", "from=2026-11-01T00:00:00&until=2026-11-30T23:59:59", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_occurrences_empty_id_bad_request(self):
        cal_system, cal_sync, deps = _deps()
        result = handle_get("/api/calendar/events//occurrences", "from=2026-11-01T00:00:00&until=2026-11-30T23:59:59", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_occurrences_with_slash_in_id_bad_request(self):
        cal_system, cal_sync, deps = _deps()
        result = handle_get("/api/calendar/events/a/b/occurrences", "from=2026-11-01T00:00:00&until=2026-11-30T23:59:59", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_occurrences_value_error_returns_not_found(self):
        cal_system, cal_sync, deps = _deps()
        cal_system.list_event_occurrences.side_effect = ValueError("invalid dates")
        result = handle_get("/api/calendar/events/evt-1/occurrences", "from=2026-11-01T00:00:00&until=2026-11-30T23:59:59", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)


class CalendarScheduleReminderTests(TestCase):
    def test_schedule_reminder_success(self):
        payload = {
            "event_id": "evt-1",
            "scheduled_at": "2026-11-15T10:00:00",
            "project": "default",
            "advance_minutes": 15,
            "delivery_channel": "local",
        }
        cal_system, cal_sync, deps = _deps(payload=payload, owner="user-1")
        cal_system.schedule_reminder.return_value = {
            "id": "rem-1",
            "event_id": "evt-1",
            "status": "scheduled",
            "scheduled_at": "2026-11-15T10:00:00",
        }
        result = handle_post("/api/calendar/reminders/schedule", deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        self.assertIn("reminder", result.payload)
        cal_system.schedule_reminder.assert_called_once()
        args, kwargs = cal_system.schedule_reminder.call_args
        self.assertEqual(args[0], "evt-1")
        self.assertEqual(args[1], "2026-11-15T10:00:00")
        self.assertEqual(kwargs.get("delivery_channel"), "local")

    def test_schedule_reminder_missing_event_id(self):
        payload = {
            "scheduled_at": "2026-11-15T10:00:00",
            "project": "default",
        }
        cal_system, cal_sync, deps = _deps(payload=payload)
        result = handle_post("/api/calendar/reminders/schedule", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("error", result.payload)

    def test_schedule_reminder_missing_scheduled_at(self):
        payload = {
            "event_id": "evt-1",
            "project": "default",
        }
        cal_system, cal_sync, deps = _deps(payload=payload)
        result = handle_post("/api/calendar/reminders/schedule", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_schedule_reminder_event_not_found(self):
        payload = {
            "event_id": "missing-evt",
            "scheduled_at": "2026-11-15T10:00:00",
            "project": "default",
        }
        cal_system, cal_sync, deps = _deps(payload=payload)
        cal_system.schedule_reminder.side_effect = KeyError("Event not found")
        result = handle_post("/api/calendar/reminders/schedule", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_schedule_reminder_invalid_timestamp(self):
        payload = {
            "event_id": "evt-1",
            "scheduled_at": "invalid-timestamp",
            "project": "default",
        }
        cal_system, cal_sync, deps = _deps(payload=payload)
        cal_system.schedule_reminder.side_effect = ValueError("Invalid timestamp")
        result = handle_post("/api/calendar/reminders/schedule", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_schedule_reminder_with_metadata(self):
        payload = {
            "event_id": "evt-1",
            "scheduled_at": "2026-11-15T10:00:00",
            "project": "default",
            "metadata": {"custom_field": "value"},
        }
        cal_system, cal_sync, deps = _deps(payload=payload)
        cal_system.schedule_reminder.return_value = {
            "id": "rem-1",
            "event_id": "evt-1",
            "status": "scheduled",
        }
        result = handle_post("/api/calendar/reminders/schedule", deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        _, kwargs = cal_system.schedule_reminder.call_args
        self.assertEqual(kwargs.get("metadata"), {"custom_field": "value"})
