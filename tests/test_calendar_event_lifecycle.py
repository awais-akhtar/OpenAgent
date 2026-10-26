from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from openagent.calendar_system import CalendarSystem, expand_rrule
from openagent.auth import required_scope_for_path
from openagent.routes import calendar_routes
from openagent.integration_hub import IntegrationHub
from openagent.storage import OpenAgentStore


class CalendarEventLifecycleTests(unittest.TestCase):
    def test_recurring_event_blocks_free_slots_even_when_series_started_before_range(self) -> None:
        with TemporaryDirectory() as directory:
            calendar = CalendarSystem(OpenAgentStore(directory))
            calendar.create_event(
                "Daily stand-up", "2026-10-24T09:00:00+00:00", "2026-10-24T10:00:00+00:00",
                project="default", reminders=[], rrule="FREQ=DAILY;COUNT=4",
            )

            slots = calendar.find_free_slots(
                project="default", from_dt="2026-10-26T09:00:00+00:00",
                to_dt="2026-10-26T11:00:00+00:00", duration_minutes=60, step_minutes=60,
            )

            self.assertEqual([slot["starts_at"] for slot in slots], ["2026-10-26T10:00:00+00:00"])

    def test_event_listing_returns_occurrences_in_range_for_a_prior_series(self) -> None:
        with TemporaryDirectory() as directory:
            calendar = CalendarSystem(OpenAgentStore(directory))
            event = calendar.create_event(
                "Daily stand-up", "2026-10-24T09:00:00+00:00", "2026-10-24T10:00:00+00:00",
                project="default", reminders=[], rrule="FREQ=DAILY;COUNT=4",
            )
            listed = calendar.list_events(
                project="default", from_dt="2026-10-26T00:00:00+00:00",
                to_dt="2026-10-27T00:00:00+00:00",
            )
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["id"], event["id"])
            self.assertEqual(listed[0]["starts_at"], "2026-10-24T09:00:00+00:00")
            self.assertEqual(listed[0]["occurrences"], [{
                "starts_at": "2026-10-26T09:00:00+00:00",
                "ends_at": "2026-10-26T10:00:00+00:00",
            }])

    def test_occurrence_override_replaces_only_the_matching_generated_occurrence(self) -> None:
        with TemporaryDirectory() as directory:
            calendar = CalendarSystem(OpenAgentStore(directory))
            master = calendar.create_event(
                "Daily stand-up", "2026-10-24T09:00:00+00:00", "2026-10-24T10:00:00+00:00",
                project="default", reminders=[], rrule="FREQ=DAILY;COUNT=4",
            )
            calendar.create_event_override(
                master["id"], "2026-10-26T09:00:00+00:00",
                updates={
                    "title": "Daily stand-up (moved)",
                    "starts_at": "2026-10-26T11:00:00+00:00",
                    "ends_at": "2026-10-26T12:00:00+00:00",
                    "location": "Room 4",
                },
                project="default",
            )

            listed = calendar.list_events(
                project="default", from_dt="2026-10-26T00:00:00+00:00",
                to_dt="2026-10-27T00:00:00+00:00",
            )

            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["id"], master["id"])
            self.assertEqual(len(listed[0]["occurrences"]), 1)
            occurrence = listed[0]["occurrences"][0]
            self.assertEqual(occurrence["recurrence_id"], "2026-10-26T09:00:00+00:00")
            self.assertEqual(occurrence["starts_at"], "2026-10-26T11:00:00+00:00")
            self.assertEqual(occurrence["ends_at"], "2026-10-26T12:00:00+00:00")
            self.assertEqual(occurrence["title"], "Daily stand-up (moved)")
            self.assertTrue(occurrence["is_override"])

    def test_occurrence_override_is_loaded_when_moved_outside_the_query_range(self) -> None:
        with TemporaryDirectory() as directory:
            calendar = CalendarSystem(OpenAgentStore(directory))
            master = calendar.create_event(
                "Daily stand-up", "2026-10-24T09:00:00+00:00", "2026-10-24T10:00:00+00:00",
                project="default", reminders=[], rrule="FREQ=DAILY;COUNT=4",
            )
            calendar.create_event_override(
                master["id"], "2026-10-26T09:00:00+00:00",
                updates={
                    "title": "Daily stand-up (rescheduled)",
                    "starts_at": "2026-12-01T11:00:00+00:00",
                    "ends_at": "2026-12-01T12:00:00+00:00",
                },
                project="default",
            )

            original_window = calendar.list_events(
                project="default", from_dt="2026-10-26T00:00:00+00:00",
                to_dt="2026-10-27T00:00:00+00:00",
            )
            self.assertEqual(original_window, [])
            original_slot = calendar.find_free_slots(
                project="default", from_dt="2026-10-26T09:00:00+00:00",
                to_dt="2026-10-26T10:00:00+00:00", duration_minutes=60, step_minutes=60,
            )
            self.assertEqual(len(original_slot), 1)

            listed = calendar.list_events(
                project="default", from_dt="2026-12-01T00:00:00+00:00",
                to_dt="2026-12-02T00:00:00+00:00",
            )
            self.assertEqual(len(listed), 1)
            occurrence = listed[0]["occurrences"][0]
            self.assertEqual(occurrence["recurrence_id"], "2026-10-26T09:00:00+00:00")
            self.assertEqual(occurrence["starts_at"], "2026-12-01T11:00:00+00:00")
            self.assertEqual(occurrence["title"], "Daily stand-up (rescheduled)")
            self.assertTrue(occurrence["is_override"])

            free_slots = calendar.find_free_slots(
                project="default", from_dt="2026-12-01T11:00:00+00:00",
                to_dt="2026-12-01T12:00:00+00:00", duration_minutes=60, step_minutes=60,
            )
            self.assertEqual(free_slots, [])

    def test_occurrence_override_rejects_timestamp_not_in_series(self) -> None:
        with TemporaryDirectory() as directory:
            calendar = CalendarSystem(OpenAgentStore(directory))
            master = calendar.create_event(
                "Daily stand-up", "2026-10-24T09:00:00+00:00", "2026-10-24T10:00:00+00:00",
                project="default", reminders=[], rrule="FREQ=DAILY;COUNT=4",
            )

            with self.assertRaisesRegex(ValueError, "does not identify an occurrence"):
                calendar.create_event_override(
                    master["id"], "2026-10-26T09:30:00+00:00",
                    updates={"title": "Invalid override"}, project="default",
                )

    def test_schedule_summary_counts_recurring_occurrences_and_finds_their_conflicts(self) -> None:
        with TemporaryDirectory() as directory:
            calendar = CalendarSystem(OpenAgentStore(directory))
            recurring = calendar.create_event(
                "Daily stand-up", "2026-10-24T09:00:00+00:00", "2026-10-24T10:00:00+00:00",
                project="default", reminders=[], rrule="FREQ=DAILY;COUNT=4",
            )
            overlapping = calendar.create_event(
                "Planning", "2026-10-26T09:30:00+00:00", "2026-10-26T10:30:00+00:00",
                project="default", reminders=[],
            )

            summary = calendar.optimize_schedule(
                project="default", from_dt="2026-10-26T00:00:00+00:00",
                to_dt="2026-10-27T00:00:00+00:00",
            )

            self.assertEqual(summary["total_events"], 2)
            self.assertEqual(summary["total_scheduled_minutes"], 120)
            self.assertEqual(summary["conflicts"], [{
                "event_a": recurring["id"], "event_b": overlapping["id"],
                "starts_at_a": "2026-10-26T09:00:00+00:00",
                "starts_at_b": "2026-10-26T09:30:00+00:00",
            }])

    def test_schedule_optimizer_route_applies_requested_date_range(self) -> None:
        with TemporaryDirectory() as directory:
            calendar = CalendarSystem(OpenAgentStore(directory))
            calendar.create_event(
                "Daily stand-up", "2026-10-24T09:00:00+00:00", "2026-10-24T10:00:00+00:00",
                project="default", reminders=[], rrule="FREQ=DAILY;COUNT=4",
            )
            calendar.create_event(
                "December planning", "2026-12-01T09:00:00+00:00", "2026-12-01T10:00:00+00:00",
                project="default", reminders=[],
            )
            deps = calendar_routes.CalendarRouteDependencies(
                calendar_system=calendar, calendar_sync=None,
                first=lambda params, name: params.get(name, [None])[0],
                parse_bool=lambda value, default=False: default,
                parse_int=lambda value, default=0: default,
                query_owner=lambda params: "local", payload_owner=lambda payload, default: default,
                auth_scope_owner=lambda: "local", read_json=lambda: {},
            )

            result = calendar_routes.handle_get(
                "/api/calendar/optimize",
                "project=default&from=2026-10-26T00%3A00%3A00%2B00%3A00&to=2026-10-27T00%3A00%3A00%2B00%3A00",
                deps,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.status, 200)
            self.assertEqual(result.payload["total_events"], 1)

    def test_monthly_recurrence_does_not_drift_and_until_accepts_ical_utc(self) -> None:
        self.assertEqual(
            expand_rrule("2026-01-31T09:00:00+00:00", "FREQ=MONTHLY;COUNT=3"),
            ["2026-01-31T09:00:00+00:00", "2026-02-28T09:00:00+00:00", "2026-03-31T09:00:00+00:00"],
        )
        self.assertEqual(
            expand_rrule("2026-10-24T09:00:00+00:00", "FREQ=DAILY;UNTIL=20261025T090000Z"),
            ["2026-10-24T09:00:00+00:00", "2026-10-25T09:00:00+00:00"],
        )

    def test_free_slot_search_respects_selected_calendar_and_project(self) -> None:
        with TemporaryDirectory() as directory:
            calendar = CalendarSystem(OpenAgentStore(directory))
            personal, work = calendar.list_calendars(project="default")[:2]
            calendar.create_event(
                "Personal busy block", "2026-10-24T10:00:00+00:00", "2026-10-24T11:00:00+00:00",
                project="default", calendar_id=personal["id"], reminders=[],
            )
            calendar.create_event(
                "Work busy block", "2026-10-24T11:00:00+00:00", "2026-10-24T12:00:00+00:00",
                project="default", calendar_id=work["id"], reminders=[],
            )
            search = dict(
                project="default", from_dt="2026-10-24T10:00:00+00:00",
                to_dt="2026-10-24T12:00:00+00:00", duration_minutes=30,
            )
            self.assertEqual(calendar.find_free_slots(**search), [])
            personal_slots = calendar.find_free_slots(**search, calendar_id=personal["id"])
            work_slots = calendar.find_free_slots(**search, calendar_id=work["id"])
            self.assertEqual([slot["starts_at"] for slot in personal_slots], [
                "2026-10-24T11:00:00+00:00", "2026-10-24T11:30:00+00:00",
            ])
            self.assertEqual([slot["starts_at"] for slot in work_slots], [
                "2026-10-24T10:00:00+00:00", "2026-10-24T10:30:00+00:00",
            ])
            foreign_calendar = calendar.create_calendar("Other project", project="elsewhere")
            with self.assertRaisesRegex(KeyError, "Calendar not found"):
                calendar.find_free_slots(**search, calendar_id=foreign_calendar["id"])
            route_deps = calendar_routes.CalendarRouteDependencies(
                calendar_system=calendar, calendar_sync=None,
                first=lambda params, name: params.get(name, [None])[0],
                parse_bool=lambda value, default=False: default,
                parse_int=lambda value, default=0: int(value) if value is not None else default,
                query_owner=lambda params: "local", payload_owner=lambda payload, default: default,
                auth_scope_owner=lambda: "local", read_json=lambda: {},
            )
            result = calendar_routes.handle_get(
                "/api/calendar/free-slots",
                f"project=default&calendar_id={personal['id']}&from=2026-10-24T10%3A00%3A00%2B00%3A00&to=2026-10-24T12%3A00%3A00%2B00%3A00&duration=30",
                route_deps,
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.status, 200)
            self.assertEqual(result.payload["slots"], personal_slots)
            denied = calendar_routes.handle_get(
                "/api/calendar/free-slots",
                f"project=default&calendar_id={foreign_calendar['id']}&from=2026-10-24T10%3A00%3A00%2B00%3A00&to=2026-10-24T12%3A00%3A00%2B00%3A00&duration=30",
                route_deps,
            )
            assert denied is not None
            self.assertEqual(denied.status, 404)
            self.assertEqual(denied.payload, {"error": "Calendar not found"})

    def test_reminder_acknowledgement_is_persistent_and_project_scoped(self) -> None:
        with TemporaryDirectory() as directory:
            calendar = CalendarSystem(OpenAgentStore(directory))
            event = calendar.create_event(
                "Due reminder", "2020-01-01T09:00:00+00:00", "2020-01-01T10:00:00+00:00",
                project="personal", reminders=[0],
            )
            reminder = calendar.get_due_reminders(project="personal")[0]
            self.assertEqual(reminder["event_id"], event["id"])
            reminder_id = reminder["reminder_id"]

            with self.assertRaises(KeyError):
                calendar.mark_reminder_triggered(reminder_id, project="other")
            self.assertEqual(len(calendar.get_due_reminders(project="personal")), 1)
            self.assertTrue(calendar.mark_reminder_triggered(reminder_id, project="personal"))
            self.assertEqual(calendar.get_due_reminders(project="personal"), [])
            self.assertTrue(calendar.mark_reminder_triggered(reminder_id, project="personal"))
            restarted_calendar = CalendarSystem(OpenAgentStore(directory))
            self.assertEqual(restarted_calendar.get_due_reminders(project="personal"), [])

    def test_notification_preference_controls_manual_delivery_without_losing_due_reminders(self) -> None:
        with TemporaryDirectory() as directory:
            store = OpenAgentStore(directory)
            calendar = CalendarSystem(store)
            integrations = IntegrationHub(
                store, health_provider=store.health, module_health_provider=lambda project: {"status": "ok"},
                metrics_provider=lambda: {"status": "ok"}, model_provider=lambda: {"status": "ok"},
                voice_provider=lambda project: {"status": "ok"}, developer_provider=lambda: {"status": "ok"},
                tool_provider=lambda: [], backup_provider=lambda: {}, backups_provider=lambda: [],
            )
            calendar.create_event(
                "Preference-controlled reminder", "2020-01-01T09:00:00+00:00", "2020-01-01T10:00:00+00:00",
                project="personal", reminders=[0],
            )
            reminder_id = calendar.get_due_reminders(project="personal")[0]["reminder_id"]

            integrations.set_preference(project="personal", scope="notifications", key="enabled", value=False)
            disabled_result = calendar.deliver_due_notifications(project="personal", integration_hub=integrations)
            self.assertEqual(disabled_result, {
                "delivered": [], "skipped": [reminder_id], "notifications_enabled": False, "project": "personal",
            })
            self.assertEqual([item["reminder_id"] for item in calendar.get_due_reminders(project="personal")], [reminder_id])
            self.assertFalse(any(event["action"] == "calendar.notification.deliver" for event in store.list_audit_events(project="personal", limit=100)))

            integrations.set_preference(project="personal", scope="notifications", key="enabled", value=True)
            enabled_result = calendar.deliver_due_notifications(project="personal", integration_hub=integrations)
            self.assertEqual(enabled_result, {
                "delivered": [reminder_id], "skipped": [], "notifications_enabled": True, "project": "personal",
            })
            self.assertEqual(calendar.get_due_reminders(project="personal"), [])
            deliveries = [event for event in store.list_audit_events(project="personal", limit=100) if event["action"] == "calendar.notification.deliver"]
            self.assertEqual(len(deliveries), 1)
            self.assertEqual(deliveries[0]["metadata"]["reminder_id"], reminder_id)

    def test_reminder_acknowledgement_route_contract_and_write_scope(self) -> None:
        routes = {(route["method"], route["path"]) for route in calendar_routes.contract()["routes"]}
        path = "/api/calendar/reminders/{reminder_id}/acknowledge"
        self.assertIn(("POST", path), routes)
        self.assertEqual(required_scope_for_path("/api/calendar/reminders/rem-1/acknowledge", "POST"), "calendar:write")

        with TemporaryDirectory() as directory:
            calendar = CalendarSystem(OpenAgentStore(directory))
            calendar.create_event("Due", "2020-01-01T09:00:00+00:00", "2020-01-01T10:00:00+00:00", project="personal", reminders=[0])
            reminder_id = calendar.get_due_reminders(project="personal")[0]["reminder_id"]
            deps = calendar_routes.CalendarRouteDependencies(
                calendar_system=calendar, calendar_sync=None,
                first=lambda params, name: params.get(name, [None])[0],
                parse_bool=lambda value, default=False: default,
                parse_int=lambda value, default=0: default,
                query_owner=lambda params: "local", payload_owner=lambda payload, default: default,
                auth_scope_owner=lambda: "local", read_json=lambda: {"project": "personal"},
            )
            result = calendar_routes.handle_post(f"/api/calendar/reminders/{reminder_id}/acknowledge", deps)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.status, 200)
            self.assertEqual(result.payload, {"acknowledged": True, "reminder_id": reminder_id})
            self.assertEqual(calendar.get_due_reminders(project="personal"), [])

    def test_update_delete_are_project_scoped_persistent_and_validated(self) -> None:
        with TemporaryDirectory() as directory:
            store = OpenAgentStore(directory)
            calendar = CalendarSystem(store)
            event = calendar.create_event(
                "Planning", "2026-10-24T09:00:00+00:00", "2026-10-24T10:00:00+00:00",
                project="personal", reminders=[15],
            )

            with self.assertRaises(KeyError):
                calendar.update_event(event["id"], project="other", updates={"title": "forged"})
            self.assertFalse(calendar.delete_event(event["id"], project="other"))

            with self.assertRaisesRegex(ValueError, "later"):
                calendar.update_event(
                    event["id"], project="personal",
                    updates={"starts_at": "2026-10-24T11:00:00+00:00"},
                )
            updated = calendar.update_event(
                event["id"], project="personal",
                updates={"title": "Planning revised", "reminders": [0, 30]},
            )
            self.assertEqual(updated["title"], "Planning revised")
            self.assertEqual(updated["reminders"], [0, 30])
            self.assertEqual(calendar.get_event(event["id"], project="personal")["title"], "Planning revised")
            self.assertEqual(len(calendar.list_events(project="personal")), 1)
            self.assertEqual(len(calendar.list_events(project="other")), 0)
            self.assertTrue(calendar.delete_event(event["id"], project="personal"))
            self.assertFalse(calendar.delete_event(event["id"], project="personal"))
            self.assertEqual(calendar.list_events(project="personal"), [])


if __name__ == "__main__":
    unittest.main()
