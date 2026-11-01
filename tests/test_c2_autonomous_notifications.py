"""Tests for C2: Autonomous Notification Scheduling with delivery state tracking."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from urllib.request import Request, urlopen

from openagent.calendar_system import CalendarSystem
from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore


class C2AutonomousNotificationsTests(TestCase):
    """Test suite for autonomous notification scheduling with state tracking."""

    def test_schedule_reminder_for_future_delivery(self) -> None:
        """Test scheduling a reminder for autonomous delivery at a specific time."""
        with TemporaryDirectory() as tmp:
            calendar = CalendarSystem(OpenAgentStore(tmp))

            # Create a calendar and event
            cal = calendar.create_calendar("test_cal", project="default")

            event_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            event = calendar.create_event(
                "Meeting with team",
                event_time,
                event_time,
                project="default",
                calendar_id=cal["id"]
            )

            # Schedule reminder for 15 minutes before the event
            reminder_time = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()
            reminder = calendar.schedule_reminder(
                event["id"],
                reminder_time,
                project="default",
                advance_minutes=15,
                delivery_channel="local"
            )

            self.assertIsNotNone(reminder["id"])
            self.assertEqual(reminder["event_id"], event["id"])
            self.assertEqual(reminder["status"], "scheduled")
            self.assertEqual(reminder["delivery_channel"], "local")

    def test_get_due_reminders_filters_by_time(self) -> None:
        """Test that only past reminders are considered due."""
        with TemporaryDirectory() as tmp:
            calendar = CalendarSystem(OpenAgentStore(tmp))

            # Create event
            cal = calendar.create_calendar("test_cal", project="default")
            event_time = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
            event = calendar.create_event(
                "Future meeting",
                event_time,
                event_time,
                project="default",
                calendar_id=cal["id"]
            )

            # Schedule reminder in the past (should be due)
            past_time = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
            past_reminder = calendar.schedule_reminder(
                event["id"],
                past_time,
                project="default",
                advance_minutes=15
            )

            # Schedule reminder in the future (should not be due)
            future_time = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
            future_reminder = calendar.schedule_reminder(
                event["id"],
                future_time,
                project="default",
                advance_minutes=15
            )

            # Get due reminders
            due = calendar.get_due_reminders(project="default")
            due_ids = [r["reminder_id"] for r in due]

            self.assertIn(past_reminder["id"], due_ids)
            self.assertNotIn(future_reminder["id"], due_ids)

    def test_mark_reminder_triggered_updates_state(self) -> None:
        """Test that marking a reminder as triggered updates its state."""
        with TemporaryDirectory() as tmp:
            calendar = CalendarSystem(OpenAgentStore(tmp))

            # Create event and reminder
            cal = calendar.create_calendar("test_cal", project="default")
            event_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            event = calendar.create_event(
                "Test event",
                event_time,
                event_time,
                project="default",
                calendar_id=cal["id"]
            )

            reminder_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
            reminder = calendar.schedule_reminder(
                event["id"],
                reminder_time,
                project="default"
            )

            # Mark as triggered
            result = calendar.mark_reminder_triggered(reminder["id"], project="default")
            self.assertTrue(result)

            # Verify it's no longer in due reminders
            due = calendar.get_due_reminders(project="default")
            due_ids = [r["reminder_id"] for r in due]
            self.assertNotIn(reminder["id"], due_ids)

    def test_deliver_due_notifications_processes_reminders(self) -> None:
        """Test that delivering due notifications marks them as triggered."""
        with TemporaryDirectory() as tmp:
            calendar = CalendarSystem(OpenAgentStore(tmp))

            # Create events and reminders
            cal = calendar.create_calendar("test_cal", project="default")
            event_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            event = calendar.create_event(
                "Delivery test event",
                event_time,
                event_time,
                project="default",
                calendar_id=cal["id"]
            )

            reminder_time = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
            reminder = calendar.schedule_reminder(
                event["id"],
                reminder_time,
                project="default"
            )

            # Deliver due notifications
            result = calendar.deliver_due_notifications(project="default")

            self.assertIn("delivered", result)
            self.assertIn("skipped", result)
            self.assertIn(reminder["id"], result.get("delivered", []))

            # Owner-initiated delivery also clears the reminder, because the
            # owner pressed Deliver on a reminder they could see.
            due_after = calendar.get_due_reminders(project="default")
            due_ids = [r["reminder_id"] for r in due_after]
            self.assertNotIn(reminder["id"], due_ids)

    def test_delivery_state_transitions(self) -> None:
        """Test reminder state transitions: scheduled -> delivered."""
        with TemporaryDirectory() as tmp:
            calendar = CalendarSystem(OpenAgentStore(tmp))

            # Create event
            cal = calendar.create_calendar("test_cal", project="default")
            event_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            event = calendar.create_event(
                "State transition event",
                event_time,
                event_time,
                project="default",
                calendar_id=cal["id"]
            )

            # Schedule reminder
            reminder_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
            reminder = calendar.schedule_reminder(
                event["id"],
                reminder_time,
                project="default",
                advance_minutes=15
            )

            # Initial state: scheduled
            self.assertEqual(reminder["status"], "scheduled")

            # Transition to delivered
            calendar.mark_reminder_triggered(reminder["id"], project="default")

            # Verify: no longer in due reminders (effectively delivered)
            due = calendar.get_due_reminders(project="default")
            due_ids = [r["reminder_id"] for r in due]
            self.assertNotIn(reminder["id"], due_ids)

    def test_multiple_reminders_independent_states(self) -> None:
        """Test that multiple reminders can have independent delivery states."""
        with TemporaryDirectory() as tmp:
            calendar = CalendarSystem(OpenAgentStore(tmp))

            # Create calendar and event
            cal = calendar.create_calendar("test_cal", project="default")
            event_time = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
            event = calendar.create_event(
                "Multi-reminder event",
                event_time,
                event_time,
                project="default",
                calendar_id=cal["id"]
            )

            # Create multiple reminders (all in the past so they're due)
            reminders = []
            for i in range(3):
                reminder_time = (datetime.now(timezone.utc) - timedelta(minutes=5-i)).isoformat()
                reminder = calendar.schedule_reminder(
                    event["id"],
                    reminder_time,
                    project="default",
                    advance_minutes=15 + i
                )
                reminders.append(reminder)

            # Verify all are initially due
            due_before = calendar.get_due_reminders(project="default")
            due_ids_before = set(r["reminder_id"] for r in due_before)
            for reminder in reminders:
                self.assertIn(reminder["id"], due_ids_before)

            # Mark first reminder as triggered
            calendar.mark_reminder_triggered(reminders[0]["id"], project="default")

            # Verify only first is no longer due
            due_after = calendar.get_due_reminders(project="default")
            due_ids_after = set(r["reminder_id"] for r in due_after)
            self.assertNotIn(reminders[0]["id"], due_ids_after)
            self.assertIn(reminders[1]["id"], due_ids_after)
            self.assertIn(reminders[2]["id"], due_ids_after)

    def test_http_schedule_reminder_endpoint(self) -> None:
        """Test the HTTP endpoint for scheduling reminders."""
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            try:
                def post(path: str, payload: dict) -> tuple[int, dict]:
                    request = Request(
                        f"{base}{path}",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(request, timeout=10) as response:
                        return response.status, json.load(response)

                def get(path: str) -> tuple[int, dict]:
                    with urlopen(f"{base}{path}", timeout=10) as response:
                        return response.status, json.load(response)

                # Create calendar
                status, cal_resp = post("/api/calendar/calendars", {
                    "project": "default",
                    "name": "Test Calendar"
                })
                self.assertEqual(status, HTTPStatus.CREATED)
                calendar_id = cal_resp["calendar"]["id"]

                # Create event
                event_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
                status, event_resp = post("/api/calendar/events", {
                    "project": "default",
                    "calendar_id": calendar_id,
                    "title": "HTTP Test Event",
                    "starts_at": event_time,
                    "ends_at": event_time
                })
                self.assertEqual(status, HTTPStatus.CREATED)
                event_id = event_resp["event"]["id"]

                # Schedule reminder
                reminder_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
                status, reminder_resp = post("/api/calendar/reminders/schedule", {
                    "project": "default",
                    "event_id": event_id,
                    "scheduled_at": reminder_time,
                    "advance_minutes": 15,
                    "delivery_channel": "local"
                })
                self.assertEqual(status, HTTPStatus.CREATED)
                self.assertIn("reminder", reminder_resp)
                self.assertEqual(reminder_resp["reminder"]["status"], "scheduled")

            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                time.sleep(0.1)

    def test_http_deliver_due_notifications_endpoint(self) -> None:
        """Test the HTTP endpoint for delivering due notifications."""
        with TemporaryDirectory() as tmp:
            server = build_http_server("127.0.0.1", 0, OpenAgentStore(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            try:
                def post(path: str, payload: dict) -> tuple[int, dict]:
                    request = Request(
                        f"{base}{path}",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(request, timeout=10) as response:
                        return response.status, json.load(response)

                # Create calendar and event
                status, cal_resp = post("/api/calendar/calendars", {
                    "project": "default",
                    "name": "Delivery Test Calendar"
                })
                calendar_id = cal_resp["calendar"]["id"]

                event_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
                status, event_resp = post("/api/calendar/events", {
                    "project": "default",
                    "calendar_id": calendar_id,
                    "title": "Delivery Test",
                    "starts_at": event_time,
                    "ends_at": event_time
                })
                event_id = event_resp["event"]["id"]

                # Schedule reminder in the past
                reminder_time = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
                status, reminder_resp = post("/api/calendar/reminders/schedule", {
                    "project": "default",
                    "event_id": event_id,
                    "scheduled_at": reminder_time,
                    "advance_minutes": 15
                })

                # Deliver notifications
                status, deliver_resp = post("/api/calendar/reminders/deliver", {
                    "project": "default"
                })
                self.assertEqual(status, HTTPStatus.OK)
                self.assertIn("delivered", deliver_resp)

            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                time.sleep(0.1)
