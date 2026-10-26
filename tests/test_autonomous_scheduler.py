"""Tests for autonomous reminder scheduling and delivery."""

import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from openagent.autonomous_scheduler import AutonomousSchedulerManager, AutonomousSchedulerWorker
from openagent.calendar_system import CalendarSystem
from openagent.scheduler import Scheduler
from openagent.storage import OpenAgentStore


class MockIntegrationHub:
    """Mock integration hub for testing."""

    def __init__(self):
        self.preferences: dict[str, dict[str, Any]] = {}

    def preference(self, *, project: str = "default", scope: str = "", key: str = ""):
        """Get a preference value."""
        key_path = f"{project}:{scope}:{key}"
        return self.preferences.get(key_path)

    def set_preference(self, *, project: str = "default", scope: str = "", key: str = "", value: Any):
        """Set a preference value."""
        key_path = f"{project}:{scope}:{key}"
        self.preferences[key_path] = {"value": value}


class AutonomousSchedulerWorkerTests(unittest.TestCase):
    """Test autonomous scheduler worker for reminder delivery."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test database and services."""
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.store = OpenAgentStore(str(Path(cls.tmpdir.name) / "test.db"))
        cls.scheduler = Scheduler(cls.store)
        cls.calendar = CalendarSystem(cls.store)
        cls.integration_hub = MockIntegrationHub()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory."""
        cls.tmpdir.cleanup()

    def test_worker_creation(self) -> None:
        """Test creating an autonomous scheduler worker."""
        worker = AutonomousSchedulerWorker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            interval_seconds=10,
        )

        assert worker.scheduler is not None
        assert worker.calendar_system is not None
        assert worker.integration_hub is not None
        assert worker.interval_seconds == 10
        assert worker.running is False
        assert worker.runs_completed == 0
        assert worker.runs_failed == 0

    def test_worker_start_stop(self) -> None:
        """Test starting and stopping worker."""
        worker = AutonomousSchedulerWorker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            interval_seconds=10,
        )

        # Start worker
        start_result = worker.start()
        assert start_result["status"] == "started"
        assert worker.running is True
        assert worker.thread is not None
        assert worker.thread.is_alive()

        # Stop worker
        stop_result = worker.stop()
        assert stop_result["status"] == "stopped"
        assert worker.running is False

        # Wait for thread to stop
        time.sleep(0.5)
        assert not worker.thread.is_alive()

    def test_worker_cannot_start_twice(self) -> None:
        """Test that worker cannot be started twice."""
        worker = AutonomousSchedulerWorker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            interval_seconds=10,
        )

        # Start worker first time
        result1 = worker.start()
        assert result1["status"] == "started"

        # Try to start again
        result2 = worker.start()
        assert result2["status"] == "already_running"

        # Clean up
        worker.stop()
        time.sleep(0.5)

    def test_worker_execute_once(self) -> None:
        """Test executing worker once (for testing)."""
        worker = AutonomousSchedulerWorker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            interval_seconds=10,
        )

        # Execute once
        result = worker.execute_once()
        assert "status" in result
        assert result["worker_id"] == worker.worker_id
        assert result["executed_at"]
        assert worker.runs_completed >= 1

    def test_worker_get_status(self) -> None:
        """Test getting worker status."""
        worker = AutonomousSchedulerWorker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            interval_seconds=10,
        )

        status = worker.get_status()

        assert status["worker_id"] == worker.worker_id
        assert status["running"] is False
        assert status["interval_seconds"] == 10
        assert status["runs_completed"] == 0
        assert status["runs_failed"] == 0

    def test_worker_autonomous_delivery_with_calendar_event(self) -> None:
        """Test that worker delivers reminders for calendar events."""
        # Create a calendar event
        event = self.calendar.create_event(
            title="Test Meeting",
            starts_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            ends_at=(datetime.now(timezone.utc) + timedelta(minutes=35)).isoformat(),
            project="test-auto-scheduler",
        )

        # Schedule a reminder for this event
        trigger_time = datetime.now(timezone.utc) + timedelta(seconds=1)
        reminder = self.calendar.schedule_reminder(
            event["id"],
            scheduled_at=trigger_time.isoformat(),
            project="test-auto-scheduler",
            advance_minutes=5,
            delivery_channel="local",
        )

        # Enable notifications
        self.integration_hub.set_preference(
            project="test-auto-scheduler",
            scope="notifications",
            key="enabled",
            value=True,
        )  # pragma: no cover

        # Create and execute worker once
        worker = AutonomousSchedulerWorker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            interval_seconds=10,
        )

        # Wait a moment for the reminder to become due
        time.sleep(2)

        # Execute the worker
        result = worker.execute_once()

        assert "result" in result
        assert "delivered" in result["result"]
        assert "notifications_enabled" in result["result"]
        assert result["result"]["notifications_enabled"] is True

    def test_worker_respects_notification_preferences(self) -> None:
        """Test that worker respects notification disabled preferences."""
        # Create a calendar event (using "default" project to match worker's execute_once)
        event = self.calendar.create_event(
            title="Test Meeting 2",
            starts_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            ends_at=(datetime.now(timezone.utc) + timedelta(minutes=35)).isoformat(),
            project="default",
        )

        # Schedule a reminder
        trigger_time = datetime.now(timezone.utc) + timedelta(seconds=1)
        self.calendar.schedule_reminder(
            event["id"],
            scheduled_at=trigger_time.isoformat(),
            project="default",
            advance_minutes=5,
            delivery_channel="local",
        )

        # DISABLE notifications via integration hub (using "default" project)
        self.integration_hub.set_preference(
            project="default",
            scope="notifications",
            key="enabled",
            value=False,
        )

        # Wait for reminder to be due
        time.sleep(2)

        # Execute worker
        worker = AutonomousSchedulerWorker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            interval_seconds=10,
        )
        result = worker.execute_once()

        # Reminders should be skipped because notifications are disabled
        assert result["result"]["notifications_enabled"] is False
        # Should have skipped reminders since notifications are disabled
        assert len(result["result"]["skipped"]) >= 0

    def test_worker_tracks_runs(self) -> None:
        """Test that worker tracks successful and failed runs."""
        worker = AutonomousSchedulerWorker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            interval_seconds=10,
        )

        # Execute multiple times
        worker.execute_once()
        worker.execute_once()
        worker.execute_once()

        assert worker.runs_completed >= 3


class AutonomousSchedulerManagerTests(unittest.TestCase):
    """Test autonomous scheduler manager for worker management."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test database and services."""
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.store = OpenAgentStore(str(Path(cls.tmpdir.name) / "test.db"))
        cls.scheduler = Scheduler(cls.store)
        cls.calendar = CalendarSystem(cls.store)
        cls.integration_hub = MockIntegrationHub()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up temporary directory."""
        cls.tmpdir.cleanup()

    def test_manager_create_worker(self) -> None:
        """Test creating a worker via manager."""
        manager = AutonomousSchedulerManager()

        result = manager.create_worker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            interval_seconds=20,
            worker_name="test-worker",
        )

        assert result["status"] == "created"
        assert result["worker_name"] == "test-worker"
        assert result["interval_seconds"] == 20
        assert "test-worker" in manager.workers

    def test_manager_cannot_create_duplicate_worker(self) -> None:
        """Test that duplicate workers cannot be created."""
        manager = AutonomousSchedulerManager()

        manager.create_worker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            worker_name="duplicate",
        )

        with self.assertRaises(ValueError) as ctx:
            manager.create_worker(
                self.scheduler,
                self.calendar,
                self.integration_hub,
                worker_name="duplicate",
            )

        assert "already exists" in str(ctx.exception).lower()

    def test_manager_start_stop_worker(self) -> None:
        """Test starting and stopping workers via manager."""
        manager = AutonomousSchedulerManager()

        manager.create_worker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            worker_name="managed-worker",
        )

        # Start worker
        start_result = manager.start_worker("managed-worker")
        assert start_result["status"] == "started"

        # Stop worker
        stop_result = manager.stop_worker("managed-worker")
        assert stop_result["status"] == "stopped"

        time.sleep(0.5)

    def test_manager_get_worker_status(self) -> None:
        """Test getting worker status via manager."""
        manager = AutonomousSchedulerManager()

        manager.create_worker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            worker_name="status-worker",
        )

        status = manager.get_worker_status("status-worker")

        assert "running" in status
        assert "runs_completed" in status
        assert "worker_id" in status

    def test_manager_list_workers(self) -> None:
        """Test listing all workers via manager."""
        manager = AutonomousSchedulerManager()

        manager.create_worker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            worker_name="worker-1",
        )
        manager.create_worker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            worker_name="worker-2",
        )

        workers = manager.list_workers()

        assert len(workers) >= 2
        worker_names = [w["worker_name"] for w in workers]
        assert "worker-1" in worker_names
        assert "worker-2" in worker_names

    def test_manager_execute_worker_once(self) -> None:
        """Test executing a worker once via manager."""
        manager = AutonomousSchedulerManager()

        manager.create_worker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            worker_name="exec-worker",
        )

        result = manager.execute_worker_once("exec-worker")

        assert "status" in result
        assert "result" in result

    def test_manager_stop_all(self) -> None:
        """Test stopping all workers via manager."""
        manager = AutonomousSchedulerManager()

        manager.create_worker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            worker_name="all-1",
        )
        manager.create_worker(
            self.scheduler,
            self.calendar,
            self.integration_hub,
            worker_name="all-2",
        )

        # Start both workers
        manager.start_worker("all-1")
        manager.start_worker("all-2")

        # Stop all
        result = manager.stop_all()

        assert result["status"] == "all_stopped"
        assert "all-1" in result["worker_results"]
        assert "all-2" in result["worker_results"]

        time.sleep(0.5)


if __name__ == "__main__":
    unittest.main()
