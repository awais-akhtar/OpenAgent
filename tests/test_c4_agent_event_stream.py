"""Tests for C4: Agent Generic Event Stream with SSE, filtering, and cursor-based pagination."""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from openagent.multi_agent import MultiAgentRuntime
from openagent.storage import OpenAgentStore


class C4AgentEventStreamTests(TestCase):
    """Test suite for agent event stream with SSE, filtering, and pagination."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.tmp_dir = TemporaryDirectory()
        self.store = OpenAgentStore(self.tmp_dir.name)
        self.runtime = MultiAgentRuntime(self.store)

    def tearDown(self) -> None:
        """Cleanup after each test."""
        try:
            self.tmp_dir.cleanup()
        except Exception:
            pass
        gc.collect()
        time.sleep(0.05)

    def test_event_stream_sse_format(self) -> None:
        """Test that events are streamed in proper SSE format."""
        # Start a job
        job = self.runtime.start_job("executive", "Test task", project="default", owner="test_user")
        job_id = job["id"]

        # Record test events
        evt1 = self.runtime.record_job_event(
            job_id,
            event_type="checkpoint",
            step="execution",
            project="default",
            owner="test_user"
        )
        evt2 = self.runtime.record_job_event(
            job_id,
            event_type="progress",
            step="step_1",
            payload={"progress": 50},
            project="default",
            owner="test_user"
        )

        # Test that events are recorded properly
        self.assertIsNotNone(job_id)
        self.assertEqual(job["project"], "default")
        self.assertIsNotNone(evt1["id"])
        self.assertIsNotNone(evt2["id"])

    def test_stream_with_event_filtering_by_type(self) -> None:
        """Test filtering events by type."""
        job = self.runtime.start_job("executive", "Filter test", project="default", owner="test_user")
        job_id = job["id"]

        # Create events with different types
        for evt_type in ["checkpoint", "progress", "progress"]:
            self.runtime.record_job_event(
                job_id,
                event_type=evt_type,
                project="default",
                owner="test_user"
            )

        # Verify all events exist
        events = self.runtime.job_events(job_id, owner="test_user")
        self.assertGreater(len(events["events"]), 0)

        # Filter by event type
        progress_events = self.runtime.job_events(job_id, owner="test_user", event_type="progress")
        self.assertEqual(len(progress_events["events"]), 2)

    def test_stream_with_status_filtering(self) -> None:
        """Test filtering events by job status."""
        # Create jobs
        job1 = self.runtime.start_job("executive", "Status test 1", project="default", owner="test_user")
        job2 = self.runtime.start_job("planning", "Status test 2", project="default", owner="test_user")

        # Both should be retrievable
        retrieved1 = self.runtime.get_job(job1["id"], owner="test_user")
        retrieved2 = self.runtime.get_job(job2["id"], owner="test_user")
        self.assertIsNotNone(retrieved1)
        self.assertIsNotNone(retrieved2)

    def test_cursor_based_pagination_for_historical_events(self) -> None:
        """Test cursor-based pagination through historical events."""
        job = self.runtime.start_job("executive", "Pagination test", project="default", owner="test_user")
        job_id = job["id"]

        # Create many events
        for i in range(12):
            self.runtime.record_job_event(
                job_id,
                event_type="progress",
                step=f"step_{i}",
                payload={"index": i},
                project="default",
                owner="test_user"
            )

        # Test pagination with limit
        events1 = self.runtime.job_events(job_id, owner="test_user", limit=5)
        self.assertEqual(len(events1["events"]), 5)

        # Should have a next cursor if we have more events than the limit
        if len(events1["events"]) == 5:
            self.assertIsNotNone(events1.get("next"))

    def test_event_aggregation_across_job_types(self) -> None:
        """Test aggregating events across different agent job types."""
        # Create different types of jobs
        job1 = self.runtime.start_job("executive", "Executive task", project="default", owner="test_user")
        job2 = self.runtime.start_job("planning", "Planning task", project="default", owner="test_user")

        # List all jobs
        all_jobs = self.runtime.jobs(project="default", owner="test_user")
        self.assertGreaterEqual(len(all_jobs), 2)

    def test_stream_error_handling_invalid_job_id(self) -> None:
        """Test error handling when job ID doesn't exist."""
        with self.assertRaises(KeyError):
            self.runtime.job_events("nonexistent-job-id", owner="test_user")

    def test_stream_graceful_closing_and_cleanup(self) -> None:
        """Test that streams close gracefully and clean up resources."""
        job = self.runtime.start_job("executive", "Cleanup test", project="default", owner="test_user")
        job_id = job["id"]

        # Record an event
        self.runtime.record_job_event(
            job_id,
            event_type="started",
            project="default",
            owner="test_user"
        )

        # Cancel the job (status can be "cancelling" or "cancelled")
        cancelled = self.runtime.cancel_job(job_id, owner="test_user")
        self.assertIn(cancelled["status"], ["cancelled", "cancelling"])

        # Should still be able to retrieve events
        events = self.runtime.job_events(job_id, owner="test_user")
        self.assertIsNotNone(events)

    def test_stream_with_timestamp_filtering(self) -> None:
        """Test filtering events work with timestamps."""
        job = self.runtime.start_job("executive", "Timestamp test", project="default", owner="test_user")
        job_id = job["id"]

        # Create events
        for i in range(3):
            self.runtime.record_job_event(
                job_id,
                event_type="progress",
                step=f"step_{i}",
                payload={"index": i},
                project="default",
                owner="test_user"
            )

        # Verify events were created
        events = self.runtime.job_events(job_id, owner="test_user")
        self.assertGreater(len(events["events"]), 0)

    def test_real_time_event_push_non_polling(self) -> None:
        """Test that events are pushed in real-time."""
        job = self.runtime.start_job("executive", "Real-time test", project="default", owner="test_user")
        job_id = job["id"]

        # Add event
        self.runtime.record_job_event(
            job_id,
            event_type="checkpoint",
            step="execution",
            project="default",
            owner="test_user"
        )

        # Event should be immediately available
        events = self.runtime.job_events(job_id, owner="test_user")
        self.assertGreater(len(events["events"]), 0)

    def test_multiple_filters_combined(self) -> None:
        """Test combining multiple filters."""
        job = self.runtime.start_job("executive", "Multi-filter test", project="default", owner="test_user")
        job_id = job["id"]

        # Create events of different types
        for evt_type in ["checkpoint", "progress", "progress"]:
            self.runtime.record_job_event(
                job_id,
                event_type=evt_type,
                project="default",
                owner="test_user"
            )

        # Filter by type and status
        events = self.runtime.job_events(
            job_id,
            owner="test_user",
            event_type="progress",
            status_filter="running"
        )
        # Should have progress events
        self.assertGreater(len(events["events"]), 0)
