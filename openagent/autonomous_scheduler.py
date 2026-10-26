"""Autonomous background scheduler for reminder delivery and task execution."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from .models import utc_now

logger = logging.getLogger(__name__)


class AutonomousSchedulerWorker:
    """Background worker that periodically executes scheduler tasks."""

    def __init__(
        self,
        scheduler: Any,
        calendar_system: Any,
        integration_hub: Any,
        interval_seconds: int = 60,
    ):
        self.scheduler = scheduler
        self.calendar_system = calendar_system
        self.integration_hub = integration_hub
        self.interval_seconds = max(10, interval_seconds)  # Minimum 10 seconds
        self.worker_id = f"scheduler-worker-{uuid4().hex[:8]}"
        self.running = False
        self.thread: threading.Thread | None = None
        self.last_run_at = ""
        self.runs_completed = 0
        self.runs_failed = 0

    def start(self) -> dict[str, Any]:
        """Start the background scheduler worker thread."""
        if self.running:
            return {
                "status": "already_running",
                "worker_id": self.worker_id,
                "interval_seconds": self.interval_seconds,
            }

        self.running = True
        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name=self.worker_id,
        )
        self.thread.start()

        return {
            "status": "started",
            "worker_id": self.worker_id,
            "interval_seconds": self.interval_seconds,
            "started_at": utc_now(),
        }

    def stop(self) -> dict[str, Any]:
        """Stop the background scheduler worker thread."""
        if not self.running:
            return {
                "status": "not_running",
                "worker_id": self.worker_id,
            }

        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=5)

        return {
            "status": "stopped",
            "worker_id": self.worker_id,
            "runs_completed": self.runs_completed,
            "runs_failed": self.runs_failed,
            "stopped_at": utc_now(),
        }

    def get_status(self) -> dict[str, Any]:
        """Get current status of the scheduler worker."""
        return {
            "worker_id": self.worker_id,
            "running": self.running,
            "interval_seconds": self.interval_seconds,
            "runs_completed": self.runs_completed,
            "runs_failed": self.runs_failed,
            "last_run_at": self.last_run_at,
            "thread_alive": self.thread.is_alive() if self.thread else False,
        }

    def _run_loop(self) -> None:
        """Main worker loop - runs autonomously in background."""
        logger.info(f"Autonomous scheduler worker {self.worker_id} started")

        while self.running:
            try:
                self.last_run_at = utc_now()

                # Run calendar due notifications for default project
                try:
                    result = self.scheduler.run_calendar_due_notifications(
                        project="default",
                        calendar_system=self.calendar_system,
                        integration_hub=self.integration_hub,
                    )

                    delivered_count = len(result.get("delivered", []))
                    skipped_count = len(result.get("skipped", []))

                    if delivered_count > 0:
                        logger.info(
                            f"Autonomous scheduler {self.worker_id}: "
                            f"delivered {delivered_count} reminders, skipped {skipped_count}"
                        )

                    self.runs_completed += 1

                except Exception as exc:
                    logger.error(f"Autonomous scheduler error: {exc}", exc_info=True)
                    self.runs_failed += 1

                # Sleep before next run
                for _ in range(self.interval_seconds * 10):
                    if not self.running:
                        break
                    time.sleep(0.1)

            except Exception as exc:
                logger.error(f"Autonomous scheduler worker fatal error: {exc}", exc_info=True)
                self.runs_failed += 1
                time.sleep(1)

        logger.info(f"Autonomous scheduler worker {self.worker_id} stopped")

    def execute_once(self) -> dict[str, Any]:
        """Execute the scheduler tasks once (for testing)."""
        try:
            result = self.scheduler.run_calendar_due_notifications(
                project="default",
                calendar_system=self.calendar_system,
                integration_hub=self.integration_hub,
            )

            self.runs_completed += 1
            self.last_run_at = utc_now()

            return {
                "status": "executed",
                "worker_id": self.worker_id,
                "result": result,
                "executed_at": self.last_run_at,
            }

        except Exception as exc:
            self.runs_failed += 1
            return {
                "status": "error",
                "worker_id": self.worker_id,
                "error": str(exc),
                "executed_at": utc_now(),
            }


class AutonomousSchedulerManager:
    """Manages autonomous scheduler workers for different projects/purposes."""

    def __init__(self):
        self.workers: dict[str, AutonomousSchedulerWorker] = {}

    def create_worker(
        self,
        scheduler: Any,
        calendar_system: Any,
        integration_hub: Any,
        interval_seconds: int = 60,
        worker_name: str = "",
    ) -> dict[str, Any]:
        """Create a new autonomous scheduler worker."""
        name = worker_name or f"default-{uuid4().hex[:8]}"

        if name in self.workers:
            raise ValueError(f"Worker '{name}' already exists")

        worker = AutonomousSchedulerWorker(
            scheduler,
            calendar_system,
            integration_hub,
            interval_seconds=interval_seconds,
        )

        self.workers[name] = worker

        return {
            "worker_name": name,
            "worker_id": worker.worker_id,
            "interval_seconds": interval_seconds,
            "status": "created",
        }

    def start_worker(self, worker_name: str) -> dict[str, Any]:
        """Start an existing worker."""
        worker = self.workers.get(worker_name)
        if worker is None:
            raise ValueError(f"Worker '{worker_name}' not found")

        return worker.start()

    def stop_worker(self, worker_name: str) -> dict[str, Any]:
        """Stop an existing worker."""
        worker = self.workers.get(worker_name)
        if worker is None:
            raise ValueError(f"Worker '{worker_name}' not found")

        return worker.stop()

    def get_worker_status(self, worker_name: str) -> dict[str, Any]:
        """Get status of a worker."""
        worker = self.workers.get(worker_name)
        if worker is None:
            raise ValueError(f"Worker '{worker_name}' not found")

        return worker.get_status()

    def list_workers(self) -> list[dict[str, Any]]:
        """List all workers and their status."""
        return [
            {
                "worker_name": name,
                "status": worker.get_status(),
            }
            for name, worker in self.workers.items()
        ]

    def execute_worker_once(self, worker_name: str) -> dict[str, Any]:
        """Execute a worker's tasks once (for testing)."""
        worker = self.workers.get(worker_name)
        if worker is None:
            raise ValueError(f"Worker '{worker_name}' not found")

        return worker.execute_once()

    def stop_all(self) -> dict[str, Any]:
        """Stop all workers."""
        results = {}
        for name, worker in self.workers.items():
            results[name] = worker.stop()

        return {
            "status": "all_stopped",
            "worker_results": results,
            "stopped_at": utc_now(),
        }
