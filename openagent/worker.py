"""Durable local background worker for OpenAgent jobs."""

from __future__ import annotations

import argparse
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .calendar_sync import CalendarSyncManager
from .calendar_system import CalendarSystem
from .communication import CommunicationHub
from .config import default_home
from .fine_tuning import FineTuningManager
from .local_ai import LocalAIInfrastructure
from .model_gateway import ModelGateway
from .models import AuditEvent, ContextEvent
from .research import ResearchEngine
from .research_lifecycle import ResearchLifecycle
from .scraper import WebScraper
from .scheduler import Scheduler
from .storage import OpenAgentStore
from .workflows import WorkflowEngine


JobHandler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class WorkerRuntime:
    """Lease and execute durable local queue jobs."""

    def __init__(
        self,
        store: OpenAgentStore,
        *,
        worker_id: str | None = None,
        gateway: ModelGateway | None = None,
    ) -> None:
        self.store = store
        self.gateway = gateway or ModelGateway()
        self.worker_id = worker_id or f"{socket.gethostname()}:{time.time_ns()}"
        self.handlers: dict[str, JobHandler] = {
            "echo": self._echo,
            "workflow.run": self._workflow_run,
            "communication.sync": self._communication_sync,
            "calendar.sync": self._calendar_sync,
            "model.download": self._model_download,
            "model.setup": self._model_setup,
            "model.quantize": self._model_quantize,
            "knowledge.ingest": self._knowledge_ingest,
            "scraper.run": self._scraper_run,
            "research.run": self._research_run,
            "tasks.run_due": self._tasks_run_due,
            "finetune.progress": self._finetune_progress,
        }

    def enqueue(
        self,
        job_type: str,
        payload: dict[str, Any] | None = None,
        *,
        project: str = "default",
        priority: int = 0,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        job = self.store.enqueue_job(
            job_type,
            payload or {},
            project=project,
            priority=priority,
            max_attempts=max_attempts,
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="worker",
                action="worker.job.queued",
                resource=f"job:{job['id']}",
                project=project,
                metadata={"job_type": job_type, "priority": priority},
            )
        )
        return job

    def run_once(self, *, lease_seconds: int = 300, retry_delay_seconds: int = 30) -> dict[str, Any]:
        job = self.store.lease_job(self.worker_id, lease_seconds=lease_seconds)
        if job is None:
            return {
                "status": "idle",
                "worker_id": self.worker_id,
                "message": "No ready job in the local queue.",
            }
        job_id = str(job["id"])
        job_type = str(job["job_type"])
        payload = job["payload"] if isinstance(job.get("payload"), dict) else {}
        try:
            self.store.update_job_progress(job_id, progress=20, message=f"Running {job_type}.")
            handler = self.handlers.get(job_type)
            if handler is None:
                raise ValueError(f"No worker handler is registered for job type '{job_type}'.")
            result = handler(payload, job)
            completed = self.store.complete_job(job_id, result)
            self.store.add_audit_event(
                AuditEvent(
                    actor="worker",
                    action="worker.job.completed",
                    resource=f"job:{job_id}",
                    project=str(job.get("project", "default")),
                    metadata={"job_type": job_type},
                )
            )
            return {"status": "completed", "job": completed}
        except Exception as exc:  # noqa: BLE001 - worker must capture all job failures
            failed = self.store.fail_job(job_id, str(exc), retry_delay_seconds=retry_delay_seconds)
            self.store.add_audit_event(
                AuditEvent(
                    actor="worker",
                    action="worker.job.failed",
                    resource=f"job:{job_id}",
                    project=str(job.get("project", "default")),
                    status="failed",
                    metadata={"job_type": job_type, "error": str(exc)},
                )
            )
            return {"status": "failed", "job": failed, "error": str(exc)}

    def run_forever(
        self,
        *,
        poll_seconds: float = 2.0,
        lease_seconds: int = 300,
        retry_delay_seconds: int = 30,
        max_jobs: int | None = None,
    ) -> dict[str, Any]:
        processed = 0
        while True:
            result = self.run_once(lease_seconds=lease_seconds, retry_delay_seconds=retry_delay_seconds)
            if result["status"] != "idle":
                processed += 1
            if max_jobs is not None and processed >= max_jobs:
                return {"status": "stopped", "processed": processed, "worker_id": self.worker_id}
            if result["status"] == "idle":
                time.sleep(max(0.1, poll_seconds))

    def _echo(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        return {"echo": payload, "job_id": job["id"]}

    def _workflow_run(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        workflow_id = str(payload.get("workflow_id", "")).strip()
        if not workflow_id:
            raise ValueError("workflow_id is required.")
        return WorkflowEngine(self.store, self.gateway).run(
            workflow_id,
            event=payload.get("event") if isinstance(payload.get("event"), dict) else None,
            allow_network=bool(payload.get("allow_network")),
            dry_run=bool(payload.get("dry_run")),
        )

    def _communication_sync(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        platform = str(payload.get("platform", "email")).strip() or "email"
        sync_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
        return CommunicationHub(self.store).sync_connector_resilient(
            platform,
            project=str(job.get("project", "default")),
            owner=str(payload.get("owner", "local")).strip() or "local",
            payload=sync_payload,
            allow_external=bool(payload.get("allow_external")),
            max_attempts=int(payload.get("max_attempts", 3)),
        )

    def _calendar_sync(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        provider_id = str(payload.get("provider_id", "")).strip()
        if not provider_id:
            raise ValueError("provider_id is required.")
        calendar = CalendarSystem(self.store)
        return CalendarSyncManager(self.store, calendar).run_sync(provider_id)

    def _model_download(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("model_id is required.")
        return LocalAIInfrastructure(self.store, self.gateway).download_model(
            model_id,
            allow_network=bool(payload.get("allow_network")),
        )

    def _model_setup(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        return LocalAIInfrastructure(self.store, self.gateway).setup_task_model(
            task=str(payload.get("task", "chat")).strip() or "chat",
            model_name=str(payload.get("model_name", "")).strip(),
            allow_network=bool(payload.get("allow_network")),
            project=str(job.get("project", "default")),
        )

    def _model_quantize(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("model_id is required.")
        return LocalAIInfrastructure(self.store, self.gateway).auto_quantize(
            model_id,
            task=str(payload.get("task", "chat")).strip() or "chat",
            target_ram_gb=float(payload["target_ram_gb"]) if payload.get("target_ram_gb") is not None else None,
            quantization_mode=str(payload.get("quantization_mode", "auto")).strip() or "auto",
            parameter_count_b=float(payload["parameter_count_b"]) if payload.get("parameter_count_b") is not None else None,
        )

    def _knowledge_ingest(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        title = str(payload.get("title", "")).strip() or "Worker knowledge item"
        content = str(payload.get("content", "")).strip()
        if not content:
            raise ValueError("content is required.")
        event = self.store.add_event(
            ContextEvent(
                title=title,
                content=content,
                type=str(payload.get("type", "knowledge")).strip() or "knowledge",
                source="worker",
                project=str(job.get("project", "default")),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
        )
        return {"event": event}

    def _scraper_run(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        return WebScraper(self.store).run_due_jobs(project=str(job.get("project", "default")))

    def _research_run(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("query is required.")
        lifecycle = ResearchLifecycle(self.store)
        session_id = str(payload.get("session_id", "")).strip()
        if session_id:
            session = lifecycle.start(session_id)
            if session["status"] == "cancelled":
                return session
        try:
            result = ResearchEngine(self.store, self.gateway).run(
                query,
                project=str(job.get("project", "default")),
                max_iterations=int(payload.get("max_iterations", 3)),
                allow_external=bool(payload.get("allow_external")),
                provider=str(payload.get("provider", "auto")).strip() or "auto",
                model=str(payload.get("model", "")).strip() or None,
            )
            return lifecycle.complete(session_id, result) if session_id else result
        except Exception as exc:
            if session_id:
                lifecycle.fail(session_id, str(exc))
            raise

    def _tasks_run_due(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        return Scheduler(self.store).run_due(
            project=str(job.get("project", "default")),
            owner=str(payload.get("owner", "")).strip(),
            limit=int(payload.get("limit", 25)),
        )

    def _finetune_progress(self, payload: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        manager = FineTuningManager(self.store)
        job_id = str(payload.get("finetune_job_id", "")).strip()
        if not job_id:
            raise ValueError("finetune_job_id is required.")
        progress = int(payload.get("progress", 100))
        status = str(payload.get("status", "completed")).strip() or "completed"
        if status == "completed":
            return manager.complete_job(
                job_id,
                metrics=payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
                artifact_path=str(payload.get("artifact_path", "")).strip(),
            )
        return manager.update_job_progress(job_id, progress=progress, status=status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenAgent durable local worker.")
    parser.add_argument("--data-dir", default=None, help="OpenAgent data directory.")
    parser.add_argument("--worker-id", default=None, help="Stable worker id to show in leases.")
    parser.add_argument("--once", action="store_true", help="Process at most one ready job and exit.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Stop after this many non-idle jobs.")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval when idle.")
    parser.add_argument("--lease-seconds", type=int, default=300, help="Lease duration per job.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else default_home()
    runtime = WorkerRuntime(OpenAgentStore(data_dir), worker_id=args.worker_id)
    if args.once:
        print(runtime.run_once(lease_seconds=args.lease_seconds))
    else:
        print(
            runtime.run_forever(
                poll_seconds=args.poll_seconds,
                lease_seconds=args.lease_seconds,
                max_jobs=args.max_jobs,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
