"""Stage 3 multi-agent runtime."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import threading
from typing import Any
from uuid import uuid4

from .agents import AgentRegistry
from .model_gateway import ModelGateway
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class PermissionDenied(RuntimeError):
    """Raised when an agent lacks a required permission."""


class ApprovalRequired(RuntimeError):
    """Raised when an action requires a human approval checkpoint."""


class MultiAgentRuntime:
    def __init__(self, store: OpenAgentStore, gateway: ModelGateway | None = None):
        self.store = store
        self.gateway = gateway
        self.registry = AgentRegistry(store, gateway)
        self._job_lock = threading.Lock()
        self._job_cancellations: dict[str, threading.Event] = {}
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_permissions (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    granted_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_permissions_lookup
                    ON agent_permissions(project, agent_id, scope);

                CREATE TABLE IF NOT EXISTS human_approvals (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_human_approvals_project_status
                    ON human_approvals(project, status, created_at DESC);

                CREATE TABLE IF NOT EXISTS agent_delegations (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    from_agent TEXT NOT NULL,
                    to_agent TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    run_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_delegations_project_created
                    ON agent_delegations(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS multi_agent_runs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tasks_json TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_multi_agent_runs_project_created
                    ON multi_agent_runs(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS agent_verifications (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    agent_run_id TEXT NOT NULL,
                    verifier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    findings_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_agent_verifications_project_created
                    ON agent_verifications(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS agent_execution_jobs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    current_step TEXT NOT NULL DEFAULT '',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    retry_of TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_agent_execution_jobs_owner_project
                    ON agent_execution_jobs(owner, project, updated_at DESC);

                CREATE TABLE IF NOT EXISTS agent_manifests (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    version TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
        self._reconcile_unfinished_jobs()
        self.seed_defaults()

    def _reconcile_unfinished_jobs(self) -> None:
        """Close jobs whose worker thread cannot have survived this process.

        Cancellation remains cancellation. Other unfinished work is marked failed
        with an explicit restart reason so the existing retry operation can resume
        it deliberately; it is never silently replayed during startup.
        """
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """UPDATE agent_execution_jobs
                   SET status='cancelled', current_step='Cancelled during application restart',
                       result_json='{}', error='', updated_at=?, completed_at=?
                   WHERE status IN ('queued','running','cancelling') AND cancel_requested=1""",
                (now, now),
            )
            connection.execute(
                """UPDATE agent_execution_jobs
                   SET status='failed', current_step='Interrupted by application restart',
                       error='The local agent worker stopped when OpenAgent restarted. Retry this job to continue.',
                       updated_at=?, completed_at=?
                   WHERE status IN ('queued','running','cancelling') AND cancel_requested=0""",
                (now, now),
            )

    def start_job(
        self, agent_id: str, goal: str, *, project: str = "default", owner: str = "local",
        retry_of: str = "", attempt: int = 1,
    ) -> dict[str, Any]:
        goal = goal.strip()
        project = project.strip() or "default"
        owner = owner.strip() or "local"
        if not goal:
            raise ValueError("A goal is required.")
        self.require_permission(agent_id, "agent.run", project=project)
        if not self.store.claim_agent_run_project(project, owner):
            raise PermissionDenied("Project not found.")
        job_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO agent_execution_jobs (
                    id, project, owner, agent, goal, status, progress, current_step,
                    attempt, retry_of, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', 0, 'Queued for local execution', ?, ?, ?, ?)
                """,
                (job_id, project, owner, agent_id, goal, max(1, int(attempt)), retry_of, now, now),
            )
        cancel = threading.Event()
        with self._job_lock:
            self._job_cancellations[job_id] = cancel
        threading.Thread(
            target=self._execute_job, args=(job_id, cancel), name=f"openagent-agent-{job_id[:8]}", daemon=True,
        ).start()
        return self.get_job(job_id, owner=owner) or {}

    def _execute_job(self, job_id: str, cancel: threading.Event) -> None:
        try:
            job = self.get_job(job_id)
            if not job or cancel.is_set() or job["cancel_requested"]:
                self._finish_cancelled(job_id)
                return
            project = str(job.get("project", "default"))
            owner = str(job.get("owner", "local"))
            # Progress is a lifecycle milestone, not an estimate of model work.
            self._update_job(job_id, status="running", progress=10, current_step="Agent invocation started")
            self.record_job_event(job_id, event_type="started", step="invocation", project=project, owner=owner)
            result = self.run_agent(
                str(job["agent"]), str(job["goal"]), project=project, owner=owner,
            )
            if cancel.is_set() or self._cancel_requested(job_id):
                self._finish_cancelled(job_id)
                return
            if not self._update_job(job_id, status="running", progress=90, current_step="Agent response recorded"):
                self._finish_cancelled(job_id)
                return
            self.record_job_event(job_id, event_type="result", step="response_recorded",
                                  payload={"result_preview": str(result)[:200]}, project=project, owner=owner)
            now = utc_now()
            with self.store.session() as connection:
                completed = connection.execute(
                    """UPDATE agent_execution_jobs SET status='completed', progress=100,
                       current_step='Result persisted', result_json=?, error='', updated_at=?, completed_at=?
                       WHERE id=? AND cancel_requested=0 AND status='running'""",
                    (json.dumps(result, sort_keys=True), now, now, job_id),
                ).rowcount
            if not completed:
                self._finish_cancelled(job_id)
            else:
                self.record_job_event(job_id, event_type="completed", step="done", project=project, owner=owner)
        except Exception as exc:
            if cancel.is_set() or self._cancel_requested(job_id):
                self._finish_cancelled(job_id)
                return
            now = utc_now()
            with self.store.session() as connection:
                connection.execute(
                    """UPDATE agent_execution_jobs SET status='failed', current_step='Failed', error=?,
                       updated_at=?, completed_at=? WHERE id=?""",
                    (str(exc), now, now, job_id),
                )
            try:
                job_meta = self.get_job(job_id)
                self.record_job_event(job_id, event_type="failed", step="error",
                                      payload={"error": str(exc)},
                                      project=str(job_meta.get("project", "default")) if job_meta else "default",
                                      owner=str(job_meta.get("owner", "local")) if job_meta else "local")
            except Exception:
                pass
        finally:
            with self._job_lock:
                self._job_cancellations.pop(job_id, None)

    def jobs(self, *, project: str, owner: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                """SELECT * FROM agent_execution_jobs WHERE project=? AND owner=?
                   ORDER BY updated_at DESC LIMIT ?""",
                (project, owner, max(1, min(int(limit), 100))),
            ).fetchall()
        return [agent_job_row(row) for row in rows]

    def get_job(self, job_id: str, *, owner: str | None = None) -> dict[str, Any] | None:
        sql = "SELECT * FROM agent_execution_jobs WHERE id=?"
        params: tuple[Any, ...] = (job_id,)
        if owner is not None:
            sql += " AND owner=?"
            params = (job_id, owner)
        with self.store.session() as connection:
            row = connection.execute(sql, params).fetchone()
        return agent_job_row(row) if row else None

    def record_job_event(
        self,
        job_id: str,
        *,
        event_type: str,
        step: str = "",
        payload: dict[str, Any] | None = None,
        project: str = "default",
        owner: str = "local",
    ) -> dict[str, Any]:
        """Append a structured event to a job's event log."""
        event_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO agent_job_events (id, job_id, project, owner, event_type, step, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, job_id, project or "default", owner or "local",
                 str(event_type or "info"), str(step or ""), json.dumps(payload or {}), now),
            )
        return {"id": event_id, "job_id": job_id, "event_type": event_type, "step": step,
                "payload": payload or {}, "created_at": now}

    def job_events(
        self,
        job_id: str,
        *,
        owner: str,
        after: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return events for a job with cursor-based pagination."""
        job = self.get_job(job_id, owner=owner)
        if not job:
            raise KeyError("Agent job not found.")
        limit = max(1, min(int(limit), 500))
        with self.store.session() as connection:
            if after:
                rows = connection.execute(
                    """SELECT * FROM agent_job_events WHERE job_id = ? AND created_at > (
                        SELECT created_at FROM agent_job_events WHERE job_id = ? AND id = ?
                    ) ORDER BY created_at ASC, rowid ASC LIMIT ?""",
                    (job_id, job_id, after, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM agent_job_events WHERE job_id = ? ORDER BY created_at ASC, rowid ASC LIMIT ?",
                    (job_id, limit),
                ).fetchall()
        events = [
            {
                "id": row["id"],
                "job_id": row["job_id"],
                "event_type": row["event_type"],
                "step": row["step"],
                "payload": json.loads(row["payload_json"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        next_cursor = events[-1]["id"] if len(events) == limit else ""
        return {"events": events, "next": next_cursor, "job_id": job_id}

    def cancel_job(self, job_id: str, *, owner: str) -> dict[str, Any]:
        job = self.get_job(job_id, owner=owner)
        if not job:
            raise KeyError("Agent job not found.")
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        with self._job_lock:
            event = self._job_cancellations.get(job_id)
            if event:
                event.set()
        now = utc_now()
        status = "cancelled" if job["status"] == "queued" else "cancelling"
        with self.store.session() as connection:
            connection.execute(
                """UPDATE agent_execution_jobs SET cancel_requested=1, status=?, progress=CASE WHEN ?='cancelled' THEN 0 ELSE progress END,
                   current_step=?, updated_at=?, completed_at=CASE WHEN ?='cancelled' THEN ? ELSE completed_at END WHERE id=?""",
                (status, status, "Cancelled before execution" if status == "cancelled" else "Cancellation requested", now, status, now, job_id),
            )
        return self.get_job(job_id, owner=owner) or job

    def retry_job(self, job_id: str, *, owner: str) -> dict[str, Any]:
        job = self.get_job(job_id, owner=owner)
        if not job:
            raise KeyError("Agent job not found.")
        if job["status"] not in {"failed", "cancelled"}:
            raise ValueError("Only failed or cancelled agent jobs can be retried.")
        return self.start_job(
            str(job["agent"]), str(job["goal"]), project=str(job["project"]), owner=owner,
            retry_of=job_id, attempt=int(job["attempt"]) + 1,
        )

    def _cancel_requested(self, job_id: str) -> bool:
        with self.store.session() as connection:
            row = connection.execute("SELECT cancel_requested FROM agent_execution_jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row[0])

    def _update_job(self, job_id: str, *, status: str, progress: int, current_step: str) -> bool:
        with self.store.session() as connection:
            changed = connection.execute(
                """UPDATE agent_execution_jobs SET status=?, progress=?, current_step=?, updated_at=?
                   WHERE id=? AND cancel_requested=0 AND status NOT IN ('completed','failed','cancelled')""",
                (status, max(0, min(100, progress)), current_step, utc_now(), job_id),
            ).rowcount
        return bool(changed)

    def _finish_cancelled(self, job_id: str) -> None:
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """UPDATE agent_execution_jobs SET status='cancelled', current_step='Cancellation confirmed',
                   result_json='{}', error='', updated_at=?, completed_at=?
                   WHERE id=? AND status NOT IN ('completed','failed','cancelled')""",
                (now, now, job_id),
            )

    def seed_defaults(self) -> None:
        scopes = [
            "agent.run",
            "agent.delegate",
            "agent.verify",
            "memory.read",
            "memory.write",
            "workflow.plan",
        ]
        for spec in self.registry.specs():
            for scope in scopes:
                self.grant(spec["id"], scope, project="*", granted_by="system")
            self.upsert_manifest(
                {
                    "id": spec["id"],
                    "name": spec["name"],
                    "description": spec["description"],
                    "version": "0.1.0",
                    "permissions": scopes,
                    "entrypoint": f"builtin:{spec['id']}",
                    "local_first": True,
                }
            )

    def grant(
        self,
        agent_id: str,
        scope: str,
        *,
        project: str = "default",
        granted_by: str = "user",
    ) -> dict[str, Any]:
        permission_id = stable_id("permission", project, agent_id, scope)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO agent_permissions (
                    id, project, agent_id, scope, granted_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (permission_id, project, agent_id, scope, granted_by, created_at),
            )
        return {
            "id": permission_id,
            "project": project,
            "agent_id": agent_id,
            "scope": scope,
            "granted_by": granted_by,
            "created_at": created_at,
        }

    def has_permission(self, agent_id: str, scope: str, *, project: str = "default") -> bool:
        with self.store.session() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM agent_permissions
                WHERE agent_id = ? AND scope = ? AND project IN (?, '*')
                LIMIT 1
                """,
                (agent_id, scope, project),
            ).fetchone()
        return row is not None

    def list_permissions(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_permissions
                WHERE project IN (?, '*')
                ORDER BY agent_id, scope
                """,
                (project,),
            ).fetchall()
        return [dict(row) for row in rows]

    def run_agent(
        self,
        agent_id: str,
        goal: str,
        *,
        project: str = "default",
        require_approval: bool = False,
        approval_id: str | None = None,
        verify: bool = True,
        owner: str | None = None,
    ) -> dict[str, Any]:
        self.require_permission(agent_id, "agent.run", project=project)
        if require_approval and not self.approval_is_approved(approval_id):
            approval = self.request_approval(
                requested_by=f"agent:{agent_id}",
                action="agent.run",
                reason=f"Run agent '{agent_id}' for goal: {goal}",
                project=project,
                payload={"agent": agent_id, "goal": goal},
            )
            raise ApprovalRequired(json.dumps(approval, sort_keys=True))
        run = self.registry.run(agent_id, goal, project=project, owner=owner)
        verification = self.verify_run(run, project=project) if verify else None
        return {"run": run, "verification": verification}

    def delegate(
        self,
        from_agent: str,
        to_agent: str,
        goal: str,
        *,
        project: str = "default",
    ) -> dict[str, Any]:
        self.require_permission(from_agent, "agent.delegate", project=project)
        self.require_permission(to_agent, "agent.run", project=project)
        result = self.run_agent(to_agent, goal, project=project, verify=True)
        delegation_id = stable_id("delegation", project, from_agent, to_agent, goal, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO agent_delegations (
                    id, project, from_agent, to_agent, goal, run_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delegation_id,
                    project,
                    from_agent,
                    to_agent,
                    goal,
                    json.dumps(result, sort_keys=True),
                    created_at,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor=f"agent:{from_agent}",
                action="agent.delegate",
                resource=f"agent:{to_agent}",
                project=project,
                metadata={"goal": goal, "delegation_id": delegation_id},
            )
        )
        return {
            "id": delegation_id,
            "project": project,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "goal": goal,
            "result": result,
            "created_at": created_at,
        }

    def run_parallel(self, tasks: list[dict[str, Any]], *, project: str = "default") -> dict[str, Any]:
        if not tasks:
            raise ValueError("At least one task is required.")
        created_at = utc_now()
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
            futures = {
                executor.submit(
                    self.run_agent,
                    str(task.get("agent", "executive")),
                    str(task.get("goal", "")),
                    project=project,
                    verify=True,
                ): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    results.append({"task": task, "status": "completed", **future.result()})
                except Exception as exc:
                    results.append({"task": task, "status": "failed", "error": str(exc)})
        status = "completed" if all(result["status"] == "completed" for result in results) else "partial"
        run_id = stable_id("parallel", project, json.dumps(tasks, sort_keys=True), created_at)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO multi_agent_runs (
                    id, project, mode, status, tasks_json, results_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    project,
                    "parallel",
                    status,
                    json.dumps(tasks, sort_keys=True),
                    json.dumps(results, sort_keys=True),
                    created_at,
                ),
            )
        return {"id": run_id, "project": project, "status": status, "results": results, "created_at": created_at}

    def verify_run(self, run: dict[str, Any], *, project: str = "default", verifier: str = "security") -> dict[str, Any]:
        self.require_permission(verifier, "agent.verify", project=project)
        confidence = float(run.get("confidence", 0.0))
        findings = []
        if confidence < 0.45:
            findings.append("Low confidence response; request more context or human review.")
        if not str(run.get("response", "")).strip():
            findings.append("Agent response is empty.")
        if not run.get("next_actions"):
            findings.append("Agent did not provide next actions.")
        status = "passed" if not findings else "warning"
        verification = {
            "status": status,
            "findings": findings,
            "reflection": "Use retrieved context, explicit permissions, and audit records for follow-up work.",
        }
        verification_id = stable_id("verification", project, str(run.get("id", "")), utc_now())
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO agent_verifications (
                    id, project, agent_run_id, verifier, status, findings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    verification_id,
                    project,
                    str(run.get("id", "")),
                    verifier,
                    status,
                    json.dumps(verification, sort_keys=True),
                    utc_now(),
                ),
            )
        return {"id": verification_id, "project": project, **verification}

    def request_approval(
        self,
        *,
        requested_by: str,
        action: str,
        reason: str,
        project: str = "default",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        approval_id = stable_id("approval", project, requested_by, action, reason, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO human_approvals (
                    id, project, requested_by, action, reason, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    project,
                    requested_by,
                    action,
                    reason,
                    "pending",
                    json.dumps(payload or {}, sort_keys=True),
                    created_at,
                ),
            )
        return {
            "id": approval_id,
            "project": project,
            "requested_by": requested_by,
            "action": action,
            "reason": reason,
            "status": "pending",
            "payload": payload or {},
            "created_at": created_at,
        }

    def resolve_approval(self, approval_id: str, *, approved: bool) -> dict[str, Any]:
        status = "approved" if approved else "rejected"
        resolved_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "UPDATE human_approvals SET status = ?, resolved_at = ? WHERE id = ?",
                (status, resolved_at, approval_id),
            )
            row = connection.execute("SELECT * FROM human_approvals WHERE id = ?", (approval_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown approval '{approval_id}'.")
        return approval_row(row)

    def list_approvals(self, *, project: str = "default", status: str | None = None) -> list[dict[str, Any]]:
        if status:
            query = "SELECT * FROM human_approvals WHERE project = ? AND status = ? ORDER BY created_at DESC"
            params: tuple[Any, ...] = (project, status)
        else:
            query = "SELECT * FROM human_approvals WHERE project = ? ORDER BY created_at DESC"
            params = (project,)
        with self.store.session() as connection:
            rows = connection.execute(query, params).fetchall()
        return [approval_row(row) for row in rows]

    def approval_is_approved(self, approval_id: str | None) -> bool:
        if not approval_id:
            return False
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT status FROM human_approvals WHERE id = ?",
                (approval_id,),
            ).fetchone()
        return bool(row and row["status"] == "approved")

    def marketplace(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM agent_manifests ORDER BY name").fetchall()
        return [json.loads(row["manifest_json"]) for row in rows]

    def upsert_manifest(self, manifest: dict[str, Any]) -> None:
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO agent_manifests (
                    id, name, description, version, manifest_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest["id"],
                    manifest["name"],
                    manifest["description"],
                    manifest["version"],
                    json.dumps(manifest, sort_keys=True),
                    created_at,
                ),
            )

    def require_permission(self, agent_id: str, scope: str, *, project: str) -> None:
        if not self.has_permission(agent_id, scope, project=project):
            raise PermissionDenied(f"Agent '{agent_id}' lacks permission '{scope}' for project '{project}'.")


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def approval_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "requested_by": row["requested_by"],
        "action": row["action"],
        "reason": row["reason"],
        "status": row["status"],
        "payload": json.loads(row["payload_json"]),
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
    }


def agent_job_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "agent": row["agent"],
        "goal": row["goal"],
        "status": row["status"],
        "progress": int(row["progress"]),
        "current_step": row["current_step"],
        "cancel_requested": bool(row["cancel_requested"]),
        "attempt": int(row["attempt"]),
        "retry_of": row["retry_of"],
        "result": json.loads(row["result_json"] or "{}"),
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
    }
