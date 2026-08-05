"""Stage 4 local workflow automation engine."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .agents import AgentRegistry
from .chat import ChatService
from .context import ContextEngine
from .model_gateway import ModelGateway
from .models import AuditEvent, ContextEvent, utc_now
from .storage import OpenAgentStore


class WorkflowSafetyError(RuntimeError):
    """Raised when a workflow action crosses a boundary without approval."""


@dataclass(frozen=True)
class StepResult:
    status: str
    action: str
    output: dict[str, Any]
    rollback: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"status": self.status, "action": self.action, **self.output}
        if self.rollback:
            result["rollback"] = self.rollback
        return result


class WorkflowEngine:
    def __init__(self, store: OpenAgentStore, gateway: ModelGateway | None = None):
        self.store = store
        self.gateway = gateway
        self.context = ContextEngine(store)
        self.agents = AgentRegistry(store, gateway)
        self.chat = ChatService(store, gateway)
        self.files_root = (self.store.data_dir / "workflow_files").resolve()
        self.files_root.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflow_triggers (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    next_run_at TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_triggers_due
                    ON workflow_triggers(trigger_type, enabled, next_run_at);

                CREATE TABLE IF NOT EXISTS workflow_rollback_records (
                    id TEXT PRIMARY KEY,
                    workflow_run_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    rollback_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    executed_at TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_rollbacks_run
                    ON workflow_rollback_records(workflow_run_id, step_index DESC);

                CREATE TABLE IF NOT EXISTS workflow_event_log (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    matched_workflows_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_event_log_project_created
                    ON workflow_event_log(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS workflow_layouts (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    layout_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_layouts_workflow
                    ON workflow_layouts(workflow_id);
                """
            )

    def create(
        self,
        *,
        name: str,
        steps: list[dict[str, Any]],
        project: str = "default",
        description: str = "",
        trigger_type: str = "manual",
        trigger: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_steps = self.validate_definition(steps)
        workflow = self.store.create_workflow(
            name,
            normalized_steps,
            project=project,
            description=description,
            trigger_type=trigger_type,
        )
        self.configure_trigger(workflow["id"], trigger_type=trigger_type, project=project, config=trigger or {})
        self.store.add_audit_event(
            AuditEvent(
                actor="workflow",
                action="workflow.create",
                resource=f"workflow:{workflow['id']}",
                project=project,
                metadata={"step_count": len(normalized_steps), "trigger_type": trigger_type},
            )
        )
        return {**workflow, "trigger": self.get_trigger(workflow["id"])}

    def validate_definition(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(steps, list):
            raise ValueError("Workflow steps must be a list.")
        allowed = set(self.builder_schema()["actions"].keys())
        normalized = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                raise ValueError(f"Workflow step {index} must be an object.")
            action = str(step.get("action", "")).strip()
            if action not in allowed:
                raise ValueError(f"Unsupported workflow action '{action}'.")
            normalized.append(dict(step))
        return normalized

    def configure_trigger(
        self,
        workflow_id: str,
        *,
        trigger_type: str,
        project: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now()
        next_run_at = ""
        if trigger_type == "scheduled":
            interval_seconds = int(config.get("interval_seconds", 3600))
            next_run_at = iso_after(interval_seconds)
            config = {**config, "interval_seconds": interval_seconds}
        trigger_id = stable_id("trigger", workflow_id, trigger_type)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO workflow_triggers (
                    id, workflow_id, project, trigger_type, config_json,
                    next_run_at, enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trigger_id,
                    workflow_id,
                    project,
                    trigger_type,
                    json.dumps(config, sort_keys=True),
                    next_run_at,
                    1,
                    now,
                    now,
                ),
            )
        return self.get_trigger(workflow_id) or {}

    def get_trigger(self, workflow_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT * FROM workflow_triggers WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
        if row is None:
            return None
        return trigger_row(row)

    def run(
        self,
        workflow_id: str,
        *,
        event: dict[str, Any] | None = None,
        allow_network: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        workflow = self.store.get_workflow(workflow_id)
        if workflow is None:
            raise ValueError(f"Unknown workflow '{workflow_id}'.")
        if not workflow["enabled"]:
            raise ValueError(f"Workflow '{workflow_id}' is disabled.")

        project = workflow["project"]
        context: dict[str, Any] = {"event": event or {}, "steps": {}, "workflow": workflow}
        outputs: list[dict[str, Any]] = []
        rollbacks: list[dict[str, Any]] = []
        status = "completed"
        for index, step in enumerate(workflow["steps"], start=1):
            try:
                if not condition_matches(step.get("if"), context):
                    output = {"status": "skipped", "action": step.get("action", ""), "reason": "condition_false"}
                    outputs.append(output)
                    context["steps"][str(index)] = output
                    continue
                if dry_run:
                    output = {"status": "planned", "action": step.get("action", ""), "step": step}
                else:
                    result = self._run_step(
                        step,
                        project=project,
                        index=index,
                        context=context,
                        allow_network=allow_network,
                    )
                    output = result.to_dict()
                    if result.rollback:
                        rollbacks.append({"step_index": index, **result.rollback})
                outputs.append(output)
                context["steps"][str(index)] = output
            except Exception as exc:
                status = "failed"
                outputs.append({"status": "failed", "action": str(step.get("action", "")), "error": str(exc)})
                if not dry_run:
                    self._execute_rollbacks(rollbacks, project=project)
                break

        result = {"workflow": workflow["name"], "steps": outputs, "event": event or {}, "dry_run": dry_run}
        run = self.store.save_workflow_run(workflow_id, project=project, status=status, result=result)
        self._store_rollbacks(run["id"], workflow_id, project, rollbacks)
        self.store.add_audit_event(
            AuditEvent(
                actor="workflow",
                action="workflow.run",
                resource=f"workflow:{workflow_id}",
                project=project,
                status=status,
                metadata={"run_id": run["id"], "dry_run": dry_run, "allow_network": allow_network},
            )
        )
        return run

    def run_due_schedules(self, *, now: str | None = None, allow_network: bool = False) -> list[dict[str, Any]]:
        now = now or utc_now()
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM workflow_triggers
                WHERE trigger_type = 'scheduled'
                  AND enabled = 1
                  AND next_run_at != ''
                  AND next_run_at <= ?
                ORDER BY next_run_at ASC
                """,
                (now,),
            ).fetchall()
        runs = []
        for row in rows:
            trigger = trigger_row(row)
            runs.append(self.run(trigger["workflow_id"], allow_network=allow_network))
            interval_seconds = int(trigger["config"].get("interval_seconds", 3600))
            self._update_next_run(trigger["id"], iso_after(interval_seconds))
        return runs

    def trigger_event(
        self,
        event_name: str,
        payload: dict[str, Any] | None = None,
        *,
        project: str = "default",
        allow_network: bool = False,
    ) -> dict[str, Any]:
        payload = payload or {}
        matched = []
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM workflow_triggers
                WHERE trigger_type = 'event' AND enabled = 1 AND project = ?
                """,
                (project,),
            ).fetchall()
        for row in rows:
            trigger = trigger_row(row)
            if trigger["config"].get("event_name") in {event_name, "*"}:
                matched.append(self.run(trigger["workflow_id"], event=payload, allow_network=allow_network))
        event_id = stable_id("event", project, event_name, utc_now())
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO workflow_event_log (
                    id, project, event_name, payload_json, matched_workflows_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    project,
                    event_name,
                    json.dumps(payload, sort_keys=True),
                    json.dumps([run["workflow_id"] for run in matched], sort_keys=True),
                    utc_now(),
                ),
            )
        return {"id": event_id, "event_name": event_name, "matched_runs": matched}

    def builder_schema(self) -> dict[str, Any]:
        return {
            "triggers": {
                "manual": {"label": "Manual"},
                "scheduled": {"label": "Scheduled", "fields": ["interval_seconds"]},
                "event": {"label": "Event", "fields": ["event_name"]},
            },
            "actions": {
                "add_event": {"label": "Add Memory", "category": "memory"},
                "run_agent": {"label": "Run Agent", "category": "agent"},
                "search_context": {"label": "Search Context", "category": "memory"},
                "chat": {"label": "Chat", "category": "model"},
                "condition": {"label": "Condition", "category": "logic"},
                "file_read": {"label": "Read File", "category": "file"},
                "file_write": {"label": "Write File", "category": "file"},
                "file_list": {"label": "List Files", "category": "file"},
                "spreadsheet_read": {"label": "Read CSV Spreadsheet", "category": "spreadsheet"},
                "spreadsheet_write": {"label": "Write CSV Spreadsheet", "category": "spreadsheet"},
                "database_query": {"label": "Read SQLite Database", "category": "database"},
                "report_generate": {"label": "Generate Report", "category": "data"},
                "monitor_check": {"label": "Monitoring Check", "category": "monitoring"},
                "alert": {"label": "Alert", "category": "monitoring"},
                "browser_fetch": {"label": "Fetch Web Page", "category": "browser", "requires_network": True},
                "api_request": {"label": "API Request", "category": "api", "requires_network": True},
                "audit": {"label": "Audit Event", "category": "security"},
            },
            "condition_ops": ["equals", "not_equals", "contains", "exists", "truthy"],
            "safety": {
                "file_root": str(self.files_root),
                "network_default": False,
                "rollback_supported": True,
            },
        }

    def visual_model(self, workflow_id: str) -> dict[str, Any]:
        workflow = self.store.get_workflow(workflow_id)
        if workflow is None:
            raise ValueError(f"Unknown workflow '{workflow_id}'.")
        nodes = [
            {"id": "trigger", "type": "trigger", "label": workflow["trigger_type"], "x": 40, "y": 80}
        ]
        edges = []
        previous = "trigger"
        for index, step in enumerate(workflow["steps"], start=1):
            node_id = f"step-{index}"
            nodes.append(
                {
                    "id": node_id,
                    "type": "action",
                    "action": step["action"],
                    "label": step.get("label", step["action"]),
                    "x": 40 + index * 180,
                    "y": 80,
                }
            )
            edges.append({"from": previous, "to": node_id})
            previous = node_id
        layout = self.get_layout(workflow_id)
        if layout:
            nodes = apply_layout(nodes, layout["layout"].get("nodes", []))
        return {"workflow": workflow, "nodes": nodes, "edges": edges, "layout": layout}

    def save_layout(self, workflow_id: str, layout: dict[str, Any]) -> dict[str, Any]:
        workflow = self.store.get_workflow(workflow_id)
        if workflow is None:
            raise ValueError(f"Unknown workflow '{workflow_id}'.")
        visual = self.visual_model(workflow_id)
        allowed_ids = {node["id"] for node in visual["nodes"]}
        raw_nodes = layout.get("nodes") if isinstance(layout.get("nodes"), list) else []
        nodes = []
        for index, node in enumerate(raw_nodes):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            if node_id not in allowed_ids:
                continue
            nodes.append(
                {
                    "id": node_id,
                    "x": int(float(node.get("x", 40 + index * 180))),
                    "y": int(float(node.get("y", 80))),
                    "order": int(float(node.get("order", index))),
                }
            )
        if not nodes:
            raise ValueError("Workflow layout must include at least one known node.")
        layout_id = stable_id("workflow-layout", workflow_id)
        now = utc_now()
        normalized = {
            "nodes": nodes,
            "privacy_boundary": "Workflow layout edits are local UI metadata and do not execute workflow actions.",
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO workflow_layouts (
                    id, workflow_id, project, layout_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM workflow_layouts WHERE id = ?), ?), ?)
                """,
                (
                    layout_id,
                    workflow_id,
                    workflow["project"],
                    json.dumps(normalized, sort_keys=True),
                    layout_id,
                    now,
                    now,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="workflow",
                action="workflow.layout.save",
                resource=f"workflow:{workflow_id}",
                project=workflow["project"],
                metadata={"layout_id": layout_id, "node_count": len(nodes)},
            )
        )
        return self.get_layout(workflow_id) or {}

    def get_layout(self, workflow_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT * FROM workflow_layouts WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
        return layout_row(row) if row else None

    def _run_step(
        self,
        step: dict[str, Any],
        *,
        project: str,
        index: int,
        context: dict[str, Any],
        allow_network: bool,
    ) -> StepResult:
        action = str(step.get("action", "")).strip()
        if action == "add_event":
            event = self.context.ingest(
                title=str(step.get("title", f"Workflow step {index}")),
                content=render_template(str(step.get("content", "")), context),
                type=str(step.get("type", "workflow")),
                source="workflow",
                project=project,
                metadata={"step": index},
            )
            return StepResult("completed", action, {"event": event})

        if action == "run_agent":
            run = self.agents.run(
                str(step.get("agent", "executive")),
                render_template(str(step.get("goal", "Review workflow state")), context),
                project=project,
            )
            return StepResult("completed", action, {"run": run})

        if action == "search_context":
            matches = self.context.search(render_template(str(step.get("query", "")), context), project=project, limit=10)
            return StepResult("completed", action, {"matches": matches})

        if action == "chat":
            chat = self.chat.send(
                render_template(str(step.get("message", "")), context),
                project=project,
                provider=str(step.get("provider", "auto")),
                allow_external=bool(step.get("allow_external", False)),
            )
            return StepResult("completed", action, {"chat": chat})

        if action == "condition":
            passed = condition_matches(step.get("condition"), context)
            return StepResult("completed", action, {"passed": passed})

        if action == "file_list":
            target = self.safe_file_path(str(step.get("path", ".")))
            files = sorted(path.name for path in target.iterdir()) if target.exists() and target.is_dir() else []
            return StepResult("completed", action, {"path": str(target), "files": files})

        if action == "file_read":
            target = self.safe_file_path(str(step.get("path", "")))
            content = target.read_text(encoding="utf-8")
            return StepResult("completed", action, {"path": str(target), "content": content})

        if action == "file_write":
            target = self.safe_file_path(str(step.get("path", "")))
            target.parent.mkdir(parents=True, exist_ok=True)
            existed = target.exists()
            previous = target.read_text(encoding="utf-8") if existed else ""
            content = render_template(str(step.get("content", "")), context)
            target.write_text(content, encoding="utf-8")
            rollback = {
                "action": "restore_file" if existed else "delete_file",
                "path": str(target),
                "previous_content": previous,
            }
            return StepResult("completed", action, {"path": str(target), "bytes": len(content)}, rollback)

        if action == "spreadsheet_read":
            target = self.safe_file_path(str(step.get("path", "")))
            with target.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            return StepResult("completed", action, {"path": str(target), "rows": rows, "row_count": len(rows)})

        if action == "spreadsheet_write":
            target = self.safe_file_path(str(step.get("path", "")))
            target.parent.mkdir(parents=True, exist_ok=True)
            existed = target.exists()
            previous = target.read_text(encoding="utf-8") if existed else ""
            rows = step.get("rows") if isinstance(step.get("rows"), list) else []
            normalized_rows = [row for row in rows if isinstance(row, dict)]
            fieldnames = sorted({key for row in normalized_rows for key in row.keys()})
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(normalized_rows)
            rollback = {
                "action": "restore_file" if existed else "delete_file",
                "path": str(target),
                "previous_content": previous,
            }
            return StepResult(
                "completed",
                action,
                {"path": str(target), "row_count": len(normalized_rows), "columns": fieldnames},
                rollback,
            )

        if action == "database_query":
            query = str(step.get("query", "")).strip()
            if not query.lower().startswith("select"):
                raise WorkflowSafetyError("Only SELECT database queries are supported.")
            database_path = str(step.get("path", "__openagent__")).strip() or "__openagent__"
            target = self.store.db_path if database_path == "__openagent__" else self.safe_file_path(database_path)
            connection = sqlite3.connect(target)
            try:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(query).fetchmany(100)
            finally:
                connection.close()
            return StepResult(
                "completed",
                action,
                {"path": str(target), "rows": [dict(row) for row in rows], "row_count": len(rows)},
            )

        if action == "report_generate":
            target = self.safe_file_path(str(step.get("path", "reports/report.md")))
            target.parent.mkdir(parents=True, exist_ok=True)
            existed = target.exists()
            previous = target.read_text(encoding="utf-8") if existed else ""
            title = str(step.get("title", "Workflow Report")).strip() or "Workflow Report"
            sections = step.get("sections") if isinstance(step.get("sections"), list) else []
            body = [f"# {title}", ""]
            for section in sections:
                if isinstance(section, dict):
                    body.extend([f"## {section.get('title', 'Section')}", "", str(section.get("content", "")), ""])
            if not sections:
                body.extend(["## Summary", "", json.dumps(context["steps"], indent=2, sort_keys=True)])
            content = "\n".join(body).strip() + "\n"
            target.write_text(content, encoding="utf-8")
            rollback = {
                "action": "restore_file" if existed else "delete_file",
                "path": str(target),
                "previous_content": previous,
            }
            return StepResult("completed", action, {"path": str(target), "bytes": len(content)}, rollback)

        if action == "monitor_check":
            passed = condition_matches(step.get("condition"), context)
            severity = str(step.get("severity", "info"))
            return StepResult(
                "completed" if passed else "warning",
                action,
                {"passed": passed, "severity": severity, "message": str(step.get("message", ""))},
            )

        if action == "alert":
            event = self.store.add_audit_event(
                AuditEvent(
                    actor=str(step.get("actor", "workflow")),
                    action=str(step.get("alert_action", "workflow.alert")),
                    resource=str(step.get("resource", f"workflow-step:{index}")),
                    project=project,
                    status=str(step.get("severity", "info")),
                    metadata={"message": str(step.get("message", "")), "step": index},
                )
            )
            return StepResult("completed", action, {"alert": event})

        if action == "browser_fetch":
            self.require_network(allow_network, action)
            url = str(step.get("url", "")).strip()
            html = http_request(url, method="GET")
            return StepResult(
                "completed",
                action,
                {"url": url, "title": extract_title(html), "content_preview": html[:500]},
            )

        if action == "api_request":
            self.require_network(allow_network, action)
            url = str(step.get("url", "")).strip()
            method = str(step.get("method", "GET")).upper()
            body = step.get("body")
            response = http_request(url, method=method, body=body)
            return StepResult("completed", action, {"url": url, "method": method, "response": response[:2000]})

        if action == "audit":
            event = self.store.add_audit_event(
                AuditEvent(
                    actor=str(step.get("actor", "workflow")),
                    action=str(step.get("audit_action", "workflow.audit")),
                    resource=str(step.get("resource", f"workflow-step:{index}")),
                    project=project,
                    status=str(step.get("status", "allowed")),
                    metadata={"step": index},
                )
            )
            return StepResult("completed", action, {"audit": event})

        raise ValueError(f"Unsupported workflow action '{action}'.")

    def safe_file_path(self, path: str) -> Path:
        if not path:
            raise ValueError("A file path is required.")
        target = (self.files_root / path).resolve()
        if target != self.files_root and self.files_root not in target.parents:
            raise WorkflowSafetyError("File action attempted to leave the workflow file root.")
        return target

    def require_network(self, allow_network: bool, action: str) -> None:
        if not allow_network:
            raise WorkflowSafetyError(f"Workflow action '{action}' requires allow_network=true.")

    def _execute_rollbacks(self, rollbacks: list[dict[str, Any]], *, project: str) -> None:
        for rollback in reversed(rollbacks):
            action = rollback.get("action")
            path = Path(str(rollback.get("path", ""))).resolve()
            if action == "restore_file":
                path.write_text(str(rollback.get("previous_content", "")), encoding="utf-8")
            elif action == "delete_file" and path.exists():
                path.unlink()
            self.store.add_audit_event(
                AuditEvent(
                    actor="workflow",
                    action=f"rollback.{action}",
                    resource=str(path),
                    project=project,
                    metadata={"rollback": rollback},
                )
            )

    def _store_rollbacks(
        self,
        workflow_run_id: int | str,
        workflow_id: str,
        project: str,
        rollbacks: list[dict[str, Any]],
    ) -> None:
        with self.store.session() as connection:
            for index, rollback in enumerate(rollbacks, start=1):
                connection.execute(
                    """
                    INSERT INTO workflow_rollback_records (
                        id, workflow_run_id, workflow_id, project, step_index,
                        action, rollback_json, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stable_id("rollback", str(workflow_run_id), str(index)),
                        str(workflow_run_id),
                        workflow_id,
                        project,
                        index,
                        str(rollback.get("action", "")),
                        json.dumps(rollback, sort_keys=True),
                        "available",
                        utc_now(),
                    ),
                )

    def _update_next_run(self, trigger_id: str, next_run_at: str) -> None:
        with self.store.session() as connection:
            connection.execute(
                "UPDATE workflow_triggers SET next_run_at = ?, updated_at = ? WHERE id = ?",
                (next_run_at, utc_now(), trigger_id),
            )


class BackgroundWorkflowWorker:
    """Tiny background scheduler loop for embedded/self-hosted deployments."""

    def __init__(self, engine: WorkflowEngine, *, poll_seconds: float = 5.0, allow_network: bool = False):
        self.engine = engine
        self.poll_seconds = poll_seconds
        self.allow_network = allow_network
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.poll_seconds + 1)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.engine.run_due_schedules(allow_network=self.allow_network)
            self._stop.wait(self.poll_seconds)


def condition_matches(condition: Any, context: dict[str, Any]) -> bool:
    if not condition:
        return True
    if isinstance(condition, bool):
        return condition
    if not isinstance(condition, dict):
        return bool(condition)
    left = resolve_path(context, str(condition.get("path", "")))
    op = str(condition.get("op", "truthy"))
    expected = condition.get("value")
    if op == "exists":
        return left is not None
    if op == "truthy":
        return bool(left)
    if op == "equals":
        return left == expected
    if op == "not_equals":
        return left != expected
    if op == "contains":
        return str(expected) in str(left)
    return False


def resolve_path(data: Any, path: str) -> Any:
    if not path:
        return data
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def render_template(template: str, context: dict[str, Any]) -> str:
    result = template
    for key in ["event", "workflow"]:
        value = context.get(key, {})
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                result = result.replace(f"{{{{{key}.{nested_key}}}}}", str(nested_value))
    return result


def iso_after(seconds: int) -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=seconds)).isoformat()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def trigger_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "workflow_id": row["workflow_id"],
        "project": row["project"],
        "trigger_type": row["trigger_type"],
        "config": json.loads(row["config_json"]),
        "next_run_at": row["next_run_at"],
        "enabled": bool(row["enabled"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def layout_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "workflow_id": row["workflow_id"],
        "project": row["project"],
        "layout": json.loads(row["layout_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def apply_layout(nodes: list[dict[str, Any]], layout_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layout_by_id = {
        str(node.get("id", "")): node
        for node in layout_nodes
        if isinstance(node, dict) and str(node.get("id", ""))
    }
    updated = []
    for index, node in enumerate(nodes):
        layout = layout_by_id.get(node["id"], {})
        updated.append(
            {
                **node,
                "x": int(layout.get("x", node["x"])),
                "y": int(layout.get("y", node["y"])),
                "order": int(layout.get("order", index)),
            }
        )
    return sorted(updated, key=lambda item: int(item.get("order", 0)))


def http_request(url: str, *, method: str = "GET", body: Any = None) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise WorkflowSafetyError("Only http and https URLs are supported.")
    data = None
    headers = {"User-Agent": "OpenAgent-Workflow/0.3"}
    if body is not None:
        data = json.dumps(body).encode("utf-8") if not isinstance(body, str) else body.encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_title(html: str) -> str:
    lower = html.lower()
    start = lower.find("<title>")
    end = lower.find("</title>")
    if start == -1 or end == -1 or end <= start:
        return ""
    return html[start + len("<title>") : end].strip()
