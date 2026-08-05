"""Model cookbook, cached-file discovery, and safe serve-session records."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from .local_ai import LocalAIInfrastructure
from .model_gateway import ModelGateway
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


DENIED_SHELL_RE = re.compile(r"[;&|`$<>]")
ALLOWED_RUNNERS = {"ollama", "llama-server", "llama.cpp", "python", "python.exe", "vllm", "vllm.exe"}


class ModelCookbook:
    """Local model operations without shell interpolation."""

    def __init__(self, store: OpenAgentStore, gateway: ModelGateway | None = None) -> None:
        self.store = store
        self.gateway = gateway or ModelGateway()
        self.local_ai = LocalAIInfrastructure(store, self.gateway)
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_hosts (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    label TEXT NOT NULL,
                    host_type TEXT NOT NULL DEFAULT 'local',
                    address TEXT NOT NULL DEFAULT '127.0.0.1',
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_serve_sessions (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    host_id TEXT NOT NULL DEFAULT '',
                    model_ref TEXT NOT NULL DEFAULT '',
                    runner TEXT NOT NULL DEFAULT '',
                    command_json TEXT NOT NULL DEFAULT '[]',
                    pid INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'planned',
                    mode TEXT NOT NULL DEFAULT 'plan',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    stopped_at TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS model_serve_logs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    level TEXT NOT NULL DEFAULT 'info',
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_cached_files (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    sha_hint TEXT NOT NULL DEFAULT '',
                    discovered_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_fit_profiles (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    model_ref TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_endpoint_certifications (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    endpoint TEXT NOT NULL,
                    model_ref TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    report_json TEXT NOT NULL DEFAULT '{}',
                    report_signature TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_model_endpoint_certifications
                    ON model_endpoint_certifications(project, updated_at DESC);
                """
            )

    def hosts(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM model_hosts WHERE project = ? ORDER BY updated_at DESC",
                (project,),
            ).fetchall()
        return [host_row(row) for row in rows]

    def add_host(self, *, label: str, project: str = "default", host_type: str = "local", address: str = "127.0.0.1", config: dict[str, Any] | None = None) -> dict[str, Any]:
        now = utc_now()
        host_id = stable_id("model-host", project, label, host_type, address)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO model_hosts (
                    id, project, label, host_type, address, config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM model_hosts WHERE id = ?), ?), ?)
                """,
                (host_id, project, label, host_type, address, json.dumps(config or {}, sort_keys=True), host_id, now, now),
            )
        return self.hosts(project=project)[0]

    def cached_files(self, *, project: str = "default", root: str = "") -> list[dict[str, Any]]:
        roots = []
        if root:
            roots.append(Path(root).expanduser())
        roots.extend([self.store.data_dir / "models", Path.home() / ".ollama" / "models"])
        discovered = []
        now = utc_now()
        with self.store.session() as connection:
            for candidate_root in roots:
                if not candidate_root.exists():
                    continue
                for path in candidate_root.rglob("*"):
                    if not path.is_file() or path.suffix.lower() not in {".gguf", ".bin", ".safetensors", ".onnx"}:
                        continue
                    stat = path.stat()
                    file_id = stable_id("cached-model", project, str(path))
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO model_cached_files (
                            id, project, path, filename, size_bytes, sha_hint, discovered_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (file_id, project, str(path), path.name, stat.st_size, "", now),
                    )
                    discovered.append(file_id)
            rows = connection.execute(
                "SELECT * FROM model_cached_files WHERE project = ? ORDER BY discovered_at DESC",
                (project,),
            ).fetchall()
        return [cached_file_row(row) for row in rows]

    def serve(self, *, project: str = "default", host_id: str = "", model_ref: str, runner: str, args: list[str] | None = None, allow_process: bool = False) -> dict[str, Any]:
        command = self._validate_command(runner, args or [])
        now = utc_now()
        session_id = uuid4().hex
        status = "planned"
        pid = 0
        mode = "plan"
        if allow_process:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)  # noqa: S603 - command is allow-listed and shell=False
            pid = int(process.pid)
            status = "running"
            mode = "local-process"
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO model_serve_sessions (
                    id, project, host_id, model_ref, runner, command_json, pid, status, mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, project, host_id, model_ref, runner, json.dumps(command), pid, status, mode, now, now),
            )
            self._log(connection, session_id, "info", f"Serve session {status}: {' '.join(command)}")
        self.store.add_audit_event(
            AuditEvent(
                actor="model-cookbook",
                action="model.serve",
                resource=f"serve:{session_id}",
                project=project,
                metadata={"runner": runner, "mode": mode, "allow_process": allow_process},
            )
        )
        return self.session(session_id)

    def adopt(self, *, project: str = "default", host_id: str = "", model_ref: str = "", runner: str = "", pid: int = 0, command: list[str] | None = None) -> dict[str, Any]:
        now = utc_now()
        session_id = uuid4().hex
        command = command or ([runner, model_ref] if runner else [])
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO model_serve_sessions (
                    id, project, host_id, model_ref, runner, command_json, pid, status, mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'adopted', 'adopted', ?, ?)
                """,
                (session_id, project, host_id, model_ref, runner, json.dumps(command), int(pid), now, now),
            )
            self._log(connection, session_id, "info", "Adopted externally-started model process.")
        return self.session(session_id)

    def stop(self, session_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM model_serve_sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                raise KeyError("serve session not found")
            connection.execute(
                "UPDATE model_serve_sessions SET status = 'stopped', stopped_at = ?, updated_at = ? WHERE id = ?",
                (now, now, session_id),
            )
            self._log(connection, session_id, "info", "Stop requested and recorded locally.")
        return self.session(session_id)

    def session(self, session_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM model_serve_sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError("serve session not found")
        return session_row(row)

    def logs(self, session_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM model_serve_logs WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, max(1, min(limit, 500))),
            ).fetchall()
        return [log_row(row) for row in rows]

    def hardware_fit(self, *, project: str = "default", model_ref: str = "", parameter_count_b: float = 3.0) -> dict[str, Any]:
        model_ref = model_ref or "local-model"
        hardware = self.local_ai.detect_hardware()
        ram_gb = float(hardware.get("ram_gb", 8.0))
        recommended_bits = 4 if ram_gb >= 8 else 3
        if ram_gb < 4:
            recommended_bits = 2
        profile = {
            "model_ref": model_ref,
            "parameter_count_b": parameter_count_b,
            "ram_gb": ram_gb,
            "recommended_bits": recommended_bits,
            "expected_memory_gb": round(parameter_count_b * recommended_bits / 8 * 1.25, 2),
            "runner_priority": ["ollama", "llama.cpp", "vllm"],
            "privacy_boundary": "Hardware fit uses local system telemetry only.",
        }
        profile_id = stable_id("fit", project, model_ref, str(parameter_count_b), utc_now())
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO model_fit_profiles (id, project, model_ref, profile_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (profile_id, project, model_ref, json.dumps(profile, sort_keys=True), utc_now()),
            )
        return profile

    def certify_endpoint(
        self,
        *,
        endpoint: str,
        project: str = "default",
        model_ref: str = "",
        allow_external: bool = False,
        api_key: str = "",
        completion_probe: bool = True,
    ) -> dict[str, Any]:
        endpoint = endpoint.strip().rstrip("/")
        parsed = urlparse(endpoint)
        loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Model endpoint must be an HTTP(S) URL.")
        if not loopback and (parsed.scheme != "https" or not allow_external):
            raise ValueError("Remote model certification requires HTTPS and allow_external=true.")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        checks = []
        started = time.perf_counter()
        models_payload = self._endpoint_json(Request(f"{endpoint}/v1/models", headers=headers), timeout=10)
        models = models_payload.get("data") if isinstance(models_payload, dict) else []
        model_ids = [str(item.get("id", "")) for item in models if isinstance(item, dict) and item.get("id")] if isinstance(models, list) else []
        checks.append({"id": "models", "passed": isinstance(models, list), "models": model_ids[:100]})
        selected_model = model_ref or (model_ids[0] if model_ids else "")
        if completion_probe and selected_model:
            body = json.dumps({
                "model": selected_model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 4,
                "temperature": 0,
                "stream": False,
            }).encode("utf-8")
            completion = self._endpoint_json(
                Request(f"{endpoint}/v1/chat/completions", data=body, headers=headers, method="POST"), timeout=30,
            )
            choices = completion.get("choices") if isinstance(completion, dict) else None
            checks.append({"id": "chat_completions", "passed": isinstance(choices, list) and bool(choices)})
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        passed = all(bool(check["passed"]) for check in checks)
        report = {
            "endpoint": endpoint,
            "loopback": loopback,
            "tls_required": not loopback,
            "tls_active": parsed.scheme == "https",
            "openai_compatible": passed,
            "model_ref": selected_model,
            "checks": checks,
            "latency_ms": latency_ms,
            "operator_controls": ["explicit-egress-approval", "bounded-timeouts", "revocable-certification", "signed-report"],
            "privacy_boundary": "Only the explicitly approved model endpoint receives the fixed certification prompt; no project context is sent.",
        }
        certification_id = uuid4().hex
        now = utc_now()
        signature = hmac.new(
            self._certification_secret().encode("utf-8"), json.dumps(report, sort_keys=True).encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO model_endpoint_certifications (id, project, endpoint, model_ref, status, report_json, report_signature, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (certification_id, project, endpoint, selected_model, "certified" if passed else "failed", json.dumps(report, sort_keys=True), signature, now, now),
            )
        self.store.add_audit_event(AuditEvent(
            actor="model-cookbook", action="model.endpoint.certify", resource=f"certification:{certification_id}",
            project=project, status="ok" if passed else "failed", metadata={"endpoint": endpoint, "loopback": loopback, "checks": len(checks)},
        ))
        return self.certification(certification_id)

    def certifications(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM model_endpoint_certifications WHERE project = ? ORDER BY updated_at DESC", (project,)
            ).fetchall()
        return [certification_row(row) for row in rows]

    def certification(self, certification_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM model_endpoint_certifications WHERE id = ?", (certification_id,)).fetchone()
        if row is None:
            raise KeyError("model endpoint certification not found")
        return certification_row(row)

    def revoke_certification(self, certification_id: str) -> dict[str, Any]:
        certification = self.certification(certification_id)
        with self.store.session() as connection:
            connection.execute(
                "UPDATE model_endpoint_certifications SET status = 'revoked', updated_at = ? WHERE id = ?",
                (utc_now(), certification_id),
            )
        self.store.add_audit_event(AuditEvent(
            actor="model-cookbook", action="model.endpoint.certification.revoke",
            resource=f"certification:{certification_id}", project=certification["project"],
            metadata={"endpoint": certification["endpoint"]},
        ))
        return self.certification(certification_id)

    @staticmethod
    def _endpoint_json(request: Request, *, timeout: int) -> dict[str, Any]:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Model endpoint response exceeded the 2 MB certification limit.")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Model endpoint returned a non-object JSON response.")
        return payload

    def _certification_secret(self) -> str:
        path = self.store.data_dir / ".model-certification-key"
        if not path.exists():
            path.write_text(secrets.token_urlsafe(48), encoding="ascii")
            try:
                path.chmod(0o600)
            except OSError:
                pass
        return path.read_text(encoding="ascii").strip()

    def _validate_command(self, runner: str, args: list[str]) -> list[str]:
        runner = Path(runner).name if runner else ""
        if runner not in ALLOWED_RUNNERS:
            raise ValueError(f"Runner '{runner}' is not allow-listed.")
        command = [runner, *[str(arg) for arg in args]]
        for part in command:
            if DENIED_SHELL_RE.search(part):
                raise ValueError("Shell metacharacters are not allowed in model serve commands.")
        return command

    def _log(self, connection: Any, session_id: str, level: str, message: str) -> None:
        connection.execute(
            "INSERT INTO model_serve_logs (id, session_id, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid4().hex, session_id, level, message, utc_now()),
        )


def host_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "label": row["label"],
        "host_type": row["host_type"],
        "address": row["address"],
        "config": json.loads(row["config_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def session_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "host_id": row["host_id"],
        "model_ref": row["model_ref"],
        "runner": row["runner"],
        "command": json.loads(row["command_json"] or "[]"),
        "pid": row["pid"],
        "status": row["status"],
        "mode": row["mode"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "stopped_at": row["stopped_at"],
    }


def log_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "level": row["level"],
        "message": row["message"],
        "created_at": row["created_at"],
    }


def cached_file_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "path": row["path"],
        "filename": row["filename"],
        "size_bytes": row["size_bytes"],
        "sha_hint": row["sha_hint"],
        "discovered_at": row["discovered_at"],
    }


def certification_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "endpoint": row["endpoint"],
        "model_ref": row["model_ref"],
        "status": row["status"],
        "report": json.loads(row["report_json"] or "{}"),
        "report_signature": row["report_signature"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def stable_id(*parts: str) -> str:
    return __import__("hashlib").sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]
