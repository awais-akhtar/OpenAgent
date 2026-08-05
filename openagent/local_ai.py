"""Stage 6 local AI infrastructure management."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from .core_runtime import CPU_MODEL_STACK, adaptive_quantization_plan, detect_total_memory_gb, model_profiles, smooth_runtime_profile
from .model_gateway import ModelGateway, local_http_available
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class ModelDownloadError(RuntimeError):
    """Raised when a model download crosses a boundary without approval."""


# Explicit model-ops lifecycle. This is additive to the simple
# registered/available status already written by register_model/
# download_model (untouched) — richer state tracking is opt-in via
# LocalAIInfrastructure.transition_model_lifecycle.
MODEL_LIFECYCLE_STATES: tuple[str, ...] = ("discovered", "downloading", "ready", "serving", "degraded", "failed")

MODEL_FAILURE_REASONS: tuple[str, ...] = (
    "download_failed",
    "checksum_mismatch",
    "insufficient_memory",
    "port_in_use",
    "dependency_missing",
    "timeout",
    "unknown",
)

_MODEL_LIFECYCLE_TRANSITIONS: dict[str, set[str]] = {
    "discovered": {"downloading", "ready", "failed"},
    "downloading": {"ready", "failed", "degraded"},
    "ready": {"serving", "degraded", "failed", "downloading"},
    "serving": {"degraded", "failed", "ready"},
    "degraded": {"ready", "serving", "failed"},
    "failed": {"discovered", "downloading"},
}


class LocalAIInfrastructure:
    def __init__(self, store: OpenAgentStore, gateway: ModelGateway | None = None):
        self.store = store
        self.gateway = gateway or ModelGateway()
        self.model_root = (self.store.data_dir / "models").resolve()
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.initialize()
        self.seed_marketplace()
        self.seed_acceleration_adapters()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS local_models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    source_url TEXT NOT NULL DEFAULT '',
                    local_path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_quantizations (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    bits INTEGER NOT NULL,
                    format TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES local_models(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS model_benchmarks (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    duration_ms REAL NOT NULL,
                    tokens_per_second REAL NOT NULL,
                    hardware_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES local_models(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS hardware_profiles (
                    id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS acceleration_adapters (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    hardware TEXT NOT NULL,
                    available INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS inference_clusters (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_marketplace (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    recommended_engine TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS model_setup_jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    project TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_name TEXT NOT NULL DEFAULT '',
                    model_id TEXT NOT NULL DEFAULT '',
                    progress INTEGER NOT NULL DEFAULT 0,
                    current_step TEXT NOT NULL DEFAULT '',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_model_setup_jobs_project_updated
                    ON model_setup_jobs(project, updated_at DESC);

                CREATE TABLE IF NOT EXISTS model_diagnostics (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_model_diagnostics_model_created
                    ON model_diagnostics(model_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS model_dependencies (
                    model_id TEXT NOT NULL,
                    dependent_kind TEXT NOT NULL,
                    dependent_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(model_id, dependent_kind, dependent_id)
                );

                CREATE TABLE IF NOT EXISTS model_tool_assignments (
                    model_id TEXT NOT NULL,
                    tool_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(model_id, tool_id)
                );
                """
            )

    def seed_marketplace(self) -> None:
        models = [
            ("smollm2-135m-q1-58", "SmolLM2 135M Q1.58", "llamacpp", "llama.cpp", {"size": "135M", "format": "gguf", "quantization": "Q1.58", "ram_gb": 3, "tier": "survival-cpu"}),
            ("gemma3-2b-q4", "Gemma 3 2B Q4", "ollama", "ollama", {"size": "2B", "quantization": "Q4_K_M", "ram_gb": 6, "tier": "speed-cpu", "roles": ["chat", "speed"], "model": "gemma3:2b", "tokens_per_second_hint": "fast"}),
            ("phi4-mini-3.8b-q4", "Phi-4 Mini 3.8B Q4", "ollama", "ollama", {"size": "3.8B", "quantization": "Q4_K_M", "ram_gb": 8, "tier": "default-cpu", "roles": ["chat", "agent", "research", "reasoning"], "model": "phi4-mini", "context_hint": "long"}),
            ("qwen2.5-coder-3b-q4", "Qwen2.5 Coder 3B Q4", "ollama", "ollama", {"size": "3B", "quantization": "Q4_K_M", "ram_gb": 8, "tier": "code-default", "roles": ["code", "debugging", "sql", "python"], "model": "qwen2.5-coder:3b"}),
            ("deepseek-coder-1.3b-q4", "DeepSeek Coder 1.3B Q4", "ollama", "ollama", {"size": "1.3B", "quantization": "Q4_K_M", "ram_gb": 4, "tier": "code-survival", "roles": ["code", "low-resource"], "model": "deepseek-coder:1.3b"}),
            ("llama3.2-3b-q4", "Llama 3.2 3B Q4", "ollama", "ollama", {"size": "3B", "quantization": "Q4_K_M", "ram_gb": 8, "tier": "rag-extraction", "roles": ["rag", "summarization", "extraction"], "model": "llama3.2:3b"}),
            ("qwen2.5-7b-q4", "Qwen2.5 7B Q4", "ollama", "ollama", {"size": "7B", "license": "community", "quantization": "Q4_K_M", "ram_gb": 16, "tier": "research-high-context", "roles": ["research", "agents", "json", "tool-calling"], "model": "qwen2.5:7b"}),
            ("qwen2.5-0.5b-q4", "Qwen2.5 0.5B Q4", "ollama", "ollama", {"size": "0.5B", "quantization": "Q4_K_M", "ram_gb": 4, "tier": "legacy-ultra-light"}),
            ("qwen2.5-0.5b-q2", "Qwen2.5 0.5B Q2", "ollama", "ollama", {"size": "0.5B", "quantization": "Q2_K", "ram_gb": 5, "tier": "legacy-micro-cpu"}),
            ("qwen2.5-coder-1.5b-q4", "Qwen2.5 Coder 1.5B Q4", "ollama", "ollama", {"size": "1.5B", "quantization": "Q4_K_M", "ram_gb": 6, "tier": "legacy-code-light"}),
            ("smollm2-1.7b-q4", "SmolLM2 1.7B Q4", "llamacpp", "llama.cpp", {"size": "1.7B", "format": "gguf", "quantization": "Q4_K_M", "ram_gb": 6, "tier": "legacy-research-light"}),
            ("llama3.1-8b", "Llama 3.1 8B", "ollama", "ollama", {"size": "8B", "license": "community", "ram_gb": 16, "tier": "large"}),
            ("qwen2.5-7b", "Qwen2.5 7B", "ollama", "ollama", {"size": "7B", "license": "community", "ram_gb": 16, "tier": "large"}),
            ("mistral-7b", "Mistral 7B", "llamacpp", "llama.cpp", {"size": "7B", "format": "gguf", "ram_gb": 16, "tier": "large"}),
            ("openagent-local", "OpenAgent Local Deterministic", "local", "local", {"offline": True}),
        ]
        with self.store.session() as connection:
            for model_id, name, provider, engine, metadata in models:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO model_marketplace (
                        id, name, provider, recommended_engine, metadata_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (model_id, name, provider, engine, json.dumps(metadata, sort_keys=True)),
                )

    def seed_acceleration_adapters(self) -> None:
        machine = platform.machine().lower()
        adapters = [
            ("cpu", "CPU", "cpu", True, {}),
            ("cuda", "NVIDIA CUDA", "nvidia", False, {}),
            ("rocm", "AMD ROCm", "amd", False, {}),
            ("metal", "Apple Metal", "apple", "arm" in machine and platform.system() == "Darwin", {}),
            ("onnxruntime", "ONNX Runtime", "cross-platform", True, {}),
            ("tensorrt", "TensorRT", "nvidia", False, {}),
        ]
        with self.store.session() as connection:
            for adapter_id, label, hardware, available, metadata in adapters:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO acceleration_adapters (
                        id, label, hardware, available, metadata_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (adapter_id, label, hardware, 1 if available else 0, json.dumps(metadata)),
                )

    def register_model(
        self,
        *,
        name: str,
        provider: str,
        source_url: str = "",
        local_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model_id = stable_id("model", name, provider, source_url or local_path)
        now = utc_now()
        status = "available" if local_path else "registered"
        size = Path(local_path).stat().st_size if local_path and Path(local_path).exists() else 0
        digest = file_sha256(Path(local_path)) if local_path and Path(local_path).exists() else ""
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO local_models (
                    id, name, provider, source_url, local_path, status, size_bytes,
                    sha256, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM local_models WHERE id = ?), ?), ?)
                """,
                (
                    model_id,
                    name,
                    provider,
                    source_url,
                    local_path,
                    status,
                    size,
                    digest,
                    json.dumps(metadata or {}, sort_keys=True),
                    model_id,
                    now,
                    now,
                ),
            )
        return self.get_model(model_id) or {}

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM local_models WHERE id = ?", (model_id,)).fetchone()
        return model_row(row) if row else None

    def models(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM local_models ORDER BY updated_at DESC").fetchall()
        return [model_row(row) for row in rows]

    def download_model(self, model_id: str, *, allow_network: bool = False) -> dict[str, Any]:
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Unknown model '{model_id}'.")
        source = model["source_url"]
        if not source:
            raise ModelDownloadError("Model has no source_url.")
        parsed = urlparse(source)
        target = self.model_root / f"{model_id}.bin"
        partial = target.with_suffix(target.suffix + ".part")
        expected_sha256 = str(model.get("metadata", {}).get("sha256", "")).strip().lower()
        is_windows_drive = len(parsed.scheme) == 1 and len(source) > 2 and source[1] == ":"
        if parsed.scheme in {"", "file"} or is_windows_drive:
            source_path = Path(parsed.path if parsed.scheme == "file" else source).resolve()
            shutil.copyfile(source_path, partial)
        elif parsed.scheme in {"http", "https"}:
            if not allow_network:
                raise ModelDownloadError("Network model downloads require allow_network=true.")
            downloaded = partial.stat().st_size if partial.exists() else 0
            request = Request(source, headers={"Range": f"bytes={downloaded}-"} if downloaded else {})
            try:
                with urlopen(request, timeout=120) as response:
                    status = int(getattr(response, "status", 200) or 200)
                    mode = "ab" if downloaded and status == 206 else "wb"
                    if mode == "wb":
                        downloaded = 0
                    with partial.open(mode) as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            downloaded += len(chunk)
            except Exception as exc:
                self._set_model_download_state(model_id, "downloading", partial.stat().st_size if partial.exists() else 0, str(exc))
                raise ModelDownloadError(f"Model download paused after {partial.stat().st_size if partial.exists() else 0} bytes: {exc}") from exc
        else:
            raise ModelDownloadError(f"Unsupported model source scheme '{parsed.scheme}'.")
        digest = file_sha256(partial)
        if expected_sha256 and digest.lower() != expected_sha256:
            self._set_model_download_state(model_id, "failed", partial.stat().st_size, "checksum_mismatch")
            raise ModelDownloadError("Downloaded model checksum does not match the registered SHA-256 value.")
        partial.replace(target)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                UPDATE local_models
                SET local_path = ?, status = ?, size_bytes = ?, sha256 = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(target), "available", target.stat().st_size, digest, now, model_id),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="local_ai",
                action="model.download",
                resource=f"model:{model_id}",
                project="default",
                metadata={"allow_network": allow_network, "source": source},
            )
        )
        return self.get_model(model_id) or {}

    def probe_model(self, model_id: str) -> dict[str, Any]:
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Unknown model '{model_id}'.")
        local_path = str(model.get("local_path", ""))
        path = Path(local_path).expanduser() if local_path and "://" not in local_path else None
        exists = bool(path and path.exists())
        digest = file_sha256(path) if path and path.is_file() else ""
        expected = str(model.get("sha256", ""))
        return {
            "model_id": model_id,
            "status": "ready" if exists or local_path.startswith(("ollama://", "llamacpp://")) else "unavailable",
            "local_path": local_path,
            "exists": exists,
            "size_bytes": path.stat().st_size if path and path.is_file() else model.get("size_bytes", 0),
            "sha256": digest,
            "checksum_valid": not expected or not digest or expected == digest,
            "dependents": self.model_dependents(model_id),
            "tools": self.model_tools(model_id),
        }

    def add_model_dependency(self, model_id: str, dependent_kind: str, dependent_id: str) -> dict[str, Any]:
        if self.get_model(model_id) is None:
            raise ValueError(f"Unknown model '{model_id}'.")
        kind = dependent_kind.strip()
        identifier = dependent_id.strip()
        if not kind or not identifier:
            raise ValueError("dependent_kind and dependent_id are required.")
        with self.store.session() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO model_dependencies (model_id, dependent_kind, dependent_id, created_at) VALUES (?, ?, ?, ?)",
                (model_id, kind, identifier, utc_now()),
            )
        return {"model_id": model_id, "dependent_kind": kind, "dependent_id": identifier}

    def model_dependents(self, model_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM model_dependencies WHERE model_id = ? ORDER BY dependent_kind, dependent_id", (model_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def assign_tool(self, model_id: str, tool_id: str, *, enabled: bool = True, config: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.get_model(model_id) is None:
            raise ValueError(f"Unknown model '{model_id}'.")
        tool_id = tool_id.strip()
        if not tool_id:
            raise ValueError("tool_id is required.")
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO model_tool_assignments (model_id, tool_id, enabled, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id, tool_id) DO UPDATE SET enabled = excluded.enabled,
                    config_json = excluded.config_json, updated_at = excluded.updated_at
                """,
                (model_id, tool_id, 1 if enabled else 0, json.dumps(config or {}, sort_keys=True), now, now),
            )
        return {"model_id": model_id, "tool_id": tool_id, "enabled": enabled, "config": config or {}}

    def model_tools(self, model_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM model_tool_assignments WHERE model_id = ? ORDER BY tool_id", (model_id,)).fetchall()
        return [
            {"model_id": row["model_id"], "tool_id": row["tool_id"], "enabled": bool(row["enabled"]), "config": json.loads(row["config_json"] or "{}")}
            for row in rows
        ]

    def _set_model_download_state(self, model_id: str, status: str, bytes_downloaded: int, error: str) -> None:
        model = self.get_model(model_id)
        metadata = dict(model.get("metadata", {})) if model else {}
        metadata["download"] = {"bytes_downloaded": bytes_downloaded, "error": error, "resumable": True}
        with self.store.session() as connection:
            connection.execute(
                "UPDATE local_models SET status = ?, size_bytes = ?, metadata_json = ?, updated_at = ? WHERE id = ?",
                (status, bytes_downloaded, json.dumps(metadata, sort_keys=True), utc_now(), model_id),
            )

    def pull_ollama_model(
        self,
        model_name: str,
        *,
        allow_network: bool = False,
        project: str = "default",
        timeout_seconds: int = 3600,
    ) -> dict[str, Any]:
        pull_name = ollama_pull_name(model_name)
        if not pull_name:
            raise ModelDownloadError("model_name is required.")
        if not allow_network:
            raise ModelDownloadError("Ollama model pulls require allow_network=true because they may download model files.")
        executable = shutil.which("ollama")
        stdout = ""
        stderr = ""
        returncode = 0
        if executable:
            completed = subprocess.run(
                [executable, "pull", pull_name],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "ollama pull failed").strip()
                raise ModelDownloadError(detail[-600:])
        else:
            base_url = os.environ.get("OPENAGENT_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
            if not local_http_available(f"{base_url}/api/tags"):
                raise ModelDownloadError(
                    "Ollama was not found on PATH and the local Ollama server is not reachable at "
                    f"{base_url}. Install/start Ollama, then use Setup Chat Model again."
                )
            request = Request(
                f"{base_url}/api/pull",
                data=json.dumps({"name": pull_name, "stream": False}).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    stdout = response.read().decode("utf-8", errors="replace")
            except Exception as exc:
                raise ModelDownloadError(f"Ollama HTTP pull failed: {exc}") from exc
        model = self.register_model(
            name=pull_name,
            provider="ollama",
            source_url=f"ollama://{pull_name}",
            local_path=f"ollama://{pull_name}",
            metadata={
                "pulled_by": "ollama",
                "downloaded_for": "openagent-chat",
                "runtime": "local-model-server",
            },
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="local_ai",
                action="model.ollama_pull",
                resource=f"model:{pull_name}",
                project=project,
                metadata={"allow_network": allow_network, "returncode": returncode, "used_cli": bool(executable)},
            )
        )
        return {
            "model": model,
            "ollama_model": pull_name,
            "command": f"ollama pull {pull_name}" if executable else f"POST {os.environ.get('OPENAGENT_OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/pull",
            "stdout": stdout[-1200:],
            "stderr": stderr[-1200:],
            "privacy_boundary": "The user explicitly requested an Ollama pull. Model files are downloaded by the local Ollama runtime into its local model store.",
        }

    def model_tool_status(self) -> dict[str, Any]:
        """Report local model runner availability without starting downloads."""
        ollama = shutil.which("ollama")
        llama_server = shutil.which("llama-server")
        llama_cli = shutil.which("llama-cli") or shutil.which("llama")
        providers = {provider["id"]: provider for provider in self.gateway.specs()}
        ollama_url = os.environ.get("OPENAGENT_OLLAMA_URL", "http://127.0.0.1:11434")
        llamacpp_url = os.environ.get("OPENAGENT_LLAMACPP_URL", "http://127.0.0.1:8080")
        ollama_server_available = local_http_available(f"{ollama_url.rstrip('/')}/api/tags")
        llamacpp_server_available = local_http_available(f"{llamacpp_url.rstrip('/')}/v1/models")
        ollama_spec = providers.get("ollama", {})
        installed_ollama_models = list(ollama_spec.get("installed_models", [])) if isinstance(ollama_spec.get("installed_models"), list) else []
        return {
            "ollama": {
                "available": bool(ollama) or ollama_server_available,
                "path": ollama or "",
                "server_available": ollama_server_available,
                "model_ready": bool(providers.get("ollama", {}).get("available")),
                "default_model_ready": bool(ollama_spec.get("default_model_ready")),
                "model": os.environ.get("OPENAGENT_OLLAMA_MODEL", CPU_MODEL_STACK["chat"]),
                "selected_model": str(ollama_spec.get("selected_model", "")),
                "installed_models": installed_ollama_models,
                "installed_model_count": len(installed_ollama_models),
                "url": ollama_url,
            },
            "llamacpp": {
                "available": bool(llama_server or llama_cli) or llamacpp_server_available,
                "server_path": llama_server or "",
                "cli_path": llama_cli or "",
                "server_available": llamacpp_server_available,
                "model_ready": bool(providers.get("llamacpp", {}).get("available")),
                "model": os.environ.get("OPENAGENT_LLAMACPP_MODEL", CPU_MODEL_STACK["chat"]),
                "url": llamacpp_url,
            },
            "recommended": {
                "chat": CPU_MODEL_STACK["chat"],
                "code": CPU_MODEL_STACK["code"],
                "research": CPU_MODEL_STACK["research"],
                "speed": CPU_MODEL_STACK["speed"],
                "low_resource_code": CPU_MODEL_STACK["low_code"],
            },
            "model_root": str(self.model_root),
            "privacy_boundary": "Tool detection checks local executables and local loopback model servers only.",
        }

    def setup_chat_model(
        self,
        *,
        model_name: str = "",
        allow_network: bool = False,
        project: str = "default",
    ) -> dict[str, Any]:
        return self.setup_task_model(
            task="chat",
            model_name=model_name,
            allow_network=allow_network,
            project=project,
        )

    def setup_task_model(
        self,
        *,
        task: str = "chat",
        model_name: str = "",
        allow_network: bool = False,
        project: str = "default",
    ) -> dict[str, Any]:
        """Prepare the default chat model using the best available local runner."""
        task = (task or "chat").strip().lower()
        default_model = {
            "chat": CPU_MODEL_STACK["chat"],
            "reasoning": CPU_MODEL_STACK["research"],
            "research": CPU_MODEL_STACK["research"],
            "code": CPU_MODEL_STACK["code"],
            "speed": CPU_MODEL_STACK["speed"],
        }.get(task, CPU_MODEL_STACK["chat"])
        requested = ollama_pull_name(model_name or default_model)
        tools = self.model_tool_status()
        if tools["ollama"]["available"]:
            pulled = self.pull_ollama_model(
                requested,
                allow_network=allow_network,
                project=project,
            )
            return {
                "runner": "ollama",
                "model": pulled["model"],
                "ollama_model": pulled["ollama_model"],
                "command": pulled["command"],
                "tool_status": self.model_tool_status(),
                "next_step": f"Keep provider routing on Auto. OpenAgent can use {requested} for {task} turns through the local Ollama server.",
                "privacy_boundary": pulled["privacy_boundary"],
            }
        if tools["llamacpp"]["available"]:
            raise ModelDownloadError(
                "llama.cpp is installed, but one-click model pulls need Ollama. "
                "Put a local .gguf path in Existing local path and click Register GGUF Model, "
                "then run llama-server with the command OpenAgent returns."
            )
        raise ModelDownloadError(
            "No supported local model runner was found on PATH. Install Ollama for one-click downloads, "
            "or install llama.cpp and register a local .gguf model path."
        )

    def create_setup_chat_job(
        self,
        *,
        model_name: str = "",
        allow_network: bool = False,
        project: str = "default",
        task: str = "chat",
        run_inline: bool = False,
    ) -> dict[str, Any]:
        task = (task or "chat").strip().lower()
        requested = ollama_pull_name(model_name or {
            "code": CPU_MODEL_STACK["code"],
            "reasoning": CPU_MODEL_STACK["research"],
            "research": CPU_MODEL_STACK["research"],
            "speed": CPU_MODEL_STACK["speed"],
        }.get(task, CPU_MODEL_STACK["chat"]))
        job_kind = f"setup-{task}"
        job_id = stable_id("model-job", job_kind, requested, project, utc_now())
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO model_setup_jobs (
                    id, kind, project, status, model_name, progress, current_step,
                    result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_kind,
                    project,
                    "queued",
                    requested,
                    1,
                    f"Queued local {task} model setup.",
                    json.dumps(
                        {
                            "allow_network": allow_network,
                            "task": task,
                            "privacy_boundary": (
                                "This job may download model files only when allow_network is true. "
                                "All job state is stored in the local OpenAgent data directory."
                            ),
                        },
                        sort_keys=True,
                    ),
                    now,
                    now,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="local_ai",
                action=f"model.setup_{task}.queued",
                resource=f"model_job:{job_id}",
                project=project,
                metadata={"model": requested, "allow_network": allow_network},
            )
        )
        if run_inline:
            self.run_setup_chat_job(job_id, model_name=requested, allow_network=allow_network, project=project, task=task)
        else:
            thread = threading.Thread(
                target=self.run_setup_chat_job,
                kwargs={"job_id": job_id, "model_name": requested, "allow_network": allow_network, "project": project, "task": task},
                daemon=True,
            )
            thread.start()
        return self.model_job(job_id) or {}

    def run_setup_chat_job(
        self,
        job_id: str,
        *,
        model_name: str,
        allow_network: bool,
        project: str = "default",
        task: str = "chat",
    ) -> None:
        task = (task or "chat").strip().lower()
        try:
            if self._job_cancel_requested(job_id):
                self._update_model_job(job_id, status="cancelled", progress=100, current_step="Cancelled before start.", completed=True)
                return
            self._update_model_job(job_id, status="running", progress=8, current_step="Checking local model runners.")
            tools = self.model_tool_status()
            self._update_model_job(job_id, progress=18, current_step="Runner check complete.", result_patch={"tool_status": tools})
            if self._job_cancel_requested(job_id):
                self._update_model_job(job_id, status="cancelled", progress=100, current_step="Cancelled before model setup.", completed=True)
                return
            self._update_model_job(job_id, progress=35, current_step=f"Preparing {model_name} for local {task}.")
            result = self.setup_task_model(task=task, model_name=model_name, allow_network=allow_network, project=project)
            model = result.get("model") if isinstance(result.get("model"), dict) else {}
            model_id = str(model.get("id", ""))
            self._update_model_job(
                job_id,
                progress=78,
                current_step="Model setup finished; running a local benchmark.",
                model_id=model_id,
                result_patch={"setup": result},
            )
            benchmark: dict[str, Any] | None = None
            if model_id and not self._job_cancel_requested(job_id):
                try:
                    benchmark = self.benchmark(model_id, prompt="Hello OpenAgent")
                except Exception as exc:  # benchmark should not make a successful pull look failed
                    benchmark = {"warning": str(exc)}
            status = "cancelled" if self._job_cancel_requested(job_id) else "completed"
            step = "Cancelled after setup." if status == "cancelled" else f"{task.title()} model is ready for Auto routing."
            self._update_model_job(
                job_id,
                status=status,
                progress=100,
                current_step=step,
                model_id=model_id,
                result_patch={"setup": result, "benchmark": benchmark, "tool_status": self.model_tool_status()},
                completed=True,
            )
            self.store.add_audit_event(
                AuditEvent(
                    actor="local_ai",
                    action=f"model.setup_{task}.{status}",
                    resource=f"model_job:{job_id}",
                    project=project,
                    metadata={"model": model_name, "model_id": model_id},
                )
            )
        except Exception as exc:
            self._update_model_job(
                job_id,
                status="failed",
                progress=100,
                current_step="Model setup failed.",
                error=str(exc),
                result_patch={"tool_status": self.model_tool_status()},
                completed=True,
            )
            self.store.add_audit_event(
                AuditEvent(
                    actor="local_ai",
                    action=f"model.setup_{task}.failed",
                    resource=f"model_job:{job_id}",
                    project=project,
                    status="failed",
                    metadata={"model": model_name, "error": str(exc)},
                )
            )

    def model_job(self, job_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM model_setup_jobs WHERE id = ?", (job_id,)).fetchone()
        return model_job_row(row) if row else None

    def model_jobs(
        self,
        *,
        project: str = "default",
        status: str = "",
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self.store.session() as connection:
            if status:
                rows = connection.execute(
                    """
                    SELECT * FROM model_setup_jobs
                    WHERE project = ? AND status = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (project, status, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM model_setup_jobs
                    WHERE project = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (project, limit),
                ).fetchall()
        return [model_job_row(row) for row in rows]

    def cancel_model_job(self, job_id: str) -> dict[str, Any]:
        job = self.model_job(job_id)
        if job is None:
            raise ValueError(f"Unknown model job '{job_id}'.")
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        self._update_model_job(
            job_id,
            cancel_requested=True,
            current_step="Cancellation requested; the current local engine call may finish first.",
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="local_ai",
                action="model.job.cancel_requested",
                resource=f"model_job:{job_id}",
                project=str(job.get("project", "default")),
                metadata={"kind": job.get("kind", "")},
            )
        )
        return self.model_job(job_id) or job

    def _job_cancel_requested(self, job_id: str) -> bool:
        job = self.model_job(job_id)
        return bool(job and job.get("cancel_requested"))

    def _update_model_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        current_step: str | None = None,
        model_id: str | None = None,
        result_patch: dict[str, Any] | None = None,
        error: str | None = None,
        cancel_requested: bool | None = None,
        completed: bool = False,
    ) -> None:
        job = self.model_job(job_id)
        if job is None:
            return
        result = dict(job.get("result", {}))
        if result_patch:
            result.update(result_patch)
        values = {
            "status": status if status is not None else job["status"],
            "progress": max(0, min(100, int(progress if progress is not None else job["progress"]))),
            "current_step": current_step if current_step is not None else job["current_step"],
            "model_id": model_id if model_id is not None else job["model_id"],
            "result_json": json.dumps(result, sort_keys=True),
            "error": error if error is not None else job["error"],
            "cancel_requested": int(cancel_requested if cancel_requested is not None else job["cancel_requested"]),
            "updated_at": utc_now(),
            "completed_at": utc_now() if completed else job["completed_at"],
        }
        with self.store.session() as connection:
            connection.execute(
                """
                UPDATE model_setup_jobs
                SET status = ?, progress = ?, current_step = ?, model_id = ?,
                    result_json = ?, error = ?, cancel_requested = ?, updated_at = ?,
                    completed_at = ?
                WHERE id = ?
                """,
                (
                    values["status"],
                    values["progress"],
                    values["current_step"],
                    values["model_id"],
                    values["result_json"],
                    values["error"],
                    values["cancel_requested"],
                    values["updated_at"],
                    values["completed_at"],
                    job_id,
                ),
            )

    def register_llamacpp_model(
        self,
        *,
        local_path: str,
        name: str = "",
        project: str = "default",
    ) -> dict[str, Any]:
        raw_path = str(local_path or "").strip()
        if not raw_path:
            raise ValueError("local_path is required for llama.cpp GGUF registration.")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"GGUF model file was not found: {path}")
        if path.suffix.lower() != ".gguf":
            raise ValueError("llama.cpp registration expects a .gguf model file.")
        model_name = name.strip() or path.stem
        server_url = os.environ.get("OPENAGENT_LLAMACPP_URL", "http://127.0.0.1:8080")
        start_command = f'llama-server -m "{path}" --host 127.0.0.1 --port 8080'
        model = self.register_model(
            name=model_name,
            provider="llamacpp",
            source_url=str(path),
            local_path=str(path),
            metadata={
                "runner": "llama.cpp",
                "format": "gguf",
                "server_url": server_url,
                "start_command": start_command,
                "registered_for": "openagent-chat",
            },
        )
        os.environ["OPENAGENT_LLAMACPP_MODEL"] = model_name
        self.store.add_audit_event(
            AuditEvent(
                actor="local_ai",
                action="model.llamacpp_register",
                resource=f"model:{model['id']}",
                project=project,
                metadata={"local_path": str(path), "server_url": server_url},
            )
        )
        return {
            "model": model,
            "runner": "llamacpp",
            "server_url": server_url,
            "start_command": start_command,
            "next_step": "Start that command in a terminal, then keep Chat provider on Auto or llama.cpp.",
            "privacy_boundary": "The GGUF model stays at the provided local path. OpenAgent stores only a local model record and never uploads the file.",
        }

    def add_quantization(
        self,
        model_id: str,
        *,
        method: str,
        bits: int,
        format: str = "gguf",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.get_model(model_id) is None:
            raise ValueError(f"Unknown model '{model_id}'.")
        quant_id = stable_id("quant", model_id, method, str(bits), format)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO model_quantizations (
                    id, model_id, method, bits, format, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (quant_id, model_id, method, bits, format, json.dumps(metadata or {}, sort_keys=True), created_at),
            )
        return {
            "id": quant_id,
            "model_id": model_id,
            "method": method,
            "bits": bits,
            "format": format,
            "metadata": metadata or {},
            "created_at": created_at,
        }

    def quantizations(self, model_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM model_quantizations WHERE model_id = ? ORDER BY bits",
                (model_id,),
            ).fetchall()
        return [quant_row(row) for row in rows]

    def auto_quantize(
        self,
        model_id: str,
        *,
        task: str = "chat",
        target_ram_gb: float | None = None,
        quantization_mode: str = "auto",
        parameter_count_b: float | None = None,
    ) -> dict[str, Any]:
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Unknown model '{model_id}'.")
        hardware = self.detect_hardware()
        if target_ram_gb is not None:
            hardware = {**hardware, "ram_gb": target_ram_gb}
        if quantization_mode:
            hardware = {**hardware, "quantization_mode": quantization_mode}
        profile = smooth_runtime_profile(task=task, hardware=hardware)
        adaptive_plan = adaptive_quantization_plan(
            task=task,
            target_ram_gb=float(hardware.get("ram_gb", profile.ram_gb)),
            quantization_mode=quantization_mode,
            parameter_count_b=parameter_count_b,
        )
        quantization = self.add_quantization(
            model_id,
            method=profile.quantization_method,
            bits=profile.quantization_bits,
            format="gguf",
            metadata={
                "task": task,
                "status": "planned",
                "smooth_profile": profile.to_dict(),
                "adaptive_plan": adaptive_plan,
                "effective_bits": profile.effective_bits,
                "source_model_size_bytes": model.get("size_bytes", 0),
                "note": (
                    "OpenAgent recorded a safe quantization plan. Actual model conversion should be performed "
                    "by the configured local engine or converter before marking a quantized artifact available."
                ),
            },
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="local_ai",
                action="model.quantization.auto_plan",
                resource=f"model:{model_id}",
                project="default",
                metadata={
                    "task": task,
                    "bits": profile.quantization_bits,
                    "effective_bits": profile.effective_bits,
                    "tier": profile.tier,
                    "mode": quantization_mode,
                },
            )
        )
        return {
            "quantization": quantization,
            "profile": profile.to_dict(),
            "adaptive_plan": adaptive_plan,
            "recommended_model": recommended_model_for_task(profile, task),
            "privacy_boundary": "Auto-quantization planning uses local hardware data only and does not download or upload models.",
        }

    def delete_model(self, model_id: str, *, delete_file: bool = False) -> dict[str, Any]:
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Unknown model '{model_id}'.")
        dependents = self.model_dependents(model_id)
        if dependents:
            raise ValueError(f"Model is still required by {len(dependents)} dependent resource(s).")
        deleted_file = ""
        if delete_file and model.get("local_path"):
            candidate = Path(str(model["local_path"])).resolve()
            model_root = self.model_root.resolve()
            try:
                candidate.relative_to(model_root)
            except ValueError:
                pass
            else:
                if candidate.exists() and candidate.is_file():
                    candidate.unlink()
                    deleted_file = str(candidate)
        with self.store.session() as connection:
            connection.execute("DELETE FROM model_quantizations WHERE model_id = ?", (model_id,))
            connection.execute("DELETE FROM model_benchmarks WHERE model_id = ?", (model_id,))
            connection.execute("DELETE FROM model_tool_assignments WHERE model_id = ?", (model_id,))
            connection.execute("DELETE FROM local_models WHERE id = ?", (model_id,))
        self.store.add_audit_event(
            AuditEvent(
                actor="local_ai",
                action="model.delete",
                resource=f"model:{model_id}",
                project="default",
                metadata={"deleted_file": bool(deleted_file), "name": model.get("name", "")},
            )
        )
        return {"deleted": True, "model": model, "deleted_file": deleted_file}

    def detect_hardware(self) -> dict[str, Any]:
        profile = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count() or 1,
            "ram_gb": detect_total_memory_gb(),
            "gpu": detect_gpu_hint(),
        }
        profile_id = stable_id("hardware", json.dumps(profile, sort_keys=True))
        with self.store.session() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO hardware_profiles (id, profile_json, created_at) VALUES (?, ?, ?)",
                (profile_id, json.dumps(profile, sort_keys=True), utc_now()),
            )
        return {"id": profile_id, **profile}

    def adapters(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM acceleration_adapters ORDER BY id").fetchall()
        return [adapter_row(row) for row in rows]

    def benchmark(self, model_id: str, *, prompt: str = "Hello OpenAgent") -> dict[str, Any]:
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Unknown model '{model_id}'.")
        start = time.perf_counter()
        provider_id = str(model.get("provider") or "local")
        request_model = str(model.get("name") or "") or None
        result = self.gateway.complete(
            [{"role": "user", "content": prompt}],
            provider_id=provider_id if provider_id in self.gateway.providers else "local",
            model=request_model,
        )
        duration_ms = max((time.perf_counter() - start) * 1000, 0.001)
        prompt_tokens = len(prompt.split())
        output_tokens = len(result.content.split())
        tps = output_tokens / (duration_ms / 1000)
        hardware = self.detect_hardware()
        benchmark_id = stable_id("benchmark", model_id, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO model_benchmarks (
                    id, model_id, prompt_tokens, output_tokens, duration_ms,
                    tokens_per_second, hardware_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    benchmark_id,
                    model_id,
                    prompt_tokens,
                    output_tokens,
                    duration_ms,
                    tps,
                    json.dumps(hardware, sort_keys=True),
                    created_at,
                ),
            )
        return {
            "id": benchmark_id,
            "model_id": model_id,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "duration_ms": round(duration_ms, 3),
            "tokens_per_second": round(tps, 3),
            "provider": result.provider,
            "model": result.model,
            "metadata": result.metadata,
            "hardware": hardware,
            "created_at": created_at,
        }

    def plan_cluster(self, *, name: str, model_id: str, nodes: int = 1) -> dict[str, Any]:
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Unknown model '{model_id}'.")
        hardware = self.detect_hardware()
        plan = {
            "model_id": model_id,
            "nodes": max(1, nodes),
            "recommended_shards": max(1, min(nodes, hardware["cpu_count"])),
            "transport": "self-hosted-private-network",
            "notes": "Keep model files and prompts inside trusted infrastructure.",
        }
        cluster_id = stable_id("cluster", name, model_id, str(nodes))
        with self.store.session() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO inference_clusters (id, name, plan_json, created_at) VALUES (?, ?, ?, ?)",
                (cluster_id, name, json.dumps(plan, sort_keys=True), utc_now()),
            )
        return {"id": cluster_id, "name": name, "plan": plan}

    def marketplace(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM model_marketplace ORDER BY name").fetchall()
        return [marketplace_row(row) for row in rows]

    def smooth_runtime_plan(
        self,
        *,
        task: str = "chat",
        target_ram_gb: float | None = None,
        quantization_mode: str = "auto",
    ) -> dict[str, Any]:
        hardware = self.detect_hardware()
        if target_ram_gb is not None:
            hardware = {**hardware, "ram_gb": target_ram_gb}
        if quantization_mode:
            hardware = {**hardware, "quantization_mode": quantization_mode}
        profile = smooth_runtime_profile(task=task, hardware=hardware)
        return {
            "task": task,
            "profile": profile.to_dict(),
            "hardware": hardware,
            "model_profiles": model_profiles(),
            "recommended_model": recommended_model_for_task(profile, task),
            "adaptive_quantization": adaptive_quantization_plan(
                task=task,
                target_ram_gb=float(hardware.get("ram_gb", profile.ram_gb)),
                quantization_mode=str(hardware.get("quantization_mode", "auto")),
            ),
            "engine_policy": {
                "chat": "Auto routes to Ollama, then llama.cpp, then deterministic local fallback; local servers receive bounded context/output limits.",
                "research": "Research uses compact iterations and staged evidence windows.",
                "voice": "Voice uses faster-whisper int8 CPU model when installed, Windows Speech only as explicit fallback.",
            },
        }

    def transition_model_lifecycle(
        self,
        model_id: str,
        new_state: str,
        *,
        reason: str = "",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Move a model through the explicit lifecycle with a failure-reason taxonomy.

        Callers that want the discovered/downloading/ready/serving/degraded/failed
        vocabulary (and a diagnostics trail) opt in by calling this explicitly; the
        existing register_model/download_model status values are untouched.
        """
        if new_state not in MODEL_LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle state '{new_state}'; expected one of {MODEL_LIFECYCLE_STATES}")
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM local_models WHERE id = ?", (model_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown model '{model_id}'.")
            current_state = row["status"]
            allowed = _MODEL_LIFECYCLE_TRANSITIONS.get(current_state)
            if allowed is not None and new_state not in allowed and new_state != current_state:
                raise ValueError(f"illegal transition from '{current_state}' to '{new_state}'")
            normalized_reason = reason.strip()
            if new_state == "failed" and normalized_reason not in MODEL_FAILURE_REASONS:
                normalized_reason = "unknown"
            now = utc_now()
            connection.execute(
                "UPDATE local_models SET status = ?, updated_at = ? WHERE id = ?",
                (new_state, now, model_id),
            )
            connection.execute(
                """
                INSERT INTO model_diagnostics (id, model_id, event_type, state, reason, detail_json, created_at)
                VALUES (?, ?, 'lifecycle_transition', ?, ?, ?, ?)
                """,
                (uuid4().hex, model_id, new_state, normalized_reason, json.dumps(detail or {}, sort_keys=True), now),
            )
        return {"model_id": model_id, "state": new_state, "reason": normalized_reason}

    def get_model_diagnostics(self, model_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            model = connection.execute("SELECT id FROM local_models WHERE id = ?", (model_id,)).fetchone()
            if model is None:
                raise KeyError(f"Unknown model '{model_id}'.")
            rows = connection.execute(
                "SELECT * FROM model_diagnostics WHERE model_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (model_id, max(1, min(limit, 200))),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "state": r["state"],
                "reason": r["reason"],
                "detail": json.loads(r["detail_json"] or "{}"),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def hardware_fit_for_model(self, model_id: str) -> dict[str, Any]:
        """Combine detected hardware with a marketplace/registered model's RAM requirement."""
        hardware = self.detect_hardware()
        detected_gb = float(hardware.get("ram_gb", 0) or 0)
        with self.store.session() as connection:
            market_row = connection.execute(
                "SELECT metadata_json FROM model_marketplace WHERE id = ?", (model_id,)
            ).fetchone()
            local_row = connection.execute(
                "SELECT metadata_json FROM local_models WHERE id = ?", (model_id,)
            ).fetchone()
        if market_row is not None:
            metadata = json.loads(market_row["metadata_json"] or "{}")
        elif local_row is not None:
            metadata = json.loads(local_row["metadata_json"] or "{}")
        else:
            raise KeyError(f"Unknown model '{model_id}'.")
        required_gb = float(metadata.get("ram_gb", 0) or 0)
        if required_gb <= 0:
            fit = "unknown"
        elif detected_gb >= required_gb * 1.5:
            fit = "comfortable"
        elif detected_gb >= required_gb:
            fit = "marginal"
        else:
            fit = "insufficient"
        return {
            "model_id": model_id,
            "required_ram_gb": required_gb,
            "detected_ram_gb": detected_gb,
            "fit": fit,
        }

    def status(self) -> dict[str, Any]:
        return {
            "models": len(self.models()),
            "marketplace": len(self.marketplace()),
            "hardware": self.detect_hardware(),
            "adapters": self.adapters(),
            "smooth_runtime": self.smooth_runtime_plan(),
            "model_root": str(self.model_root),
        }


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def ollama_pull_name(model_name: str) -> str:
    value = str(model_name or "").strip()
    aliases = {
        "phi4-mini:3.8b-instruct-q4_k_m": "phi4-mini",
        "phi4-mini:3.8b-instruct-q4_K_M": "phi4-mini",
        "qwen2.5-coder:3b-instruct-q4_k_m": "qwen2.5-coder:3b",
        "qwen2.5-coder:3b-instruct-q4_K_M": "qwen2.5-coder:3b",
        "gemma3:2b-it-q4_k_m": "gemma3:2b",
        "gemma3:2b-it-q4_K_M": "gemma3:2b",
        "deepseek-coder:1.3b-instruct-q4_k_m": "deepseek-coder:1.3b",
        "deepseek-coder:1.3b-instruct-q4_K_M": "deepseek-coder:1.3b",
        "llama3.2:3b-instruct-q4_k_m": "llama3.2:3b",
        "llama3.2:3b-instruct-q4_K_M": "llama3.2:3b",
        "qwen2.5:7b-instruct-q4_k_m": "qwen2.5:7b",
        "qwen2.5:7b-instruct-q4_K_M": "qwen2.5:7b",
    }
    return aliases.get(value, aliases.get(value.lower(), value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_gpu_hint() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and "arm" in machine:
        return {"vendor": "apple", "available": True, "adapter": "metal"}
    return {"vendor": "unknown", "available": False, "adapter": "cpu"}


def recommended_model_for_task(profile: Any, task: str) -> str:
    if task == "agent":
        return profile.recommended_agent_model
    if task == "code":
        return profile.recommended_code_model
    if task == "research":
        return profile.recommended_research_model
    if task == "speed":
        return profile.recommended_speed_model
    if task == "extraction":
        return profile.recommended_extraction_model
    if task in {"high-research", "high_context_research"}:
        return profile.recommended_high_context_research_model
    if task == "voice":
        return profile.recommended_voice_engine
    return profile.recommended_chat_model


def model_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "provider": row["provider"],
        "source_url": row["source_url"],
        "local_path": row["local_path"],
        "status": row["status"],
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def model_job_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "project": row["project"],
        "status": row["status"],
        "model_name": row["model_name"],
        "model_id": row["model_id"],
        "progress": row["progress"],
        "current_step": row["current_step"],
        "cancel_requested": bool(row["cancel_requested"]),
        "result": json.loads(row["result_json"]),
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
    }


def quant_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "model_id": row["model_id"],
        "method": row["method"],
        "bits": row["bits"],
        "format": row["format"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }


def adapter_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "label": row["label"],
        "hardware": row["hardware"],
        "available": bool(row["available"]),
        "metadata": json.loads(row["metadata_json"]),
    }


def marketplace_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "provider": row["provider"],
        "recommended_engine": row["recommended_engine"],
        "metadata": json.loads(row["metadata_json"]),
    }
