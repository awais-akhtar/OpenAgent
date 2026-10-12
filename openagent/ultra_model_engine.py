"""Ultra-scale local model planning, lifecycle control, and output verification.

The engine coordinates llama.cpp rather than reimplementing tensor kernels. Its
job is to make resource limits explicit, choose a safe execution tier, keep a
local server warm, and refuse to describe a merely launchable model as fast.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any
from uuid import uuid4

from .egress import EgressClient, EgressError
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore
from .system_metrics import system_metrics


ENGINE_ID = "openagent-tiered-inference-v1"
SHARD_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)
CLAIM_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
CITATION_PATTERN = re.compile(r"\[S(\d+)\]", re.IGNORECASE)
NEGATIONS = {"no", "not", "never", "without", "cannot", "can't", "isn't", "doesn't", "won't"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "will", "with", "you", "your",
}
ALLOWED_KV_TYPES = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}


class UltraModelError(RuntimeError):
    """Raised when an ultra-model operation cannot be performed safely."""


class LaunchApprovalRequired(UltraModelError):
    """Raised when process or distributed execution lacks explicit approval."""


@dataclass(frozen=True)
class HardwareBudget:
    total_ram_gb: float
    available_ram_gb: float
    memory_bandwidth_gbps: float
    disk_bandwidth_gbps: float
    network_bandwidth_gbps: float
    cpu_threads: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ram_gb": round(self.total_ram_gb, 3),
            "available_ram_gb": round(self.available_ram_gb, 3),
            "memory_bandwidth_gbps": round(self.memory_bandwidth_gbps, 3),
            "disk_bandwidth_gbps": round(self.disk_bandwidth_gbps, 3),
            "network_bandwidth_gbps": round(self.network_bandwidth_gbps, 3),
            "cpu_threads": self.cpu_threads,
        }


class UltraModelEngine:
    """Plan and supervise large sharded GGUF models on trusted infrastructure."""

    def __init__(self, store: OpenAgentStore, *, llama_server: str | None = None) -> None:
        self.store = store
        self.runtime_root = store.data_dir / "ultra-model-runtime"
        self.log_root = self.runtime_root / "logs"
        self.cache_root = self.runtime_root / "prompt-cache"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.llama_server = llama_server or os.environ.get("OPENAGENT_LLAMA_SERVER", "")
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._log_handles: dict[str, Any] = {}
        self._keepalive_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ultra_model_manifests (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    name TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    parameter_count_b REAL NOT NULL,
                    active_parameter_count_b REAL NOT NULL,
                    quantization_bits REAL NOT NULL,
                    shard_paths_json TEXT NOT NULL,
                    draft_model_path TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ultra_manifests_project
                    ON ultra_model_manifests(project, updated_at DESC);

                CREATE TABLE IF NOT EXISTS ultra_model_plans (
                    id TEXT PRIMARY KEY,
                    manifest_id TEXT NOT NULL DEFAULT '',
                    project TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ultra_plans_project
                    ON ultra_model_plans(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS ultra_model_sessions (
                    id TEXT PRIMARY KEY,
                    manifest_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pid INTEGER NOT NULL DEFAULT 0,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    command_json TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    stopped_at TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_ultra_sessions_project
                    ON ultra_model_sessions(project, started_at DESC);

                CREATE TABLE IF NOT EXISTS ultra_model_verifications (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    status TEXT NOT NULL,
                    support_score REAL NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ultra_model_calibrations (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    calibration_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            session_ids = list(self._processes)
        for session_id in session_ids:
            try:
                self.stop(session_id, reason="OpenAgent runtime shutdown")
            except UltraModelError:
                pass

    # ------------------------------------------------------------------
    # Manifests and shard validation
    # ------------------------------------------------------------------
    def register_manifest(
        self,
        *,
        name: str,
        shard_paths: list[str],
        parameter_count_b: float,
        quantization_bits: float,
        architecture: str = "dense",
        active_parameter_count_b: float | None = None,
        draft_model_path: str = "",
        project: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = " ".join(name.split()).strip()
        project = project.strip() or "default"
        if not name:
            raise ValueError("name is required.")
        architecture = architecture.strip().lower()
        if architecture not in {"dense", "moe"}:
            raise ValueError("architecture must be dense or moe.")
        parameter_count_b = _bounded_float(parameter_count_b, "parameter_count_b", 0.01, 100000.0)
        quantization_bits = _bounded_float(quantization_bits, "quantization_bits", 1.0, 32.0)
        active = parameter_count_b if architecture == "dense" else float(active_parameter_count_b or 0)
        active = _bounded_float(active, "active_parameter_count_b", 0.01, parameter_count_b)
        paths = discover_gguf_shards(shard_paths)
        draft_path = ""
        if draft_model_path.strip():
            draft = Path(draft_model_path).expanduser().resolve()
            if not draft.is_file() or draft.suffix.lower() != ".gguf":
                raise ValueError("draft_model_path must be an existing local GGUF file.")
            draft_path = str(draft)
        manifest_id = hashlib.sha256(
            json.dumps(
                {
                    "project": project,
                    "name": name,
                    "shards": paths,
                    "params": parameter_count_b,
                    "bits": quantization_bits,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:24]
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO ultra_model_manifests (
                    id, project, name, architecture, parameter_count_b,
                    active_parameter_count_b, quantization_bits, shard_paths_json,
                    draft_model_path, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    architecture = excluded.architecture,
                    active_parameter_count_b = excluded.active_parameter_count_b,
                    quantization_bits = excluded.quantization_bits,
                    shard_paths_json = excluded.shard_paths_json,
                    draft_model_path = excluded.draft_model_path,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    manifest_id,
                    project,
                    name,
                    architecture,
                    parameter_count_b,
                    active,
                    quantization_bits,
                    json.dumps(paths),
                    draft_path,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                    now,
                ),
            )
        self._audit(
            "ultra_model.manifest.register",
            f"ultra-model:{manifest_id}",
            project,
            {"shards": len(paths), "parameter_count_b": parameter_count_b, "architecture": architecture},
        )
        return self.get_manifest(manifest_id) or {}

    def get_manifest(self, manifest_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT * FROM ultra_model_manifests WHERE id = ?", (manifest_id,)
            ).fetchone()
        return _manifest_row(row) if row else None

    def manifests(self, *, project: str = "default", limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM ultra_model_manifests WHERE project = ? ORDER BY updated_at DESC LIMIT ?",
                (project.strip() or "default", limit),
            ).fetchall()
        return [_manifest_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Constraint-aware execution planning
    # ------------------------------------------------------------------
    def plan(
        self,
        *,
        manifest_id: str = "",
        specification: dict[str, Any] | None = None,
        project: str = "default",
        context_tokens: int = 4096,
        expected_output_tokens: int = 256,
        hardware: dict[str, Any] | None = None,
        trusted_nodes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        project = project.strip() or "default"
        manifest = self.get_manifest(manifest_id) if manifest_id else None
        if manifest_id and manifest is None:
            raise ValueError(f"Unknown ultra-model manifest '{manifest_id}'.")
        spec = dict(specification or {})
        if manifest:
            spec = {
                "name": manifest["name"],
                "architecture": manifest["architecture"],
                "parameter_count_b": manifest["parameter_count_b"],
                "active_parameter_count_b": manifest["active_parameter_count_b"],
                "quantization_bits": manifest["quantization_bits"],
                "shard_paths": manifest["shard_paths"],
                "draft_model_path": manifest["draft_model_path"],
                "metadata": manifest["metadata"],
            }
            project = manifest["project"]
        total_params = _bounded_float(spec.get("parameter_count_b", 0), "parameter_count_b", 0.01, 100000.0)
        bits = _bounded_float(spec.get("quantization_bits", 4), "quantization_bits", 1.0, 32.0)
        architecture = str(spec.get("architecture", "dense")).strip().lower()
        if architecture not in {"dense", "moe"}:
            raise ValueError("architecture must be dense or moe.")
        active_params = total_params if architecture == "dense" else _bounded_float(
            spec.get("active_parameter_count_b", 0), "active_parameter_count_b", 0.01, total_params
        )
        context_tokens = max(256, min(int(context_tokens), 1_048_576))
        expected_output_tokens = max(1, min(int(expected_output_tokens), 32768))
        shard_paths = [str(item) for item in spec.get("shard_paths", []) if str(item).strip()]
        actual_bytes = sum(Path(path).stat().st_size for path in shard_paths if Path(path).is_file())
        theoretical_bytes = int(total_params * 1_000_000_000 * bits / 8 * 1.06)
        # Guard against incorrect metadata that would make a tiny placeholder file
        # look like a real 200B model. Real GGUF size remains the primary evidence.
        effective_weight_bytes = max(actual_bytes, int(theoretical_bytes * 0.65))
        active_weight_bytes = int(effective_weight_bytes * (active_params / total_params))
        budget = self._hardware_budget(hardware)
        kv_type = str(spec.get("metadata", {}).get("kv_type", "q8_0"))
        if kv_type not in ALLOWED_KV_TYPES:
            kv_type = "q8_0"
        kv_bytes = estimate_kv_cache_bytes(active_params, context_tokens, kv_type)
        runtime_bytes = int(min(4 * 1024**3, max(768 * 1024**2, active_weight_bytes * 0.012)))
        reserve_gb = max(1.5, budget.total_ram_gb * 0.15)
        usable_ram_gb = max(0.0, min(budget.available_ram_gb * 0.85, budget.total_ram_gb - reserve_gb))
        usable_bytes = int(usable_ram_gb * 1024**3)
        resident_requirement = effective_weight_bytes + kv_bytes + runtime_bytes
        local_resident_capacity = max(0, usable_bytes - kv_bytes - runtime_bytes)
        node_report = validate_trusted_nodes(trusted_nodes or [])
        remote_ram_bytes = int(sum(node["ram_gb"] for node in node_report) * 1024**3)
        files_ready = bool(shard_paths) and all(Path(path).is_file() for path in shard_paths)

        if resident_requirement <= usable_bytes:
            mode = "resident-mmap"
        elif node_report and local_resident_capacity + remote_ram_bytes >= effective_weight_bytes:
            mode = "trusted-rpc-sharded"
        elif files_ready:
            mode = "out-of-core-mmap"
        else:
            mode = "planning-only"

        resident_active = min(active_weight_bytes, local_resident_capacity)
        remote_active = 0
        if mode == "trusted-rpc-sharded":
            remote_active = min(active_weight_bytes - resident_active, remote_ram_bytes)
        disk_active = max(0, active_weight_bytes - resident_active - remote_active)
        seconds_per_token = 0.0
        if resident_active:
            seconds_per_token += resident_active / max(1.0, budget.memory_bandwidth_gbps * 1_000_000_000)
        if remote_active:
            remote_limit = min(budget.network_bandwidth_gbps, budget.memory_bandwidth_gbps)
            seconds_per_token += remote_active / max(1.0, remote_limit * 1_000_000_000)
        if disk_active:
            seconds_per_token += disk_active / max(1.0, budget.disk_bandwidth_gbps * 1_000_000_000)
        efficiency = 0.58 if mode == "resident-mmap" else 0.38 if mode == "trusted-rpc-sharded" else 0.32
        base_tps = efficiency / max(0.000001, seconds_per_token)
        draft_path = str(spec.get("draft_model_path", "")).strip()
        speculation = "draft-simple" if draft_path and Path(draft_path).is_file() else "ngram-mod"
        speculative_factor = 1.35 if speculation == "draft-simple" else 1.12
        predicted_tps = max(0.001, base_tps * speculative_factor)
        warm_first_token_ms = max(40.0, 1000.0 / predicted_tps)
        load_bandwidth = budget.memory_bandwidth_gbps if mode == "resident-mmap" else budget.disk_bandwidth_gbps
        cold_load_seconds = effective_weight_bytes / max(1.0, load_bandwidth * 1_000_000_000)
        interactive = predicted_tps >= 2.0 and warm_first_token_ms <= 1500
        launchable = files_ready and mode != "planning-only"
        quality_risk = "high" if bits < 2.0 else "elevated" if bits < 3.0 else "normal"
        bottleneck = "memory-bandwidth"
        if disk_active:
            bottleneck = "storage-bandwidth"
        elif remote_active:
            bottleneck = "trusted-network-bandwidth"
        warnings: list[str] = []
        if mode == "planning-only":
            warnings.append("Register every local GGUF shard before launch.")
        if mode == "out-of-core-mmap":
            warnings.append("The model can launch through mmap, but non-resident pages are read repeatedly and interactive speed is unlikely.")
        if total_params >= 200 and predicted_tps < 1:
            warnings.append("This 200B-class configuration is physically bandwidth-bound and is not predicted to be smooth on the supplied CPU system.")
        if quality_risk != "normal":
            warnings.append("Extreme quantization can reduce answer quality; grounded verification is mandatory.")
        if architecture == "moe":
            warnings.append("MoE throughput uses active parameters, while all experts still require storage and loading capacity.")

        runtime = {
            "threads": max(1, min(budget.cpu_threads, 16 if total_params < 70 else budget.cpu_threads)),
            "threads_batch": max(1, budget.cpu_threads),
            "context_tokens": context_tokens,
            "expected_output_tokens": expected_output_tokens,
            "batch_size": 128 if usable_ram_gb < 16 else 256,
            "ubatch_size": 64 if usable_ram_gb < 16 else 128,
            "parallel_slots": 1 if mode != "resident-mmap" else 2,
            "kv_type": kv_type,
            "mmap": True,
            "mlock": mode == "resident-mmap" and resident_requirement < usable_bytes * 0.85,
            "prompt_cache": True,
            "cache_reuse_tokens": 64,
            "warmup": True,
            "keep_warm": True,
            "speculation": speculation,
        }
        plan_id = uuid4().hex
        result = {
            "id": plan_id,
            "engine": ENGINE_ID,
            "project": project,
            "manifest_id": manifest_id,
            "name": str(spec.get("name", "Unregistered model")),
            "mode": mode,
            "launchable": launchable,
            "interactive_predicted": interactive,
            "architecture": architecture,
            "parameter_count_b": total_params,
            "active_parameter_count_b": active_params,
            "quantization_bits": bits,
            "shards": {"count": len(shard_paths), "files_ready": files_ready, "actual_bytes": actual_bytes},
            "memory": {
                "effective_weight_gb": _gib(effective_weight_bytes),
                "active_weight_gb_per_token": _gib(active_weight_bytes),
                "kv_cache_gb": _gib(kv_bytes),
                "runtime_buffer_gb": _gib(runtime_bytes),
                "resident_requirement_gb": _gib(resident_requirement),
                "usable_local_ram_gb": round(usable_ram_gb, 3),
                "resident_active_gb": _gib(resident_active),
                "remote_active_gb": _gib(remote_active),
                "disk_streamed_active_gb": _gib(disk_active),
            },
            "performance": {
                "bottleneck": bottleneck,
                "predicted_tokens_per_second": round(predicted_tps, 4),
                "predicted_warm_first_token_ms": round(warm_first_token_ms, 1),
                "predicted_cold_load_seconds_lower_bound": round(cold_load_seconds, 2),
                "speculative_factor": speculative_factor,
                "estimation": "bandwidth lower-bound model; benchmark the real GGUF before relying on it",
            },
            "hardware": budget.to_dict(),
            "runtime": runtime,
            "trusted_nodes": node_report,
            "quality_gate": {
                "quantization_risk": quality_risk,
                "verification_required": quality_risk != "normal",
                "guarantee": "No inference engine can guarantee factual correctness. Grounded outputs can be checked and unsupported claims can be rejected.",
            },
            "request_cascade": {
                "enabled": True,
                "policy": "short deterministic turns use the existing small local model; complex or high-assurance turns escalate to this target model",
                "quality_preservation": "The cascade changes which model handles a request, never truncates a completed target-model answer.",
            },
            "warnings": warnings,
            "privacy_boundary": "Model shards, prompts, caches, process logs, and verification reports remain local. RPC is accepted only for explicitly trusted private endpoints.",
            "created_at": utc_now(),
        }
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO ultra_model_plans (id, manifest_id, project, mode, plan_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (plan_id, manifest_id, project, mode, json.dumps(result, sort_keys=True), result["created_at"]),
            )
        self._audit(
            "ultra_model.plan",
            f"ultra-plan:{plan_id}",
            project,
            {"mode": mode, "predicted_tps": result["performance"]["predicted_tokens_per_second"], "launchable": launchable},
        )
        return result

    def plans(self, *, project: str = "default", limit: int = 20) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT plan_json FROM ultra_model_plans WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project.strip() or "default", max(1, min(int(limit), 100))),
            ).fetchall()
        return [json.loads(row["plan_json"]) for row in rows]

    def route_request(self, prompt: str, *, high_assurance: bool = False) -> dict[str, Any]:
        clean = " ".join(prompt.split()).strip()
        if not clean:
            raise ValueError("prompt is required.")
        words = TOKEN_PATTERN.findall(clean.lower())
        complex_markers = {"analyze", "architecture", "compare", "debug", "derive", "evidence", "implement", "prove", "research", "security", "verify"}
        score = min(1.0, len(words) / 180 + sum(1 for word in words if word in complex_markers) * 0.12)
        if any(char in clean for char in "{}[]`\n"):
            score = min(1.0, score + 0.18)
        target = high_assurance or score >= 0.28
        return {
            "route": "target-model" if target else "fast-local-model",
            "complexity_score": round(score, 4),
            "high_assurance": high_assurance,
            "reason": "Complex or high-assurance request" if target else "Short low-complexity request",
            "quality_policy": "Escalate uncertainty to the target model; never substitute cached text for a non-identical prompt.",
        }

    # ------------------------------------------------------------------
    # Safe llama.cpp process lifecycle
    # ------------------------------------------------------------------
    def build_command(
        self,
        manifest_id: str,
        *,
        plan: dict[str, Any] | None = None,
        executable: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8081,
        allow_distributed: bool = False,
    ) -> list[str]:
        manifest = self.get_manifest(manifest_id)
        if manifest is None:
            raise ValueError(f"Unknown ultra-model manifest '{manifest_id}'.")
        if not _is_loopback_host(host):
            raise UltraModelError("Ultra-model servers bind to loopback only.")
        port = max(1024, min(int(port), 65535))
        selected_plan = plan or self.plan(manifest_id=manifest_id, project=manifest["project"])
        if not selected_plan.get("launchable"):
            raise UltraModelError("The selected plan is not launchable; register all GGUF shards first.")
        binary = resolve_llama_server(executable or self.llama_server)
        runtime = selected_plan["runtime"]
        command = [
            binary,
            "--model", manifest["shard_paths"][0],
            "--host", host,
            "--port", str(port),
            "--threads", str(runtime["threads"]),
            "--threads-batch", str(runtime["threads_batch"]),
            "--ctx-size", str(runtime["context_tokens"]),
            "--batch-size", str(runtime["batch_size"]),
            "--ubatch-size", str(runtime["ubatch_size"]),
            "--parallel", str(runtime["parallel_slots"]),
            "--cache-type-k", str(runtime["kv_type"]),
            "--cache-type-v", str(runtime["kv_type"]),
            "--mmap",
            "--warmup",
            "--cont-batching",
            "--cache-prompt",
            "--cache-reuse", str(runtime["cache_reuse_tokens"]),
            "--slot-save-path", str(self.cache_root),
            "--metrics",
            "--offline",
            "--alias", manifest["id"],
        ]
        if runtime.get("mlock"):
            command.append("--mlock")
        draft = manifest.get("draft_model_path", "")
        if draft:
            command.extend(["--spec-type", "draft-simple", "--spec-draft-model", draft, "--spec-draft-n-max", "8"])
        else:
            command.extend(
                [
                    "--spec-type", "ngram-mod",
                    "--spec-ngram-mod-n-match", "24",
                    "--spec-ngram-mod-n-min", "32",
                    "--spec-ngram-mod-n-max", "64",
                ]
            )
        nodes = selected_plan.get("trusted_nodes", [])
        if nodes:
            if not allow_distributed:
                raise LaunchApprovalRequired("Distributed RPC requires allow_distributed=true for this launch.")
            if os.environ.get("OPENAGENT_ENABLE_LLAMA_RPC", "").strip().lower() not in {"1", "true", "yes", "on"}:
                raise LaunchApprovalRequired("Set OPENAGENT_ENABLE_LLAMA_RPC=1 after securing the private RPC network.")
            command.extend(["--rpc", ",".join(node["endpoint"] for node in nodes)])
        return command

    def launch(
        self,
        manifest_id: str,
        *,
        approved: bool,
        project: str = "default",
        executable: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8081,
        allow_distributed: bool = False,
        keep_warm: bool = True,
        monitor: bool = True,
    ) -> dict[str, Any]:
        if not approved:
            raise LaunchApprovalRequired("Starting a local model process requires approved=true.")
        manifest = self.get_manifest(manifest_id)
        if manifest is None:
            raise ValueError(f"Unknown ultra-model manifest '{manifest_id}'.")
        plan = self.plan(manifest_id=manifest_id, project=project)
        command = self.build_command(
            manifest_id,
            plan=plan,
            executable=executable,
            host=host,
            port=port,
            allow_distributed=allow_distributed,
        )
        session_id = uuid4().hex
        log_path = self.log_root / f"{session_id}.log"
        log_handle = log_path.open("ab", buffering=0)
        creationflags = 0
        startupinfo = None
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.runtime_root),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=creationflags,
                startupinfo=startupinfo,
            )
        except OSError as exc:
            log_handle.close()
            raise UltraModelError(f"Unable to start llama-server: {exc}") from exc
        with self._lock:
            self._processes[session_id] = process
            self._log_handles[session_id] = log_handle
        started_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO ultra_model_sessions (
                    id, manifest_id, project, status, pid, host, port, command_json,
                    log_path, started_at, stopped_at, last_error
                ) VALUES (?, ?, ?, 'starting', ?, ?, ?, ?, ?, ?, '', '')
                """,
                (
                    session_id,
                    manifest_id,
                    manifest["project"],
                    int(process.pid),
                    host,
                    int(port),
                    json.dumps(command),
                    str(log_path),
                    started_at,
                ),
            )
        if monitor:
            thread = threading.Thread(
                target=self._monitor_startup,
                args=(session_id,),
                name=f"openagent-ultra-monitor-{session_id[:8]}",
                daemon=True,
            )
            thread.start()
        if keep_warm:
            self._start_keepalive(session_id)
        self._audit(
            "ultra_model.launch",
            f"ultra-session:{session_id}",
            manifest["project"],
            {"manifest_id": manifest_id, "pid": process.pid, "host": host, "port": int(port), "mode": plan["mode"]},
        )
        return {"session": self.get_session(session_id), "plan": plan}

    def stop(self, session_id: str, *, reason: str = "user requested stop") -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown ultra-model session '{session_id}'.")
        event = self._keepalive_events.pop(session_id, None)
        if event:
            event.set()
        with self._lock:
            process = self._processes.pop(session_id, None)
            log_handle = self._log_handles.pop(session_id, None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if log_handle is not None:
            log_handle.close()
        with self.store.session() as connection:
            connection.execute(
                "UPDATE ultra_model_sessions SET status = 'stopped', stopped_at = ?, last_error = ? WHERE id = ?",
                (utc_now(), reason, session_id),
            )
        self._audit(
            "ultra_model.stop",
            f"ultra-session:{session_id}",
            session["project"],
            {"reason": reason},
        )
        return self.get_session(session_id) or {}

    def warmup(self, session_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown ultra-model session '{session_id}'.")
        if session["status"] not in {"starting", "ready", "warming"}:
            raise UltraModelError(f"Session is {session['status']}, not running.")
        payload = json.dumps(
            {
                "model": session["manifest_id"],
                "messages": [{"role": "user", "content": "Respond with OK."}],
                "max_tokens": 1,
                "temperature": 0,
                "cache_prompt": True,
            }
        ).encode("utf-8")
        started = time.perf_counter()
        try:
            response = EgressClient().fetch(
                f"http://{session['host']}:{session['port']}/v1/chat/completions",
                approved=False,
                allow_loopback=True,
                purpose="ultra-model local warmup",
                method="POST",
                body=payload,
                headers={"Content-Type": "application/json"},
                timeout_seconds=max(1.0, min(float(timeout), 120.0)),
                max_response_bytes=4096,
                max_redirects=0,
            )
            if response.status != 200:
                raise UltraModelError(f"Local model warmup returned HTTP {response.status}.")
        except (EgressError, PermissionError, TimeoutError, OSError) as exc:
            raise UltraModelError(f"Local model warmup failed: {exc}") from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        with self.store.session() as connection:
            connection.execute(
                "UPDATE ultra_model_sessions SET status = 'ready', last_error = '' WHERE id = ?",
                (session_id,),
            )
        self._audit(
            "ultra_model.warmup",
            f"ultra-session:{session_id}",
            session["project"],
            {"latency_ms": elapsed_ms},
        )
        return {"status": "ready", "session_id": session_id, "latency_ms": elapsed_ms, "kept_warm": True}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        self._reconcile_process(session_id)
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT * FROM ultra_model_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return _session_row(row) if row else None

    def sessions(self, *, project: str = "default", limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            active_ids = list(self._processes)
        for session_id in active_ids:
            self._reconcile_process(session_id)
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM ultra_model_sessions WHERE project = ? ORDER BY started_at DESC LIMIT ?",
                (project.strip() or "default", max(1, min(int(limit), 100))),
            ).fetchall()
        return [_session_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Grounding and correctness gates
    # ------------------------------------------------------------------
    def verify_output(
        self,
        answer: str,
        *,
        evidence: list[dict[str, Any]] | list[str],
        project: str = "default",
        require_citations: bool = True,
        expected_json: bool = False,
    ) -> dict[str, Any]:
        answer = answer.strip()
        normalized_evidence = [
            str(item.get("content", item.get("text", ""))) if isinstance(item, dict) else str(item)
            for item in evidence
        ]
        normalized_evidence = [item for item in normalized_evidence if item.strip()]
        claims = [part.strip() for part in CLAIM_PATTERN.split(answer) if part.strip()]
        claim_reports: list[dict[str, Any]] = []
        invalid_citations: set[int] = set()
        for cited in CITATION_PATTERN.findall(answer):
            index = int(cited)
            if index < 1 or index > len(normalized_evidence):
                invalid_citations.add(index)
        for claim in claims:
            tokens = _meaningful_tokens(CITATION_PATTERN.sub("", claim))
            if len(tokens) < 3:
                continue
            scores = [_support_score(tokens, _meaningful_tokens(source)) for source in normalized_evidence]
            best_score = max(scores, default=0.0)
            best_source = scores.index(best_score) + 1 if scores else 0
            citations = [int(value) for value in CITATION_PATTERN.findall(claim)]
            citation_valid = bool(citations) and all(1 <= value <= len(normalized_evidence) for value in citations)
            contradiction = False
            if best_source:
                source_tokens = set(_meaningful_tokens(normalized_evidence[best_source - 1]))
                claim_negated = bool(set(tokens) & NEGATIONS)
                source_negated = bool(source_tokens & NEGATIONS)
                contradiction = best_score >= 0.45 and claim_negated != source_negated
            claim_reports.append(
                {
                    "claim": claim,
                    "support_score": round(best_score, 4),
                    "best_source": best_source or None,
                    "citation_valid": citation_valid,
                    "contradiction_flag": contradiction,
                }
            )
        supported = [report for report in claim_reports if report["support_score"] >= 0.45 and not report["contradiction_flag"]]
        support_score = len(supported) / max(1, len(claim_reports))
        citation_coverage = (
            sum(1 for report in claim_reports if report["citation_valid"]) / max(1, len(claim_reports))
        )
        json_valid: bool | None = None
        if expected_json:
            try:
                json.loads(answer)
                json_valid = True
            except json.JSONDecodeError:
                json_valid = False
        contradiction_count = sum(1 for report in claim_reports if report["contradiction_flag"])
        if not answer or (not normalized_evidence and claim_reports):
            status = "abstain"
        elif expected_json and not json_valid:
            status = "rejected"
        elif invalid_citations or contradiction_count:
            status = "rejected"
        elif support_score >= 0.8 and (not require_citations or citation_coverage >= 0.75):
            status = "verified"
        elif support_score >= 0.5 and (not require_citations or citation_coverage >= 0.4):
            status = "review"
        else:
            status = "abstain"
        report_id = uuid4().hex
        report = {
            "id": report_id,
            "status": status,
            "support_score": round(support_score, 4),
            "citation_coverage": round(citation_coverage, 4),
            "invalid_citations": sorted(invalid_citations),
            "contradiction_count": contradiction_count,
            "json_valid": json_valid,
            "claims": claim_reports,
            "policy": "Unsupported, contradictory, or malformed output is rejected or marked for review; correctness is never assumed from model confidence.",
            "created_at": utc_now(),
        }
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO ultra_model_verifications (id, project, status, support_score, report_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (report_id, project, status, support_score, json.dumps(report, sort_keys=True), report["created_at"]),
            )
        self._audit(
            "ultra_model.output.verify",
            f"ultra-verification:{report_id}",
            project,
            {"status": status, "support_score": support_score, "claims": len(claim_reports)},
        )
        return report

    def calibrate(self, *, project: str = "default", size_mb: int = 32) -> dict[str, Any]:
        size_mb = max(8, min(int(size_mb), 256))
        block = bytearray(size_mb * 1024 * 1024)
        destination = bytearray(len(block))
        memory_started = time.perf_counter()
        for _ in range(4):
            destination[:] = block
        memory_seconds = max(0.000001, time.perf_counter() - memory_started)
        memory_gbps = len(block) * 4 / memory_seconds / 1_000_000_000
        path = self.runtime_root / f"calibration-{uuid4().hex}.bin"
        write_started = time.perf_counter()
        with path.open("wb") as handle:
            handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        write_seconds = max(0.000001, time.perf_counter() - write_started)
        read_started = time.perf_counter()
        with path.open("rb") as handle:
            while handle.read(4 * 1024 * 1024):
                pass
        read_seconds = max(0.000001, time.perf_counter() - read_started)
        path.unlink(missing_ok=True)
        result = {
            "id": uuid4().hex,
            "project": project,
            "sample_mb": size_mb,
            "memory_copy_gbps": round(memory_gbps, 3),
            "disk_write_gbps": round(len(block) / write_seconds / 1_000_000_000, 3),
            "disk_buffered_read_gbps": round(len(block) / read_seconds / 1_000_000_000, 3),
            "planning_disk_gbps": round(max(0.05, len(block) / read_seconds / 1_000_000_000 * 0.6), 3),
            "warning": "Buffered microbenchmarks are directional. Benchmark the target GGUF for deployment decisions.",
            "created_at": utc_now(),
        }
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO ultra_model_calibrations (id, project, calibration_json, created_at) VALUES (?, ?, ?, ?)",
                (result["id"], project, json.dumps(result, sort_keys=True), result["created_at"]),
            )
        self._audit("ultra_model.calibrate", f"ultra-calibration:{result['id']}", project, {"sample_mb": size_mb})
        return result

    def status(self, *, project: str = "default") -> dict[str, Any]:
        binary = ""
        try:
            binary = resolve_llama_server(self.llama_server)
        except UltraModelError:
            pass
        manifests = self.manifests(project=project, limit=200)
        sessions = self.sessions(project=project, limit=100)
        active = [session for session in sessions if session["status"] in {"starting", "warming", "ready"}]
        budget = self._hardware_budget(None)
        return {
            "status": "ready",
            "engine": ENGINE_ID,
            "llama_server": {"available": bool(binary), "path": binary},
            "manifests": len(manifests),
            "active_sessions": len(active),
            "hardware": budget.to_dict(),
            "execution_tiers": ["fast-local-model", "resident-mmap", "out-of-core-mmap", "trusted-rpc-sharded"],
            "techniques": [
                "bandwidth-constrained-feasibility",
                "sharded-gguf-auto-discovery",
                "memory-mapped-out-of-core-loading",
                "quantized-kv-cache",
                "prompt-prefix-cache-reuse",
                "warm-process-pool",
                "draft-or-ngram-speculative-decoding",
                "complexity-aware-model-cascade",
                "grounded-output-rejection-gate",
            ],
            "hard_limit": "A dense model still touches essentially all active weights per generated token. mmap can make an oversized model launch, but cannot turn storage bandwidth into RAM bandwidth.",
            "privacy_boundary": "The engine starts only local loopback processes. Distributed execution requires explicit approval, an opt-in environment flag, and trusted private endpoints.",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hardware_budget(self, override: dict[str, Any] | None) -> HardwareBudget:
        metrics = system_metrics(self.store.data_dir)
        memory = metrics.get("memory", {})
        cpu = metrics.get("cpu", {})
        values = dict(override or {})
        return HardwareBudget(
            total_ram_gb=max(0.5, float(values.get("total_ram_gb", memory.get("total_gb", 8) or 8))),
            available_ram_gb=max(0.25, float(values.get("available_ram_gb", memory.get("available_gb", 4) or 4))),
            memory_bandwidth_gbps=max(0.1, float(values.get("memory_bandwidth_gbps", os.environ.get("OPENAGENT_MEMORY_BANDWIDTH_GBPS", 25)))),
            disk_bandwidth_gbps=max(0.01, float(values.get("disk_bandwidth_gbps", os.environ.get("OPENAGENT_DISK_READ_GBPS", 1.5)))),
            network_bandwidth_gbps=max(0.01, float(values.get("network_bandwidth_gbps", os.environ.get("OPENAGENT_TRUSTED_NETWORK_GBPS", 1.0)))),
            cpu_threads=max(1, int(values.get("cpu_threads", cpu.get("count", os.cpu_count() or 1)))),
        )

    def _monitor_startup(self, session_id: str) -> None:
        deadline = time.monotonic() + max(30, int(os.environ.get("OPENAGENT_ULTRA_STARTUP_TIMEOUT", "900")))
        while time.monotonic() < deadline:
            session = self.get_session(session_id)
            if session is None or session["status"] in {"failed", "stopped", "exited"}:
                return
            try:
                response = EgressClient().fetch(
                    f"http://{session['host']}:{session['port']}/health",
                    approved=False,
                    allow_loopback=True,
                    purpose="ultra-model local health",
                    timeout_seconds=2,
                    max_response_bytes=4096,
                    max_redirects=0,
                )
                if response.status == 200:
                    with self.store.session() as connection:
                        connection.execute(
                            "UPDATE ultra_model_sessions SET status = 'ready', last_error = '' WHERE id = ?",
                            (session_id,),
                        )
                    return
            except (EgressError, PermissionError, TimeoutError, OSError):
                time.sleep(1.0)
        with self.store.session() as connection:
            connection.execute(
                "UPDATE ultra_model_sessions SET status = 'failed', last_error = ? WHERE id = ?",
                ("llama-server startup timed out", session_id),
            )

    def _start_keepalive(self, session_id: str) -> None:
        event = threading.Event()
        self._keepalive_events[session_id] = event

        def loop() -> None:
            interval = max(30, int(os.environ.get("OPENAGENT_ULTRA_KEEPALIVE_SECONDS", "300")))
            if event.wait(min(10, interval)):
                return
            while not event.is_set():
                try:
                    self.warmup(session_id, timeout=60)
                except UltraModelError:
                    pass
                if event.wait(interval):
                    return

        threading.Thread(
            target=loop,
            name=f"openagent-ultra-keepalive-{session_id[:8]}",
            daemon=True,
        ).start()

    def _reconcile_process(self, session_id: str) -> None:
        with self._lock:
            process = self._processes.get(session_id)
        if process is None:
            return
        return_code = process.poll()
        if return_code is None:
            return
        with self._lock:
            self._processes.pop(session_id, None)
            log_handle = self._log_handles.pop(session_id, None)
        if log_handle is not None:
            log_handle.close()
        event = self._keepalive_events.pop(session_id, None)
        if event:
            event.set()
        with self.store.session() as connection:
            connection.execute(
                "UPDATE ultra_model_sessions SET status = 'exited', stopped_at = ?, last_error = ? WHERE id = ? AND status != 'stopped'",
                (utc_now(), f"llama-server exited with code {return_code}", session_id),
            )

    def _audit(self, action: str, resource: str, project: str, metadata: dict[str, Any]) -> None:
        self.store.add_audit_event(
            AuditEvent(
                actor="ultra-model-engine",
                action=action,
                resource=resource,
                project=project,
                metadata=metadata,
            )
        )


def discover_gguf_shards(raw_paths: list[str]) -> list[str]:
    paths = [Path(str(item)).expanduser().resolve() for item in raw_paths if str(item).strip()]
    if not paths:
        raise ValueError("At least one local GGUF shard path is required.")
    first_match = SHARD_PATTERN.match(paths[0].name)
    if len(paths) == 1 and first_match:
        count = int(first_match.group("count"))
        prefix = first_match.group("prefix")
        paths = [paths[0].with_name(f"{prefix}-{index:05d}-of-{count:05d}.gguf") for index in range(1, count + 1)]
    unique = list(dict.fromkeys(paths))
    for path in unique:
        if path.suffix.lower() != ".gguf":
            raise ValueError(f"Model shard is not GGUF: {path}")
        if not path.is_file():
            raise ValueError(f"Model shard is missing: {path}")
    indexed: list[tuple[int, Path]] = []
    expected_count = 0
    for path in unique:
        match = SHARD_PATTERN.match(path.name)
        if match:
            indexed.append((int(match.group("index")), path))
            expected_count = max(expected_count, int(match.group("count")))
        else:
            indexed.append((1, path))
    if expected_count and (len(indexed) != expected_count or sorted(index for index, _ in indexed) != list(range(1, expected_count + 1))):
        raise ValueError(f"Expected all {expected_count} GGUF shards with contiguous indexes.")
    return [str(path) for _, path in sorted(indexed, key=lambda item: item[0])]


def validate_trusted_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for node in nodes:
        if not bool(node.get("trusted", False)):
            raise ValueError("Every RPC node must be explicitly marked trusted.")
        endpoint = str(node.get("endpoint", "")).strip()
        if ":" not in endpoint:
            raise ValueError("RPC endpoint must use host:port.")
        host, port_text = endpoint.rsplit(":", 1)
        try:
            address = ipaddress.ip_address(host.strip("[]"))
            port = int(port_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid RPC endpoint '{endpoint}'.") from exc
        if not (address.is_private or address.is_loopback):
            raise ValueError("RPC endpoints must be private or loopback addresses.")
        if not 1 <= port <= 65535:
            raise ValueError("RPC endpoint port is out of range.")
        ram_gb = _bounded_float(node.get("ram_gb", 0), "node ram_gb", 0.25, 1048576.0)
        validated.append({"endpoint": endpoint, "ram_gb": ram_gb, "trusted": True})
    return validated


def resolve_llama_server(configured: str = "") -> str:
    candidate = configured.strip()
    resolved = shutil.which(candidate) if candidate else shutil.which("llama-server")
    if not resolved and candidate:
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            resolved = str(path)
    if not resolved:
        raise UltraModelError("llama-server was not found. Install llama.cpp or set OPENAGENT_LLAMA_SERVER.")
    basename = Path(resolved).name.lower()
    if basename not in {"llama-server", "llama-server.exe"}:
        raise UltraModelError("Only the llama-server executable is allowed for ultra-model launch.")
    return str(Path(resolved).resolve())


def estimate_kv_cache_bytes(active_parameter_count_b: float, context_tokens: int, kv_type: str) -> int:
    type_factor = {
        "f32": 2.0,
        "f16": 1.0,
        "bf16": 1.0,
        "q8_0": 0.55,
        "q5_0": 0.36,
        "q5_1": 0.38,
        "q4_0": 0.30,
        "q4_1": 0.32,
        "iq4_nl": 0.30,
    }.get(kv_type, 0.55)
    bytes_per_token_f16 = 0.5 * 1024**2 * math.sqrt(max(0.1, active_parameter_count_b) / 7.0)
    return int(bytes_per_token_f16 * type_factor * max(1, context_tokens))


def _manifest_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "name": row["name"],
        "architecture": row["architecture"],
        "parameter_count_b": float(row["parameter_count_b"]),
        "active_parameter_count_b": float(row["active_parameter_count_b"]),
        "quantization_bits": float(row["quantization_bits"]),
        "shard_paths": json.loads(row["shard_paths_json"]),
        "draft_model_path": row["draft_model_path"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _session_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "manifest_id": row["manifest_id"],
        "project": row["project"],
        "status": row["status"],
        "pid": int(row["pid"]),
        "host": row["host"],
        "port": int(row["port"]),
        "command": json.loads(row["command_json"]),
        "log_path": row["log_path"],
        "started_at": row["started_at"],
        "stopped_at": row["stopped_at"],
        "last_error": row["last_error"],
    }


def _bounded_float(value: object, label: str, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _gib(value: int | float) -> float:
    return round(float(value) / 1024**3, 3)


def _is_loopback_host(host: str) -> bool:
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _meaningful_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


def _support_score(claim_tokens: list[str], evidence_tokens: list[str]) -> float:
    if not claim_tokens or not evidence_tokens:
        return 0.0
    claim = set(claim_tokens)
    evidence = set(evidence_tokens)
    return len(claim & evidence) / max(1, min(len(claim), 12))
