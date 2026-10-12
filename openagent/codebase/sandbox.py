"""Workspace isolation, snapshots, and command execution backends."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from typing import Any
from uuid import uuid4

from ..process_policy import ProcessPolicy, run_process
from .repository import IGNORED_NAMES, RepositoryBoundaryError, RepositoryIntelligence


SECRET_NAME_PARTS = ("TOKEN", "SECRET", "PASSWORD", "PRIVATE_KEY", "API_KEY", "ACCESS_KEY")
ALLOWED_ENV = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG"}


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": round(self.duration_ms, 3),
            "timed_out": self.timed_out,
        }


class SandboxManager:
    """Create disposable local or Docker-backed project workspaces."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "snapshots").mkdir(exist_ok=True)
        self._processes: dict[str, dict[str, Any]] = {}
        self._process_lock = threading.RLock()

    def create(self, workspace_id: str, *, source: str | Path, backend: str = "local", branch: str = "") -> dict[str, Any]:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_dir():
            raise ValueError("source repository must be an existing directory")
        if backend not in {"local", "docker", "hosted"}:
            raise ValueError("sandbox backend must be local, docker, or hosted")
        if backend == "docker" and shutil.which("docker") is None:
            raise ValueError("Docker backend requested but docker is not installed")
        if backend == "hosted" and not os.getenv("OPENAGENT_HOSTED_SANDBOX_COMMAND", "").strip():
            raise ValueError("Hosted backend requires OPENAGENT_HOSTED_SANDBOX_COMMAND")
        destination = (self.root / workspace_id / "workspace").resolve()
        if destination.exists():
            raise ValueError("workspace already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, destination, ignore=self._copy_ignore, symlinks=False)
        repository = RepositoryIntelligence(destination)
        git = shutil.which("git")
        if git and (destination / ".git").is_dir():
            self._run_local([git, "status", "--short"], destination, timeout=20)
            if branch:
                safe_branch = self.safe_branch(branch)
                existing = self._run_local([git, "branch", "--list", safe_branch], destination, timeout=20)
                command = [git, "checkout", safe_branch] if existing.stdout.strip() else [git, "checkout", "-b", safe_branch]
                result = self._run_local(command, destination, timeout=30)
                if result.exit_code != 0:
                    raise ValueError(f"could not create sandbox branch: {result.stderr.strip()}")
        return {
            "id": workspace_id,
            "path": str(destination),
            "source": str(source_path),
            "backend": backend,
            "branch": branch,
            "repository": repository.scan(),
        }

    def path(self, workspace_id: str) -> Path:
        path = (self.root / workspace_id / "workspace").resolve()
        expected = self.root / workspace_id
        if expected.resolve() not in path.parents or not path.is_dir():
            raise KeyError("workspace not found")
        return path

    def remove(self, workspace_id: str) -> None:
        path = (self.root / workspace_id).resolve()
        if path.parent != self.root or not path.exists():
            raise KeyError("workspace not found")
        shutil.rmtree(path)

    def snapshot(self, workspace_id: str, snapshot_id: str) -> dict[str, Any]:
        workspace = self.path(workspace_id)
        target = (self.root / "snapshots" / f"{snapshot_id}.tar.gz").resolve()
        if target.parent != (self.root / "snapshots").resolve():
            raise RepositoryBoundaryError("invalid snapshot id")
        with tarfile.open(target, "w:gz") as archive:
            for path in workspace.rglob("*"):
                rel = path.relative_to(workspace)
                if any(part in IGNORED_NAMES - {".git"} for part in rel.parts):
                    continue
                if path.is_symlink():
                    continue
                archive.add(path, arcname=rel.as_posix(), recursive=False)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return {"id": snapshot_id, "workspace_id": workspace_id, "path": str(target), "sha256": digest, "size_bytes": target.stat().st_size}

    def restore(self, workspace_id: str, snapshot_path: str | Path) -> None:
        workspace = self.path(workspace_id)
        archive_path = Path(snapshot_path).resolve()
        if (self.root / "snapshots").resolve() not in archive_path.parents:
            raise RepositoryBoundaryError("snapshot must belong to this sandbox manager")
        staging = Path(tempfile.mkdtemp(prefix="openagent-restore-", dir=str(workspace.parent)))
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                for member in archive.getmembers():
                    destination = (staging / member.name).resolve()
                    if destination != staging and staging not in destination.parents:
                        raise RepositoryBoundaryError("snapshot contains an unsafe path")
                # Snapshots are created by this manager and addressed from its
                # private snapshot directory. Path validation above keeps the
                # restore compatible with Python 3.11 without relying on the
                # newer tarfile extraction filter API.
                archive.extractall(staging)
            replacement = workspace.with_name("workspace-restored")
            if replacement.exists():
                shutil.rmtree(replacement)
            shutil.move(str(staging), replacement)
            old = workspace.with_name("workspace-old")
            if old.exists():
                shutil.rmtree(old)
            workspace.rename(old)
            replacement.rename(workspace)
            shutil.rmtree(old)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def run(
        self,
        workspace_id: str,
        command: list[str],
        *,
        cwd: str = "",
        backend: str = "local",
        timeout_seconds: int = 120,
        environment: dict[str, str] | None = None,
        docker_image: str = "python:3.12-slim",
    ) -> CommandResult:
        workspace = self.path(workspace_id)
        resolved_cwd = RepositoryIntelligence(workspace).resolve(cwd) if cwd else workspace
        timeout = max(1, min(int(timeout_seconds), 1800))
        env = self.safe_environment(environment or {})
        if backend == "docker":
            if shutil.which("docker") is None:
                raise ValueError("Docker backend is not available")
            relative = resolved_cwd.relative_to(workspace).as_posix()
            docker_cwd = "/workspace" if relative == "." else f"/workspace/{relative}"
            mounted = f"{workspace}:/workspace"
            wrapped = ["docker", "run", "--rm", "--network", "none", "--memory", "1g", "--cpus", "2", "-v", mounted, "-w", docker_cwd]
            for key, value in env.items():
                if key not in ALLOWED_ENV:
                    wrapped.extend(["-e", f"{key}={value}"])
            wrapped.extend([docker_image, *command])
            return self._run_local(wrapped, workspace, timeout=timeout, environment=self.safe_environment({}))
        if backend == "hosted":
            return self._run_hosted(
                workspace_id,
                command,
                cwd=resolved_cwd.relative_to(workspace).as_posix(),
                timeout=timeout,
                environment=environment or {},
            )
        return self._run_local(self._native_command(command), resolved_cwd, timeout=timeout, environment=env)

    def start_process(
        self,
        workspace_id: str,
        command: list[str],
        *,
        cwd: str = "",
        backend: str = "local",
        environment: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if backend != "local":
            raise ValueError("managed background processes currently require the local copied-workspace backend")
        workspace = self.path(workspace_id)
        resolved_cwd = RepositoryIntelligence(workspace).resolve(cwd) if cwd else workspace
        process_id = uuid4().hex
        process_dir = workspace / ".openagent" / "processes"
        process_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = process_dir / f"{process_id}.stdout.log"
        stderr_path = process_dir / f"{process_id}.stderr.log"
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        try:
            native_command = self._native_command(command)
            proc = subprocess.Popen(
                native_command, cwd=str(resolved_cwd), env=self.safe_environment(environment or {}),
                stdout=stdout_handle, stderr=stderr_handle, text=True, shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            raise
        record = {
            "id": process_id,
            "workspace_id": workspace_id,
            "command": native_command,
            "cwd": str(resolved_cwd),
            "pid": int(proc.pid),
            "started_at": time.time(),
            "process": proc,
            "stdout_handle": stdout_handle,
            "stderr_handle": stderr_handle,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        with self._process_lock:
            self._processes[process_id] = record
        return self.process_status(workspace_id, process_id)

    def process_status(self, workspace_id: str, process_id: str) -> dict[str, Any]:
        record = self._managed_process(workspace_id, process_id)
        proc: subprocess.Popen[Any] = record["process"]
        exit_code = proc.poll()
        if exit_code is not None:
            self._close_process_logs(record)
        return {
            "id": process_id,
            "pid": record["pid"],
            "command": record["command"],
            "cwd": record["cwd"],
            "status": "running" if exit_code is None else "exited",
            "exit_code": exit_code,
            "duration_ms": round((time.time() - float(record["started_at"])) * 1000, 3),
        }

    def stop_process(self, workspace_id: str, process_id: str) -> dict[str, Any]:
        record = self._managed_process(workspace_id, process_id)
        proc: subprocess.Popen[Any] = record["process"]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._close_process_logs(record)
        result = self.process_status(workspace_id, process_id)
        with self._process_lock:
            self._processes.pop(process_id, None)
        return result

    def process_logs(self, workspace_id: str, process_id: str, *, limit: int = 100_000) -> dict[str, Any]:
        record = self._managed_process(workspace_id, process_id)
        for key in ("stdout_handle", "stderr_handle"):
            try:
                record[key].flush()
            except ValueError:
                pass
        bounded = max(1_000, min(int(limit), 500_000))
        def read(path: str) -> str:
            try:
                return Path(path).read_text(encoding="utf-8", errors="replace")[-bounded:]
            except OSError:
                return ""
        return {**self.process_status(workspace_id, process_id), "stdout": read(record["stdout_path"]), "stderr": read(record["stderr_path"])}

    def _managed_process(self, workspace_id: str, process_id: str) -> dict[str, Any]:
        with self._process_lock:
            record = self._processes.get(process_id)
        if record is None or record["workspace_id"] != workspace_id:
            raise KeyError("managed process not found")
        return record

    @staticmethod
    def _close_process_logs(record: dict[str, Any]) -> None:
        for key in ("stdout_handle", "stderr_handle"):
            handle = record.get(key)
            if handle is None or getattr(handle, "closed", True):
                continue
            try:
                handle.close()
            except OSError:
                pass

    @staticmethod
    def _native_command(command: list[str]) -> list[str]:
        parts = [str(item) for item in command]
        if os.name == "nt" and parts and parts[0].lower() in {"npm", "npx", "pnpm", "yarn"}:
            command_file = shutil.which(f"{parts[0]}.cmd")
            if command_file:
                parts[0] = command_file
        return parts

    def _run_hosted(
        self,
        workspace_id: str,
        command: list[str],
        *,
        cwd: str,
        timeout: int,
        environment: dict[str, str],
    ) -> CommandResult:
        adapter = os.getenv("OPENAGENT_HOSTED_SANDBOX_COMMAND", "").strip()
        if not adapter:
            raise ValueError("Hosted sandbox adapter is not configured")
        adapter_command = shlex.split(adapter, posix=os.name != "nt")
        if os.name == "nt":
            adapter_command = [item[1:-1] if len(item) >= 2 and item[0] == item[-1] == '"' else item for item in adapter_command]
        payload = {
            "protocol": "openagent.sandbox.v1",
            "action": "run",
            "workspace_id": workspace_id,
            "workspace_path": str(self.path(workspace_id)),
            "command": command,
            "cwd": cwd,
            "timeout_seconds": timeout,
            "environment": {key: value for key, value in environment.items() if not any(part in key.upper() for part in SECRET_NAME_PARTS)},
        }
        started = time.perf_counter()
        result = subprocess.run(
            adapter_command, input=json.dumps(payload), capture_output=True, text=True, errors="replace",
            timeout=timeout + 15, shell=False, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            return CommandResult(command, cwd, int(result.returncode), result.stdout[-200_000:], result.stderr[-200_000:], (time.perf_counter() - started) * 1000)
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("Hosted sandbox adapter returned invalid JSON") from exc
        return CommandResult(
            command,
            cwd,
            int(response.get("exit_code", -1)),
            str(response.get("stdout", ""))[-200_000:],
            str(response.get("stderr", ""))[-200_000:],
            float(response.get("duration_ms", (time.perf_counter() - started) * 1000)),
            bool(response.get("timed_out", False)),
        )

    @staticmethod
    def safe_environment(injected: dict[str, str]) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key in ALLOWED_ENV}
        for key, value in injected.items():
            clean_key = str(key).strip().upper()
            if not clean_key or any(part in clean_key for part in SECRET_NAME_PARTS):
                continue
            env[clean_key] = str(value)
        env["OPENAGENT_SANDBOX"] = "1"
        return env

    @staticmethod
    def safe_branch(value: str) -> str:
        cleaned = "".join(char if char.isalnum() or char in "-_/" else "-" for char in value.strip())
        cleaned = cleaned.strip("-/. ")[:120]
        if not cleaned or ".." in cleaned or cleaned.startswith("-"):
            raise ValueError("invalid Git branch name")
        return cleaned

    @staticmethod
    def _copy_ignore(path: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in IGNORED_NAMES - {".git"}}
        ignored.update({name for name in names if name.endswith((".pyc", ".pyo"))})
        ignored.update({name for name in names if (Path(path) / name).is_symlink()})
        return ignored

    @staticmethod
    def _run_local(command: list[str], cwd: Path, *, timeout: int, environment: dict[str, str] | None = None) -> CommandResult:
        started = time.perf_counter()
        try:
            process = run_process(
                [str(item) for item in command], cwd=cwd, env=environment,
                policy=ProcessPolicy(
                    timeout_seconds=max(1, int(timeout)),
                    max_output_bytes=200_000,
                    allowed_cwd_roots=(cwd.resolve(),),
                    env_keys=frozenset((environment or {}).keys()) | frozenset(ALLOWED_ENV) | frozenset({"OPENAGENT_SANDBOX"}),
                ),
            )
            stderr = process.stderr
            if process.output_truncated:
                stderr = (stderr + "\nCommand output was truncated at the workspace policy limit.").strip()
            return CommandResult(
                command, str(cwd), process.returncode, process.stdout, stderr,
                (time.perf_counter() - started) * 1000, process.timed_out,
            )
        except (PermissionError, ValueError) as exc:
            return CommandResult(command, str(cwd), 127, "", str(exc), (time.perf_counter() - started) * 1000)
