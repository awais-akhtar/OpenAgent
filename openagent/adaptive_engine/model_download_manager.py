"""Stage 6 — Model download manager.

Provides a thread-safe download queue with:
  - Model registry (family → source URLs, expected sizes, SHA-256 hashes)
  - Download queue with priority and concurrency control
  - Resumable HTTP downloads (Range requests)
  - SHA-256 hash verification on completion
  - Progress callbacks (bytes_downloaded, total_bytes, speed_bps)
  - HuggingFace Hub URL helpers (repo_id/filename → resolved URL)
  - Persistent download state (survives process restart)

No network I/O happens on import.  Downloads are started explicitly by
calling ModelDownloadManager.enqueue() and then .run_until_complete() or
managing the queue from a background thread.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Model registry entries
# ---------------------------------------------------------------------------

class SourceKind(str, Enum):
    HUGGINGFACE = "huggingface"
    DIRECT_URL = "direct_url"
    LOCAL_PATH = "local_path"


@dataclass(frozen=True)
class ModelSource:
    """One resolvable download source for a model file."""
    kind: SourceKind
    uri: str                     # HF repo_id/filename OR direct URL OR local path
    expected_sha256: Optional[str] = None   # lowercase hex, None = skip verify
    expected_bytes: Optional[int] = None    # None = unknown
    priority: int = 0            # higher = preferred when multiple sources exist


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Registry record for one model artifact."""
    model_id: str                # stable identifier, e.g. "mixtral-8x7b-v0.1-fp8"
    family: str                  # adaptive engine family name
    sources: Tuple[ModelSource, ...]
    description: str = ""
    min_disk_bytes: Optional[int] = None


# ---------------------------------------------------------------------------
# Download state
# ---------------------------------------------------------------------------

class DownloadStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadJob:
    job_id: str
    model_id: str
    source: ModelSource
    dest_path: Path
    status: DownloadStatus = DownloadStatus.PENDING
    bytes_downloaded: int = 0
    total_bytes: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    attempts: int = 0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "source_kind": self.source.kind.value,
            "source_uri": self.source.uri,
            "source_expected_sha256": self.source.expected_sha256,
            "source_expected_bytes": self.source.expected_bytes,
            "source_priority": self.source.priority,
            "dest_path": str(self.dest_path),
            "status": self.status.value,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: dict, source: ModelSource) -> "DownloadJob":
        job = cls(
            job_id=data["job_id"],
            model_id=data["model_id"],
            source=source,
            dest_path=Path(data["dest_path"]),
            status=DownloadStatus(data["status"]),
            bytes_downloaded=data.get("bytes_downloaded", 0),
            total_bytes=data.get("total_bytes"),
            error=data.get("error"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            attempts=data.get("attempts", 0),
        )
        return job


ProgressCallback = Callable[[str, int, Optional[int], float], None]
# (job_id, bytes_downloaded, total_bytes_or_None, speed_bps)


# ---------------------------------------------------------------------------
# HuggingFace URL helpers
# ---------------------------------------------------------------------------

_HF_ENDPOINT = "https://huggingface.co"


def resolve_huggingface_url(repo_id: str, filename: str, *,
                             revision: str = "main",
                             endpoint: str = _HF_ENDPOINT) -> str:
    """Build a HuggingFace resolve URL without network I/O."""
    return f"{endpoint.rstrip('/')}/{repo_id}/resolve/{revision}/{filename}"


def huggingface_source(repo_id: str, filename: str, *,
                        revision: str = "main",
                        expected_sha256: Optional[str] = None,
                        expected_bytes: Optional[int] = None,
                        priority: int = 0) -> ModelSource:
    """Convenience constructor for a HuggingFace model source."""
    uri = f"{repo_id}/{filename}@{revision}"
    return ModelSource(
        kind=SourceKind.HUGGINGFACE,
        uri=uri,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        priority=priority,
    )


def _resolve_source_url(source: ModelSource) -> str:
    """Return a download URL for any source kind."""
    if source.kind == SourceKind.DIRECT_URL:
        return source.uri
    if source.kind == SourceKind.HUGGINGFACE:
        # uri format: "repo_id/filename@revision" or "repo_id/filename"
        if "@" in source.uri:
            path_part, revision = source.uri.rsplit("@", 1)
        else:
            path_part, revision = source.uri, "main"
        # repo_id is everything before the last slash separator
        parts = path_part.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid HuggingFace URI: {source.uri!r}")
        # Handle org/repo/path/to/file
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return resolve_huggingface_url(repo_id, filename, revision=revision)
    raise ValueError(f"Cannot resolve URL for source kind {source.kind!r}")


# ---------------------------------------------------------------------------
# Download engine
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 1 << 17  # 128 KiB
_CONNECT_TIMEOUT = 30
_READ_TIMEOUT = 60
_MAX_ATTEMPTS = 3


def _download_file(
    url: str,
    dest_path: Path,
    *,
    resume_from: int = 0,
    progress_cb: Optional[ProgressCallback] = None,
    job_id: str = "",
    cancel_event: Optional[threading.Event] = None,
) -> int:
    """Download url → dest_path, resuming from resume_from bytes.

    Returns total bytes written (from the start, not just this call).
    Raises on HTTP error or cancellation.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "openagent/1.0"})
    if resume_from > 0:
        req.add_header("Range", f"bytes={resume_from}-")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=_CONNECT_TIMEOUT) as resp:
        response_status = getattr(resp, "status", None) or resp.getcode()
        effective_resume = resume_from
        if response_status == 206:
            content_range = resp.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range.strip(), re.IGNORECASE)
            expected_start = resume_from if resume_from > 0 else 0
            if not match or int(match.group(1)) != expected_start or int(match.group(2)) < expected_start:
                raise RuntimeError("server returned an invalid Content-Range for resumed download")
            effective_resume = expected_start
        elif response_status == 200:
            # Some servers ignore Range. Restart from byte zero instead of
            # appending a full response to a partial artifact.
            effective_resume = 0
        else:
            raise RuntimeError(f"unexpected HTTP status {response_status} for model download")

        mode = "ab" if effective_resume > 0 else "wb"
        total: Optional[int] = None
        content_length = resp.headers.get("Content-Length")
        if content_length:
            try:
                total = int(content_length) + effective_resume
            except ValueError:
                pass
        written = effective_resume
        last_time = time.monotonic()
        last_bytes = effective_resume

        with dest_path.open(mode) as fh:
            while True:
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("download cancelled")
                chunk = resp.read(_CHUNK_SIZE)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
                now = time.monotonic()
                dt = now - last_time
                speed = (written - last_bytes) / dt if dt > 0.1 else 0.0
                if progress_cb:
                    try:
                        progress_cb(job_id, written, total, speed)
                    except Exception:
                        pass
                if dt > 0.5:
                    last_time = now
                    last_bytes = written

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("download cancelled")

    return written


def _verify_sha256(path: Path, expected_hex: str) -> bool:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower() == expected_hex.lower()


# ---------------------------------------------------------------------------
# Download manager
# ---------------------------------------------------------------------------

class ModelDownloadManager:
    """Thread-safe model download manager with resume and hash verification.

    Usage::

        mgr = ModelDownloadManager(storage_root=Path("models"))
        mgr.register(entry)
        job_id = mgr.enqueue("mixtral-8x7b-v0.1-fp8", dest_dir=Path("models"))
        mgr.run_until_complete(job_id)
        path = mgr.get_path(job_id)
    """

    def __init__(
        self,
        storage_root: Path,
        *,
        max_concurrent: int = 2,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self._root = Path(storage_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_concurrent = max(1, max_concurrent)
        self._progress_cb = progress_callback
        self._lock = threading.Lock()
        self._registry: Dict[str, ModelRegistryEntry] = {}
        self._jobs: Dict[str, DownloadJob] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._state_file = self._root / ".download_state.json"
        self._load_state()

    # ── Registry ──────────────────────────────────────────────────────────

    def register(self, entry: ModelRegistryEntry) -> None:
        """Add or replace a model registry entry."""
        with self._lock:
            self._registry[entry.model_id] = entry

    def get_entry(self, model_id: str) -> Optional[ModelRegistryEntry]:
        with self._lock:
            return self._registry.get(model_id)

    def registered_models(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._registry))

    # ── Queue ─────────────────────────────────────────────────────────────

    def enqueue(
        self,
        model_id: str,
        dest_dir: Optional[Path] = None,
        *,
        source_index: int = 0,
    ) -> str:
        """Add a download job and return its job_id."""
        import uuid
        with self._lock:
            entry = self._registry.get(model_id)
            if entry is None:
                raise KeyError(f"model {model_id!r} is not registered")
            sorted_sources = sorted(entry.sources, key=lambda s: -s.priority)
            if source_index >= len(sorted_sources):
                raise IndexError(
                    f"source_index {source_index} out of range for {model_id!r}")
            source = sorted_sources[source_index]
            dest = (dest_dir or self._root) / model_id / Path(source.uri).name
            job_id = f"dl-{uuid.uuid4().hex[:12]}"
            job = DownloadJob(
                job_id=job_id,
                model_id=model_id,
                source=source,
                dest_path=dest,
            )
            self._jobs[job_id] = job
            self._cancel_events[job_id] = threading.Event()
            self._save_state()
            return job_id

    def cancel(self, job_id: str) -> None:
        """Signal a running download to stop."""
        with self._lock:
            ev = self._cancel_events.get(job_id)
            job = self._jobs.get(job_id)
            if ev:
                ev.set()
            if job and job.status in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING):
                job.status = DownloadStatus.CANCELLED
                job.finished_at = time.time()
                self._save_state()

    def get_job(self, job_id: str) -> Optional[DownloadJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_path(self, job_id: str) -> Optional[Path]:
        """Return the destination path if the download is complete."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job and job.status == DownloadStatus.COMPLETE:
            return job.dest_path
        return None

    def list_jobs(self) -> List[DownloadJob]:
        with self._lock:
            return list(self._jobs.values())

    # ── Execution ─────────────────────────────────────────────────────────

    def run_job(self, job_id: str) -> DownloadJob:
        """Run one download job synchronously; return the finished job."""
        with self._lock:
            job = self._jobs.get(job_id)
            ev = self._cancel_events.get(job_id)
        if job is None:
            raise KeyError(f"job {job_id!r} not found")
        if job.status == DownloadStatus.COMPLETE:
            return job
        if job.status == DownloadStatus.CANCELLED:
            return job

        source = job.source
        if source.kind == SourceKind.LOCAL_PATH:
            return self._run_local_job(job)

        url = _resolve_source_url(source)
        # The partial file is authoritative after a process restart; state
        # snapshots may lag behind its last progress callback.
        resume_from = job.dest_path.stat().st_size if job.dest_path.exists() else 0
        job.status = DownloadStatus.DOWNLOADING
        job.started_at = time.time()
        job.attempts += 1
        self._save_state()

        def on_progress(progress_job_id: str, downloaded: int, total: Optional[int], speed: float) -> None:
            with self._lock:
                job.bytes_downloaded = downloaded
                job.total_bytes = total
            if self._progress_cb:
                self._progress_cb(progress_job_id, downloaded, total, speed)

        for attempt in range(_MAX_ATTEMPTS):
            try:
                written = _download_file(
                    url,
                    job.dest_path,
                    resume_from=resume_from,
                    progress_cb=on_progress,
                    job_id=job_id,
                    cancel_event=ev,
                )
                job.bytes_downloaded = written
                break
            except RuntimeError as exc:
                if "cancelled" in str(exc):
                    job.status = DownloadStatus.CANCELLED
                    job.finished_at = time.time()
                    self._save_state()
                    return job
                if attempt == _MAX_ATTEMPTS - 1:
                    job.status = DownloadStatus.FAILED
                    job.error = str(exc)
                    job.finished_at = time.time()
                    self._save_state()
                    return job
                resume_from = job.dest_path.stat().st_size if job.dest_path.exists() else 0
            except Exception as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    job.status = DownloadStatus.FAILED
                    job.error = type(exc).__name__ + ": " + str(exc)[:200]
                    job.finished_at = time.time()
                    self._save_state()
                    return job
                time.sleep(2 ** attempt)
                resume_from = job.dest_path.stat().st_size if job.dest_path.exists() else 0

        if ev and ev.is_set():
            job.status = DownloadStatus.CANCELLED
            job.finished_at = time.time()
            self._save_state()
            return job

        actual_bytes = job.dest_path.stat().st_size if job.dest_path.exists() else 0
        job.bytes_downloaded = actual_bytes
        if source.expected_bytes is not None and actual_bytes != source.expected_bytes:
            job.status = DownloadStatus.FAILED
            job.error = f"size mismatch: expected {source.expected_bytes} bytes, received {actual_bytes}"
            job.finished_at = time.time()
            self._save_state()
            return job

        # Verify
        if source.expected_sha256:
            job.status = DownloadStatus.VERIFYING
            self._save_state()
            if not _verify_sha256(job.dest_path, source.expected_sha256):
                job.status = DownloadStatus.FAILED
                job.error = "SHA-256 mismatch"
                job.finished_at = time.time()
                self._save_state()
                return job

        if ev and ev.is_set():
            job.status = DownloadStatus.CANCELLED
            job.finished_at = time.time()
            self._save_state()
            return job

        job.status = DownloadStatus.COMPLETE
        job.finished_at = time.time()
        self._save_state()
        return job

    def run_until_complete(self, job_id: str) -> DownloadJob:
        """Alias for run_job — blocks until the job finishes."""
        return self.run_job(job_id)

    def _run_local_job(self, job: DownloadJob) -> DownloadJob:
        job.status = DownloadStatus.DOWNLOADING
        job.started_at = time.time()
        job.attempts += 1
        self._save_state()
        src = Path(job.source.uri)
        if not src.exists():
            job.status = DownloadStatus.FAILED
            job.error = f"local path not found: {src}"
            job.finished_at = time.time()
            self._save_state()
            return job
        job.dest_path.parent.mkdir(parents=True, exist_ok=True)
        if job.dest_path.resolve() != src.resolve():
            shutil.copy2(src, job.dest_path)
        job.bytes_downloaded = job.dest_path.stat().st_size
        if job.source.expected_bytes is not None and job.bytes_downloaded != job.source.expected_bytes:
            job.status = DownloadStatus.FAILED
            job.error = f"size mismatch: expected {job.source.expected_bytes} bytes, received {job.bytes_downloaded}"
            job.finished_at = time.time()
            self._save_state()
            return job
        if job.source.expected_sha256:
            if not _verify_sha256(job.dest_path, job.source.expected_sha256):
                job.status = DownloadStatus.FAILED
                job.error = "SHA-256 mismatch"
                job.finished_at = time.time()
                self._save_state()
                return job
        job.status = DownloadStatus.COMPLETE
        job.finished_at = time.time()
        self._save_state()
        return job

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_state(self) -> None:
        data = [job.to_dict() for job in self._jobs.values()]
        tmp = self._state_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self._state_file)
        except OSError:
            pass

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for record in data:
            try:
                status = DownloadStatus(record.get("status", "pending"))
                if status in (DownloadStatus.DOWNLOADING, DownloadStatus.VERIFYING):
                    status = DownloadStatus.PENDING
                source = ModelSource(
                    kind=SourceKind(record.get("source_kind", "direct_url")),
                    uri=record.get("source_uri", ""),
                    expected_sha256=record.get("source_expected_sha256"),
                    expected_bytes=record.get("source_expected_bytes"),
                    priority=record.get("source_priority", 0),
                )
                job = DownloadJob(
                    job_id=record["job_id"],
                    model_id=record["model_id"],
                    source=source,
                    dest_path=Path(record["dest_path"]),
                    status=status,
                    bytes_downloaded=record.get("bytes_downloaded", 0),
                    total_bytes=record.get("total_bytes"),
                    error=record.get("error"),
                    started_at=record.get("started_at"),
                    finished_at=record.get("finished_at"),
                    attempts=record.get("attempts", 0),
                )
                self._jobs[job.job_id] = job
                self._cancel_events[job.job_id] = threading.Event()
            except (KeyError, ValueError):
                continue

    def close(self) -> None:
        """Cancel all running jobs and flush state."""
        for ev in self._cancel_events.values():
            ev.set()
        self._save_state()


# ---------------------------------------------------------------------------
# Built-in family registry entries (Phase 1 families, no actual SHA-256s yet)
# ---------------------------------------------------------------------------

def default_registry_entries() -> Tuple[ModelRegistryEntry, ...]:
    """Return ModelRegistryEntry stubs for all Phase 1 admitted families.

    Source URIs are placeholders — replace expected_sha256 and expected_bytes
    with real values once checkpoints are available.
    """
    def _hf(repo: str, file: str, sha: Optional[str] = None,
            size: Optional[int] = None) -> ModelSource:
        return ModelSource(
            kind=SourceKind.HUGGINGFACE,
            uri=f"{repo}/{file}@main",
            expected_sha256=sha,
            expected_bytes=size,
            priority=10,
        )

    return (
        ModelRegistryEntry(
            model_id="mixtral-8x7b-v0.1-fp8",
            family="mixtral",
            sources=(_hf("mistralai/Mixtral-8x7B-v0.1", "model.safetensors"),),
            description="Mixtral 8×7B MoE, FP8 quantised",
        ),
        ModelRegistryEntry(
            model_id="olmoe-1b-7b-0924-fp8",
            family="olmoe",
            sources=(_hf("allenai/OLMoE-1B-7B-0924", "model.safetensors"),),
            description="OLMoE 1B/7B September 2024, FP8",
        ),
        ModelRegistryEntry(
            model_id="qwen3-30b-a3b-fp8",
            family="qwen35",
            sources=(_hf("Qwen/Qwen3-30B-A3B", "model.safetensors"),),
            description="Qwen3 30B MoE (3B active), FP8",
        ),
        ModelRegistryEntry(
            model_id="qwen3-235b-a22b-fp8",
            family="qwen4",
            sources=(_hf("Qwen/Qwen3-235B-A22B", "model.safetensors"),),
            description="Qwen3 235B MoE (22B active), FP8",
        ),
        ModelRegistryEntry(
            model_id="glm5-next-fp8",
            family="glm5-next",
            sources=(_hf("THUDM/GLM-4", "model.safetensors"),),
            description="GLM-5-Next dense, FP8",
        ),
        ModelRegistryEntry(
            model_id="inkling-v1-fp8",
            family="inkling",
            sources=(_hf("inkling-ai/inkling-v1", "model.safetensors"),),
            description="Inkling V1 MTP, FP8",
        ),
        ModelRegistryEntry(
            model_id="glm-dsa-fp8",
            family="glm-dsa",
            sources=(_hf("THUDM/GLM-DSA", "model.safetensors"),),
            description="GLM-DSA reasoning, FP8",
        ),
        ModelRegistryEntry(
            model_id="kimi-k3-fp8",
            family="kimi-k3",
            sources=(_hf("moonshotai/Kimi-K3", "model.safetensors"),),
            description="Kimi K3 MoE, FP8",
        ),
        ModelRegistryEntry(
            model_id="deepseek-v41-bf16",
            family="deepseek-v41",
            sources=(_hf("deepseek-ai/DeepSeek-V4.1", "model.safetensors"),),
            description="DeepSeek V4.1 MoE, BF16",
        ),
    )
