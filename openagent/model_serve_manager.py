"""G20: Model serve/stop lifecycle with PID tracking and verification."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ModelServeSession:
    """Represents a running model serve session with process tracking."""

    def __init__(self, session_id: str, model_id: str, model_path: str, project: str = "default", owner: str = "default"):
        self.session_id = session_id
        self.model_id = model_id
        self.model_path = model_path
        self.pid: int | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self.status = "planned"  # planned, starting, running, stopped, failed
        self.logs: list[str] = []
        self.process_identity: dict[str, Any] = {}
        self.project = project
        self.owner = owner

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "pid": self.pid,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "status": self.status,
            "logs": self.logs[-20:],  # Last 20 log lines
            "process_identity": self.process_identity,
            "project": self.project,
            "owner": self.owner,
        }


class ModelServeManager:
    """Manage model serve sessions with real process lifecycle."""

    def __init__(self, models_dir: str | None = None):
        self.models_dir = Path(models_dir or "models")
        self.sessions: dict[str, ModelServeSession] = {}
        self.session_metadata: dict[str, dict[str, Any]] = {}
        # Keep state beside the models it describes; a bare relative name would
        # drop this file into whatever directory the server was started from.
        self.state_file = self.models_dir / "model_sessions_state.json"
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted session state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    for session_dict in state.get("sessions", []):
                        session = ModelServeSession(
                            session_dict["session_id"],
                            session_dict["model_id"],
                            session_dict["model_path"],
                        )
                        session.pid = session_dict.get("pid")
                        session.status = session_dict.get("status", "stopped")
                        session.created_at = session_dict.get("created_at", time.time())
                        self.sessions[session.session_id] = session
            except Exception as e:
                logger.error(f"Failed to load session state: {e}")

    def _save_state(self) -> None:
        """Save session state to disk."""
        state = {"sessions": [s.to_dict() for s in self.sessions.values()]}
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def create_serve_plan(
        self, model_id: str, model_path: str, host: str = "127.0.0.1", port: int = 8000,
        project: str = "default", owner: str = "default"
    ) -> dict[str, Any]:
        """Create a plan for serving a model (no actual process started)."""
        session_id = str(uuid4())
        session = ModelServeSession(session_id, model_id, model_path, project, owner)
        session.status = "planned"

        self.sessions[session_id] = session
        self.session_metadata[session_id] = {"project": project, "owner": owner}
        self._save_state()

        return {
            "session_id": session_id,
            "model_id": model_id,
            "status": "planned",
            "command": f"llama-server -m {model_path} -ngl 999 --host {host} --port {port}",
            "created_at": session.created_at,
            "project": project,
            "owner": owner,
        }

    def start_serve(self, session_id: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Start actual model serving process (with permission)."""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}

        meta = self.session_metadata.get(session_id, {})
        if meta.get("project") != project or meta.get("owner") != owner:
            return {"error": "Session does not belong to this project/owner"}

        session = self.sessions[session_id]

        # Verify model file exists
        if not Path(session.model_path).exists():
            session.status = "failed"
            session.logs.append(f"Model file not found: {session.model_path}")
            self._save_state()
            return {"error": f"Model file not found: {session.model_path}"}

        try:
            session.status = "starting"
            session.logs.append("Starting model serve process...")

            # Build command - using llama-server which can be mocked for testing
            cmd = [
                "llama-server",
                "-m",
                session.model_path,
                "-ngl",
                "999",  # GPU layers
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ]

            # Try to start process
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                session.pid = process.pid
                session.started_at = time.time()
                # Recorded from what was actually launched. An earlier version
                # also shelled out to tasklist/ps here and then threw the output
                # away, and its Windows arguments were malformed besides.
                session.process_identity = {
                    "pid": process.pid,
                    "executable": cmd[0],
                    "started_at": session.started_at,
                }
                session.status = "running"
                session.logs.append(f"Process started with PID {process.pid}")

            except FileNotFoundError:
                # No runtime, so nothing is serving. This used to report the
                # session as "ready" with a pid of 0 and call it mock mode, so
                # the screen showed a model as ready to answer when no process
                # existed. Say it is unavailable instead.
                session.status = "unavailable"
                session.pid = None
                session.process_identity = {}
                session.logs.append(f"{cmd[0]} is not installed, so no model is being served")
                session.logs.append(f"Install {cmd[0]} to serve a downloaded model")

        except Exception as e:
            session.status = "failed"
            session.logs.append(f"Failed to start: {str(e)}")
            logger.error(f"Failed to start serve for {session_id}: {e}")

        self._save_state()
        return session.to_dict()

    def stop_serve(self, session_id: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Stop model serving process with PID verification."""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}

        meta = self.session_metadata.get(session_id, {})
        if meta.get("project") != project or meta.get("owner") != owner:
            return {"error": "Session does not belong to this project/owner"}

        session = self.sessions[session_id]

        if session.status == "stopped":
            return {"already_stopped": True, **session.to_dict()}

        try:
            if session.pid and session.pid > 0:
                session.logs.append(f"Stopping process with PID {session.pid}")

                try:
                    if os.name == "nt":
                        # Windows: use taskkill
                        subprocess.run(["taskkill", "/PID", str(session.pid), "/T", "/F"], timeout=5)
                    else:
                        # Unix: use kill
                        os.kill(session.pid, 15)  # SIGTERM
                        time.sleep(0.5)
                        # Force kill if still running
                        try:
                            os.kill(session.pid, 9)  # SIGKILL
                        except ProcessLookupError:
                            pass

                    session.logs.append(f"Process {session.pid} terminated")

                except Exception as e:
                    session.logs.append(f"Error terminating process: {str(e)}")
                    logger.warning(f"Failed to terminate PID {session.pid}: {e}")

            session.status = "stopped"
            session.stopped_at = time.time()
            session.pid = None

        except Exception as e:
            session.logs.append(f"Stop failed: {str(e)}")
            logger.error(f"Failed to stop serve {session_id}: {e}")

        self._save_state()
        return session.to_dict()

    def get_session(self, session_id: str, project: str = "default", owner: str = "default") -> dict[str, Any] | None:
        """Get session status."""
        if session_id not in self.sessions:
            return None
        meta = self.session_metadata.get(session_id, {})
        if meta.get("project") != project or meta.get("owner") != owner:
            return None
        return self.sessions[session_id].to_dict()

    def list_sessions(self, project: str = "default", owner: str = "default") -> list[dict[str, Any]]:
        """List all sessions."""
        return [s.to_dict() for session_id, s in self.sessions.items()
                if self.session_metadata.get(session_id, {}).get("project") == project
                and self.session_metadata.get(session_id, {}).get("owner") == owner]

    def cleanup_stopped_sessions(self, older_than_seconds: int = 3600) -> int:
        """Clean up stopped sessions older than threshold."""
        now = time.time()
        to_delete = []

        for session_id, session in self.sessions.items():
            if session.status == "stopped" and session.stopped_at:
                age = now - session.stopped_at
                if age > older_than_seconds:
                    to_delete.append(session_id)

        for session_id in to_delete:
            del self.sessions[session_id]

        if to_delete:
            self._save_state()

        return len(to_delete)
