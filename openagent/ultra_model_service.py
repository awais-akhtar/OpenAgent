"""G22: Ultra Model service with real model downloads and session management."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from .model_downloader import ModelDownloader
from .model_serve_manager import ModelServeManager

logger = logging.getLogger(__name__)


class UltraModelService:
    """Manage ultra model sessions with downloads, serving, and recovery."""

    def __init__(self, models_dir: str | None = None):
        self.downloader = ModelDownloader(models_dir or "models")
        self.serve_manager = ModelServeManager(models_dir)
        self.sessions: dict[str, dict[str, Any]] = {}
        self.session_metadata: dict[str, dict[str, Any]] = {}
        # Keep the manifest beside the models; a bare relative name would
        # drop this file into whatever directory the server was started from.
        self.manifest_file = Path(models_dir or "models") / "ultra_model_manifest.json"
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load persisted sessions."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file) as f:
                    data = json.load(f)
                    self.sessions = data.get("sessions", {})
            except Exception as e:
                logger.error(f"Failed to load manifest: {e}")

    def _save_manifest(self) -> None:
        """Save sessions to manifest."""
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_file, "w") as f:
            json.dump({"sessions": self.sessions}, f, indent=2)

    def get_available_models(self, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Get all available models and their cache status."""
        # Note: downloader is global, models are project-aware in storage
        return self.downloader.get_cached_models()

    def download_model(self, model_id: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Download a model (may take time for 2GB+ downloads)."""
        logger.info(f"Starting download of {model_id} (project={project}, owner={owner})")

        try:
            model_info = self.downloader.get_model_info(model_id)
            if not model_info:
                return {"error": f"Unknown model: {model_id}"}

            # Check if already cached
            if model_info["cached"]:
                return {
                    "status": "cached",
                    "model_id": model_id,
                    "path": model_info["path"],
                    "size_mb": model_info["size_mb"],
                }

            # Start download
            path = self.downloader.download_model(model_id)

            if path:
                return {
                    "status": "downloaded",
                    "model_id": model_id,
                    "path": str(path),
                    "size_mb": model_info["size_mb"],
                }
            else:
                return {"error": f"Failed to download {model_id}"}

        except Exception as e:
            logger.error(f"Download error: {e}")
            return {"error": str(e)}

    def create_session(
        self, model_id: str, session_name: str | None = None,
        project: str = "default", owner: str = "default"
    ) -> dict[str, Any]:
        """Create a new ultra model session."""
        try:
            # Check model is available
            model_info = self.downloader.get_model_info(model_id)
            if not model_info:
                return {"error": f"Unknown model: {model_id}"}

            # Download if not cached
            if not model_info["cached"]:
                logger.info(f"Model {model_id} not cached, attempting download...")
                download_result = self.download_model(model_id, project, owner)
                if "error" in download_result:
                    return download_result

            # Get model path
            model_path = self.downloader.get_model_path(model_id)
            if not model_path:
                return {"error": f"Could not find model path for {model_id}"}

            # Create serve plan
            serve_plan = self.serve_manager.create_serve_plan(
                model_id=model_id, model_path=str(model_path),
                project=project, owner=owner
            )

            # Create session record
            session_id = serve_plan["session_id"]
            self.sessions[session_id] = {
                "session_id": session_id,
                "model_id": model_id,
                "name": session_name or f"UltraModel-{model_id}",
                "status": "ready",
                "serve_plan": serve_plan,
                "created_at": serve_plan["created_at"],
                "project": project,
                "owner": owner,
            }
            self.session_metadata[session_id] = {"project": project, "owner": owner}

            self._save_manifest()

            return self.sessions[session_id]

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return {"error": str(e)}

    def launch_session(self, session_id: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Launch a session (start serving the model)."""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}

        meta = self.session_metadata.get(session_id, {})
        if meta.get("project") != project or meta.get("owner") != owner:
            return {"error": "Session does not belong to this project/owner"}

        try:
            # Start the serve process
            serve_result = self.serve_manager.start_serve(session_id, project, owner)

            if "error" in serve_result:
                self.sessions[session_id]["status"] = "failed"
            else:
                self.sessions[session_id]["status"] = serve_result.get("status", "running")
                self.sessions[session_id]["serve_session"] = serve_result

            self._save_manifest()
            return self.sessions[session_id]

        except Exception as e:
            logger.error(f"Failed to launch session: {e}")
            self.sessions[session_id]["status"] = "failed"
            self._save_manifest()
            return {"error": str(e)}

    def stop_session(self, session_id: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Stop a running session."""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}

        meta = self.session_metadata.get(session_id, {})
        if meta.get("project") != project or meta.get("owner") != owner:
            return {"error": "Session does not belong to this project/owner"}

        try:
            # Stop the serve process
            serve_result = self.serve_manager.stop_serve(session_id, project, owner)

            self.sessions[session_id]["status"] = serve_result.get("status", "stopped")
            self.sessions[session_id]["serve_session"] = serve_result

            self._save_manifest()
            return self.sessions[session_id]

        except Exception as e:
            logger.error(f"Failed to stop session: {e}")
            return {"error": str(e)}

    def reload_session(self, session_id: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Reload a session (stop and restart)."""
        stop_result = self.stop_session(session_id, project, owner)
        if "error" in stop_result:
            return stop_result

        # Wait a moment for cleanup
        import time

        time.sleep(1)

        return self.launch_session(session_id, project, owner)

    def get_session(self, session_id: str, project: str = "default", owner: str = "default") -> dict[str, Any] | None:
        """Get session details."""
        if session_id not in self.sessions:
            return None
        meta = self.session_metadata.get(session_id, {})
        if meta.get("project") != project or meta.get("owner") != owner:
            return None
        # Refresh serve status
        serve_status = self.serve_manager.get_session(session_id, project, owner)
        if serve_status:
            self.sessions[session_id]["serve_session"] = serve_status
        return self.sessions[session_id]

    def list_sessions(self, project: str = "default", owner: str = "default") -> list[dict[str, Any]]:
        """List all sessions."""
        result = []
        for session_id, session in self.sessions.items():
            meta = self.session_metadata.get(session_id, {})
            if meta.get("project") != project or meta.get("owner") != owner:
                continue
            # Refresh serve status
            serve_status = self.serve_manager.get_session(session_id, project, owner)
            if serve_status:
                session["serve_session"] = serve_status
            result.append(session)
        return result

    def get_readiness_status(self, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Check infrastructure readiness."""
        models = self.get_available_models()
        sessions = self.list_sessions()

        return {
            "models_available": len(models),
            "models_cached": sum(1 for m in models.values() if m.get("cached")),
            "sessions_total": len(sessions),
            "sessions_running": sum(1 for s in sessions if s.get("status") == "running"),
            "downloader_ready": True,
            "serve_manager_ready": True,
            "models": models,
            "sample_command": "llama-server -m model.gguf -ngl 999 --host 127.0.0.1 --port 8000",
        }

    def calibrate_for_hardware(self, model_id: str, project: str = "default", owner: str = "default") -> dict[str, Any]:
        """Suggest hardware settings for a model."""
        model_info = self.downloader.get_model_info(model_id)
        if not model_info:
            return {"error": f"Unknown model: {model_id}"}

        size_mb = model_info["size_mb"]

        # Hardware recommendations based on model size
        if size_mb < 1000:
            return {
                "model_id": model_id,
                "model_size_mb": size_mb,
                "recommended_ram_gb": 4,
                "recommended_vram_gb": 2,
                "context_size": 2048,
                "gpu_layers": 20,
                "batch_size": 8,
            }
        elif size_mb < 5000:
            return {
                "model_id": model_id,
                "model_size_mb": size_mb,
                "recommended_ram_gb": 8,
                "recommended_vram_gb": 6,
                "context_size": 4096,
                "gpu_layers": 40,
                "batch_size": 16,
            }
        else:
            return {
                "model_id": model_id,
                "model_size_mb": size_mb,
                "recommended_ram_gb": 16,
                "recommended_vram_gb": 12,
                "context_size": 8192,
                "gpu_layers": 60,
                "batch_size": 32,
            }
