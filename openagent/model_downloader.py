"""Model downloader and cache manager for production ML models."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Download and cache ML models with versioning support."""

    # Production models: ~2GB quantized formats
    AVAILABLE_MODELS = {
        "tinyllama-q4": {
            "name": "TinyLlama-1.1B-Chat (Q4)",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size_mb": 680,
            "format": "gguf",
            "quantization": "Q4_K_M",
            "checksummed": False,
        },
        "mistral-7b-q4": {
            "name": "Mistral-7B-Instruct (Q4)",
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "size_mb": 4500,
            "format": "gguf",
            "quantization": "Q4_K_M",
            "checksummed": False,
        },
        "neural-chat-q4": {
            "name": "Neural-Chat-7B (Q4)",
            "url": "https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf",
            "size_mb": 4200,
            "format": "gguf",
            "quantization": "Q4_K_M",
            "checksummed": False,
        },
    }

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or "models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        """Load or create manifest of cached models."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def get_model_path(self, model_id: str) -> Path | None:
        """Get path to cached model, or None if not cached."""
        if model_id not in self.AVAILABLE_MODELS:
            return None

        model_info = self.AVAILABLE_MODELS[model_id]
        model_path = self.cache_dir / f"{model_id}.gguf"

        if model_path.exists():
            return model_path

        return None

    def download_model(self, model_id: str, on_progress=None) -> Path | None:
        """Download model from HuggingFace and cache it."""
        if model_id not in self.AVAILABLE_MODELS:
            logger.error(f"Unknown model: {model_id}")
            return None

        model_info = self.AVAILABLE_MODELS[model_id]
        model_path = self.cache_dir / f"{model_id}.gguf"

        # Check if already cached
        if model_path.exists():
            logger.info(f"Model {model_id} already cached at {model_path}")
            return model_path

        logger.info(f"Downloading {model_info['name']} from {model_info['url']}")

        try:
            # Download with progress tracking
            downloaded_size = 0
            expected_size = model_info["size_mb"] * 1024 * 1024

            def report_progress(block_num, block_size, total_size):
                nonlocal downloaded_size
                downloaded_size = block_num * block_size
                if on_progress:
                    on_progress(downloaded_size, total_size)
                if downloaded_size % (100 * 1024 * 1024) == 0:  # Every 100MB
                    pct = (downloaded_size / total_size * 100) if total_size else 0
                    logger.info(f"Download progress: {pct:.1f}%")

            # Simple download without urlretrieve to avoid blocking
            logger.info(f"Starting download of {model_info['name']} ({model_info['size_mb']}MB)")

            # Save to temp file first
            temp_path = model_path.with_suffix(".tmp")

            with urlopen(model_info["url"], timeout=300) as response:
                with open(temp_path, "wb") as f:
                    chunk_size = 8 * 1024 * 1024  # 8MB chunks
                    downloaded = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = (downloaded / expected_size * 100) if expected_size else 0
                        logger.info(f"Download {model_id}: {pct:.1f}% ({downloaded / 1024 / 1024:.0f}MB)")

            # Verify and move to final location
            if temp_path.exists():
                shutil.move(str(temp_path), str(model_path))
                logger.info(f"Model cached at {model_path}")

                # Update manifest
                self.manifest[model_id] = {
                    "path": str(model_path),
                    "size_mb": model_info["size_mb"],
                    "format": model_info["format"],
                    "quantization": model_info["quantization"],
                    "downloaded_at": str(Path(__file__).stat().st_mtime),
                }
                self._save_manifest()

                return model_path

        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            if Path(temp_path).exists():
                Path(temp_path).unlink()
            return None

    def get_cached_models(self) -> dict[str, Any]:
        """List all cached models."""
        cached = {}
        for model_id, info in self.AVAILABLE_MODELS.items():
            path = self.get_model_path(model_id)
            cached[model_id] = {
                **info,
                "cached": path is not None,
                "path": str(path) if path else None,
            }
        return cached

    def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """Get info about a model."""
        if model_id not in self.AVAILABLE_MODELS:
            return None

        info = self.AVAILABLE_MODELS[model_id].copy()
        path = self.get_model_path(model_id)
        info["cached"] = path is not None
        info["path"] = str(path) if path else None

        return info

    def delete_model(self, model_id: str) -> bool:
        """Delete cached model."""
        path = self.get_model_path(model_id)
        if path and path.exists():
            path.unlink()
            if model_id in self.manifest:
                del self.manifest[model_id]
            self._save_manifest()
            return True
        return False
