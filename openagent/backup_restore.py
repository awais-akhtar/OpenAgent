"""Local backup, restore, and disaster-recovery helpers."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class BackupRestoreManager:
    """Create auditable local backups and validate restores before applying them."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.backup_dir = self.store.data_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, *, project: str = "default", include_media: bool = True) -> dict[str, Any]:
        created_at = utc_now()
        backup_id = created_at.replace(":", "").replace("+", "Z").replace("-", "")
        archive_path = self.backup_dir / f"openagent-backup-{backup_id}.zip"
        manifest = {
            "backup_id": backup_id,
            "created_at": created_at,
            "project": project,
            "database": self.store.db_path.name,
            "include_media": include_media,
            "health": self.store.health(),
            "privacy_boundary": "Backup archive is written to the local OpenAgent data directory.",
        }
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
            if self.store.db_path.exists():
                archive.write(self.store.db_path, f"database/{self.store.db_path.name}")
            media_dir = self.store.data_dir / "media"
            if include_media and media_dir.exists():
                for path in media_dir.rglob("*"):
                    if path.is_file():
                        archive.write(path, f"media/{path.relative_to(media_dir).as_posix()}")
        result = self._backup_record(archive_path)
        self.store.add_audit_event(
            AuditEvent(
                actor="backup-restore",
                action="backup.create",
                resource=f"backup:{archive_path.name}",
                project=project,
                metadata={"include_media": include_media, "bytes": result["size_bytes"]},
            )
        )
        return result

    def backups(self) -> list[dict[str, Any]]:
        return [
            self._backup_record(path)
            for path in sorted(self.backup_dir.glob("openagent-backup-*.zip"), key=lambda item: item.stat().st_mtime, reverse=True)
        ]

    def restore_dry_run(self, archive_path: str | Path) -> dict[str, Any]:
        path = self._resolve_backup_path(archive_path)
        manifest, names = self._inspect_archive(path)
        conflicts = []
        database_entries = [name for name in names if name.startswith("database/")]
        media_entries = [name for name in names if name.startswith("media/")]
        if not database_entries:
            conflicts.append("archive_missing_database")
        if self.store.db_path.exists():
            conflicts.append("current_database_would_be_replaced")
        return {
            "status": "ready" if database_entries else "invalid",
            "archive": str(path),
            "manifest": manifest,
            "database_entries": database_entries,
            "media_entries": len(media_entries),
            "conflicts": conflicts,
            "will_write": False,
            "privacy_boundary": "Dry-run reads only the local backup archive and does not modify current data.",
        }

    def restore(self, archive_path: str | Path, *, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            raise ValueError("confirm=true is required to restore a backup.")
        path = self._resolve_backup_path(archive_path)
        dry_run = self.restore_dry_run(path)
        if dry_run["status"] != "ready":
            raise ValueError("Backup archive is not restorable.")
        safety_copy = self.backup_dir / f"pre-restore-{utc_now().replace(':', '').replace('+', 'Z')}.sqlite3"
        if self.store.db_path.exists():
            shutil.copy2(self.store.db_path, safety_copy)
        with zipfile.ZipFile(path) as archive:
            db_entries = [name for name in archive.namelist() if name.startswith("database/")]
            with archive.open(db_entries[0]) as source, self.store.db_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        self.store.add_audit_event(
            AuditEvent(
                actor="backup-restore",
                action="backup.restore",
                resource=f"backup:{path.name}",
                project=str(dry_run["manifest"].get("project", "default")),
                metadata={"safety_copy": str(safety_copy) if safety_copy.exists() else ""},
            )
        )
        return {
            "restored": True,
            "archive": str(path),
            "safety_copy": str(safety_copy) if safety_copy.exists() else "",
        }

    def _resolve_backup_path(self, archive_path: str | Path) -> Path:
        path = Path(archive_path)
        if not path.is_absolute():
            path = self.backup_dir / path
        path = path.resolve()
        if not path.exists() or path.suffix.lower() != ".zip":
            raise FileNotFoundError(f"Backup archive not found: {archive_path}")
        return path

    def _inspect_archive(self, path: Path) -> tuple[dict[str, Any], list[str]]:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if "manifest.json" not in names:
                raise ValueError("Backup archive is missing manifest.json.")
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        return manifest, names

    def _backup_record(self, path: Path) -> dict[str, Any]:
        try:
            manifest, names = self._inspect_archive(path)
        except Exception as exc:  # noqa: BLE001 - corrupted archives should still be visible
            manifest = {"error": str(exc)}
            names = []
        return {
            "name": path.name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "manifest": manifest,
            "entry_count": len(names),
        }
