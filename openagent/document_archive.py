"""Document archive lifecycle management."""

from __future__ import annotations

from typing import Any


class DocumentArchiveService:
    """Manages document archival based on canonical control."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def archive_document(self, document_id: str, project: str, reason: str = "") -> dict[str, Any]:
        """Archive a document when canonical control requires it."""
        if not reason:
            raise ValueError("Archive reason is required.")

        with self.store.session() as connection:
            connection.execute(
                "UPDATE documents SET archived_at = ?, archive_reason = ? WHERE id = ? AND project = ?",
                (self._now(), reason.strip(), document_id, project),
            )

        return {"document_id": document_id, "archived": True, "reason": reason.strip()}

    def restore_document(self, document_id: str, project: str) -> dict[str, Any]:
        """Restore an archived document."""
        with self.store.session() as connection:
            connection.execute(
                "UPDATE documents SET archived_at = '', archive_reason = '' WHERE id = ? AND project = ?",
                (document_id, project),
            )

        return {"document_id": document_id, "restored": True}

    def list_archived(self, project: str, limit: int = 100) -> list[dict[str, Any]]:
        """List archived documents."""
        limit = max(1, min(limit, 500))
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE project = ? AND archived_at != '' ORDER BY archived_at DESC LIMIT ?",
                (project, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_archive_policy(self) -> dict[str, Any]:
        """Get document archive policy."""
        return {
            "name": "Document Archive Policy",
            "conditions": [
                "Canonical control explicitly requires archival",
                "Document is indexed in semantic collection",
                "Retention period has expired",
                "User requests explicit archival",
            ],
            "scope": "Optional when product workflows require it",
            "retention": "Archived documents remain in audit trail",
        }

    def _now(self) -> str:
        from .models import utc_now
        return utc_now()
