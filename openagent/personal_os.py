"""Personal local app surface for notes, drafts, presets, gallery, and compare."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class PersonalOSError(RuntimeError):
    """Raised when a personal OS action is invalid."""


class PersonalOS:
    """Local-first personal productivity surface.

    This manager groups small but important app features that users expect in a
    mature local assistant: prompt/model presets, reusable signatures, editable
    drafts with versions, notes with due dates, gallery metadata, blind response
    comparison, and portable local export/import.
    """

    def __init__(self, store: OpenAgentStore):
        self.store = store
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS personal_presets (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT '',
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'chat',
                    instructions TEXT NOT NULL DEFAULT '',
                    provider TEXT NOT NULL DEFAULT 'auto',
                    model TEXT NOT NULL DEFAULT '',
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_personal_presets_scope
                    ON personal_presets(project, owner, kind, updated_at DESC);

                CREATE TABLE IF NOT EXISTS personal_signatures (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT '',
                    label TEXT NOT NULL,
                    content TEXT NOT NULL,
                    is_default INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_personal_signatures_scope
                    ON personal_signatures(project, owner, updated_at DESC);

                CREATE TABLE IF NOT EXISTS personal_drafts (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_personal_drafts_scope
                    ON personal_drafts(project, owner, updated_at DESC);

                CREATE TABLE IF NOT EXISTS personal_draft_versions (
                    id TEXT PRIMARY KEY,
                    draft_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_personal_draft_versions
                    ON personal_draft_versions(draft_id, version DESC);

                CREATE TABLE IF NOT EXISTS personal_notes (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    body TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    due_at TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_personal_notes_scope
                    ON personal_notes(project, owner, status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS personal_gallery_assets (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    media_type TEXT NOT NULL DEFAULT 'image',
                    source_path TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    favorite INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_personal_gallery_scope
                    ON personal_gallery_assets(project, owner, updated_at DESC);

                CREATE TABLE IF NOT EXISTS personal_compare_runs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT '',
                    prompt TEXT NOT NULL,
                    left_label TEXT NOT NULL,
                    left_output TEXT NOT NULL,
                    right_label TEXT NOT NULL,
                    right_output TEXT NOT NULL,
                    vote TEXT NOT NULL DEFAULT '',
                    winner TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_personal_compare_scope
                    ON personal_compare_runs(project, owner, updated_at DESC);
                """
            )

    def status(self, *, project: str = "default", owner: str = "") -> dict[str, Any]:
        return {
            "status": "ready",
            "project": project,
            "owner": owner,
            "counts": {
                "presets": len(self.presets(project=project, owner=owner)),
                "signatures": len(self.signatures(project=project, owner=owner)),
                "drafts": len(self.drafts(project=project, owner=owner)),
                "notes": len(self.notes(project=project, owner=owner)),
                "gallery": len(self.gallery(project=project, owner=owner)),
                "compares": len(self.compare_history(project=project, owner=owner)),
            },
            "privacy_boundary": "Personal OS records live in local SQLite only and are filtered by project and owner.",
        }

    def upsert_preset(
        self,
        *,
        name: str,
        project: str = "default",
        owner: str = "",
        kind: str = "chat",
        instructions: str = "",
        provider: str = "auto",
        model: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise PersonalOSError("Preset name is required.")
        now = utc_now()
        record_id = stable_id("preset", project, owner, kind, name)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO personal_presets (
                    id, project, owner, name, kind, instructions, provider, model, parameters_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    kind = excluded.kind,
                    instructions = excluded.instructions,
                    provider = excluded.provider,
                    model = excluded.model,
                    parameters_json = excluded.parameters_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record_id,
                    project,
                    owner,
                    name,
                    kind.strip() or "chat",
                    instructions,
                    provider.strip() or "auto",
                    model.strip(),
                    json.dumps(parameters or {}, sort_keys=True),
                    now,
                    now,
                ),
            )
        self._audit("personal.preset.upsert", record_id, project=project, owner=owner)
        return self.get_preset(record_id) or {}

    def get_preset(self, preset_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM personal_presets WHERE id = ?", (preset_id,)).fetchone()
        return preset_row(row) if row else None

    def presets(self, *, project: str = "default", owner: str = "", kind: str = "") -> list[dict[str, Any]]:
        if kind:
            sql = "SELECT * FROM personal_presets WHERE project = ? AND owner = ? AND kind = ? ORDER BY updated_at DESC"
            params: tuple[Any, ...] = (project, owner, kind)
        else:
            sql = "SELECT * FROM personal_presets WHERE project = ? AND owner = ? ORDER BY updated_at DESC"
            params = (project, owner)
        with self.store.session() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [preset_row(row) for row in rows]

    def upsert_signature(
        self,
        *,
        label: str,
        content: str,
        project: str = "default",
        owner: str = "",
        is_default: bool = False,
    ) -> dict[str, Any]:
        label = label.strip()
        content = content.strip()
        if not label or not content:
            raise PersonalOSError("Signature label and content are required.")
        record_id = stable_id("signature", project, owner, label)
        now = utc_now()
        with self.store.session() as connection:
            if is_default:
                connection.execute(
                    "UPDATE personal_signatures SET is_default = 0 WHERE project = ? AND owner = ?",
                    (project, owner),
                )
            connection.execute(
                """
                INSERT INTO personal_signatures (
                    id, project, owner, label, content, is_default, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    is_default = excluded.is_default,
                    updated_at = excluded.updated_at
                """,
                (record_id, project, owner, label, content, 1 if is_default else 0, now, now),
            )
        self._audit("personal.signature.upsert", record_id, project=project, owner=owner)
        return self.get_signature(record_id) or {}

    def get_signature(self, signature_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM personal_signatures WHERE id = ?", (signature_id,)).fetchone()
        return signature_row(row) if row else None

    def signatures(self, *, project: str = "default", owner: str = "") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM personal_signatures WHERE project = ? AND owner = ? ORDER BY is_default DESC, updated_at DESC",
                (project, owner),
            ).fetchall()
        return [signature_row(row) for row in rows]

    def save_draft(
        self,
        *,
        title: str,
        content: str,
        project: str = "default",
        owner: str = "",
        draft_id: str = "",
        status: str = "draft",
    ) -> dict[str, Any]:
        title = title.strip() or "Untitled draft"
        now = utc_now()
        record_id = draft_id.strip() or stable_id("draft", project, owner, title, now)
        current = self.get_draft(record_id)
        version = int(current["version"]) + 1 if current else 1
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO personal_drafts (
                    id, project, owner, title, content, status, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    status = excluded.status,
                    version = excluded.version,
                    updated_at = excluded.updated_at
                """,
                (record_id, project, owner, title, content, status.strip() or "draft", version, now, now),
            )
            connection.execute(
                """
                INSERT INTO personal_draft_versions (id, draft_id, version, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (stable_id("draft-version", record_id, str(version), now), record_id, version, content, now),
            )
        self._audit("personal.draft.save", record_id, project=project, owner=owner, metadata={"version": version})
        return self.get_draft(record_id) or {}

    def get_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM personal_drafts WHERE id = ?", (draft_id,)).fetchone()
        return draft_row(row) if row else None

    def drafts(self, *, project: str = "default", owner: str = "", status: str = "") -> list[dict[str, Any]]:
        if status:
            sql = "SELECT * FROM personal_drafts WHERE project = ? AND owner = ? AND status = ? ORDER BY updated_at DESC"
            params: tuple[Any, ...] = (project, owner, status)
        else:
            sql = "SELECT * FROM personal_drafts WHERE project = ? AND owner = ? ORDER BY updated_at DESC"
            params = (project, owner)
        with self.store.session() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [draft_row(row) for row in rows]

    def draft_versions(self, draft_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM personal_draft_versions WHERE draft_id = ? ORDER BY version DESC",
                (draft_id,),
            ).fetchall()
        return [draft_version_row(row) for row in rows]

    def save_note(
        self,
        *,
        title: str,
        body: str = "",
        project: str = "default",
        owner: str = "",
        note_id: str = "",
        tags: list[str] | None = None,
        due_at: str = "",
        status: str = "open",
    ) -> dict[str, Any]:
        title = title.strip()
        if not title:
            raise PersonalOSError("Note title is required.")
        now = utc_now()
        record_id = note_id.strip() or stable_id("note", project, owner, title, now)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO personal_notes (
                    id, project, owner, title, body, tags_json, due_at, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    body = excluded.body,
                    tags_json = excluded.tags_json,
                    due_at = excluded.due_at,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    record_id,
                    project,
                    owner,
                    title,
                    body,
                    json.dumps(clean_tags(tags), sort_keys=True),
                    due_at.strip(),
                    status.strip() or "open",
                    now,
                    now,
                ),
            )
        self._audit("personal.note.save", record_id, project=project, owner=owner, metadata={"status": status})
        return self.get_note(record_id) or {}

    def get_note(self, note_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM personal_notes WHERE id = ?", (note_id,)).fetchone()
        return note_row(row) if row else None

    def notes(self, *, project: str = "default", owner: str = "", status: str = "", query: str = "") -> list[dict[str, Any]]:
        rows = self._rows(
            table="personal_notes",
            project=project,
            owner=owner,
            status_column="status" if status else "",
            status=status,
        )
        notes = [note_row(row) for row in rows]
        if query.strip():
            q = query.lower().strip()
            notes = [
                note for note in notes
                if q in note["title"].lower() or q in note["body"].lower() or any(q in tag.lower() for tag in note["tags"])
            ]
        return notes

    def add_gallery_asset(
        self,
        *,
        title: str,
        project: str = "default",
        owner: str = "",
        media_type: str = "image",
        source_path: str = "",
        content: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        title = title.strip()
        if not title:
            raise PersonalOSError("Gallery title is required.")
        now = utc_now()
        content_hash = hashlib.sha256((content or source_path or title).encode("utf-8")).hexdigest()
        record_id = stable_id("gallery", project, owner, content_hash)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO personal_gallery_assets (
                    id, project, owner, title, media_type, source_path, content_hash, tags_json, favorite, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    media_type = excluded.media_type,
                    source_path = excluded.source_path,
                    tags_json = excluded.tags_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record_id,
                    project,
                    owner,
                    title,
                    media_type.strip() or "image",
                    source_path.strip(),
                    content_hash,
                    json.dumps(clean_tags(tags), sort_keys=True),
                    0,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                    now,
                ),
            )
        self._audit("personal.gallery.add", record_id, project=project, owner=owner, metadata={"media_type": media_type})
        return self.get_gallery_asset(record_id) or {}

    def get_gallery_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM personal_gallery_assets WHERE id = ?", (asset_id,)).fetchone()
        return gallery_row(row) if row else None

    def gallery(self, *, project: str = "default", owner: str = "", tag: str = "") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM personal_gallery_assets WHERE project = ? AND owner = ? ORDER BY updated_at DESC",
                (project, owner),
            ).fetchall()
        items = [gallery_row(row) for row in rows]
        if tag.strip():
            tag_lower = tag.lower().strip()
            items = [item for item in items if tag_lower in {t.lower() for t in item["tags"]}]
        return items

    def set_gallery_favorite(self, *, asset_id: str, favorite: bool, project: str = "default", owner: str = "") -> dict[str, Any]:
        asset = self.get_gallery_asset(asset_id)
        if not asset or asset["project"] != project or asset["owner"] != owner:
            raise PersonalOSError("Gallery asset not found.")
        with self.store.session() as connection:
            connection.execute(
                "UPDATE personal_gallery_assets SET favorite = ?, updated_at = ? WHERE id = ?",
                (1 if favorite else 0, utc_now(), asset_id),
            )
        self._audit("personal.gallery.favorite", asset_id, project=project, owner=owner, metadata={"favorite": favorite})
        return self.get_gallery_asset(asset_id) or {}

    def gallery_stats(self, *, project: str = "default", owner: str = "") -> dict[str, Any]:
        items = self.gallery(project=project, owner=owner)
        tags: dict[str, int] = {}
        media_types: dict[str, int] = {}
        for item in items:
            media_types[item["media_type"]] = media_types.get(item["media_type"], 0) + 1
            for tag in item["tags"]:
                tags[tag] = tags.get(tag, 0) + 1
        return {
            "total": len(items),
            "favorites": sum(1 for item in items if item["favorite"]),
            "media_types": media_types,
            "tags": dict(sorted(tags.items(), key=lambda kv: (-kv[1], kv[0]))),
        }

    def start_compare(
        self,
        *,
        prompt: str,
        left_output: str,
        right_output: str,
        project: str = "default",
        owner: str = "",
        left_label: str = "A",
        right_label: str = "B",
    ) -> dict[str, Any]:
        if not prompt.strip() or not left_output.strip() or not right_output.strip():
            raise PersonalOSError("Prompt and both outputs are required.")
        now = utc_now()
        record_id = stable_id("compare", project, owner, prompt, now)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO personal_compare_runs (
                    id, project, owner, prompt, left_label, left_output, right_label, right_output, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (record_id, project, owner, prompt, left_label, left_output, right_label, right_output, now, now),
            )
        self._audit("personal.compare.start", record_id, project=project, owner=owner)
        return self.get_compare(record_id) or {}

    def vote_compare(self, *, compare_id: str, vote: str, project: str = "default", owner: str = "") -> dict[str, Any]:
        compare = self.get_compare(compare_id)
        if not compare or compare["project"] != project or compare["owner"] != owner:
            raise PersonalOSError("Compare run not found.")
        vote = vote.strip().upper()
        if vote not in {"A", "B", "TIE"}:
            raise PersonalOSError("Vote must be A, B, or TIE.")
        winner = compare["left_label"] if vote == "A" else compare["right_label"] if vote == "B" else "tie"
        with self.store.session() as connection:
            connection.execute(
                "UPDATE personal_compare_runs SET vote = ?, winner = ?, updated_at = ? WHERE id = ?",
                (vote, winner, utc_now(), compare_id),
            )
        self._audit("personal.compare.vote", compare_id, project=project, owner=owner, metadata={"vote": vote, "winner": winner})
        return self.get_compare(compare_id) or {}

    def get_compare(self, compare_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM personal_compare_runs WHERE id = ?", (compare_id,)).fetchone()
        return compare_row(row) if row else None

    def compare_history(self, *, project: str = "default", owner: str = "") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM personal_compare_runs WHERE project = ? AND owner = ? ORDER BY updated_at DESC",
                (project, owner),
            ).fetchall()
        return [compare_row(row) for row in rows]

    def export_bundle(self, *, project: str = "default", owner: str = "") -> dict[str, Any]:
        drafts = []
        for draft in self.drafts(project=project, owner=owner):
            drafts.append({**draft, "versions": self.draft_versions(draft["id"])})
        bundle = {
            "schema": "openagent-personal-os-v1",
            "project": project,
            "owner": owner,
            "created_at": utc_now(),
            "presets": self.presets(project=project, owner=owner),
            "signatures": self.signatures(project=project, owner=owner),
            "drafts": drafts,
            "notes": self.notes(project=project, owner=owner),
            "gallery": self.gallery(project=project, owner=owner),
            "compares": self.compare_history(project=project, owner=owner),
        }
        self._audit("personal.export", stable_id("personal-export", project, owner, bundle["created_at"]), project=project, owner=owner)
        return bundle

    def import_bundle(self, bundle: dict[str, Any], *, project: str = "default", owner: str = "") -> dict[str, Any]:
        imported = {"presets": 0, "signatures": 0, "drafts": 0, "notes": 0, "gallery": 0, "compares": 0}
        for preset in as_list(bundle.get("presets")):
            self.upsert_preset(
                name=str(preset.get("name", "")),
                kind=str(preset.get("kind", "chat")),
                instructions=str(preset.get("instructions", "")),
                provider=str(preset.get("provider", "auto")),
                model=str(preset.get("model", "")),
                parameters=preset.get("parameters") if isinstance(preset.get("parameters"), dict) else {},
                project=project,
                owner=owner,
            )
            imported["presets"] += 1
        for sig in as_list(bundle.get("signatures")):
            self.upsert_signature(
                label=str(sig.get("label", "")),
                content=str(sig.get("content", "")),
                is_default=bool(sig.get("is_default")),
                project=project,
                owner=owner,
            )
            imported["signatures"] += 1
        for draft in as_list(bundle.get("drafts")):
            versions = sorted(as_list(draft.get("versions")), key=lambda item: int(item.get("version", 0) or 0))
            saved: dict[str, Any] | None = None
            if versions:
                for version in versions:
                    saved = self.save_draft(
                        title=str(draft.get("title", "Untitled draft")),
                        content=str(version.get("content", "")),
                        status=str(draft.get("status", "draft")),
                        draft_id=str(saved.get("id", "")) if saved else "",
                        project=project,
                        owner=owner,
                    )
            else:
                self.save_draft(
                    title=str(draft.get("title", "Untitled draft")),
                    content=str(draft.get("content", "")),
                    status=str(draft.get("status", "draft")),
                    project=project,
                    owner=owner,
                )
            imported["drafts"] += 1
        for note in as_list(bundle.get("notes")):
            self.save_note(
                title=str(note.get("title", "")),
                body=str(note.get("body", "")),
                tags=note.get("tags") if isinstance(note.get("tags"), list) else [],
                due_at=str(note.get("due_at", "")),
                status=str(note.get("status", "open")),
                project=project,
                owner=owner,
            )
            imported["notes"] += 1
        for item in as_list(bundle.get("gallery")):
            self.add_gallery_asset(
                title=str(item.get("title", "")),
                media_type=str(item.get("media_type", "image")),
                source_path=str(item.get("source_path", "")),
                content=str(item.get("content_hash", "")),
                tags=item.get("tags") if isinstance(item.get("tags"), list) else [],
                metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                project=project,
                owner=owner,
            )
            imported["gallery"] += 1
        for compare in as_list(bundle.get("compares")):
            imported_compare = self.start_compare(
                prompt=str(compare.get("prompt", "")),
                left_label=str(compare.get("left_label", "A")),
                left_output=str(compare.get("left_output", "")),
                right_label=str(compare.get("right_label", "B")),
                right_output=str(compare.get("right_output", "")),
                project=project,
                owner=owner,
            )
            if str(compare.get("vote", "")).strip():
                self.vote_compare(
                    compare_id=imported_compare["id"],
                    vote=str(compare.get("vote", "")),
                    project=project,
                    owner=owner,
                )
            imported["compares"] += 1
        self._audit("personal.import", stable_id("personal-import", project, owner, utc_now()), project=project, owner=owner, metadata=imported)
        return {"imported": imported, "privacy_boundary": "Import writes only local personal OS records."}

    def _rows(self, *, table: str, project: str, owner: str, status_column: str = "", status: str = "") -> list[Any]:
        if status_column and status:
            sql = f"SELECT * FROM {table} WHERE project = ? AND owner = ? AND {status_column} = ? ORDER BY updated_at DESC"
            params: tuple[Any, ...] = (project, owner, status)
        else:
            sql = f"SELECT * FROM {table} WHERE project = ? AND owner = ? ORDER BY updated_at DESC"
            params = (project, owner)
        with self.store.session() as connection:
            return connection.execute(sql, params).fetchall()

    def _audit(
        self,
        action: str,
        resource: str,
        *,
        project: str,
        owner: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store.add_audit_event(
            AuditEvent(
                actor="personal_os",
                action=action,
                resource=resource,
                project=project,
                metadata={"owner": owner, **(metadata or {})},
            )
        )


def preset_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "name": row["name"],
        "kind": row["kind"],
        "instructions": row["instructions"],
        "provider": row["provider"],
        "model": row["model"],
        "parameters": json.loads(row["parameters_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def signature_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "label": row["label"],
        "content": row["content"],
        "is_default": bool(row["is_default"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def draft_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "title": row["title"],
        "content": row["content"],
        "status": row["status"],
        "version": row["version"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def draft_version_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "draft_id": row["draft_id"],
        "version": row["version"],
        "content": row["content"],
        "created_at": row["created_at"],
    }


def note_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "title": row["title"],
        "body": row["body"],
        "tags": json.loads(row["tags_json"]),
        "due_at": row["due_at"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def gallery_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "title": row["title"],
        "media_type": row["media_type"],
        "source_path": row["source_path"],
        "content_hash": row["content_hash"],
        "tags": json.loads(row["tags_json"]),
        "favorite": bool(row["favorite"]),
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def compare_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "prompt": row["prompt"],
        "left_label": row["left_label"],
        "left_output": row["left_output"],
        "right_label": row["right_label"],
        "right_output": row["right_output"],
        "vote": row["vote"],
        "winner": row["winner"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def clean_tags(tags: list[str] | None) -> list[str]:
    cleaned = []
    for tag in tags or []:
        text = str(tag).strip().lower()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:25]


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def as_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
