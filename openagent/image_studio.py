"""Local image studio library, transforms, and audit records."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from pathlib import Path
import struct
from typing import Any
from uuid import uuid4

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class ImageStudio:
    """Owns local image assets and deterministic image operation records."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.media_root = self.store.data_dir / "image_studio"
        self.media_root.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS image_assets (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_image_assets_project_owner
                    ON image_assets(project, owner, updated_at DESC);

                CREATE TABLE IF NOT EXISTS image_operations (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    asset_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    output_path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'completed',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(asset_id) REFERENCES image_assets(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_image_operations_asset
                    ON image_operations(asset_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS image_albums (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS image_album_assets (
                    album_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(album_id, asset_id)
                );
                CREATE TABLE IF NOT EXISTS image_versions (
                    id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_image_versions_asset ON image_versions(asset_id, created_at DESC);
                """
            )

    def upload_asset(
        self,
        *,
        filename: str,
        content: bytes,
        project: str = "default",
        owner: str = "local",
        content_type: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not content:
            raise ValueError("image content is required")
        safe_name = safe_filename(filename or "image.bin")
        content_type = content_type or detect_content_type(safe_name, content)
        width, height = image_dimensions(content)
        asset_id = uuid4().hex
        now = utc_now()
        folder = self._asset_folder(project, owner)
        path = folder / f"{asset_id}-{safe_name}"
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO image_assets (
                    id, project, owner, filename, content_type, width, height, size_bytes,
                    sha256, local_path, tags_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    project,
                    owner,
                    safe_name,
                    content_type,
                    width,
                    height,
                    len(content),
                    digest,
                    str(path),
                    json.dumps(tags or []),
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO image_versions (id, asset_id, operation, local_path, sha256, width, height, created_at)
                VALUES (?, ?, 'upload', ?, ?, ?, ?, ?)
                """,
                (uuid4().hex, asset_id, str(path), digest, width, height, now),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="image-studio",
                action="image.upload",
                resource=f"image:{asset_id}",
                project=project,
                metadata={"owner": owner, "filename": safe_name, "sha256": digest, "size_bytes": len(content)},
            )
        )
        return self.get_asset(asset_id, owner=owner)

    def library(
        self,
        *,
        project: str = "default",
        owner: str = "",
        query: str = "",
        tag: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        clauses = ["project = ?"]
        args: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        if query:
            clauses.append("(filename LIKE ? OR metadata_json LIKE ?)")
            needle = f"%{query}%"
            args.extend([needle, needle])
        if tag:
            clauses.append("tags_json LIKE ?")
            args.append(f"%{tag}%")
        sql = "SELECT * FROM image_assets WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 500)))
        with self.store.session() as connection:
            rows = connection.execute(sql, args).fetchall()
        assets = [asset_row(row) for row in rows]
        return {
            "assets": assets,
            "count": len(assets),
            "privacy_boundary": "Image files stay in the configured local image_studio media directory.",
        }

    def get_asset(self, asset_id: str, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            row = self._asset_row(connection, asset_id, owner=owner)
        return asset_row(row)

    def operations(self, asset_id: str, *, owner: str = "") -> list[dict[str, Any]]:
        with self.store.session() as connection:
            self._asset_row(connection, asset_id, owner=owner)
            rows = connection.execute(
                "SELECT * FROM image_operations WHERE asset_id = ? ORDER BY created_at DESC",
                (asset_id,),
            ).fetchall()
        return [operation_row(row) for row in rows]

    def rotate(self, asset_id: str, *, degrees: int = 90, owner: str = "") -> dict[str, Any]:
        degrees = int(degrees) % 360
        if degrees not in {0, 90, 180, 270}:
            raise ValueError("degrees must be one of 0, 90, 180, 270")
        return self.transform(asset_id, "rotate", owner=owner, params={"degrees": degrees})

    def replace(
        self,
        asset_id: str,
        *,
        content: bytes,
        filename: str = "",
        content_type: str = "",
        owner: str = "",
    ) -> dict[str, Any]:
        if not content:
            raise ValueError("replacement image content is required")
        with self.store.session() as connection:
            row = self._asset_row(connection, asset_id, owner=owner)
            asset = asset_row(row)
            safe_name = safe_filename(filename or str(asset["filename"]))
            path = self._derived_path(Path(str(asset["local_path"])), "replace")
            path.write_bytes(content)
            width, height = image_dimensions(content)
            digest = hashlib.sha256(content).hexdigest()
            now = utc_now()
            connection.execute(
                """
                UPDATE image_assets SET filename = ?, content_type = ?, width = ?, height = ?, size_bytes = ?,
                    sha256 = ?, local_path = ?, updated_at = ?
                WHERE id = ?
                """,
                (safe_name, content_type or detect_content_type(safe_name, content), width, height, len(content), digest, str(path), now, asset_id),
            )
            connection.execute(
                "INSERT INTO image_versions (id, asset_id, operation, local_path, sha256, width, height, created_at) VALUES (?, ?, 'replace', ?, ?, ?, ?, ?)",
                (uuid4().hex, asset_id, str(path), digest, width, height, now),
            )
        updated = self.get_asset(asset_id, owner=owner)
        operation = self._record_operation(
            asset_id,
            project=str(updated["project"]),
            owner=str(updated["owner"]),
            operation="replace",
            params={"filename": updated["filename"], "sha256": updated["sha256"]},
            output_path=Path(str(updated["local_path"])),
        )
        self.store.add_audit_event(
            AuditEvent(
                actor="image-studio",
                action="image.replace",
                resource=f"image:{asset_id}",
                project=str(updated["project"]),
                metadata={"owner": updated["owner"], "sha256": updated["sha256"]},
            )
        )
        return {"asset": updated, "operation": operation}

    def upscale(self, asset_id: str, *, scale: int = 2, owner: str = "") -> dict[str, Any]:
        scale = max(1, min(int(scale), 8))
        return self.transform(asset_id, "upscale", owner=owner, params={"scale": scale})

    def create_album(self, name: str, *, project: str = "default", owner: str = "local", description: str = "") -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("album name is required")
        album_id = uuid4().hex
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO image_albums (id, project, owner, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (album_id, project, owner, name, description, now, now),
            )
        return {"id": album_id, "project": project, "owner": owner, "name": name, "description": description, "created_at": now}

    def albums(self, *, project: str = "default", owner: str = "") -> list[dict[str, Any]]:
        query = "SELECT a.*, COUNT(aa.asset_id) AS asset_count FROM image_albums a LEFT JOIN image_album_assets aa ON aa.album_id = a.id WHERE a.project = ?"
        values: list[Any] = [project]
        if owner:
            query += " AND a.owner = ?"
            values.append(owner)
        query += " GROUP BY a.id ORDER BY a.updated_at DESC"
        with self.store.session() as connection:
            rows = connection.execute(query, values).fetchall()
        return [{**dict(row), "asset_count": int(row["asset_count"])} for row in rows]

    def add_to_album(self, album_id: str, asset_id: str, *, owner: str = "") -> dict[str, Any]:
        asset = self.get_asset(asset_id, owner=owner)
        with self.store.session() as connection:
            album = connection.execute("SELECT * FROM image_albums WHERE id = ?", (album_id,)).fetchone()
            if album is None:
                raise KeyError("album not found")
            if owner and album["owner"] != owner:
                raise KeyError("album not found")
            connection.execute(
                "INSERT OR IGNORE INTO image_album_assets (album_id, asset_id, created_at) VALUES (?, ?, ?)",
                (album_id, asset_id, utc_now()),
            )
            connection.execute("UPDATE image_albums SET updated_at = ? WHERE id = ?", (utc_now(), album_id))
        return {"album_id": album_id, "asset": asset}

    def edit_metadata(
        self,
        asset_id: str,
        *,
        owner: str = "",
        filename: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        favorite: bool | None = None,
    ) -> dict[str, Any]:
        asset = self.get_asset(asset_id, owner=owner)
        new_metadata = {**asset["metadata"], **(metadata or {})}
        if favorite is not None:
            new_metadata["favorite"] = bool(favorite)
        clean_tags = list(dict.fromkeys(str(tag).strip().lower() for tag in (tags if tags is not None else asset["tags"]) if str(tag).strip()))
        with self.store.session() as connection:
            connection.execute(
                "UPDATE image_assets SET filename = ?, tags_json = ?, metadata_json = ?, updated_at = ? WHERE id = ?",
                (safe_filename(filename) if filename else asset["filename"], json.dumps(clean_tags), json.dumps(new_metadata, sort_keys=True), utc_now(), asset_id),
            )
        return self.get_asset(asset_id, owner=owner)

    def batch_tag(
        self,
        asset_ids: list[str],
        tags: list[str],
        *,
        owner: str = "",
        replace: bool = False,
    ) -> dict[str, Any]:
        clean_ids = list(dict.fromkeys(str(item).strip() for item in asset_ids if str(item).strip()))
        clean_tags = list(dict.fromkeys(str(tag).strip().lower() for tag in tags if str(tag).strip()))
        if not clean_ids:
            raise ValueError("asset_ids must contain at least one image.")
        if not clean_tags and not replace:
            raise ValueError("tags must contain at least one tag.")
        updated = []
        for asset_id in clean_ids:
            asset = self.get_asset(asset_id, owner=owner)
            next_tags = clean_tags if replace else list(dict.fromkeys([*asset["tags"], *clean_tags]))
            updated.append(self.edit_metadata(asset_id, owner=owner, tags=next_tags))
        project = updated[0]["project"]
        self.store.add_audit_event(
            AuditEvent(
                actor="image-studio",
                action="image.tags.batch",
                resource=f"images:{len(updated)}",
                project=project,
                metadata={"owner": owner, "asset_ids": clean_ids, "tags": clean_tags, "replace": replace},
            )
        )
        return {"updated": updated, "count": len(updated), "tags": clean_tags, "replace": replace}

    def deduplicate(self, *, project: str = "default", owner: str = "") -> dict[str, Any]:
        library = self.library(project=project, owner=owner, limit=500)["assets"]
        groups: dict[str, list[dict[str, Any]]] = {}
        for asset in library:
            groups.setdefault(asset["sha256"], []).append(asset)
        duplicates = [items for items in groups.values() if len(items) > 1]
        return {"groups": duplicates, "duplicate_assets": sum(len(items) - 1 for items in duplicates)}

    def versions(self, asset_id: str, *, owner: str = "") -> list[dict[str, Any]]:
        self.get_asset(asset_id, owner=owner)
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM image_versions WHERE asset_id = ? ORDER BY rowid DESC", (asset_id,)).fetchall()
        return [dict(row) for row in rows]

    def undo(self, asset_id: str, *, owner: str = "") -> dict[str, Any]:
        versions = self.versions(asset_id, owner=owner)
        if len(versions) < 2:
            raise ValueError("No prior image version is available.")
        current, previous = versions[0], versions[1]
        with self.store.session() as connection:
            connection.execute("DELETE FROM image_versions WHERE id = ?", (current["id"],))
            path = Path(previous["local_path"])
            connection.execute(
                "UPDATE image_assets SET local_path = ?, sha256 = ?, width = ?, height = ?, size_bytes = ?, updated_at = ? WHERE id = ?",
                (str(path), previous["sha256"], previous["width"], previous["height"], path.stat().st_size, utc_now(), asset_id),
            )
        return {"asset": self.get_asset(asset_id, owner=owner), "restored_version": previous}

    def transform(
        self,
        asset_id: str,
        operation: str,
        *,
        owner: str = "",
        params: dict[str, Any] | None = None,
        mask: bytes | None = None,
    ) -> dict[str, Any]:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps

        params = params or {}
        supported = {"rotate", "upscale", "inpaint", "remove-background", "restore", "style", "sharpen", "denoise"}
        if operation not in supported:
            raise ValueError(f"Unsupported image operation '{operation}'.")
        asset = self.get_asset(asset_id, owner=owner)
        source = Path(asset["local_path"])
        image = Image.open(source).convert("RGBA")
        if operation == "rotate":
            image = image.rotate(-int(params.get("degrees", 90)), expand=True)
        elif operation == "upscale":
            scale = max(1, min(int(params.get("scale", 2)), 8))
            image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
        elif operation == "inpaint":
            if not mask:
                raise ValueError("inpainting requires a local mask image")
            from io import BytesIO

            mask_image = Image.open(BytesIO(mask)).convert("L").resize(image.size)
            replacement = image.filter(ImageFilter.GaussianBlur(radius=max(2, int(params.get("radius", 8)))))
            image = Image.composite(replacement, image, mask_image)
        elif operation == "remove-background":
            pixels = image.load()
            background = image.getpixel((0, 0))[:3]
            threshold = max(5, min(int(params.get("threshold", 36)), 180))
            for y in range(image.height):
                for x in range(image.width):
                    color = pixels[x, y]
                    distance = sum((int(color[index]) - int(background[index])) ** 2 for index in range(3)) ** 0.5
                    if distance <= threshold:
                        pixels[x, y] = (*color[:3], 0)
        elif operation == "restore":
            image = ImageOps.autocontrast(image.convert("RGB")).filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.UnsharpMask(2, 150, 3)).convert("RGBA")
        elif operation == "style":
            preset = str(params.get("preset", "vivid"))
            if preset == "mono":
                image = ImageOps.grayscale(image).convert("RGBA")
            elif preset == "warm":
                rgb = image.convert("RGB")
                red, green, blue = rgb.split()
                image = Image.merge("RGB", (red.point(lambda v: min(255, int(v * 1.08))), green, blue.point(lambda v: int(v * 0.88)))).convert("RGBA")
            else:
                image = ImageEnhance.Color(ImageEnhance.Contrast(image,).enhance(1.08)).enhance(1.25)
        elif operation == "sharpen":
            image = image.filter(ImageFilter.UnsharpMask(2, 180, 2))
        elif operation == "denoise":
            image = image.filter(ImageFilter.MedianFilter(max(3, int(params.get("size", 3)) | 1)))
        output = self._derived_path(source, f"{operation}-{uuid4().hex[:8]}").with_suffix(".png")
        image.save(output, format="PNG", optimize=True)
        return self._commit_derivative(asset, operation, params, output, image.width, image.height)

    def _commit_derivative(
        self, asset: dict[str, Any], operation: str, params: dict[str, Any], output: Path, width: int, height: int
    ) -> dict[str, Any]:
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "UPDATE image_assets SET local_path = ?, content_type = 'image/png', width = ?, height = ?, size_bytes = ?, sha256 = ?, updated_at = ? WHERE id = ?",
                (str(output), width, height, output.stat().st_size, digest, now, asset["id"]),
            )
            connection.execute(
                "INSERT INTO image_versions (id, asset_id, operation, local_path, sha256, width, height, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (uuid4().hex, asset["id"], operation, str(output), digest, width, height, now),
            )
        operation_record = self._record_operation(
            asset["id"], project=asset["project"], owner=asset["owner"], operation=operation,
            params=params, output_path=output,
        )
        return {"asset": self.get_asset(asset["id"], owner=asset["owner"]), "operation": operation_record, "output_path": str(output)}

    def decode_base64(self, value: str) -> bytes:
        try:
            return base64.b64decode(value.encode("ascii"), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("content_base64 must be valid base64") from exc

    def _asset_folder(self, project: str, owner: str) -> Path:
        folder = self.media_root / safe_filename(project or "default") / safe_filename(owner or "local")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _derived_path(self, source: Path, label: str) -> Path:
        return source.with_name(f"{source.stem}-{label}{source.suffix or '.bin'}")

    def _record_operation(
        self,
        asset_id: str,
        *,
        project: str,
        owner: str,
        operation: str,
        params: dict[str, Any],
        output_path: Path,
    ) -> dict[str, Any]:
        operation_id = uuid4().hex
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO image_operations (
                    id, project, owner, asset_id, operation, params_json, output_path, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?)
                """,
                (operation_id, project, owner, asset_id, operation, json.dumps(params, sort_keys=True), str(output_path), utc_now()),
            )
            row = connection.execute("SELECT * FROM image_operations WHERE id = ?", (operation_id,)).fetchone()
        self.store.add_audit_event(
            AuditEvent(
                actor="image-studio",
                action=f"image.{operation}",
                resource=f"image:{asset_id}",
                project=project,
                metadata={"owner": owner, "operation_id": operation_id, "output_path": str(output_path)},
            )
        )
        return operation_row(row)

    def _asset_row(self, connection: Any, asset_id: str, *, owner: str = "") -> Any:
        if owner:
            row = connection.execute("SELECT * FROM image_assets WHERE id = ? AND owner = ?", (asset_id, owner)).fetchone()
        else:
            row = connection.execute("SELECT * FROM image_assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            raise KeyError("image asset not found")
        return row


def asset_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "filename": row["filename"],
        "content_type": row["content_type"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "size_bytes": int(row["size_bytes"]),
        "sha256": row["sha256"],
        "local_path": row["local_path"],
        "tags": json.loads(row["tags_json"] or "[]"),
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def operation_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "asset_id": row["asset_id"],
        "operation": row["operation"],
        "params": json.loads(row["params_json"] or "{}"),
        "output_path": row["output_path"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


def safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return cleaned.strip("._") or "asset"


def detect_content_type(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if content.startswith(b"\x89PNG\r\n\x1a\n") or lower.endswith(".png"):
        return "image/png"
    if content.startswith(b"\xff\xd8") or lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if content.startswith(b"GIF8") or lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def image_dimensions(content: bytes) -> tuple[int, int]:
    if content.startswith(b"\x89PNG\r\n\x1a\n") and len(content) >= 24:
        return struct.unpack(">II", content[16:24])
    if content.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(content):
            if content[index] != 0xFF:
                index += 1
                continue
            marker = content[index + 1]
            length = int.from_bytes(content[index + 2:index + 4], "big")
            if marker in {0xC0, 0xC1, 0xC2, 0xC3} and index + 9 < len(content):
                height = int.from_bytes(content[index + 5:index + 7], "big")
                width = int.from_bytes(content[index + 7:index + 9], "big")
                return width, height
            index += max(length + 2, 2)
    return 0, 0
