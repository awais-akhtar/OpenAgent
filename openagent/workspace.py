"""Workspace objects for notes, docs, wiki pages, canvases, and templates."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import secrets
from typing import Any

from .models import AuditEvent, ContextEvent, utc_now
from .storage import OpenAgentStore


WORKSPACE_KINDS = {
    "folder",
    "document",
    "note",
    "wiki_page",
    "dashboard",
    "research_canvas",
    "whiteboard",
    "mind_map",
    "template",
}


class WorkspaceManager:
    def __init__(self, store: OpenAgentStore):
        self.store = store
        self.initialize()

    def initialize(self) -> None:
        self.store.initialize()

    def create_item(
        self,
        *,
        kind: str,
        title: str,
        project: str = "default",
        parent_id: str = "",
        content: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kind = kind.strip()
        if kind not in WORKSPACE_KINDS:
            raise ValueError(f"Unsupported workspace item kind '{kind}'.")
        title = title.strip()
        if not title:
            raise ValueError("Workspace item title is required.")
        project = project.strip() or "default"
        parent_id = parent_id.strip()
        if parent_id:
            parent = self.get_item(parent_id)
            if parent is None or parent["project"] != project:
                raise ValueError(f"Unknown parent workspace item '{parent_id}'.")
        item_id = unique_id("workspace", project, kind, title)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO workspace_items (
                    id, project, kind, title, parent_id, content,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    project,
                    kind,
                    title,
                    parent_id,
                    content,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                    now,
                ),
            )
        item = self.get_item(item_id) or {}
        self._index_item(item)
        self._audit("workspace.item.create", item_id, project=project, metadata={"kind": kind})
        return item

    def update_item(
        self,
        item_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        item = self.get_item(item_id)
        if item is None:
            raise ValueError(f"Unknown workspace item '{item_id}'.")
        updated = {
            "title": item["title"] if title is None else title.strip() or item["title"],
            "content": item["content"] if content is None else content,
            "metadata": item["metadata"] if metadata is None else metadata,
        }
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                UPDATE workspace_items
                SET title = ?, content = ?, metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    updated["title"],
                    updated["content"],
                    json.dumps(updated["metadata"], sort_keys=True),
                    now,
                    item_id,
                ),
            )
        refreshed = self.get_item(item_id) or {}
        self._index_item(refreshed)
        self._audit("workspace.item.update", item_id, project=refreshed["project"], metadata={"kind": refreshed["kind"]})
        return refreshed

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM workspace_items WHERE id = ?", (item_id,)).fetchone()
        return workspace_item_row(row) if row else None

    def list_items(
        self,
        *,
        project: str = "default",
        kind: str = "",
        parent_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if parent_id is not None:
            clauses.append("parent_id = ?")
            params.append(parent_id)
        params.append(max(1, min(limit, 500)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM workspace_items
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [workspace_item_row(row) for row in rows]

    def search(self, query: str, *, project: str = "default", limit: int = 50) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return self.list_items(project=project, limit=limit)
        pattern = f"%{query.lower()}%"
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM workspace_items
                WHERE project = ?
                  AND (
                    lower(title) LIKE ?
                    OR lower(content) LIKE ?
                    OR lower(metadata_json) LIKE ?
                  )
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (project, pattern, pattern, pattern, max(1, min(limit, 200))),
            ).fetchall()
        return [workspace_item_row(row) for row in rows]

    def instantiate_template(
        self,
        template_id: str,
        *,
        title: str,
        project: str = "default",
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        template = self.get_item(template_id)
        if template is None or template["kind"] != "template":
            raise ValueError(f"Unknown workspace template '{template_id}'.")
        variables = variables or {}
        content = template["content"]
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", str(value))
        metadata = {
            "template_id": template_id,
            "template_title": template["title"],
            "variables": variables,
        }
        return self.create_item(
            kind="document",
            title=title or f"{template['title']} instance",
            project=project or template["project"],
            content=content,
            metadata=metadata,
        )

    def dashboard(self, *, project: str = "default") -> dict[str, Any]:
        items = self.list_items(project=project, limit=500)
        by_kind = Counter(item["kind"] for item in items)
        recent = items[:10]
        pinned = [item for item in items if item["metadata"].get("pinned")][:10]
        return {
            "project": project,
            "total_items": len(items),
            "by_kind": dict(by_kind),
            "recent": recent,
            "pinned": pinned,
        }

    def canvas_model(self, *, project: str = "default") -> dict[str, Any]:
        canvas_items = [
            item
            for item in self.list_items(project=project, limit=500)
            if item["kind"] in {"research_canvas", "whiteboard", "mind_map", "dashboard"}
        ]
        nodes = []
        edges = []
        for index, item in enumerate(canvas_items):
            nodes.append(
                {
                    "id": item["id"],
                    "type": item["kind"],
                    "label": item["title"],
                    "x": 80 + (index % 4) * 220,
                    "y": 80 + (index // 4) * 140,
                    "metadata": item["metadata"],
                }
            )
            for linked_id in item["metadata"].get("links", []):
                edges.append({"from": item["id"], "to": str(linked_id)})
        return {"project": project, "nodes": nodes, "edges": edges}

    def _index_item(self, item: dict[str, Any]) -> None:
        if not item:
            return
        if item["kind"] == "folder":
            return
        text = item["content"] or item["title"]
        self.store.add_event(
            ContextEvent(
                title=f"Workspace {item['kind']}: {item['title']}",
                content=text[:4000],
                type=f"workspace_{item['kind']}",
                source="workspace",
                project=item["project"],
                metadata={"workspace_item_id": item["id"], "kind": item["kind"]},
            )
        )

    def _audit(
        self,
        action: str,
        resource: str,
        *,
        project: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store.add_audit_event(
            AuditEvent(
                actor="workspace",
                action=action,
                resource=f"workspace:{resource}",
                project=project,
                metadata=metadata or {},
            )
        )


def workspace_item_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "kind": row["kind"],
        "title": row["title"],
        "parent_id": row["parent_id"],
        "content": row["content"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:32]


def unique_id(*parts: str) -> str:
    return stable_id(*parts, utc_now(), secrets.token_hex(8))
