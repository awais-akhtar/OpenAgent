"""Versioned, normalized SQLite storage for the repository code graph (roadmap Phase 1).

Each saved graph is a revision: its nodes and edges are rows keyed by revision, and every
row carries the owner and CodeQ project it belongs to. Reads filter by both, so one
owner's graph is never returned to another, and a project's graph is never mixed with
a sibling project's.

The schema is additive. `migrate` only creates missing tables and records a version, so
databases written before this module existed keep working unchanged.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from uuid import uuid4

from ..models import utc_now

GRAPH_SCHEMA_VERSION = 1


def migrate(connection: sqlite3.Connection) -> None:
    """Create the graph tables if absent and record the schema version. Safe to run repeatedly."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS codebase_graph_schema (
            component TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS codebase_graph_revisions (
            id TEXT PRIMARY KEY,
            codebase_project_id TEXT NOT NULL,
            owner TEXT NOT NULL,
            revision TEXT NOT NULL,
            node_count INTEGER NOT NULL,
            edge_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (codebase_project_id, owner, revision)
        );
        CREATE INDEX IF NOT EXISTS idx_codebase_graph_revisions_scope
            ON codebase_graph_revisions(codebase_project_id, owner, created_at DESC);
        CREATE TABLE IF NOT EXISTS codebase_graph_nodes (
            revision_id TEXT NOT NULL REFERENCES codebase_graph_revisions(id) ON DELETE CASCADE,
            id TEXT NOT NULL,
            codebase_project_id TEXT NOT NULL,
            owner TEXT NOT NULL,
            kind TEXT NOT NULL,
            node_key TEXT NOT NULL,
            label TEXT NOT NULL,
            path TEXT NOT NULL DEFAULT '',
            content_sha256 TEXT NOT NULL DEFAULT '',
            data_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (revision_id, id)
        );
        CREATE INDEX IF NOT EXISTS idx_codebase_graph_nodes_path
            ON codebase_graph_nodes(codebase_project_id, owner, path);
        CREATE TABLE IF NOT EXISTS codebase_graph_edges (
            revision_id TEXT NOT NULL REFERENCES codebase_graph_revisions(id) ON DELETE CASCADE,
            id TEXT NOT NULL,
            codebase_project_id TEXT NOT NULL,
            owner TEXT NOT NULL,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relation TEXT NOT NULL,
            provenance TEXT NOT NULL DEFAULT '',
            confidence TEXT NOT NULL DEFAULT 'verified',
            PRIMARY KEY (revision_id, id)
        );
        CREATE INDEX IF NOT EXISTS idx_codebase_graph_edges_source
            ON codebase_graph_edges(revision_id, source);
        """
    )
    connection.execute(
        "INSERT INTO codebase_graph_schema (component, version, updated_at) VALUES ('code_graph', ?, ?) "
        "ON CONFLICT(component) DO UPDATE SET version = excluded.version, updated_at = excluded.updated_at",
        (GRAPH_SCHEMA_VERSION, utc_now()),
    )


def schema_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT version FROM codebase_graph_schema WHERE component = 'code_graph'").fetchone()
    return int(row["version"]) if row else 0


def save_revision(
    connection: sqlite3.Connection,
    project_id: str,
    owner: str,
    graph: dict[str, Any],
    *,
    previous_graph: dict[str, Any] | None = None,
) -> str:
    """Store a graph as a revision and return its row id.

    Saving the same graph revision twice returns the existing row, so an unchanged
    repository adds no rows.
    """
    revision = str(graph.get("revision", ""))
    if not revision:
        raise ValueError("graph has no revision hash")
    existing = connection.execute(
        "SELECT id FROM codebase_graph_revisions WHERE codebase_project_id = ? AND owner = ? AND revision = ?",
        (project_id, owner, revision),
    ).fetchone()
    if existing is not None:
        return str(existing["id"])
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    previous_revision_id = str((previous_graph or {}).get("revision_id", ""))
    previous_is_scoped = bool(previous_revision_id) and connection.execute(
        "SELECT 1 FROM codebase_graph_revisions WHERE id = ? AND codebase_project_id = ? AND owner = ?",
        (previous_revision_id, project_id, owner),
    ).fetchone() is not None
    previous_nodes = {
        str(node.get("id")): node for node in (previous_graph or {}).get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    } if previous_is_scoped else {}
    previous_edges = {
        str(edge.get("id")): edge for edge in (previous_graph or {}).get("edges", [])
        if isinstance(edge, dict) and edge.get("id")
    } if previous_is_scoped else {}
    current_nodes = {str(node["id"]): node for node in nodes}
    current_edges = {str(edge["id"]): edge for edge in edges}
    unchanged_node_ids = {
        node_id for node_id, node in current_nodes.items()
        if previous_nodes.get(node_id) == node
    }

    def edge_storage_value(edge: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(edge.get(key, default)) for key, default in (
            ("source", ""), ("target", ""), ("relation", ""),
            ("provenance", ""), ("confidence", "verified"),
        ))

    unchanged_edge_ids = {
        edge_id for edge_id, edge in current_edges.items()
        if edge_id in previous_edges and edge_storage_value(previous_edges[edge_id]) == edge_storage_value(edge)
    }
    revision_id = uuid4().hex
    try:
        connection.execute(
            "INSERT INTO codebase_graph_revisions (id, codebase_project_id, owner, revision, node_count, edge_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (revision_id, project_id, owner, revision, len(nodes), len(edges), utc_now()),
        )
    except sqlite3.IntegrityError:
        # A concurrent task saved the same graph between our check and our insert. That row is
        # the identical revision, so reuse it rather than failing the run.
        existing = connection.execute(
            "SELECT id FROM codebase_graph_revisions WHERE codebase_project_id = ? AND owner = ? AND revision = ?",
            (project_id, owner, revision),
        ).fetchone()
        if existing is None:
            raise
        return str(existing["id"])
    if previous_is_scoped:
        _copy_unchanged_rows(connection, "nodes", revision_id, previous_revision_id, project_id, owner, unchanged_node_ids)
        _copy_unchanged_rows(connection, "edges", revision_id, previous_revision_id, project_id, owner, unchanged_edge_ids)
    connection.executemany(
        "INSERT INTO codebase_graph_nodes (revision_id, id, codebase_project_id, owner, kind, node_key, label, path, content_sha256, data_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                revision_id, str(node["id"]), project_id, owner, str(node.get("kind", "")), str(node.get("key", "")),
                str(node.get("label", "")), str(node.get("path", "")), str(node.get("content_sha256", "")),
                json.dumps(node, sort_keys=True, default=str),
            )
            for node in nodes if str(node["id"]) not in unchanged_node_ids
        ],
    )
    connection.executemany(
        "INSERT INTO codebase_graph_edges (revision_id, id, codebase_project_id, owner, source, target, relation, provenance, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                revision_id, str(edge["id"]), project_id, owner, str(edge["source"]), str(edge["target"]),
                str(edge.get("relation", "")), str(edge.get("provenance", "")), str(edge.get("confidence", "verified")),
            )
            for edge in edges if str(edge["id"]) not in unchanged_edge_ids
        ],
    )
    return revision_id


def _copy_unchanged_rows(
    connection: sqlite3.Connection,
    entity: str,
    revision_id: str,
    previous_revision_id: str,
    project_id: str,
    owner: str,
    ids: set[str],
) -> None:
    """Copy unchanged records into an immutable revision inside SQLite.

    Changed and new records are serialized from the fresh graph; deleted records are
    omitted. The previous revision is already scope-checked by ``save_revision``.
    """
    if not ids:
        return
    table, columns = {
        "nodes": (
            "codebase_graph_nodes",
            "id, codebase_project_id, owner, kind, node_key, label, path, content_sha256, data_json",
        ),
        "edges": (
            "codebase_graph_edges",
            "id, codebase_project_id, owner, source, target, relation, provenance, confidence",
        ),
    }[entity]
    id_table = f"graph_keep_{entity}_{uuid4().hex}"
    connection.execute(f"CREATE TEMP TABLE {id_table} (id TEXT PRIMARY KEY) WITHOUT ROWID")
    try:
        connection.executemany(f"INSERT INTO {id_table} (id) VALUES (?)", [(item,) for item in sorted(ids)])
        connection.execute(
            f"INSERT INTO {table} (revision_id, {columns}) "
            f"SELECT ?, old.{columns.replace(', ', ', old.')} FROM {table} AS old "
            f"JOIN {id_table} AS keep ON keep.id = old.id "
            "WHERE old.revision_id = ? AND old.codebase_project_id = ? AND old.owner = ?",
            (revision_id, previous_revision_id, project_id, owner),
        )
    finally:
        connection.execute(f"DROP TABLE IF EXISTS {id_table}")


def latest_revision_id(connection: sqlite3.Connection, project_id: str, owner: str) -> str | None:
    row = connection.execute(
        "SELECT id FROM codebase_graph_revisions WHERE codebase_project_id = ? AND owner = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (project_id, owner),
    ).fetchone()
    return str(row["id"]) if row else None


def load_revision(connection: sqlite3.Connection, project_id: str, owner: str, revision_id: str | None = None) -> dict[str, Any] | None:
    """Return a stored graph in the same shape `build_code_knowledge_graph` produces.

    Returns None when the revision does not exist for this owner and project.
    """
    target = revision_id or latest_revision_id(connection, project_id, owner)
    if not target:
        return None
    head = connection.execute(
        "SELECT revision FROM codebase_graph_revisions WHERE id = ? AND codebase_project_id = ? AND owner = ?",
        (target, project_id, owner),
    ).fetchone()
    if head is None:
        return None
    nodes = [
        json.loads(row["data_json"] or "{}")
        for row in connection.execute(
            "SELECT data_json FROM codebase_graph_nodes WHERE revision_id = ? AND codebase_project_id = ? AND owner = ? ORDER BY id",
            (target, project_id, owner),
        ).fetchall()
    ]
    edges = [
        {"id": row["id"], "source": row["source"], "target": row["target"], "relation": row["relation"],
         "provenance": row["provenance"], "confidence": row["confidence"]}
        for row in connection.execute(
            "SELECT * FROM codebase_graph_edges WHERE revision_id = ? AND codebase_project_id = ? AND owner = ? ORDER BY id",
            (target, project_id, owner),
        ).fetchall()
    ]
    return {"revision": str(head["revision"]), "revision_id": target, "nodes": nodes, "edges": edges}


def changed_files(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, list[str]]:
    """Files added, changed or deleted between two graphs, by content hash.

    Only file nodes are compared. Unchanged files keep their node ids, so a refresh needs to
    re-derive only the nodes and edges of the paths listed here.
    """
    def files(graph: dict[str, Any] | None) -> dict[str, str]:
        if not graph:
            return {}
        return {
            str(node.get("path", "")): str(node.get("content_sha256", ""))
            for node in graph.get("nodes", [])
            if isinstance(node, dict) and node.get("kind") == "file" and node.get("path")
        }

    before, after = files(previous), files(current)
    return {
        "added": sorted(path for path in after if path not in before),
        "changed": sorted(path for path in after if path in before and before[path] != after[path]),
        "deleted": sorted(path for path in before if path not in after),
    }


def refresh_scope(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    """What must be re-derived between two graph revisions, and nothing more.

    `changed_node_ids` lists every node that is new, removed or whose data differs. Unchanged
    files keep their nodes, so they are absent from that list. `neighbour_paths` are files
    linked to a changed file in the current graph: their outgoing references may need a fresh
    look, but their own content did not change.
    """
    changes = changed_files(previous, current)
    changed_paths = set(changes["added"]) | set(changes["changed"]) | set(changes["deleted"])
    previous_nodes = {str(node["id"]): node for node in (previous or {}).get("nodes", []) if isinstance(node, dict)}
    current_nodes = {str(node["id"]): node for node in current.get("nodes", []) if isinstance(node, dict)}
    changed_node_ids = sorted(
        node_id for node_id in set(previous_nodes) | set(current_nodes)
        if previous_nodes.get(node_id) != current_nodes.get(node_id)
    )
    path_of = {node_id: str(node.get("path", "")) for node_id, node in current_nodes.items() if node.get("path")}
    previous_path_of = {
        str(node["id"]): str(node.get("path", ""))
        for node in (previous or {}).get("nodes", [])
        if isinstance(node, dict) and node.get("id") and node.get("path")
    }
    current_file_paths = {
        str(node.get("path", "")) for node in current_nodes.values()
        if node.get("kind") == "file" and node.get("path")
    }
    neighbours: set[str] = set()
    # Inspect old and new edges: when an edit removes an import/reference, the
    # former neighbor still needs a refresh to clear the stale relationship.
    for graph, paths in ((previous or {}, previous_path_of), (current, path_of)):
        for edge in graph.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source, target = str(edge.get("source", "")), str(edge.get("target", ""))
            source_path, target_path = paths.get(source, ""), paths.get(target, "")
            if source_path in changed_paths and target_path in current_file_paths:
                neighbours.add(target_path)
            if target_path in changed_paths and source_path in current_file_paths:
                neighbours.add(source_path)
    return {
        "changed_files": changes,
        "changed_node_ids": changed_node_ids,
        "neighbour_paths": sorted(neighbours - changed_paths),
    }
