"""Phase 1 storage: the code graph round-trips through normalized SQLite, scoped by owner and project."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openagent.codebase import graph_store
from openagent.codebase.work_graph import build_code_knowledge_graph
from openagent.storage import OpenAgentStore


def _project_map(hashes: dict[str, str]) -> dict:
    return {
        "root": "/repo",
        "file_inventory": [
            {"path": path, "content_sha256": digest, "index_status": "fresh", "language": "python", "size_bytes": 10}
            for path, digest in sorted(hashes.items())
        ],
        "symbols": [],
    }


class GraphStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.store = OpenAgentStore(self.base / "data")
        with self.store.session() as connection:
            graph_store.migrate(connection)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _save(self, project: str, owner: str, graph: dict) -> str:
        with self.store.session() as connection:
            return graph_store.save_revision(connection, project, owner, graph)

    def test_graph_round_trips_with_the_same_nodes_and_edges(self) -> None:
        graph = build_code_knowledge_graph(_project_map({"a.py": "h1", "b.py": "h2"}))
        revision_id = self._save("proj-1", "alice", graph)
        with self.store.session() as connection:
            loaded = graph_store.load_revision(connection, "proj-1", "alice", revision_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(graph["revision"], loaded["revision"])
        self.assertEqual(sorted(node["id"] for node in graph["nodes"]), sorted(node["id"] for node in loaded["nodes"]))
        self.assertEqual(sorted(edge["id"] for edge in graph["edges"]), sorted(edge["id"] for edge in loaded["edges"]))

    def test_unchanged_graph_adds_no_rows(self) -> None:
        graph = build_code_knowledge_graph(_project_map({"a.py": "h1"}))
        first = self._save("proj-1", "alice", graph)
        second = self._save("proj-1", "alice", copy.deepcopy(graph))
        self.assertEqual(first, second)
        with self.store.session() as connection:
            count = connection.execute("SELECT COUNT(*) FROM codebase_graph_revisions").fetchone()[0]
        self.assertEqual(1, count)

    def test_another_owner_or_project_cannot_read_the_graph(self) -> None:
        graph = build_code_knowledge_graph(_project_map({"a.py": "h1"}))
        revision_id = self._save("proj-1", "alice", graph)
        with self.store.session() as connection:
            self.assertIsNone(graph_store.load_revision(connection, "proj-1", "mallory", revision_id))
            self.assertIsNone(graph_store.load_revision(connection, "proj-2", "alice", revision_id))
            self.assertIsNotNone(graph_store.load_revision(connection, "proj-1", "alice", revision_id))

    def test_latest_revision_is_the_most_recent_for_that_scope(self) -> None:
        self._save("proj-1", "alice", build_code_knowledge_graph(_project_map({"a.py": "h1"})))
        newest = self._save("proj-1", "alice", build_code_knowledge_graph(_project_map({"a.py": "h2"})))
        self._save("proj-1", "bob", build_code_knowledge_graph(_project_map({"z.py": "hz"})))
        with self.store.session() as connection:
            self.assertEqual(newest, graph_store.latest_revision_id(connection, "proj-1", "alice"))

    def test_migration_is_repeatable_and_records_its_version(self) -> None:
        with self.store.session() as connection:
            graph_store.migrate(connection)
            graph_store.migrate(connection)
            self.assertEqual(graph_store.GRAPH_SCHEMA_VERSION, graph_store.schema_version(connection))

    def test_stored_graph_survives_reopening_the_database(self) -> None:
        graph = build_code_knowledge_graph(_project_map({"a.py": "h1"}))
        revision_id = self._save("proj-1", "alice", graph)
        reopened = OpenAgentStore(self.base / "data")
        with reopened.session() as connection:
            loaded = graph_store.load_revision(connection, "proj-1", "alice", revision_id)
        self.assertEqual(graph["revision"], loaded["revision"])

    def test_unchanged_files_keep_their_node_ids_and_changes_are_reported(self) -> None:
        before = build_code_knowledge_graph(_project_map({"a.py": "h1", "gone.py": "g1"}))
        after = build_code_knowledge_graph(_project_map({"a.py": "h1-edited", "new.py": "n1"}))
        before_ids = {node["path"]: node["id"] for node in before["nodes"] if node["kind"] == "file"}
        after_ids = {node["path"]: node["id"] for node in after["nodes"] if node["kind"] == "file"}
        self.assertEqual(before_ids["a.py"], after_ids["a.py"], "a changed file keeps its node id")
        self.assertEqual(
            {"added": ["new.py"], "changed": ["a.py"], "deleted": ["gone.py"]},
            graph_store.changed_files(before, after),
        )

    def test_incremental_revision_copies_unchanged_rows_and_replaces_changed_scope(self) -> None:
        before = build_code_knowledge_graph(_project_map({"a.py": "h1", "gone.py": "g1"}))
        first_revision_id = self._save("proj-1", "alice", before)
        with self.store.session() as connection:
            previous = graph_store.load_revision(connection, "proj-1", "alice", first_revision_id)
        after = build_code_knowledge_graph(_project_map({"a.py": "h2", "new.py": "n1"}))
        with self.store.session() as connection, patch.object(
            graph_store, "_copy_unchanged_rows", wraps=graph_store._copy_unchanged_rows,
        ) as copied_rows:
            next_revision_id = graph_store.save_revision(
                connection, "proj-1", "alice", after, previous_graph=previous,
            )
            self.assertEqual(["nodes", "edges"], [call.args[1] for call in copied_rows.call_args_list])
            stored = graph_store.load_revision(connection, "proj-1", "alice", next_revision_id)
        self.assertEqual(after["revision"], stored["revision"])
        self.assertEqual(
            {node["id"]: node for node in after["nodes"]},
            {node["id"]: node for node in stored["nodes"]},
        )
        self.assertEqual(
            {edge["id"]: edge for edge in after["edges"]},
            {edge["id"]: edge for edge in stored["edges"]},
        )

    def test_incremental_revision_never_copies_from_another_owner(self) -> None:
        alice_graph = build_code_knowledge_graph(_project_map({"private.py": "alice"}))
        alice_revision_id = self._save("proj-1", "alice", alice_graph)
        with self.store.session() as connection:
            alice_previous = graph_store.load_revision(connection, "proj-1", "alice", alice_revision_id)
            bob_graph = build_code_knowledge_graph(_project_map({"public.py": "bob"}))
            bob_revision_id = graph_store.save_revision(
                connection, "proj-1", "bob", bob_graph, previous_graph=alice_previous,
            )
            bob_stored = graph_store.load_revision(connection, "proj-1", "bob", bob_revision_id)
        self.assertEqual(bob_graph["nodes"], bob_stored["nodes"])
        self.assertEqual({"public.py"}, {node.get("path") for node in bob_stored["nodes"] if node.get("kind") == "file"})


if __name__ == "__main__":
    unittest.main()
